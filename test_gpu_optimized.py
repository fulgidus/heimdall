#!/usr/bin/env python3
"""
Fine-grained test between batch_size 800 and 1200

Tests: 900, 1000, 1100 to find the exact limit
"""

import requests
import time
import sys
import subprocess


def update_batch_size(batch_size):
    """Update batch_size and restart"""
    print(f"\nSetting batch_size={batch_size}...")
    
    try:
        subprocess.run([
            "docker", "exec", "heimdall-training",
            "sed", "-i",
            f"s/batch_size = min([0-9]*, num_samples)/batch_size = min({batch_size}, num_samples)/",
            "/app/src/data/synthetic_generator.py"
        ], capture_output=True, timeout=10)
        
        subprocess.run(["docker", "compose", "restart", "training"], 
                      capture_output=True, timeout=60)
        
        for i in range(20):
            time.sleep(3)
            try:
                if requests.get("http://localhost:8002/health", timeout=5).status_code == 200:
                    print(f"  ✅ Ready")
                    return True
            except:
                pass
        return True
    except:
        return False


def check_gpu():
    """Check GPU memory"""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            return int(used), int(total)
    except:
        pass
    return None, None


def test_batch(batch_size):
    """Test a batch size"""
    print(f"\n{'='*70}")
    print(f"Testing batch_size={batch_size}")
    print('='*70)
    
    payload = {
        "name": f"test_{batch_size}",
        "num_samples": batch_size,
        "frequency_mhz": 145.0,
        "tx_power_dbm": 50.0,
        "min_snr_db": 5.0,
        "min_receivers": 3,
        "max_gdop": 100.0,
        "dataset_type": "feature_based",
        "use_random_receivers": True,
        "seed": 42 + batch_size
    }
    
    try:
        start = time.time()
        resp = requests.post("http://localhost:8002/v1/training/synthetic/generate", 
                            json=payload, timeout=30)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        
        print(f"Job: {job_id}")
        
        max_wait = 180  # 3 minutes
        while True:
            elapsed = time.time() - start
            if elapsed > max_wait:
                print(f"  ⏱️  Timeout")
                return {"success": False, "error": "timeout", "batch_size": batch_size}
            
            time.sleep(5)
            
            try:
                status = requests.get(f"http://localhost:8002/v1/training/synthetic/jobs/{job_id}", 
                                     timeout=10).json()
            except:
                continue
            
            st = status["status"]
            prog = status.get("current_progress", 0)
            total = status.get("total_progress", batch_size)
            
            print(f"  [{st}] {prog}/{total} - {elapsed:.0f}s")
            
            if st in ["completed", "failed", "cancelled"]:
                break
        
        total_time = time.time() - start
        
        if st == "failed":
            err = status.get("error", "Unknown")
            print(f"  ❌ Failed: {err[:100]}")
            
            if "memory" in err.lower() or "cuda" in err.lower():
                return {"success": False, "error": "oom", "batch_size": batch_size}
            
            return {"success": False, "error": "failed", "batch_size": batch_size}
        
        if st != "completed":
            return {"success": False, "error": st, "batch_size": batch_size}
        
        samples = status.get("current_progress", batch_size)
        throughput = samples / total_time
        
        print(f"  ✅ {samples} samples in {total_time:.1f}s = {throughput:.2f} samples/sec")
        
        return {
            "success": True,
            "batch_size": batch_size,
            "samples": samples,
            "time": total_time,
            "throughput": throughput
        }
        
    except Exception as e:
        print(f"  ❌ {e}")
        return {"success": False, "error": str(e), "batch_size": batch_size}


def main():
    print("\n" + "="*70)
    print("FINE-GRAINED BATCH SIZE TEST (800-1200)")
    print("="*70)
    print("\nTesting: 900, 1000, 1100")
    
    used, total = check_gpu()
    if used:
        print(f"\nGPU: {used}/{total} MB ({used/total*100:.1f}%)")
    
    batch_sizes = [900, 1000, 1100]
    results = []
    
    for bs in batch_sizes:
        print(f"\n{'#'*70}")
        print(f"# TEST {len(results)+1}/{len(batch_sizes)}: batch_size={bs}")
        print(f"{'#'*70}")
        
        if not update_batch_size(bs):
            print("Config update failed")
            break
        
        used_before, _ = check_gpu()
        if used_before:
            print(f"GPU before: {used_before} MB")
        
        result = test_batch(bs)
        results.append(result)
        
        used_after, total = check_gpu()
        if used_after:
            print(f"GPU after: {used_after} MB ({used_after/total*100:.1f}%)")
        
        if not result["success"]:
            if result.get("error") == "oom":
                print(f"\n💥 OOM at batch_size={bs}")
                break
            print(f"\n⚠️  Failed, stopping")
            break
        
        if bs < batch_sizes[-1]:
            print(f"\n⏸️  Wait 10s...")
            time.sleep(10)
    
    print(f"\n\n{'='*70}")
    print("RESULTS")
    print('='*70)
    
    successful = [r for r in results if r["success"]]
    
    if successful:
        print("\n✅ Successful:\n")
        print(f"{'Batch':<8} {'Samples':<8} {'Time':<8} {'Throughput':<12}")
        print('-'*70)
        for r in successful:
            print(f"{r['batch_size']:<8} {r['samples']:<8} {r['time']:<8.1f} {r['throughput']:<12.2f}")
        
        best = max(successful, key=lambda x: x['throughput'])
        max_bs = max([r['batch_size'] for r in successful])
        
        print(f"\n🏆 Best: {best['batch_size']} @ {best['throughput']:.2f} samples/sec")
        print(f"💡 Max safe: {max_bs}")
        print(f"💡 Recommended: {int(max_bs * 0.9)} (90% of max)")
    
    failed = [r for r in results if not r["success"]]
    if failed:
        print("\n❌ Failed:")
        for r in failed:
            print(f"   {r['batch_size']}: {r.get('error')}")
    
    print('\n' + '='*70)
    
    return len(successful) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
