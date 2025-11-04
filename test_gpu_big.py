#!/usr/bin/env python3
"""
Test larger batch_sizes to push GPU limits

This script tests batch_sizes: 1200, 1600, 2000
to find the true maximum for the 24GB GPU.
"""

import requests
import json
import time
import sys
import subprocess


def update_batch_size(batch_size):
    """Update batch_size in container and restart service"""
    print(f"\nUpdating batch_size to {batch_size}...")
    
    try:
        # Update batch_size
        result = subprocess.run([
            "docker", "exec", "heimdall-training",
            "sed", "-i",
            f"s/batch_size = min([0-9]*, num_samples)/batch_size = min({batch_size}, num_samples)/",
            "/app/src/data/synthetic_generator.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"  ❌ Failed: {result.stderr}")
            return False
        
        # Verify
        result = subprocess.run([
            "docker", "exec", "heimdall-training",
            "grep", "batch_size = min", "/app/src/data/synthetic_generator.py"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            matching = [line for line in result.stdout.split('\n') if 'batch_size = min' in line]
            if matching:
                print(f"  ✅ Verified: {matching[0].strip()}")
        
        # Restart
        print(f"  Restarting service...")
        subprocess.run(["docker", "compose", "restart", "training"], 
                      capture_output=True, timeout=60)
        
        # Wait for health
        for i in range(20):
            time.sleep(3)
            try:
                response = requests.get("http://localhost:8002/health", timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ Service healthy (took {(i+1)*3}s)")
                    return True
            except:
                pass
        
        print(f"  ⚠️  Timeout waiting for health")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_gpu_memory():
    """Check GPU memory"""
    try:
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            total, used, free = result.stdout.strip().split(',')
            return {
                "total": int(total),
                "used": int(used),
                "free": int(free),
                "used_percent": (int(used) / int(total)) * 100
            }
    except:
        pass
    return None


def test_batch_size(batch_size, base_url="http://localhost:8002"):
    """Test a specific batch_size"""
    
    print(f"\n{'='*80}")
    print(f"Testing batch_size={batch_size}")
    print('='*80)
    
    num_samples = batch_size
    
    payload = {
        "name": f"test_big_batch{batch_size}",
        "description": f"Test large batch_size={batch_size}",
        "num_samples": num_samples,
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
        # Submit job
        print(f"Submitting job...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/v1/training/synthetic/generate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        job_info = response.json()
        job_id = job_info["job_id"]
        
        print(f"  ✅ Job: {job_id}")
        
        # Poll with timeout
        max_wait = 300  # 5 minutes
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait:
                return {
                    "success": False,
                    "error": "timeout",
                    "batch_size": batch_size
                }
            
            time.sleep(5)
            
            try:
                status_response = requests.get(
                    f"{base_url}/v1/training/synthetic/jobs/{job_id}",
                    timeout=10
                )
                status_response.raise_for_status()
                status_info = status_response.json()
            except Exception as e:
                print(f"  ⚠️  Status check failed: {e}")
                continue
            
            status = status_info["status"]
            progress = status_info.get("current_progress", 0)
            total = status_info.get("total_progress", num_samples)
            
            print(f"  [{status}] {progress}/{total} ({progress/total*100 if total > 0 else 0:.1f}%) - {elapsed:.1f}s")
            
            if status in ["completed", "failed", "cancelled"]:
                break
        
        end_time = time.time()
        total_elapsed = end_time - start_time
        
        if status == "failed":
            error_msg = status_info.get("error", "Unknown error")
            print(f"\n  ❌ Failed: {error_msg}")
            
            # Check for OOM
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print(f"  💥 GPU OUT OF MEMORY!")
                return {
                    "success": False,
                    "error": "oom",
                    "batch_size": batch_size,
                    "error_detail": error_msg
                }
            
            return {
                "success": False,
                "error": "failed",
                "batch_size": batch_size,
                "error_detail": error_msg
            }
        
        if status != "completed":
            return {
                "success": False,
                "error": status,
                "batch_size": batch_size
            }
        
        # Success
        samples_generated = status_info.get("current_progress", num_samples)
        throughput = samples_generated / total_elapsed
        time_per_sample = total_elapsed / samples_generated if samples_generated > 0 else 0
        
        print(f"\n  ✅ SUCCESS!")
        print(f"    Samples: {samples_generated}")
        print(f"    Time: {total_elapsed:.2f}s")
        print(f"    Throughput: {throughput:.2f} samples/sec")
        print(f"    Per sample: {time_per_sample*1000:.1f}ms")
        
        return {
            "success": True,
            "batch_size": batch_size,
            "samples": samples_generated,
            "time": total_elapsed,
            "throughput": throughput,
            "time_per_sample": time_per_sample
        }
        
    except Exception as e:
        print(f"\n  ❌ Exception: {e}")
        return {
            "success": False,
            "error": "exception",
            "batch_size": batch_size,
            "error_detail": str(e)
        }


def main():
    print()
    print("="*80)
    print("GPU BATCH SIZE STRESS TEST - LARGE BATCHES")
    print("="*80)
    print()
    print("Testing batch sizes: 1200, 1600, 2000")
    print("Goal: Find the true maximum for 24GB GPU")
    print()
    
    # Check initial GPU state
    gpu = check_gpu_memory()
    if gpu:
        print(f"GPU Memory (initial):")
        print(f"  Total: {gpu['total']} MB")
        print(f"  Used: {gpu['used']} MB ({gpu['used_percent']:.1f}%)")
        print(f"  Free: {gpu['free']} MB")
    
    # Test progressively larger batches
    batch_sizes = [1200, 1600, 2000]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'#'*80}")
        print(f"# TEST {len(results)+1}/{len(batch_sizes)}: batch_size={batch_size}")
        print(f"{'#'*80}")
        
        # Update config
        if not update_batch_size(batch_size):
            print(f"\n⚠️  Config update failed, stopping")
            break
        
        # Check GPU before
        gpu_before = check_gpu_memory()
        if gpu_before:
            print(f"\nGPU before: {gpu_before['used']} MB ({gpu_before['used_percent']:.1f}%)")
        
        # Run test
        result = test_batch_size(batch_size)
        results.append(result)
        
        # Check GPU after
        gpu_after = check_gpu_memory()
        if gpu_after:
            print(f"GPU after: {gpu_after['used']} MB ({gpu_after['used_percent']:.1f}%)")
        
        # Stop on OOM
        if not result["success"] and result.get("error") == "oom":
            print(f"\n💥 Hit GPU limit at batch_size={batch_size}")
            break
        
        # Stop on any failure
        if not result["success"]:
            print(f"\n⚠️  Test failed, stopping")
            break
        
        # Wait between tests
        if batch_size < batch_sizes[-1]:
            print(f"\n⏸️  Cooling down 15s...")
            time.sleep(15)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("STRESS TEST SUMMARY")
    print('='*80)
    print()
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if successful:
        print("✅ Successful Tests:\n")
        print(f"{'Batch':<10} {'Samples':<10} {'Time':<10} {'Throughput':<15} {'ms/sample':<12}")
        print('-'*80)
        for r in successful:
            print(f"{r['batch_size']:<10} {r['samples']:<10} {r['time']:<10.2f} "
                  f"{r['throughput']:<15.2f} {r['time_per_sample']*1000:<12.1f}")
        
        best = max(successful, key=lambda x: x['throughput'])
        print()
        print(f"🏆 Best: batch_size={best['batch_size']} at {best['throughput']:.2f} samples/sec")
        
        max_batch = max([r['batch_size'] for r in successful])
        recommended = int(max_batch * 0.85)
        print()
        print(f"💡 Recommendation:")
        print(f"   Maximum tested: {max_batch}")
        print(f"   Production safe: {recommended} (85% of max)")
        print()
    
    if failed:
        print("❌ Failed Tests:")
        for r in failed:
            print(f"   batch_size={r['batch_size']}: {r.get('error', 'unknown')}")
        print()
    
    # Final GPU
    gpu_final = check_gpu_memory()
    if gpu_final:
        print(f"GPU Memory (final): {gpu_final['used']} MB ({gpu_final['used_percent']:.1f}%)")
    
    print('='*80)
    
    return len(successful) > 0


if __name__ == "__main__":
    print()
    print("🚀 Starting large batch stress test")
    print()
    
    success = main()
    
    sys.exit(0 if success else 1)
