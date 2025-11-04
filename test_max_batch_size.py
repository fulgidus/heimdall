#!/usr/bin/env python3
"""
Test maximum batch_size for GPU processing

This script progressively tests larger batch sizes to find the hardware limit.
It will test: 200 (baseline), 400, 600, 800

For each batch size:
1. Update configuration in container
2. Restart training service
3. Submit test job
4. Measure throughput and memory usage
5. Stop if OOM error occurs
"""

import requests
import json
import time
import sys
import subprocess


def update_batch_size(batch_size):
    """Update batch_size in container and restart service"""
    print(f"Updating batch_size to {batch_size}...")
    
    try:
        # Update batch_size in synthetic_generator.py
        result = subprocess.run([
            "docker", "exec", "heimdall-training",
            "sed", "-i",
            f"s/batch_size = min([0-9]*, num_samples)/batch_size = min({batch_size}, num_samples)/",
            "/app/src/data/synthetic_generator.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"  ❌ Failed to update batch_size: {result.stderr}")
            return False
        
        print(f"  ✅ Updated batch_size in synthetic_generator.py")
        
        # Verify the change
        result = subprocess.run([
            "docker", "exec", "heimdall-training",
            "grep", "batch_size = min", "/app/src/data/synthetic_generator.py"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            matching_lines = [line for line in result.stdout.split('\n') if 'batch_size = min' in line]
            if matching_lines:
                print(f"  ✅ Verified: {matching_lines[0].strip()}")
        
        # Restart training service
        print(f"  Restarting training service...")
        result = subprocess.run([
            "docker", "compose", "restart", "training"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"  ❌ Failed to restart service: {result.stderr}")
            return False
        
        print(f"  ✅ Service restarted")
        
        # Wait for service to be healthy using external API
        print(f"  Waiting for service to be healthy...")
        for i in range(20):
            time.sleep(3)
            try:
                response = requests.get("http://localhost:8002/health", timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ Service is healthy (took {(i+1)*3}s)")
                    return True
            except:
                pass  # Keep trying
        
        print(f"  ⚠️  Service did not become healthy in 60s, continuing anyway...")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout during configuration update")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_batch_size(batch_size, base_url="http://localhost:8002"):
    """Test a specific batch_size with actual job submission"""
    
    print(f"\n{'='*80}")
    print(f"Testing batch_size={batch_size}")
    print('='*80)
    
    # Use the batch_size as num_samples to test one full batch
    num_samples = batch_size
    
    payload = {
        "name": f"test_batch{batch_size}",
        "description": f"Test maximum batch_size={batch_size}",
        "num_samples": num_samples,
        "frequency_mhz": 145.0,
        "tx_power_dbm": 50.0,
        "min_snr_db": 5.0,
        "min_receivers": 3,
        "max_gdop": 100.0,
        "dataset_type": "feature_based",
        "use_random_receivers": True,
        "seed": 42 + batch_size  # Different seed per test
    }
    
    print(f"Configuration:")
    print(f"  - Samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Expected batches: 1")
    print()
    
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
        
        print(f"  ✅ Job submitted: {job_id}")
        print(f"  Polling for completion...")
        
        # Poll for completion with timeout
        max_wait_time = 300  # 5 minutes max
        poll_interval = 5
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                print(f"\n  ⏱️  Timeout after {max_wait_time}s")
                return {
                    "success": False,
                    "error": "timeout",
                    "batch_size": batch_size
                }
            
            time.sleep(poll_interval)
            
            try:
                status_response = requests.get(
                    f"{base_url}/v1/training/synthetic/jobs/{job_id}",
                    timeout=10
                )
                status_response.raise_for_status()
                status_info = status_response.json()
            except Exception as e:
                print(f"  ⚠️  Failed to get status: {e}")
                continue
            
            status = status_info["status"]
            progress = status_info.get("current_progress", 0)
            total = status_info.get("total_progress", num_samples)
            
            print(f"    Status: {status} - Progress: {progress}/{total} ({(progress/total*100 if total > 0 else 0):.1f}%) - Elapsed: {elapsed:.1f}s")
            
            if status in ["completed", "failed", "cancelled"]:
                break
        
        end_time = time.time()
        total_elapsed = end_time - start_time
        
        if status == "failed":
            error_msg = status_info.get("error", "Unknown error")
            print(f"\n  ❌ Job failed: {error_msg}")
            
            # Check for OOM error
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                print(f"  💥 GPU OUT OF MEMORY - Maximum batch_size exceeded!")
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
            print(f"\n  ⚠️  Job ended with status: {status}")
            return {
                "success": False,
                "error": status,
                "batch_size": batch_size
            }
        
        # Success! Calculate metrics
        samples_generated = status_info.get("current_progress", num_samples)
        throughput = samples_generated / total_elapsed
        time_per_sample = total_elapsed / samples_generated if samples_generated > 0 else 0
        
        print(f"\n  ✅ SUCCESS!")
        print(f"    - Samples generated: {samples_generated}")
        print(f"    - Total time: {total_elapsed:.2f}s")
        print(f"    - Throughput: {throughput:.2f} samples/sec")
        print(f"    - Time per sample: {time_per_sample*1000:.1f}ms")
        
        return {
            "success": True,
            "batch_size": batch_size,
            "samples": samples_generated,
            "time": total_elapsed,
            "throughput": throughput,
            "time_per_sample": time_per_sample
        }
        
    except requests.exceptions.Timeout:
        print(f"\n  ❌ Request timeout")
        return {
            "success": False,
            "error": "request_timeout",
            "batch_size": batch_size
        }
    
    except requests.exceptions.RequestException as e:
        print(f"\n  ❌ Request failed: {e}")
        error_detail = str(e)
        
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
            except:
                error_detail = e.response.text
        
        return {
            "success": False,
            "error": "request_error",
            "batch_size": batch_size,
            "error_detail": error_detail
        }
    
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": "exception",
            "batch_size": batch_size,
            "error_detail": str(e)
        }


def check_gpu_memory():
    """Check available GPU memory using nvidia-smi"""
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
    except Exception as e:
        print(f"  ⚠️  Could not check GPU memory: {e}")
    
    return None


def main():
    print()
    print("="*80)
    print("GPU BATCH SIZE MAXIMUM TEST")
    print("="*80)
    print()
    print("This test will progressively increase batch_size to find your GPU limit.")
    print("Testing batch sizes: 200 (baseline), 400, 600, 800")
    print()
    
    # Check GPU memory before starting
    gpu_mem = check_gpu_memory()
    if gpu_mem:
        print(f"GPU Memory (before test):")
        print(f"  - Total: {gpu_mem['total']} MB")
        print(f"  - Used: {gpu_mem['used']} MB ({gpu_mem['used_percent']:.1f}%)")
        print(f"  - Free: {gpu_mem['free']} MB")
        print()
    
    # Test batch sizes
    batch_sizes = [200, 400, 600, 800]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'#'*80}")
        print(f"# TEST {len(results)+1}/{len(batch_sizes)}: batch_size={batch_size}")
        print(f"{'#'*80}\n")
        
        # Update configuration
        if not update_batch_size(batch_size):
            print(f"\n⚠️  Failed to update configuration, stopping tests")
            break
        
        # Check GPU memory before test
        gpu_mem_before = check_gpu_memory()
        if gpu_mem_before:
            print(f"\nGPU Memory (before batch {batch_size}):")
            print(f"  - Used: {gpu_mem_before['used']} MB ({gpu_mem_before['used_percent']:.1f}%)")
        
        # Run test
        result = test_batch_size(batch_size)
        results.append(result)
        
        # Check GPU memory after test
        gpu_mem_after = check_gpu_memory()
        if gpu_mem_after:
            print(f"\nGPU Memory (after batch {batch_size}):")
            print(f"  - Used: {gpu_mem_after['used']} MB ({gpu_mem_after['used_percent']:.1f}%)")
        
        # Stop if we hit OOM
        if not result["success"] and result.get("error") == "oom":
            print(f"\n💥 Hit GPU memory limit at batch_size={batch_size}")
            print(f"   Maximum safe batch_size is {batch_sizes[batch_sizes.index(batch_size)-1] if batch_sizes.index(batch_size) > 0 else 'unknown'}")
            break
        
        # Stop if test failed for other reasons
        if not result["success"]:
            print(f"\n⚠️  Test failed for batch_size={batch_size}, stopping")
            break
        
        # Wait a bit between tests to let GPU cool down
        if batch_size < batch_sizes[-1]:
            print(f"\n⏸️  Waiting 10s before next test...")
            time.sleep(10)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    print()
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        print("✅ Successful Tests:")
        print(f"\n{'Batch Size':<12} {'Samples':<10} {'Time (s)':<12} {'Throughput':<15} {'Time/Sample':<15}")
        print('-'*80)
        for r in successful_results:
            print(f"{r['batch_size']:<12} {r['samples']:<10} {r['time']:<12.2f} {r['throughput']:<15.2f} {r['time_per_sample']*1000:<15.1f}")
        
        # Find best throughput
        best = max(successful_results, key=lambda x: x['throughput'])
        print()
        print(f"🏆 Best Performance:")
        print(f"   - Batch size: {best['batch_size']}")
        print(f"   - Throughput: {best['throughput']:.2f} samples/sec")
        print(f"   - Speedup vs batch_size=200: {best['throughput']/successful_results[0]['throughput']:.2f}x")
        print()
        
        # Recommendation
        max_successful = max([r['batch_size'] for r in successful_results])
        recommended = int(max_successful * 0.8)  # 80% of max for safety
        print(f"💡 Recommendation:")
        print(f"   - Maximum tested: batch_size={max_successful}")
        print(f"   - Recommended for production: batch_size={recommended} (80% of max)")
        print()
    
    if failed_results:
        print("❌ Failed Tests:")
        for r in failed_results:
            error_type = r.get('error', 'unknown')
            print(f"   - Batch size {r['batch_size']}: {error_type}")
            if 'error_detail' in r:
                print(f"     Detail: {r['error_detail'][:100]}")
        print()
    
    # Final GPU memory check
    gpu_mem_final = check_gpu_memory()
    if gpu_mem_final:
        print(f"GPU Memory (final):")
        print(f"  - Used: {gpu_mem_final['used']} MB ({gpu_mem_final['used_percent']:.1f}%)")
        print()
    
    print('='*80)
    
    # Return success if at least one test passed
    return len(successful_results) > 0


if __name__ == "__main__":
    print()
    print("🚀 Starting maximum batch_size test")
    print()
    
    success = main()
    
    sys.exit(0 if success else 1)
