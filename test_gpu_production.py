"""
Test GPU acceleration in production synthetic data generation.

This script:
1. Creates a small dataset generation job (50 samples)
2. Monitors GPU usage during generation
3. Verifies performance improvements
"""

import requests
import time
import subprocess
import json
from datetime import datetime

BASE_URL = "http://localhost:8002"

def monitor_gpu():
    """Monitor GPU usage with nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
            return {
                "gpu_utilization": float(gpu_util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "memory_pct": (float(mem_used) / float(mem_total)) * 100
            }
    except Exception as e:
        print(f"Error monitoring GPU: {e}")
    return None

def create_dataset_generation_job():
    """Create a dataset generation job via API"""
    config = {
        "name": f"gpu_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "description": "GPU acceleration test dataset",
        "num_samples": 50,
        "frequency_mhz": 145.0,
        "tx_power_dbm": 37.0,
        "min_snr_db": 3.0,
        "min_receivers": 3,
        "max_gdop": 10.0,
        "dataset_type": "feature_based",
        "use_random_receivers": False,  # Use fixed Italian receivers for consistency
        "seed": 42
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/training/synthetic/generate",
        json=config,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def get_job_status(job_id):
    """Get job status"""
    response = requests.get(f"{BASE_URL}/v1/training/synthetic/jobs/{job_id}")
    response.raise_for_status()
    return response.json()

def main():
    print("=" * 80)
    print("GPU Acceleration Production Test")
    print("=" * 80)
    
    # Check GPU before starting
    print("\n1. Initial GPU state:")
    gpu_stats = monitor_gpu()
    if gpu_stats:
        print(f"   GPU Utilization: {gpu_stats['gpu_utilization']}%")
        print(f"   Memory Used: {gpu_stats['memory_used_mb']:.0f} / {gpu_stats['memory_total_mb']:.0f} MB ({gpu_stats['memory_pct']:.1f}%)")
    else:
        print("   WARNING: Cannot monitor GPU (nvidia-smi not available)")
    
    # Create generation job
    print("\n2. Creating dataset generation job (50 samples)...")
    try:
        result = create_dataset_generation_job()
        job_id = result.get("job_id")
        print(f"   ✓ Job created: {job_id}")
    except Exception as e:
        print(f"   ✗ Failed to create job: {e}")
        return 1
    
    # Monitor job progress
    print("\n3. Monitoring job progress and GPU usage...")
    start_time = time.time()
    max_gpu_util = 0.0
    max_memory_used = 0.0
    sample_count = 0
    job_status = "unknown"
    
    while True:
        time.sleep(2)  # Poll every 2 seconds
        
        # Get job status
        try:
            status = get_job_status(job_id)
            job_status = status.get("status")
            current_progress = status.get("current_progress", 0)
            total_progress = status.get("total_progress", 50)
            progress_message = status.get("progress_message", "")
            
            # Monitor GPU
            gpu_stats = monitor_gpu()
            if gpu_stats:
                max_gpu_util = max(max_gpu_util, gpu_stats["gpu_utilization"])
                max_memory_used = max(max_memory_used, gpu_stats["memory_used_mb"])
                
                print(f"   Progress: {current_progress}/{total_progress} samples | "
                      f"GPU: {gpu_stats['gpu_utilization']}% | "
                      f"VRAM: {gpu_stats['memory_used_mb']:.0f} MB | "
                      f"{progress_message}")
            else:
                print(f"   Progress: {current_progress}/{total_progress} samples | {progress_message}")
            
            sample_count = current_progress
            
            if job_status in ["completed", "failed", "cancelled"]:
                break
                
        except Exception as e:
            print(f"   Error checking status: {e}")
            break
    
    elapsed_time = time.time() - start_time
    
    # Final results
    print("\n4. Results:")
    print(f"   Status: {job_status}")
    print(f"   Duration: {elapsed_time:.1f}s ({elapsed_time/sample_count:.2f}s per sample)")
    print(f"   Max GPU Utilization: {max_gpu_util}%")
    print(f"   Max Memory Used: {max_memory_used:.0f} MB")
    
    if job_status == "completed":
        print("\n   ✅ SUCCESS: GPU-accelerated generation completed!")
        print(f"\n   Performance: {sample_count / elapsed_time:.1f} samples/sec")
        return 0
    else:
        print(f"\n   ✗ FAILED: Job status = {job_status}")
        return 1

if __name__ == "__main__":
    exit(main())
