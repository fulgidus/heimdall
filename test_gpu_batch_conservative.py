#!/usr/bin/env python3
"""
Test conservative GPU-batched feature extraction with memory monitoring.
"""

import requests
import time
import json
import subprocess
import threading


def monitor_memory(stop_event, interval=2):
    """Monitor Docker container memory usage."""
    print("\n=== Memory Monitor Started ===")
    max_memory_mb = 0
    
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", "heimdall-training"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                mem_str = result.stdout.strip().split('/')[0].strip()
                
                # Parse memory (handle both MiB and GiB)
                if 'GiB' in mem_str:
                    mem_mb = float(mem_str.replace('GiB', '')) * 1024
                elif 'MiB' in mem_str:
                    mem_mb = float(mem_str.replace('MiB', ''))
                else:
                    mem_mb = 0
                
                max_memory_mb = max(max_memory_mb, mem_mb)
                print(f"Memory: {mem_mb:.1f} MiB (max: {max_memory_mb:.1f} MiB)", end='\r')
        
        except Exception as e:
            print(f"\nMemory monitoring error: {e}")
        
        time.sleep(interval)
    
    print(f"\n=== Memory Monitor Stopped === Max memory: {max_memory_mb:.1f} MiB\n")
    return max_memory_mb


def test_synthetic_generation(num_samples=10):
    """Test synthetic dataset generation."""
    
    print(f"\n{'='*60}")
    print(f"Testing Conservative GPU Batch Feature Extraction")
    print(f"{'='*60}\n")
    
    # Start memory monitoring thread
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory, 
        args=(stop_monitoring,),
        daemon=True
    )
    monitor_thread.start()
    
    try:
        # Submit dataset generation job
        print(f"Submitting job for {num_samples} samples...")
        
        payload = {
            "name": f"test_batch_conservative_{int(time.time())}",
            "description": "Conservative GPU batch feature extraction test",
            "num_samples": num_samples,
            "frequency_mhz": 145.0,
            "tx_power_dbm": 37.0,
            "min_snr_db": 3.0,
            "min_receivers": 3,
            "max_gdop": 100.0,
            "dataset_type": "feature_based",
            "use_random_receivers": False
        }
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8002/v1/training/synthetic/generate",
            json=payload,
            timeout=600
        )
        
        if response.status_code != 200:
            print(f"\n❌ ERROR: Failed to submit job (status {response.status_code})")
            print(response.text)
            return False
        
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ Job submitted: {job_id}\n")
        
        # Poll for completion
        print("Waiting for completion...")
        while True:
            status_response = requests.get(
                f"http://localhost:8002/v1/training/synthetic/jobs/{job_id}",
                timeout=10
            )
            
            if status_response.status_code != 200:
                print(f"\n❌ ERROR: Failed to get status")
                return False
            
            status_data = status_response.json()
            state = status_data.get('status', 'UNKNOWN')
            progress = status_data.get('progress_percent', 0)
            
            if state == 'completed':
                elapsed = time.time() - start_time
                print(f"\n\n✅ Generation complete!")
                print(f"Time: {elapsed:.2f}s")
                print(f"Throughput: {num_samples / elapsed:.2f} samples/sec")
                
                result = status_data.get('result_data', {})
                print(f"\nResults:")
                print(f"  Current progress: {status_data.get('current_progress', 0)} samples")
                print(f"  Total progress: {status_data.get('total_progress', 0)} samples")
                
                return True
            
            elif state == 'failed':
                print(f"\n❌ Generation FAILED")
                print(f"Error: {status_data.get('error_message', 'Unknown error')}")
                return False
            
            elif state in ['pending', 'running']:
                msg = status_data.get('progress_message', '')
                print(f"Progress: {progress:.0f}% ({state}) - {msg}", end='\r')
                time.sleep(2)
            
            else:
                print(f"\n⚠️  Unknown state: {state}")
                time.sleep(2)
    
    finally:
        # Stop memory monitoring
        stop_monitoring.set()
        monitor_thread.join(timeout=5)


if __name__ == "__main__":
    success = test_synthetic_generation(num_samples=10)
    exit(0 if success else 1)
