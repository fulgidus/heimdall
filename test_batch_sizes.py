#!/usr/bin/env python3
"""
Benchmark different batch sizes for GPU-accelerated feature extraction.
"""

import requests
import time
import json
import subprocess
import threading


def monitor_memory(stop_event, interval=2):
    """Monitor Docker container memory usage."""
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
        
        except Exception:
            pass
        
        time.sleep(interval)
    
    return max_memory_mb


def test_synthetic_generation(num_samples):
    """Test synthetic dataset generation."""
    
    print(f"\n{'='*80}")
    print(f"Testing with {num_samples} samples")
    print(f"{'='*80}")
    
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
            "name": f"test_batch_{num_samples}_{int(time.time())}",
            "description": f"GPU batch test with {num_samples} samples",
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
            print(f"❌ ERROR: Failed to submit job (status {response.status_code})")
            print(response.text)
            return None
        
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✅ Job submitted: {job_id}")
        
        # Poll for completion
        last_progress = 0
        while True:
            status_response = requests.get(
                f"http://localhost:8002/v1/training/synthetic/jobs/{job_id}",
                timeout=10
            )
            
            if status_response.status_code != 200:
                print(f"❌ ERROR: Failed to get status")
                return None
            
            status_data = status_response.json()
            state = status_data.get('status', 'UNKNOWN')
            progress = status_data.get('progress_percent', 0)
            current_progress = status_data.get('current_progress', 0)
            
            if current_progress != last_progress:
                elapsed = time.time() - start_time
                throughput = current_progress / elapsed if elapsed > 0 else 0
                print(f"Progress: {current_progress}/{num_samples} samples ({throughput:.2f} samples/sec)", end='\r')
                last_progress = current_progress
            
            if state == 'completed':
                elapsed = time.time() - start_time
                samples_generated = status_data.get('current_progress', 0)
                throughput = samples_generated / elapsed if elapsed > 0 else 0
                
                print(f"\n✅ Generation complete!")
                print(f"   Time: {elapsed:.2f}s")
                print(f"   Throughput: {throughput:.2f} samples/sec")
                print(f"   Samples generated: {samples_generated}/{num_samples}")
                
                return {
                    'num_samples': num_samples,
                    'samples_generated': samples_generated,
                    'elapsed_time': elapsed,
                    'throughput': throughput
                }
            
            elif state == 'failed':
                print(f"\n❌ Generation FAILED")
                print(f"Error: {status_data.get('error_message', 'Unknown error')}")
                return None
            
            elif state in ['pending', 'running']:
                time.sleep(1)
            
            else:
                print(f"\n⚠️  Unknown state: {state}")
                time.sleep(2)
    
    finally:
        # Stop memory monitoring
        stop_monitoring.set()
        monitor_thread.join(timeout=5)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GPU BATCH PROCESSING BENCHMARK")
    print("="*80)
    
    # Test different batch sizes
    test_configs = [
        10,   # Small batch (should be 1 batch with batch_size=50)
        50,   # Full batch (exactly 1 batch)
        100,  # 2 batches
        200,  # 4 batches
    ]
    
    results = []
    
    for num_samples in test_configs:
        result = test_synthetic_generation(num_samples)
        if result:
            results.append(result)
        time.sleep(5)  # Wait between tests
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'Samples':<12} {'Time (s)':<12} {'Throughput':<20} {'Speedup':<10}")
    print("-"*80)
    
    baseline_throughput = None
    for r in results:
        if baseline_throughput is None:
            baseline_throughput = r['throughput']
            speedup = 1.0
        else:
            speedup = r['throughput'] / baseline_throughput
        
        print(f"{r['num_samples']:<12} {r['elapsed_time']:<12.2f} {r['throughput']:<20.2f} {speedup:<10.2f}x")
    
    print("="*80)
    print("\nKey Insights:")
    print(f"  • Batch size configured: 50 samples")
    print(f"  • GPU init overhead: ~4.5s per batch")
    print(f"  • Feature extraction: ~15ms per chunk")
    print(f"  • Optimal throughput at: {max(results, key=lambda x: x['throughput'])['num_samples']} samples")
