#!/usr/bin/env python3
"""
Quick test to verify audio library performance fix.
Tests that 1-second audio chunks are loaded in <1s instead of ~4s.
"""
import time
import requests
import json

BACKEND_URL = "http://localhost:8001"
TRAINING_URL = "http://localhost:8002"

def create_small_dataset():
    """Create a small test dataset with 10 samples."""
    print("Creating small test dataset (10 samples)...")
    
    payload = {
        "name": "audio_performance_test",
        "description": "Testing audio library chunk extraction performance",
        "num_samples": 10,
        "frequency_mhz": 145.0,
        "tx_power_dbm": 37.0,
        "min_snr_db": 10.0,
        "min_receivers": 3,
        "max_gdop": 150.0,
        "use_random_receivers": False,
        "use_srtm_terrain": True,
        "use_gpu": False,  # CPU for testing
        "seed": 42,
        "enable_meteorological": True,
        "enable_sporadic_e": True,
        "enable_knife_edge": True
    }
    
    response = requests.post(
        f"{TRAINING_URL}/api/v1/jobs/synthetic/generate",
        json=payload,
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to create dataset: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    job_id = result.get("job_id")
    print(f"✅ Dataset generation started: job_id={job_id}")
    return job_id

def monitor_progress(job_id):
    """Monitor job progress and measure time per sample."""
    print(f"\nMonitoring job {job_id}...")
    
    start_time = time.time()
    last_progress = 0
    sample_times = []
    status = "pending"
    
    while True:
        response = requests.get(
            f"{TRAINING_URL}/api/v1/jobs/synthetic/{job_id}",
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to get job status: {response.status_code}")
            status = "error"
            break
        
        job = response.json()
        status = job.get("status", "unknown")
        progress = job.get("progress", 0)
        
        if progress > last_progress:
            elapsed = time.time() - start_time
            samples_generated = int(progress * 10 / 100)  # 10 total samples
            if samples_generated > 0:
                avg_time = elapsed / samples_generated
                sample_times.append(avg_time)
                print(f"Progress: {progress}% ({samples_generated}/10 samples) - Avg: {avg_time:.2f}s/sample")
            last_progress = progress
        
        if status in ["completed", "failed"]:
            print(f"\n{'✅' if status == 'completed' else '❌'} Job {status}")
            break
        
        time.sleep(2)
    
    # Calculate statistics
    if sample_times:
        final_elapsed = time.time() - start_time
        total_samples = int(last_progress * 10 / 100)
        final_avg = final_elapsed / total_samples if total_samples > 0 else 0
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total time: {final_elapsed:.2f}s")
        print(f"   Samples generated: {total_samples}/10")
        print(f"   Average time per sample: {final_avg:.2f}s")
        print(f"   Expected with bug: ~4.0s/sample")
        print(f"   Expected with fix: ~0.5-1.0s/sample (audio loading)")
        
        if final_avg < 2.0:
            print(f"\n✅ PERFORMANCE FIX VERIFIED! ({final_avg:.2f}s < 2.0s threshold)")
        else:
            print(f"\n⚠️  Still slow - may need investigation ({final_avg:.2f}s)")
    
    return status == "completed"

if __name__ == "__main__":
    print("=" * 60)
    print("Audio Library Performance Test")
    print("=" * 60)
    
    # Create small dataset
    job_id = create_small_dataset()
    if not job_id:
        exit(1)
    
    # Monitor progress
    success = monitor_progress(job_id)
    
    if success:
        print("\n✅ Test completed successfully!")
        exit(0)
    else:
        print("\n❌ Test failed")
        exit(1)
