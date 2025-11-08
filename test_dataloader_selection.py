#!/usr/bin/env python3
"""
Test script to trigger a training job with iq_resnet18 model.
This verifies that IQ models correctly use the IQ dataloader.
"""

import requests
import json
import time

# Training service API endpoint
API_BASE = "http://localhost:8002"

# Training configuration for iq_resnet18 with IQ dataset
training_config = {
    "job_name": "Dataloader Selection Test - iq_resnet18",
    "dataset_id": "22a722a3-5508-48a9-8657-d75bbe5629bd",  # "Basic" IQ dataset
    "model_architecture": "iq_resnet18",
    "batch_size": 16,  # Small batch for quick test
    "total_epochs": 2,   # Just 2 epochs for testing
    "learning_rate": 0.001,
    "val_split": 0.2,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "early_stopping_patience": 0  # Disable early stopping for test
}

def submit_training_job():
    """Submit the training job."""
    print("=" * 80)
    print("SUBMITTING TRAINING JOB: iq_resnet18 with IQ Dataset")
    print("=" * 80)
    print()
    print("Configuration:")
    print(json.dumps(training_config, indent=2))
    print()
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/jobs/training",
            json=training_config,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        job_id = result.get("id")
        print(f"✅ Job submitted successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Status: {result.get('status')}")
        print(f"   Model: {result.get('model_architecture')}")
        print()
        
        return job_id
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error submitting job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return None

def check_job_status(job_id, timeout=120):
    """Check job status and look for dataloader selection log."""
    print("=" * 80)
    print("MONITORING TRAINING JOB")
    print("=" * 80)
    print()
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{API_BASE}/api/v1/jobs/training/{job_id}",
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            status = result.get("status")
            if status != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
                last_status = status
            
            # Check if job is complete
            if status in ["SUCCESS", "FAILURE", "CANCELLED"]:
                print()
                print(f"Job finished with status: {status}")
                if status == "FAILURE":
                    print(f"Error: {result.get('error', 'Unknown error')}")
                return result
            
            time.sleep(5)
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Error checking status: {e}")
            time.sleep(5)
    
    print()
    print("⚠️  Timeout waiting for job completion")
    return None

def check_training_logs(job_id):
    """Check training logs for dataloader selection message."""
    print()
    print("=" * 80)
    print("CHECKING DATALOADER SELECTION IN LOGS")
    print("=" * 80)
    print()
    
    # Use docker logs to check for the dataloader selection message
    import subprocess
    
    try:
        cmd = f"docker logs heimdall-training 2>&1 | grep -A 5 -B 5 'Using IQ/Spectrogram dataloader' | tail -20"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout:
            print("✅ Found dataloader selection log:")
            print(result.stdout)
            return True
        else:
            print("⚠️  Dataloader selection log not found yet")
            return False
            
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
        return False

if __name__ == "__main__":
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "     DATALOADER SELECTION TEST - iq_resnet18 with IQ Dataset".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Submit job
    job_id = submit_training_job()
    if not job_id:
        print("❌ Failed to submit job")
        exit(1)
    
    # Wait a bit for job to start
    print("Waiting for job to start...")
    time.sleep(10)
    
    # Check logs for dataloader selection
    check_training_logs(job_id)
    
    # Monitor job status (with timeout)
    print()
    print("Monitoring job status (timeout: 120s)...")
    result = check_job_status(job_id, timeout=120)
    
    # Final log check
    print()
    print("Final log check...")
    check_training_logs(job_id)
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
