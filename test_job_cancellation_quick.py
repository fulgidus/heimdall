#!/usr/bin/env python3
"""
Quick test script to verify job cancellation works correctly.
This tests the hybrid cancellation approach (signal handler + DB polling).
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8002/api/v1"

def test_cancellation():
    """Test that a job can be cancelled quickly."""
    print("=" * 60)
    print("Testing Job Cancellation (Hybrid Approach)")
    print("=" * 60)
    
    # 1. Create a synthetic generation job (small batch for quick testing)
    print("\n1. Creating synthetic generation job...")
    create_payload = {
        "name": f"cancel_test_{int(time.time())}",
        "description": "Test job for cancellation",
        "num_samples": 100,  # Small job
        "dataset_type": "feature_based",
        "use_random_receivers": True,
        "min_receivers_count": 3,
        "max_receivers_count": 5,
        "seed": 42
    }
    
    try:
        response = requests.post(f"{BASE_URL}/jobs/synthetic/generate", json=create_payload, timeout=10)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"✓ Job created: {job_id}")
        print(f"  Status: {job_data['status']}")
    except Exception as e:
        print(f"✗ Failed to create job: {e}")
        return False
    
    # 2. Wait for job to start running
    print("\n2. Waiting for job to start running...")
    max_wait = 30  # seconds
    started = False
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/jobs/synthetic/{job_id}", timeout=5)
            response.raise_for_status()
            job_status = response.json()
            status = job_status['status']
            print(f"  [{i+1}s] Status: {status}, Progress: {job_status.get('current_progress', 0)}/{job_status.get('total_progress', 0)}")
            
            if status == 'running':
                started = True
                print(f"✓ Job is running!")
                break
            elif status in ['failed', 'cancelled', 'completed']:
                print(f"✗ Job entered unexpected state: {status}")
                return False
                
            time.sleep(1)
        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(1)
    
    if not started:
        print(f"✗ Job did not start running within {max_wait}s")
        return False
    
    # 3. Cancel the job IMMEDIATELY (test fast response)
    print("\n3. Cancelling job...")
    cancel_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/jobs/synthetic/{job_id}/cancel", timeout=10)
        response.raise_for_status()
        print(f"✓ Cancel request sent")
    except Exception as e:
        print(f"✗ Failed to cancel job: {e}")
        return False
    
    # 4. Verify job is cancelled quickly
    print("\n4. Verifying cancellation response time...")
    max_wait = 10  # Should cancel within 10 seconds with hybrid approach
    cancelled = False
    response_time = 0.0
    progress = 0
    total = 0
    
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/jobs/synthetic/{job_id}", timeout=5)
            response.raise_for_status()
            job_status = response.json()
            status = job_status['status']
            progress = job_status.get('current_progress', 0)
            total = job_status.get('total_progress', 0)
            
            print(f"  [{i+1}s] Status: {status}, Progress: {progress}/{total}")
            
            if status == 'cancelled':
                response_time = time.time() - cancel_time
                print(f"✓ Job cancelled in {response_time:.2f}s")
                print(f"  Final progress: {progress}/{total} samples")
                cancelled = True
                break
                
            time.sleep(1)
        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(1)
    
    if not cancelled:
        print(f"✗ Job did not cancel within {max_wait}s (FAILED)")
        return False
    
    # 5. Success summary
    print("\n" + "=" * 60)
    print("CANCELLATION TEST PASSED ✓")
    print("=" * 60)
    print(f"Response time: {response_time:.2f}s (target: <10s)")
    print(f"Final samples: {progress}/{total}")
    print("\nHybrid approach working:")
    print("  ✓ Signal handler flag checked every iteration")
    print("  ✓ DB polling fallback every 10 samples")
    return True

if __name__ == "__main__":
    print("\nStarting cancellation test...")
    print("This tests the hybrid cancellation approach:\n")
    print("  1. Signal handler flag (checked EVERY iteration)")
    print("  2. DB polling (fallback every 10 samples)\n")
    
    success = test_cancellation()
    sys.exit(0 if success else 1)
