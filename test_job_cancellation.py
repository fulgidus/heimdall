#!/usr/bin/env python3
"""
Test script to diagnose job cancellation issues.
Tests:
1. Create a small synthetic dataset generation job
2. Cancel it after a few seconds
3. Verify cancellation works properly
"""
import asyncio
import httpx
import time
import sys

BASE_URL = "http://localhost:8002"  # Training service

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("=" * 60)
        print("Job Cancellation Test")
        print("=" * 60)
        
        # Step 1: Create a small job (50k samples for ~30-60 seconds runtime)
        print("\n[1/4] Creating synthetic generation job...")
        create_payload = {
            "name": "CANCEL_TEST",
            "description": "Test job for cancellation debugging",
            "num_samples": 50000,  # Should take ~30-60 seconds
            "frequency_mhz": 144.0,
            "tx_power_dbm": 10.0,
            "min_snr_db": 5.0,
            "min_receivers": 3,
            "dataset_type": "feature_based",
            "use_gpu": False  # CPU for consistent timing
        }
        
        try:
            response = await client.post(f"{BASE_URL}/v1/jobs/synthetic/generate", json=create_payload)
            response.raise_for_status()
            job_data = response.json()
            job_id = job_data["job_id"]
            print(f"✓ Job created: {job_id}")
        except Exception as e:
            print(f"✗ Failed to create job: {e}")
            return 1
        
        # Step 2: Wait for job to start running
        print("\n[2/4] Waiting for job to start running...")
        max_wait = 15  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = await client.get(f"{BASE_URL}/v1/jobs/synthetic/{job_id}")
                response.raise_for_status()
                job_status = response.json()
                
                status = job_status["status"]
                progress = job_status.get("current_progress", 0)
                total = job_status.get("total_progress", 0)
                
                print(f"  Status: {status}, Progress: {progress}/{total}")
                
                if status == "running" and progress > 0:
                    print(f"✓ Job is running with {progress} samples generated")
                    break
                
                if status in ["failed", "cancelled", "completed"]:
                    print(f"✗ Job ended prematurely with status: {status}")
                    return 1
                    
                await asyncio.sleep(1)
            except Exception as e:
                print(f"  Error checking status: {e}")
                await asyncio.sleep(1)
        else:
            print(f"✗ Job didn't start running after {max_wait} seconds")
            return 1
        
        # Step 3: Cancel the job
        print(f"\n[3/4] Cancelling job {job_id}...")
        try:
            response = await client.post(f"{BASE_URL}/v1/jobs/synthetic/{job_id}/cancel")
            response.raise_for_status()
            cancel_data = response.json()
            print(f"✓ Cancel request accepted: {cancel_data.get('message', 'OK')}")
            print(f"  Celery task ID: {cancel_data.get('celery_task_id', 'N/A')}")
        except Exception as e:
            print(f"✗ Failed to cancel job: {e}")
            if hasattr(e, 'response'):
                print(f"  Response: {e.response.text}")
            return 1
        
        # Step 4: Verify cancellation (poll for up to 30 seconds)
        print("\n[4/4] Verifying cancellation...")
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = await client.get(f"{BASE_URL}/v1/jobs/synthetic/{job_id}")
                response.raise_for_status()
                job_status = response.json()
                
                status = job_status["status"]
                progress = job_status.get("current_progress", 0)
                total = job_status.get("total_progress", 0)
                
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] Status: {status}, Progress: {progress}/{total}")
                
                if status == "cancelled":
                    print(f"✓ Job cancelled successfully after {elapsed:.1f} seconds")
                    print(f"  Final progress: {progress}/{total} samples")
                    return 0
                
                if status in ["failed", "completed"]:
                    print(f"✗ Job ended with unexpected status: {status}")
                    if job_status.get("error_message"):
                        print(f"  Error: {job_status['error_message']}")
                    return 1
                
                await asyncio.sleep(2)
            except Exception as e:
                print(f"  Error checking status: {e}")
                await asyncio.sleep(2)
        
        # Timeout - cancellation didn't work
        print(f"✗ Job did not cancel after {max_wait} seconds")
        print("\nDIAGNOSTICS:")
        print("- Check if signal handlers are registered (look for 'Signal handlers registered' in logs)")
        print("- Check if SIGTERM is being received by Celery worker")
        print("- Check if database status check is happening (every 10 samples)")
        print("\nRun these commands to debug:")
        print(f"  docker compose logs training | grep -i 'cancel\\|sigterm\\|{job_id[:8]}'")
        print(f"  docker compose exec postgres psql -U heimdall -d heimdall -c \"SELECT id, status, current_progress FROM heimdall.training_jobs WHERE id = '{job_id}';\"")
        
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
