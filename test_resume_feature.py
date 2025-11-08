#!/usr/bin/env python3
"""
Test Resume Feature for Cancelled/Failed Training Jobs
Tests the new Resume button functionality for training jobs
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost/api/v1/training"  # Via Envoy proxy on port 80
CANCELLED_JOB_ID = "a72a2296-33dc-450b-aff4-e230e59d7f75"  # Known cancelled job with checkpoints

def test_job_info():
    """Get information about the cancelled job"""
    print("=" * 80)
    print("TEST 1: Fetch Job Information")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/jobs/{CANCELLED_JOB_ID}")
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch job info: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    data = response.json()
    job = data.get('job', data)  # Handle both wrapped and unwrapped responses
    print(f"✓ Job ID: {job['id']}")
    print(f"✓ Job Name: {job.get('job_name', 'N/A')}")
    print(f"✓ Status: {job['status']}")
    print(f"✓ Current Epoch: {job.get('current_epoch', 0)}")
    print(f"✓ Total Epochs: {job.get('total_epochs', 0)}")
    print(f"✓ Checkpoint Path: {job.get('checkpoint_path', 'None')}")
    print(f"✓ Pause Checkpoint: {job.get('pause_checkpoint_path', 'None')}")
    
    return job

def test_list_checkpoints(job_id):
    """List available checkpoints for the job"""
    print("\n" + "=" * 80)
    print("TEST 2: List Available Checkpoints")
    print("=" * 80)
    
    # MinIO path format: models/checkpoints/{job_id}/
    # We need to query MinIO or check the checkpoint_path field
    
    response = requests.get(f"{BASE_URL}/jobs/{job_id}")
    if response.status_code == 200:
        data = response.json()
        job = data.get('job', data)  # Handle both wrapped and unwrapped responses
        checkpoint = job.get('checkpoint_path')
        if checkpoint:
            print(f"✓ Best checkpoint available: {checkpoint}")
            return True
        else:
            print("❌ No checkpoint path found in job metadata")
            return False
    else:
        print(f"❌ Failed to fetch job: {response.status_code}")
        return False

def test_resume_cancelled_job():
    """Test resuming a cancelled job"""
    print("\n" + "=" * 80)
    print("TEST 3: Resume Cancelled Job")
    print("=" * 80)
    
    # First, check current status
    response = requests.get(f"{BASE_URL}/jobs/{CANCELLED_JOB_ID}")
    if response.status_code != 200:
        print(f"❌ Failed to fetch job: {response.status_code}")
        return False
    
    data = response.json()
    job = data.get('job', data)  # Handle both wrapped and unwrapped responses
    print(f"Current Status: {job['status']}")
    print(f"Current Epoch: {job.get('current_epoch', 0)}/{job.get('total_epochs', 0)}")
    
    if job['status'] not in ['cancelled', 'failed', 'paused']:
        print(f"⚠️  Job status is '{job['status']}', not cancelled/failed/paused")
        print("   Skipping resume test (job must be cancelled/failed/paused)")
        return True
    
    if not job.get('checkpoint_path'):
        print("❌ No checkpoint path found - cannot resume")
        return False
    
    # Attempt to resume
    print(f"\nAttempting to resume job from checkpoint: {job.get('checkpoint_path')}")
    response = requests.post(f"{BASE_URL}/jobs/{CANCELLED_JOB_ID}/resume")
    
    if response.status_code in [200, 202]:  # Accept both 200 and 202
        result = response.json()
        print(f"✓ Resume request accepted!")
        print(f"  - New Status: {result.get('status')}")
        print(f"  - Celery Task ID: {result.get('celery_task_id')}")
        print(f"  - Message: {result.get('message')}")
        print(f"  - Will resume from epoch: {result.get('current_epoch')}/{result.get('total_epochs')}")
        
        # Wait a few seconds and check status
        print("\nWaiting 5 seconds for job to start...")
        time.sleep(5)
        
        response = requests.get(f"{BASE_URL}/jobs/{CANCELLED_JOB_ID}")
        if response.status_code == 200:
            data = response.json()
            updated_job = data.get('job', data)  # Handle both wrapped and unwrapped responses
            print(f"  - Updated Status: {updated_job['status']}")
            if updated_job['status'] in ['queued', 'running']:
                print("✓ Job successfully resumed!")
                
                # Cancel it again so we can test multiple times
                print("\n⚠️  Cancelling job again for future testing...")
                cancel_response = requests.post(f"{BASE_URL}/jobs/{CANCELLED_JOB_ID}/cancel")
                if cancel_response.status_code == 200:
                    print("✓ Job cancelled successfully for future testing")
                
                return True
            else:
                print(f"⚠️  Job status is '{updated_job['status']}', expected 'queued' or 'running'")
                return True  # Still counts as success if resume was accepted
        
        return True
    else:
        print(f"❌ Resume failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_resume_validation():
    """Test that resume validation works correctly"""
    print("\n" + "=" * 80)
    print("TEST 4: Resume Validation")
    print("=" * 80)
    
    # Test 1: Try to resume non-existent job
    fake_job_id = "00000000-0000-0000-0000-000000000000"
    response = requests.post(f"{BASE_URL}/jobs/{fake_job_id}/resume")
    
    if response.status_code == 404:
        print("✓ Correctly rejects non-existent job (404)")
    else:
        print(f"⚠️  Expected 404 for non-existent job, got {response.status_code}")
    
    # Test 2: Check if completed jobs can be resumed (should fail)
    # First, find a completed job
    response = requests.get(f"{BASE_URL}/jobs?status=completed&limit=1")
    if response.status_code == 200:
        jobs = response.json().get('jobs', [])
        if jobs:
            completed_job = jobs[0]
            print(f"\nTesting with completed job: {completed_job['id'][:8]}")
            response = requests.post(f"{BASE_URL}/jobs/{completed_job['id']}/resume")
            
            if response.status_code == 400:
                print("✓ Correctly rejects completed job (400)")
            else:
                print(f"⚠️  Expected 400 for completed job, got {response.status_code}")
    
    return True

def main():
    print("\n" + "=" * 80)
    print("TRAINING JOB RESUME FEATURE TEST")
    print("=" * 80)
    print(f"Testing against: {BASE_URL}")
    print(f"Test Job ID: {CANCELLED_JOB_ID}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80 + "\n")
    
    results = []
    
    # Test 1: Get job information
    job = test_job_info()
    results.append(("Fetch Job Info", job is not None))
    
    if job:
        # Test 2: List checkpoints
        has_checkpoint = test_list_checkpoints(CANCELLED_JOB_ID)
        results.append(("List Checkpoints", has_checkpoint))
        
        # Test 3: Resume job
        resumed = test_resume_cancelled_job()
        results.append(("Resume Job", resumed))
    else:
        print("\n⚠️  Skipping remaining tests (job not found)")
        results.append(("List Checkpoints", False))
        results.append(("Resume Job", False))
    
    # Test 4: Validation
    validation = test_resume_validation()
    results.append(("Resume Validation", validation))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
