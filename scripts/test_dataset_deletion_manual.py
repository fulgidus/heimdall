#!/usr/bin/env python3
"""
Manual test script for dataset deletion with MinIO cleanup.

This script can be run against a live Heimdall instance to verify that
dataset and job deletion properly cleans up MinIO data.

Usage:
    python scripts/test_dataset_deletion_manual.py

Prerequisites:
    - docker-compose services running (make dev-up)
    - Training service accessible at http://localhost:8002
    - MinIO accessible at http://localhost:9000
"""

import sys
import os
import time
import uuid
import requests
import boto3
from typing import List, Tuple

# Configuration - use environment variables with defaults
TRAINING_API_URL = os.environ.get("TRAINING_API_URL", "http://localhost:8002/api/v1/jobs/synthetic")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "heimdall-synthetic-iq")


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_step(message: str):
    """Print a test step."""
    print(f"\n{Colors.BLUE}▶ {message}{Colors.RESET}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def check_services() -> bool:
    """Check if required services are accessible."""
    print_step("Checking service availability...")
    
    # Check training service
    try:
        response = requests.get(f"{TRAINING_API_URL}/datasets", timeout=5)
        if response.status_code == 200:
            print_success("Training service is accessible")
        else:
            print_error(f"Training service returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Training service not accessible: {e}")
        return False
    
    # Check MinIO
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY
        )
        s3_client.head_bucket(Bucket=MINIO_BUCKET)
        print_success("MinIO is accessible")
        return True
    except Exception as e:
        print_error(f"MinIO not accessible: {e}")
        return False


def list_minio_objects(dataset_id: str) -> List[str]:
    """List MinIO objects for a dataset."""
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )
    
    objects = []
    prefixes = [f"synthetic/{dataset_id}/", f"synthetic/dataset-{dataset_id}/"]
    
    for prefix in prefixes:
        try:
            response = s3_client.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
            if 'Contents' in response:
                objects.extend([obj['Key'] for obj in response['Contents']])
        except Exception:
            # Prefix may not exist, continue to next prefix
            pass
    
    return objects


def create_test_dataset() -> Tuple[str, str]:
    """
    Create a test iq_raw dataset.
    
    Returns:
        Tuple of (job_id, dataset_id) or (None, None) on failure
    """
    print_step("Creating test iq_raw dataset...")
    
    # Generate a small test dataset (100 samples)
    payload = {
        "name": f"test_deletion_{uuid.uuid4().hex[:8]}",
        "description": "Test dataset for deletion verification",
        "num_samples": 100,
        "dataset_type": "iq_raw",
        "frequency_mhz": 145.0,
        "tx_power_dbm": 37.0,
        "min_snr_db": 3.0,
        "min_receivers": 3,
        "max_gdop": 150.0,
        "use_random_receivers": True,
        "min_receivers_count": 5,
        "max_receivers_count": 7
    }
    
    try:
        response = requests.post(f"{TRAINING_API_URL}/generate", json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print_success(f"Dataset generation job created: {job_id}")
            
            # Wait for job to complete (with timeout)
            print_step("Waiting for dataset generation to complete...")
            max_wait = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = requests.get(f"{TRAINING_API_URL}/{job_id}", timeout=5)
                if status_response.status_code == 200:
                    job_data = status_response.json()
                    status = job_data['status']
                    
                    if status == 'completed':
                        # Find the dataset ID
                        datasets_response = requests.get(f"{TRAINING_API_URL}/datasets", timeout=5)
                        if datasets_response.status_code == 200:
                            datasets = datasets_response.json()['datasets']
                            # Find dataset created by this job
                            for dataset in datasets:
                                if dataset.get('created_by_job_id') == job_id:
                                    dataset_id = dataset['id']
                                    print_success(f"Dataset created: {dataset_id}")
                                    return job_id, dataset_id
                        
                        print_error("Could not find dataset created by job")
                        return None, None
                    
                    elif status in ['failed', 'cancelled']:
                        print_error(f"Job {status}: {job_data.get('error_message', 'Unknown error')}")
                        return None, None
                    
                    # Still running
                    progress = job_data.get('progress_percent', 0)
                    print(f"  Progress: {progress:.1f}%", end='\r')
                
                time.sleep(5)
            
            print_error("Dataset generation timed out")
            return None, None
            
        else:
            print_error(f"Failed to create dataset: {response.status_code} - {response.text}")
            return None, None
            
    except Exception as e:
        print_error(f"Exception creating dataset: {e}")
        return None, None


def test_dataset_deletion(dataset_id: str) -> bool:
    """
    Test dataset deletion and verify MinIO cleanup.
    
    Returns:
        True if test passed, False otherwise
    """
    print_step(f"Testing dataset deletion for {dataset_id}...")
    
    # List MinIO objects before deletion
    print_step("Checking MinIO objects before deletion...")
    objects_before = list_minio_objects(dataset_id)
    
    if not objects_before:
        print_warning("No MinIO objects found for dataset (might be feature_based)")
    else:
        print_success(f"Found {len(objects_before)} MinIO objects")
        for obj in objects_before[:5]:  # Show first 5
            print(f"  - {obj}")
        if len(objects_before) > 5:
            print(f"  ... and {len(objects_before) - 5} more")
    
    # Delete the dataset
    print_step("Deleting dataset...")
    try:
        response = requests.delete(f"{TRAINING_API_URL}/datasets/{dataset_id}", timeout=10)
        
        if response.status_code == 204:
            print_success("Dataset deleted successfully")
        else:
            print_error(f"Deletion failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print_error(f"Exception deleting dataset: {e}")
        return False
    
    # Verify MinIO objects are gone
    print_step("Verifying MinIO cleanup...")
    time.sleep(2)  # Give MinIO a moment
    
    objects_after = list_minio_objects(dataset_id)
    
    if not objects_after:
        print_success("MinIO objects successfully deleted")
        return True
    else:
        print_error(f"MinIO cleanup incomplete: {len(objects_after)} objects remain")
        for obj in objects_after[:5]:
            print(f"  - {obj}")
        return False


def test_job_deletion(job_id: str) -> bool:
    """
    Test job deletion with dataset cascade.
    
    Returns:
        True if test passed, False otherwise
    """
    print_step(f"Testing job deletion for {job_id}...")
    
    # Delete the job (with dataset cascade enabled by default)
    try:
        response = requests.delete(f"{TRAINING_API_URL}/{job_id}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            datasets_deleted = data.get('datasets_deleted', 0)
            minio_files_deleted = data.get('minio_files_deleted', 0)
            
            print_success(f"Job deleted: {datasets_deleted} datasets, {minio_files_deleted} MinIO files")
            return True
        else:
            print_error(f"Deletion failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print_error(f"Exception deleting job: {e}")
        return False


def main():
    """Run the manual test suite."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("Dataset Deletion with MinIO Cleanup - Manual Test")
    print(f"{'='*60}{Colors.RESET}\n")
    
    # Check services
    if not check_services():
        print_error("\nServices not available. Please run 'make dev-up' first.")
        sys.exit(1)
    
    # Test 1: Create and delete dataset
    print(f"\n{Colors.BLUE}{'='*60}")
    print("TEST 1: Dataset Deletion")
    print(f"{'='*60}{Colors.RESET}")
    
    job_id, dataset_id = create_test_dataset()
    
    if not dataset_id:
        print_error("\nFailed to create test dataset")
        sys.exit(1)
    
    test1_passed = test_dataset_deletion(dataset_id)
    
    # Test 2: Create and delete job (cascades to dataset)
    print(f"\n{Colors.BLUE}{'='*60}")
    print("TEST 2: Job Deletion (with dataset cascade)")
    print(f"{'='*60}{Colors.RESET}")
    
    job_id2, dataset_id2 = create_test_dataset()
    
    if not dataset_id2:
        print_error("\nFailed to create test dataset for test 2")
        test2_passed = False
    else:
        test2_passed = test_job_deletion(job_id2)
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}")
    print("Test Summary")
    print(f"{'='*60}{Colors.RESET}")
    
    if test1_passed:
        print_success("Test 1 (Dataset Deletion): PASSED")
    else:
        print_error("Test 1 (Dataset Deletion): FAILED")
    
    if test2_passed:
        print_success("Test 2 (Job Deletion): PASSED")
    else:
        print_error("Test 2 (Job Deletion): FAILED")
    
    if test1_passed and test2_passed:
        print(f"\n{Colors.GREEN}All tests passed!{Colors.RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}Some tests failed.{Colors.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
