#!/usr/bin/env python3
"""
Integration test for synthetic data generation continuation feature.

Tests the complete flow:
1. Start a synthetic data generation job (200 samples)
2. Cancel it after ~100 samples
3. Verify samples were persisted in DB
4. Call the continue endpoint
5. Verify new job is created with parent_job_id reference
6. Let continuation job complete
7. Verify total of 200 samples exist

Usage:
    python scripts/test_synthetic_continuation.py
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
BACKEND_URL = "http://localhost:8000"
DB_URL = "postgresql://heimdall:heimdall@localhost:5432/heimdall"
DATASET_NAME = "test_continuation_dataset"
NUM_SAMPLES = 200
CANCEL_THRESHOLD = 100  # Cancel after this many samples


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_step(step_num: int, message: str):
    """Print a test step header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}[Step {step_num}] {message}{Colors.ENDC}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


async def create_synthetic_job(client: httpx.AsyncClient, dataset_name: str) -> dict:
    """Create a synthetic data generation job."""
    payload = {
        "dataset_name": dataset_name,
        "training_type": "synthetic_generation",
        "num_samples": NUM_SAMPLES,
        "synthetic_config": {
            "num_samples": NUM_SAMPLES,
            "bands": ["2m", "70cm"],
            "modulations": ["fm", "am", "ssb"],
            "num_workers": 4,
            "batch_size": 10
        }
    }
    
    response = await client.post(f"{BACKEND_URL}/training/jobs", json=payload)
    response.raise_for_status()
    return response.json()


async def get_job_status(client: httpx.AsyncClient, job_id: str) -> dict:
    """Get job status."""
    response = await client.get(f"{BACKEND_URL}/training/jobs/{job_id}")
    response.raise_for_status()
    return response.json()


async def cancel_job(client: httpx.AsyncClient, job_id: str):
    """Cancel a job."""
    response = await client.post(f"{BACKEND_URL}/training/jobs/{job_id}/cancel")
    response.raise_for_status()
    return response.json()


async def continue_job(client: httpx.AsyncClient, job_id: str) -> dict:
    """Continue a cancelled synthetic job."""
    response = await client.post(f"{BACKEND_URL}/training/jobs/{job_id}/continue")
    response.raise_for_status()
    return response.json()


def count_samples_in_db(engine, dataset_id: str) -> int:
    """Count samples in database for a dataset."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM heimdall.training_samples WHERE dataset_id = :dataset_id"),
            {"dataset_id": dataset_id}
        )
        return result.scalar()


def get_parent_job_id(engine, job_id: str) -> str | None:
    """Get parent_job_id for a job."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT parent_job_id FROM heimdall.training_jobs WHERE id = :job_id"),
            {"job_id": job_id}
        )
        return result.scalar()


async def wait_for_samples(client: httpx.AsyncClient, job_id: str, threshold: int, timeout: int = 300) -> dict:
    """Wait for job to generate at least threshold samples."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        job = await get_job_status(client, job_id)
        current_progress = job.get("current_progress", 0)
        
        print_info(f"Job {job_id[:8]}... progress: {current_progress}/{NUM_SAMPLES} samples")
        
        if current_progress >= threshold:
            return job
        
        if job["status"] in ["failed", "cancelled", "completed"]:
            raise RuntimeError(f"Job ended unexpectedly with status: {job['status']}")
        
        await asyncio.sleep(2)
    
    raise TimeoutError(f"Job did not reach {threshold} samples within {timeout}s")


async def wait_for_completion(client: httpx.AsyncClient, job_id: str, timeout: int = 600) -> dict:
    """Wait for job to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        job = await get_job_status(client, job_id)
        status = job["status"]
        current_progress = job.get("current_progress", 0)
        
        print_info(f"Job {job_id[:8]}... status: {status}, progress: {current_progress}/{NUM_SAMPLES}")
        
        if status == "completed":
            return job
        
        if status in ["failed", "cancelled"]:
            error_msg = job.get("error_message", "Unknown error")
            raise RuntimeError(f"Job failed: {error_msg}")
        
        await asyncio.sleep(3)
    
    raise TimeoutError(f"Job did not complete within {timeout}s")


async def main():
    """Run the integration test."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}")
    print("Synthetic Data Continuation Integration Test")
    print(f"{'='*70}{Colors.ENDC}\n")
    
    engine = create_engine(DB_URL)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Step 1: Create synthetic data generation job
            print_step(1, f"Creating synthetic data job ({NUM_SAMPLES} samples)")
            job1 = await create_synthetic_job(client, DATASET_NAME)
            job1_id = job1["job_id"]
            dataset_id = job1["dataset_id"]
            print_success(f"Job created: {job1_id}")
            print_info(f"Dataset ID: {dataset_id}")
            
            # Step 2: Wait for ~100 samples then cancel
            print_step(2, f"Waiting for job to reach {CANCEL_THRESHOLD} samples")
            await wait_for_samples(client, job1_id, CANCEL_THRESHOLD)
            print_success(f"Job reached {CANCEL_THRESHOLD}+ samples")
            
            # Give it a moment to save samples
            await asyncio.sleep(2)
            
            print_info("Cancelling job...")
            await cancel_job(client, job1_id)
            print_success("Job cancelled")
            
            # Step 3: Verify samples were persisted
            print_step(3, "Verifying samples persisted in database")
            await asyncio.sleep(2)  # Give DB time to commit
            
            samples_count = count_samples_in_db(engine, dataset_id)
            print_info(f"Samples in DB: {samples_count}")
            
            if samples_count < 50:
                print_error(f"Expected at least 50 samples, got {samples_count}")
                return False
            
            print_success(f"Samples persisted: {samples_count}")
            
            # Step 4: Continue the job
            print_step(4, "Calling continue endpoint")
            continue_response = await continue_job(client, job1_id)
            job2_id = continue_response["job_id"]
            samples_existing = continue_response["samples_existing"]
            samples_remaining = continue_response["samples_remaining"]
            
            print_success(f"Continuation job created: {job2_id}")
            print_info(f"Existing samples: {samples_existing}")
            print_info(f"Remaining samples: {samples_remaining}")
            print_info(f"Total target: {samples_existing + samples_remaining}")
            
            # Step 5: Verify parent_job_id reference
            print_step(5, "Verifying parent_job_id reference")
            parent_id = get_parent_job_id(engine, job2_id)
            
            if parent_id != job1_id:
                print_error(f"Expected parent_job_id={job1_id}, got {parent_id}")
                return False
            
            print_success(f"Parent job reference correct: {parent_id}")
            
            # Step 6: Wait for continuation job to complete
            print_step(6, "Waiting for continuation job to complete")
            job2_final = await wait_for_completion(client, job2_id)
            print_success(f"Continuation job completed")
            
            # Step 7: Verify total sample count
            print_step(7, "Verifying total sample count")
            final_samples_count = count_samples_in_db(engine, dataset_id)
            print_info(f"Final samples in DB: {final_samples_count}")
            
            if final_samples_count < NUM_SAMPLES:
                print_error(f"Expected {NUM_SAMPLES} samples, got {final_samples_count}")
                return False
            
            print_success(f"Total samples correct: {final_samples_count}/{NUM_SAMPLES}")
            
            # Summary
            print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'='*70}")
            print("✓ ALL TESTS PASSED")
            print(f"{'='*70}{Colors.ENDC}\n")
            
            print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
            print(f"  Original job: {job1_id}")
            print(f"  Samples before cancel: {samples_existing}")
            print(f"  Continuation job: {job2_id}")
            print(f"  Final sample count: {final_samples_count}")
            print(f"  Dataset ID: {dataset_id}")
            
            return True
            
        except Exception as e:
            print_error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
