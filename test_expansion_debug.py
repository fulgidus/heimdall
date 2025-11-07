#!/usr/bin/env python3
"""Test dataset expansion with debug logging to verify GDOP inheritance."""

import requests
import time
import sys
import subprocess

# Configuration
API_URL = "http://localhost:8001"  # Backend service directly (no API gateway running)
DATASET_ID = "c8ffc1fa-1553-4a66-b62e-43b9069eeed7"  # UHF@5W_100MAXGDOP with max_gdop=100.0
NUM_ADDITIONAL_SAMPLES = 100  # Minimum required

def main():
    print(f"Testing dataset expansion inheritance bug fix")
    print(f"Original dataset ID: {DATASET_ID}")
    print(f"Expected max_gdop: 100.0 (inherited)")
    print(f"Adding: {NUM_ADDITIONAL_SAMPLES} samples\n")
    
    # Create expansion request
    payload = {
        "name": f"Debug Test Expansion",
        "description": f"Testing GDOP inheritance with debug logging",
        "num_samples": NUM_ADDITIONAL_SAMPLES,
        "expand_dataset_id": DATASET_ID,
        "use_gpu": False
    }
    
    print("Sending expansion request...")
    response = requests.post(
        f"{API_URL}/api/v1/training/synthetic/generate",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"ERROR: Request failed with status {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    result = response.json()
    job_id = result.get("job_id")
    print(f"✓ Job created: {job_id}\n")
    print("Check backend logs with:")
    print(f"  docker logs heimdall-backend --tail 50 | grep 'EXPANSION DEBUG\\|Inherited.*gdop'")
    
    # Wait a moment for the job to be picked up
    time.sleep(2)
    
    # Query the job config from database
    print("\nQuerying job config from database...")
    result = subprocess.run([
        "docker", "exec", "-i", "heimdall-postgres",
        "psql", "-U", "heimdall_user", "-d", "heimdall",
        "-c", f"SELECT config->>'max_gdop' as stored_gdop FROM heimdall.training_jobs WHERE id = '{job_id}';"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    if "100.0" in result.stdout:
        print("✓ SUCCESS: max_gdop=100.0 correctly inherited!")
    elif "150.0" in result.stdout:
        print("✗ FAILURE: max_gdop=150.0 (Pydantic default, inheritance failed!)")
    else:
        print("? Unable to verify GDOP value")

if __name__ == "__main__":
    main()
