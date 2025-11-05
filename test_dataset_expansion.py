#!/usr/bin/env python3
"""
Test dataset expansion to verify samples are saved correctly.
"""
import requests
import time
import json

BACKEND_URL = "http://localhost:8001"
EXPAND_DATASET_ID = "1fd1bf91-04d0-41a7-9309-2425ed157a7a"

def check_dataset_samples(dataset_id: str) -> int:
    """Check current number of samples in dataset"""
    import psycopg2
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="heimdall_user",
        password="heimdall_password",
        database="heimdall"
    )
    cursor = conn.cursor()
    
    # Get count from measurement_features
    cursor.execute(
        "SELECT COUNT(*) FROM heimdall.measurement_features WHERE dataset_id = %s",
        (dataset_id,)
    )
    result = cursor.fetchone()
    count = result[0] if result else 0
    
    # Get dataset info
    cursor.execute(
        "SELECT name, num_samples FROM heimdall.synthetic_datasets WHERE id = %s",
        (dataset_id,)
    )
    dataset_info = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if dataset_info:
        print(f"\nDataset: {dataset_info[0]}")
        print(f"  synthetic_datasets.num_samples: {dataset_info[1]}")
        print(f"  measurement_features COUNT(*): {count}")
    else:
        print(f"\nDataset {dataset_id} not found!")
    
    return count

def test_expansion():
    """Test expanding existing dataset"""
    
    print("=" * 80)
    print("DATASET EXPANSION TEST")
    print("=" * 80)
    
    # Check initial state
    print("\n1. Checking initial dataset state...")
    initial_count = check_dataset_samples(EXPAND_DATASET_ID)
    
    # Create expansion job
    print("\n2. Creating expansion job (adding 20 samples)...")
    payload = {
        "name": "Test Expansion - Should Add 20 Samples",
        "description": "Testing expansion bug fix",
        "num_samples": 20,
        "expand_dataset_id": EXPAND_DATASET_ID,
        "frequency_mhz": 145.0,
        "signal_types": ["FM"],
        "antenna_configs": [
            {
                "type": "dipole",
                "height_m": 10.0,
                "gain_dbi": 2.15,
                "efficiency": 0.9
            }
        ],
        "path_loss_model": "fspl",
        "terrain_enabled": False,
        "noise_floor_dbm": -120.0,
        "include_iq": False,
        "dataset_type": "feature_based"
    }
    
    response = requests.post(
        f"{BACKEND_URL}/api/v1/training/synthetic/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to create job: {response.status_code}")
        print(response.text)
        return
    
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✓ Job created: {job_id}")
    
    # Monitor job progress
    print("\n3. Monitoring job progress...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status_response = requests.get(f"{BACKEND_URL}/api/v1/training/jobs/{job_id}")
        
        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.status_code}")
            break
        
        status_data = status_response.json()
        state = status_data["state"]
        progress = status_data.get("progress", {})
        
        print(f"  Status: {state} | Progress: {progress.get('current', 0)}/{progress.get('total', 20)} | "
              f"Attempted: {progress.get('total_attempted', 0)}")
        
        if state in ["SUCCESS", "FAILURE"]:
            print(f"\n✓ Job completed with state: {state}")
            if state == "FAILURE":
                print(f"Error: {status_data.get('error')}")
            break
        
        time.sleep(5)
    
    # Check final state
    print("\n4. Checking final dataset state...")
    final_count = check_dataset_samples(EXPAND_DATASET_ID)
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Initial samples: {initial_count}")
    print(f"Final samples: {final_count}")
    print(f"Difference: {final_count - initial_count}")
    print(f"Expected: 20 new samples")
    
    if final_count - initial_count == 20:
        print("\n✅ SUCCESS: All samples saved correctly!")
    elif final_count - initial_count > 0:
        print(f"\n⚠️  PARTIAL: Only {final_count - initial_count} samples saved (expected 20)")
    else:
        print("\n❌ FAILURE: No samples saved!")

if __name__ == "__main__":
    test_expansion()
