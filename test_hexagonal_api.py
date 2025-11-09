"""
API-based test for hexagonal receiver placement.

Tests the complete flow:
1. Generate a small dataset with hexagonal placement
2. Generate a small dataset with random placement  
3. Compare GDOP metrics
"""

import requests
import time
import json

BASE_URL = "http://localhost:8001"

def create_dataset(name, placement_strategy):
    """Create a dataset with specified placement strategy."""
    print(f"\n📊 Creating dataset '{name}' with {placement_strategy} placement...")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/training/synthetic/generate",
        json={
            "name": name,
            "description": f"Test dataset with {placement_strategy} placement",
            "dataset_type": "iq_raw",
            "num_samples": 100,  # Small test dataset
            "frequency_mhz": 144.0,
            "tx_power_dbm": 33.0,
            "min_snr_db": 3.0,
            "min_receivers": 3,
            "max_gdop": 150.0,
            "use_random_receivers": True,
            "receiver_placement_strategy": placement_strategy,
            "min_receivers_count": 7,
            "max_receivers_count": 7,
            "use_gpu": False,
        }
    )
    
    if response.status_code != 202:
        print(f"   ❌ Failed to create dataset: {response.status_code}")
        print(f"   Response: {response.text}")
        return None
    
    job_data = response.json()
    job_id = job_data.get('job_id')
    print(f"   ✅ Job created: {job_id}")
    return job_id

def wait_for_job(job_id, timeout=300):
    """Wait for job to complete."""
    print(f"   ⏳ Waiting for job {job_id} to complete...")
    
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(f"{BASE_URL}/api/v1/training/jobs/{job_id}")
        if response.status_code != 200:
            print(f"   ❌ Failed to get job status: {response.status_code}")
            return None
        
        data = response.json()
        job = data.get('job', data)  # Handle both nested and flat response
        status = job.get('status')
        progress = job.get('progress_percent', 0)
        
        if status == 'completed':
            print(f"   ✅ Job completed! Dataset ID: {job.get('dataset_id')}")
            return job.get('dataset_id')
        elif status == 'failed':
            print(f"   ❌ Job failed: {job.get('error_message')}")
            return None
        elif status == 'running':
            print(f"   ⏳ Progress: {progress:.1f}%")
        
        time.sleep(5)
    
    print(f"   ⏱️  Timeout after {timeout}s")
    return None

def get_dataset_metrics(dataset_id):
    """Get quality metrics from dataset."""
    response = requests.get(f"{BASE_URL}/api/v1/training/synthetic/datasets/{dataset_id}")
    if response.status_code != 200:
        print(f"   ❌ Failed to get dataset: {response.status_code}")
        return None
    
    dataset = response.json()
    metrics = dataset.get('quality_metrics', {})
    return metrics

def main():
    print("=" * 80)
    print("HEXAGONAL PLACEMENT API TEST")
    print("=" * 80)
    
    # Test hexagonal placement
    hex_job = create_dataset("test_hexagonal", "hexagonal")
    if not hex_job:
        return False
    
    hex_dataset = wait_for_job(hex_job)
    if not hex_dataset:
        return False
    
    hex_metrics = get_dataset_metrics(hex_dataset)
    if not hex_metrics:
        return False
    
    # Test random placement
    rand_job = create_dataset("test_random", "random")
    if not rand_job:
        return False
    
    rand_dataset = wait_for_job(rand_job)
    if not rand_dataset:
        return False
    
    rand_metrics = get_dataset_metrics(rand_dataset)
    if not rand_metrics:
        return False
    
    # Compare metrics
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 80}")
    
    hex_gdop = hex_metrics.get('mean_gdop', 999)
    rand_gdop = rand_metrics.get('mean_gdop', 999)
    
    print(f"\n   Hexagonal GDOP:  {hex_gdop:.1f}")
    print(f"   Random GDOP:     {rand_gdop:.1f}")
    
    if hex_gdop < rand_gdop:
        improvement = ((rand_gdop - hex_gdop) / rand_gdop) * 100
        print(f"   📈 Improvement:  {improvement:.1f}%")
        print(f"\n   ✅ Hexagonal placement is BETTER ({hex_gdop:.1f} < {rand_gdop:.1f})")
        success = True
    else:
        print(f"\n   ❌ Hexagonal placement is NOT better ({hex_gdop:.1f} >= {rand_gdop:.1f})")
        success = False
    
    print(f"\n{'=' * 80}\n")
    return success

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
