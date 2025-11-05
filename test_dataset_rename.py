#!/usr/bin/env python3
"""
Test script for dataset rename functionality.

Tests the PATCH /v1/training/jobs/synthetic/datasets/{dataset_id} endpoint.
"""

import sys
import requests
from uuid import uuid4

# Configuration
BASE_URL = "http://localhost:8001"
DATASET_ENDPOINT = f"{BASE_URL}/api/v1/training/synthetic/datasets"

def test_rename_nonexistent_dataset():
    """Test renaming a dataset that doesn't exist (should return 404)."""
    print("Test 1: Rename non-existent dataset")
    
    fake_id = str(uuid4())
    url = f"{DATASET_ENDPOINT}/{fake_id}"
    params = {"dataset_name": "New Name"}
    
    response = requests.patch(url, params=params)
    
    if response.status_code == 404:
        print("✓ Test 1 PASSED: Got expected 404 for non-existent dataset")
        return True
    else:
        print(f"✗ Test 1 FAILED: Expected 404, got {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def test_rename_with_empty_name():
    """Test renaming with empty name (should return 400)."""
    print("\nTest 2: Rename with empty name")
    
    # Use a fake UUID since we're testing validation
    fake_id = str(uuid4())
    url = f"{DATASET_ENDPOINT}/{fake_id}"
    params = {"dataset_name": ""}
    
    response = requests.patch(url, params=params)
    
    if response.status_code == 400:
        print("✓ Test 2 PASSED: Got expected 400 for empty name")
        return True
    else:
        print(f"✗ Test 2 FAILED: Expected 400, got {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def test_rename_with_long_name():
    """Test renaming with name exceeding 200 characters (should return 400)."""
    print("\nTest 3: Rename with name > 200 characters")
    
    fake_id = str(uuid4())
    url = f"{DATASET_ENDPOINT}/{fake_id}"
    params = {"dataset_name": "A" * 201}  # 201 characters
    
    response = requests.patch(url, params=params)
    
    if response.status_code == 400:
        print("✓ Test 3 PASSED: Got expected 400 for name > 200 chars")
        return True
    else:
        print(f"✗ Test 3 FAILED: Expected 400, got {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def test_list_datasets():
    """List existing datasets to find one for actual rename test."""
    print("\nTest 4: List existing datasets")
    
    url = f"{DATASET_ENDPOINT}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        datasets = data.get('datasets', [])
        print(f"✓ Found {len(datasets)} datasets")
        
        if datasets:
            # Return the first dataset for testing
            dataset = datasets[0]
            print(f"  Using dataset: {dataset['id']} (current name: '{dataset['name']}')")
            return dataset
        else:
            print("  No datasets available for rename testing")
            return None
    else:
        print(f"✗ Failed to list datasets: {response.status_code}")
        return None

def test_rename_existing_dataset(dataset_id: str, original_name: str):
    """Test renaming an existing dataset and reverting."""
    print(f"\nTest 5: Rename existing dataset {dataset_id}")
    
    # Rename to a new name
    new_name = f"{original_name} (Renamed Test)"
    url = f"{DATASET_ENDPOINT}/{dataset_id}"
    params = {"dataset_name": new_name}
    
    print(f"  Renaming '{original_name}' → '{new_name}'")
    response = requests.patch(url, params=params)
    
    if response.status_code == 200:
        print(f"✓ Rename successful")
        data = response.json()
        print(f"  Response: {data}")
        
        # Verify the name was changed
        verify_response = requests.get(DATASET_ENDPOINT)
        if verify_response.status_code == 200:
            datasets = verify_response.json().get('datasets', [])
            updated_dataset = next((d for d in datasets if d['id'] == dataset_id), None)
            
            if updated_dataset and updated_dataset['name'] == new_name:
                print(f"✓ Verified: Dataset name updated correctly")
                
                # Revert to original name
                print(f"  Reverting to original name: '{original_name}'")
                revert_params = {"dataset_name": original_name}
                revert_response = requests.patch(url, params=revert_params)
                
                if revert_response.status_code == 200:
                    print(f"✓ Reverted to original name successfully")
                    return True
                else:
                    print(f"⚠ Warning: Failed to revert name (status {revert_response.status_code})")
                    return False
            else:
                print(f"✗ Verification failed: Name not updated in database")
                return False
        else:
            print(f"⚠ Warning: Could not verify rename (status {verify_response.status_code})")
            return False
    else:
        print(f"✗ Test 5 FAILED: Expected 200, got {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Dataset Rename Functionality Tests")
    print("=" * 60)
    
    try:
        # Check if backend is running
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print(f"✓ Backend is running at {BASE_URL}\n")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to backend at {BASE_URL}")
        print(f"  Error: {e}")
        print("\nPlease ensure the backend service is running:")
        print("  docker-compose up -d backend")
        sys.exit(1)
    
    results = []
    
    # Run validation tests
    results.append(test_rename_nonexistent_dataset())
    results.append(test_rename_with_empty_name())
    results.append(test_rename_with_long_name())
    
    # Try to test with a real dataset
    dataset = test_list_datasets()
    if dataset:
        results.append(test_rename_existing_dataset(dataset['id'], dataset['name']))
    else:
        print("\n⚠ Skipping actual rename test (no datasets available)")
        print("  To create a dataset, use the Training UI or API")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests PASSED!")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
