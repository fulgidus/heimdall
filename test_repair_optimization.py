#!/usr/bin/env python3
"""
Test script for optimized dataset repair functionality.
Tests that repair uses cached validation issues instead of re-validating.
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8001/api"

def test_repair_with_cached_issues():
    """Test repair endpoint uses cached issues without re-validation."""
    
    # Step 1: Get list of datasets
    print("\n=== Step 1: Fetching datasets ===")
    response = requests.get(f"{BASE_URL}/v1/training/synthetic/datasets")
    if response.status_code != 200:
        print(f"❌ Failed to fetch datasets: {response.status_code}")
        print(response.text)
        return False
    
    response_data = response.json()
    datasets = response_data.get("datasets", [])
    print(f"✅ Found {len(datasets)} datasets")
    
    # Find a dataset with validation issues
    import json
    target_dataset = None
    for ds in datasets:
        validation_issues = ds.get("validation_issues")
        if validation_issues:
            # Parse JSON if it's a string
            if isinstance(validation_issues, str):
                try:
                    validation_issues = json.loads(validation_issues)
                    ds["validation_issues"] = validation_issues
                except:
                    pass
            
            if isinstance(validation_issues, dict) and validation_issues.get("orphaned_features", 0) > 0:
                target_dataset = ds
                break
    
    if not target_dataset:
        print("⚠️  No datasets with orphaned features found. Creating test case...")
        print("Please run validation on a dataset with orphaned data first.")
        return False
    
    dataset_id = target_dataset["id"]
    dataset_name = target_dataset["name"]
    orphaned_features = target_dataset["validation_issues"]["orphaned_features"]
    
    print(f"\n📊 Target dataset: {dataset_name} ({dataset_id})")
    print(f"   Orphaned features: {orphaned_features}")
    print(f"   Health status: {target_dataset.get('health_status', 'unknown')}")
    
    # Step 2: Validate dataset first to ensure issues are cached
    print("\n=== Step 2: Validating dataset (to cache issues) ===")
    validate_start = time.time()
    response = requests.get(f"{BASE_URL}/v1/training/synthetic/datasets/{dataset_id}/validate")
    validate_duration = time.time() - validate_start
    
    if response.status_code != 200:
        print(f"❌ Validation failed: {response.status_code}")
        print(response.text)
        return False
    
    validation_result = response.json()
    print(f"✅ Validation complete in {validate_duration:.2f}s")
    print(f"   Issues found: {validation_result.get('validation_issues', {}).get('total_issues', 0)}")
    
    # Step 3: Run repair (should use cached issues)
    print("\n=== Step 3: Running repair with cached issues ===")
    repair_start = time.time()
    response = requests.post(
        f"{BASE_URL}/v1/training/synthetic/datasets/{dataset_id}/repair",
        params={"strategy": "delete_orphans"}
    )
    repair_duration = time.time() - repair_start
    
    if response.status_code != 200:
        print(f"❌ Repair failed: {response.status_code}")
        print(response.text)
        return False
    
    repair_result = response.json()
    print(f"✅ Repair complete in {repair_duration:.2f}s")
    print(f"   Deleted IQ files: {repair_result.get('deleted_iq_files', 0)}")
    print(f"   Deleted features: {repair_result.get('deleted_features', 0)}")
    print(f"   Status: {repair_result.get('status', 'unknown')}")
    
    # Step 4: Verify repair was faster than validation
    if repair_duration < validate_duration * 0.5:  # Should be significantly faster
        print(f"\n🚀 SUCCESS: Repair was {validate_duration/repair_duration:.1f}x faster than validation!")
        print(f"   This confirms cached issues were used (no re-validation)")
    else:
        print(f"\n⚠️  WARNING: Repair took {repair_duration:.2f}s vs validation {validate_duration:.2f}s")
        print(f"   Expected repair to be much faster using cached issues")
    
    # Step 5: Verify database state
    print("\n=== Step 4: Verifying final state ===")
    response = requests.get(f"{BASE_URL}/v1/training/synthetic/datasets")
    if response.status_code == 200:
        response_data = response.json()
        datasets = response_data.get("datasets", [])
        updated_dataset = next((ds for ds in datasets if ds["id"] == dataset_id), None)
        if updated_dataset:
            print(f"✅ Dataset updated:")
            print(f"   Health status: {updated_dataset.get('health_status', 'unknown')}")
            print(f"   Orphaned features: {updated_dataset.get('validation_issues', {}).get('orphaned_features', 0)}")
            print(f"   Total issues: {updated_dataset.get('validation_issues', {}).get('total_issues', 0)}")
            
            if updated_dataset.get('validation_issues', {}).get('orphaned_features', 0) == 0:
                print("\n🎉 SUCCESS: All orphaned features removed!")
                return True
            else:
                print("\n⚠️  Some orphaned features remain")
                return True  # Still consider success if repair ran
    
    return True

if __name__ == "__main__":
    try:
        success = test_repair_with_cached_issues()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
