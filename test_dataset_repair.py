#!/usr/bin/env python3
"""
Test the dataset repair functionality on VHF@5W_100MAXGDOP dataset.
This script validates the dataset, runs repair, and verifies the results.
Run this INSIDE the backend Docker container.
"""

import asyncio
import sys
from uuid import UUID

# Correct imports for backend container
from validators.dataset_validator import DatasetValidator
from storage.db_manager import DatabaseManager
from storage.minio_client import MinIOClient
from config import settings


async def main():
    """Test dataset repair on the problematic dataset."""
    
    # Dataset ID from the summary
    dataset_id = UUID("0d3b82ec-edba-4721-842c-c45d60a0f795")
    dataset_name = "VHF@5W_100MAXGDOP"
    
    print(f"\n{'='*80}")
    print(f"Testing Dataset Repair on: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    print(f"{'='*80}\n")
    
    # Initialize validator
    db_manager = DatabaseManager()
    minio_client = MinIOClient(
        endpoint_url=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY
    )
    validator = DatasetValidator(db_manager, minio_client)
    
    # Step 1: Validate dataset
    print("Step 1: Running validation...")
    print("-" * 80)
    report = validator.validate_dataset(dataset_id)
    
    print(f"\nValidation Results:")
    print(f"  Healthy: {report.is_healthy}")
    print(f"  Total Issues: {len(report.issues)}")
    
    # Count issues by type
    issue_counts = {}
    for issue in report.issues:
        issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
    
    print(f"\nIssue Breakdown:")
    for issue_type, count in issue_counts.items():
        print(f"  {issue_type}: {count}")
    
    if report.is_healthy:
        print("\n✅ Dataset is already healthy! No repair needed.")
        return
    
    # Step 2: Run repair
    print("\n" + "="*80)
    print("Step 2: Running repair with strategy 'delete_orphans'...")
    print("-" * 80)
    
    # Convert issues to dict format for caching
    cached_issues = [
        {
            "issue_type": issue.issue_type,
            "path": issue.path,
            "details": issue.details
        }
        for issue in report.issues
    ]
    
    result = validator.repair_dataset(
        dataset_id=dataset_id,
        strategy="delete_orphans",
        cached_issues=cached_issues
    )
    
    print(f"\nRepair Results:")
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print(f"  Deleted IQ Files: {result['deleted_iq_files']}")
    print(f"  Deleted Features: {result['deleted_features']}")
    
    # Step 3: Re-validate to confirm health
    print("\n" + "="*80)
    print("Step 3: Re-validating dataset after repair...")
    print("-" * 80)
    
    report_after = validator.validate_dataset(dataset_id)
    
    print(f"\nValidation Results After Repair:")
    print(f"  Healthy: {report_after.is_healthy}")
    print(f"  Total Issues: {len(report_after.issues)}")
    
    if report_after.issues:
        issue_counts_after = {}
        for issue in report_after.issues:
            issue_counts_after[issue.issue_type] = issue_counts_after.get(issue.issue_type, 0) + 1
        
        print(f"\nRemaining Issues:")
        for issue_type, count in issue_counts_after.items():
            print(f"  {issue_type}: {count}")
    
    # Step 4: Check database state
    print("\n" + "="*80)
    print("Step 4: Checking database state...")
    print("-" * 80)
    
    from sqlalchemy import text
    with db_manager.get_session() as session:
        # Count samples in synthetic_iq_samples
        samples_query = text("""
            SELECT COUNT(*) 
            FROM heimdall.synthetic_iq_samples 
            WHERE dataset_id = :dataset_id
        """)
        samples_count = session.execute(samples_query, {"dataset_id": str(dataset_id)}).scalar()
        
        # Count features in measurement_features
        features_query = text("""
            SELECT COUNT(*) 
            FROM heimdall.measurement_features 
            WHERE dataset_id = :dataset_id
        """)
        features_count = session.execute(features_query, {"dataset_id": str(dataset_id)}).scalar()
        
        # Get dataset info
        dataset_query = text("""
            SELECT name, dataset_type, total_samples
            FROM heimdall.synthetic_datasets
            WHERE id = :dataset_id
        """)
        dataset_info = session.execute(dataset_query, {"dataset_id": str(dataset_id)}).fetchone()
    
    print(f"\nDatabase State:")
    print(f"  Dataset Name: {dataset_info[0]}")
    print(f"  Dataset Type: {dataset_info[1]}")
    print(f"  Total Samples (metadata): {dataset_info[2]}")
    print(f"  Samples in synthetic_iq_samples: {samples_count}")
    print(f"  Features in measurement_features: {features_count}")
    
    # Summary
    print("\n" + "="*80)
    if report_after.is_healthy:
        print("✅ SUCCESS: Dataset is now healthy!")
    else:
        print("⚠️  WARNING: Dataset still has issues after repair")
        print(f"    Remaining issues: {len(report_after.issues)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
