#!/usr/bin/env python3
"""
Debug script to verify dataset validation persistence.
This script checks if validation_issues are properly saved and retrieved.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'backend', 'src'))

from backend.src.database import get_db_manager
from sqlalchemy import text
import json

def check_validation_issues():
    """Check validation_issues in database."""
    db_manager = get_db_manager()
    
    print("=" * 80)
    print("CHECKING DATASET VALIDATION PERSISTENCE")
    print("=" * 80)
    
    with db_manager.get_session() as session:
        # Get all datasets with their validation_issues
        query = text("""
            SELECT 
                id,
                name,
                health_status,
                last_validated_at,
                validation_issues,
                storage_size_bytes,
                num_samples
            FROM heimdall.synthetic_datasets
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        results = session.execute(query).fetchall()
        
        if not results:
            print("\n❌ NO DATASETS FOUND")
            return
        
        print(f"\nFound {len(results)} dataset(s)\n")
        
        for row in results:
            dataset_id = row[0]
            name = row[1]
            health_status = row[2]
            last_validated_at = row[3]
            validation_issues = row[4]  # This is JSONB, should be a dict
            storage_size_bytes = row[5]
            num_samples = row[6]
            
            print(f"Dataset: {name}")
            print(f"  ID: {dataset_id}")
            print(f"  Health Status: {health_status}")
            print(f"  Last Validated: {last_validated_at}")
            print(f"  Num Samples: {num_samples}")
            print(f"  Storage Size: {storage_size_bytes}")
            print(f"  Validation Issues Type: {type(validation_issues)}")
            
            if validation_issues:
                print(f"  Validation Issues Content:")
                if isinstance(validation_issues, dict):
                    print(f"    ✅ Is dict (correct)")
                    for key, value in validation_issues.items():
                        print(f"      {key}: {value}")
                else:
                    print(f"    ❌ Not a dict: {validation_issues}")
            else:
                print(f"  Validation Issues: NULL")
            
            print()

if __name__ == "__main__":
    check_validation_issues()
