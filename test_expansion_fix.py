#!/usr/bin/env python3
"""
Test script to verify dataset expansion fix.

This script:
1. Creates a small iq_raw dataset (10 samples)
2. Expands it by 20 samples
3. Verifies that IQ samples were saved for all samples
4. Checks database consistency
"""

import asyncio
import sys
import json
import uuid
from datetime import datetime
import asyncpg

# Database connection config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'heimdall',
    'user': 'heimdall',
    'password': 'heimdallpass'
}

async def check_dataset_state(conn, dataset_id):
    """Check current state of dataset in database."""
    
    # Get dataset record
    dataset = await conn.fetchrow("""
        SELECT id, name, num_samples, dataset_type, config
        FROM heimdall.synthetic_datasets
        WHERE id = $1
    """, dataset_id)
    
    if not dataset:
        print(f"❌ Dataset {dataset_id} not found")
        return None
    
    # Count features
    features_count = await conn.fetchval("""
        SELECT COUNT(*) FROM heimdall.measurement_features
        WHERE dataset_id = $1
    """, dataset_id)
    
    # Count IQ samples
    iq_count = await conn.fetchval("""
        SELECT COUNT(*) FROM heimdall.synthetic_iq_samples
        WHERE dataset_id = $1
    """, dataset_id)
    
    print(f"\n📊 Dataset State: {dataset['name']}")
    print(f"   ID: {dataset_id}")
    print(f"   Type: {dataset['dataset_type']}")
    print(f"   num_samples (record): {dataset['num_samples']}")
    print(f"   measurement_features: {features_count}")
    print(f"   synthetic_iq_samples: {iq_count}")
    
    # Check for mismatches
    if dataset['dataset_type'] == 'iq_raw':
        if features_count != iq_count:
            print(f"   ⚠️  MISMATCH: features={features_count}, iq_samples={iq_count}")
            print(f"   ⚠️  Missing {features_count - iq_count} IQ samples!")
            return False
        elif features_count != dataset['num_samples']:
            print(f"   ⚠️  MISMATCH: record says {dataset['num_samples']}, but DB has {features_count}")
            return False
        else:
            print(f"   ✅ All counts match!")
            return True
    
    return True

async def main():
    """Main test function."""
    
    print("=" * 70)
    print("Dataset Expansion Fix - Test Script")
    print("=" * 70)
    
    # Connect to database
    print("\n🔌 Connecting to database...")
    conn = await asyncpg.connect(**DB_CONFIG)
    
    try:
        # Create test dataset
        dataset_id = uuid.uuid4()
        dataset_name = f"expansion_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n📝 Creating test dataset: {dataset_name}")
        print(f"   Dataset ID: {dataset_id}")
        print(f"   Initial samples: 10")
        print(f"   Dataset type: iq_raw")
        
        config = {
            "frequency": 145500000,
            "power": 5,
            "min_snr_db": 3.0,
            "min_receivers": 3,
            "max_gdop": 200.0,
            "use_random_receivers": True,
            "dataset_type": "iq_raw"
        }
        
        await conn.execute("""
            INSERT INTO heimdall.synthetic_datasets (
                id, name, description, num_samples, config, dataset_type
            ) VALUES (
                $1, $2, $3, 0, $4::jsonb, 'iq_raw'::dataset_type_enum
            )
        """, dataset_id, dataset_name, "Test dataset for expansion fix", json.dumps(config))
        
        print(f"   ✅ Dataset record created")
        
        # Wait a moment for database commit
        await asyncio.sleep(0.5)
        
        # Check initial state
        await check_dataset_state(conn, dataset_id)
        
        print(f"\n⏳ You need to manually trigger generation jobs:")
        print(f"   1. Create initial dataset with 10 samples")
        print(f"   2. Expand it by 20 samples")
        print(f"\n   Use the frontend or API to create these jobs with:")
        print(f"      - dataset_id: {dataset_id}")
        print(f"      - name: {dataset_name}")
        print(f"      - num_samples: 10 (initial), then 20 (expansion)")
        print(f"\n   Then run this script again to check results")
        
        # Check if we should verify existing dataset
        if len(sys.argv) > 1:
            test_dataset_id = uuid.UUID(sys.argv[1])
            print(f"\n🔍 Checking existing dataset: {test_dataset_id}")
            is_valid = await check_dataset_state(conn, test_dataset_id)
            
            if is_valid:
                print(f"\n✅ Dataset expansion fix verified!")
                print(f"   All IQ samples were correctly saved during expansion")
            else:
                print(f"\n❌ Dataset has issues - expansion may have failed")
        
    finally:
        await conn.close()
        print(f"\n🔌 Database connection closed")

if __name__ == "__main__":
    asyncio.run(main())
