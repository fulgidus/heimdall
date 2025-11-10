#!/usr/bin/env python3
"""
Migrate existing synthetic datasets from 1000ms to 200ms samples.

This script:
1. Deletes all existing training samples and IQ files from MinIO
2. Regenerates ALL synthetic datasets with 200ms duration
3. Updates database records accordingly

WARNING: This will REGENERATE all training data!
"""

import asyncio
import asyncpg
import sys
from urllib.parse import urlparse


async def migrate_to_200ms():
    """Migrate all synthetic datasets to 200ms samples."""
    
    # Database connection (use postgres for Docker network)
    DATABASE_URL = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    db_url = urlparse(DATABASE_URL)
    
    pool = await asyncpg.create_pool(
        user=db_url.username,
        password=db_url.password,
        database=db_url.path.lstrip("/"),
        host=db_url.hostname,
        port=db_url.port or 5432,
        min_size=2,
        max_size=10
    )
    
    try:
        # Step 1: Get all synthetic dataset IDs
        datasets = await pool.fetch("""
            SELECT id, name, config, sample_count, created_at
            FROM synthetic_datasets
            WHERE status = 'READY'
            ORDER BY created_at DESC
        """)
        
        print(f"\n🔍 Found {len(datasets)} synthetic datasets to migrate\n")
        
        for dataset in datasets:
            print(f"📦 Dataset: {dataset['name']}")
            print(f"   ID: {dataset['id']}")
            print(f"   Samples: {dataset['sample_count']}")
            print(f"   Created: {dataset['created_at']}")
        
        # Ask for confirmation
        print("\n⚠️  WARNING: This will DELETE all existing samples and REGENERATE them with 200ms duration!")
        print("⚠️  All MinIO IQ files will be deleted and recreated.")
        print("⚠️  This operation cannot be undone!\n")
        
        response = input("Type 'YES' to proceed: ")
        if response != "YES":
            print("❌ Migration cancelled")
            return
        
        # Step 2: Delete all training samples
        print("\n🗑️  Deleting existing training samples from database...")
        deleted_count = await pool.fetchval("""
            DELETE FROM training_samples
            WHERE dataset_id IN (SELECT id FROM synthetic_datasets)
            RETURNING COUNT(*)
        """)
        print(f"✅ Deleted {deleted_count} training samples")
        
        # Step 3: Update datasets status to PENDING for regeneration
        print("\n🔄 Marking datasets for regeneration...")
        await pool.execute("""
            UPDATE synthetic_datasets
            SET status = 'PENDING',
                sample_count = 0,
                updated_at = NOW()
            WHERE status = 'READY'
        """)
        print("✅ Datasets marked as PENDING")
        
        # Step 4: MinIO cleanup instructions
        print("\n📝 Next steps:")
        print("1. Clean up MinIO buckets manually or via script:")
        print("   docker exec heimdall-minio-1 mc rm --recursive --force minio/heimdall-training-data/")
        print("   docker exec heimdall-minio-1 mc rm --recursive --force minio/heimdall-audio-chunks/")
        print("\n2. Regenerate datasets using the training service API:")
        print("   curl -X POST http://localhost:8003/api/v1/jobs/synthetic/generate \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"num_samples\": <count>, \"config\": {...}}'")
        print("\n3. Verify new samples have 200ms duration:")
        print("   SELECT iq_metadata->>'duration_ms' FROM training_samples LIMIT 1;")
        
        print("\n✅ Database migration complete!")
        print("⚠️  Remember to regenerate datasets via API after MinIO cleanup\n")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(migrate_to_200ms())
