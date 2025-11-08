#!/usr/bin/env python3
"""
Cleanup orphaned files in MinIO by comparing with database references.
This script identifies files in MinIO that have no corresponding DB entry.
"""
import psycopg2
import boto3
from typing import Set, List
import os

# Database connection (use Docker service names when running inside container)
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'heimdall'),
    'user': os.getenv('POSTGRES_USER', 'heimdall_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'changeme')
}

# MinIO connection
MINIO_CONFIG = {
    'endpoint_url': os.getenv('MINIO_ENDPOINT', 'http://minio:9000'),
    'aws_access_key_id': os.getenv('MINIO_ROOT_USER', 'minioadmin'),
    'aws_secret_access_key': os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
}

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

def get_minio_client():
    """Get MinIO S3 client"""
    return boto3.client('s3', **MINIO_CONFIG)

def get_referenced_audio_chunks(conn) -> Set[str]:
    """Get all audio chunk paths referenced in DB"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT audio_id, chunk_index 
            FROM heimdall.audio_chunks
        """)
        chunks = set()
        for audio_id, chunk_index in cur.fetchall():
            # Format: {audio_id}/chunk_{chunk_index:04d}.npy
            path = f"{audio_id}/chunk_{chunk_index:04d}.npy"
            chunks.add(path)
        return chunks

def get_referenced_synthetic_iq(conn) -> Set[str]:
    """Get all synthetic IQ paths referenced in DB"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT dataset_id, sample_idx, iq_storage_paths 
            FROM heimdall.synthetic_iq_samples
        """)
        paths = set()
        for dataset_id, sample_idx, storage_paths in cur.fetchall():
            # storage_paths is a dict like {"RX_000": "synthetic/.../RX_000.npy", ...}
            if storage_paths:
                for rx_key, path in storage_paths.items():
                    paths.add(path)
        return paths

def list_minio_files(s3_client, bucket: str, prefix: str = '') -> List[str]:
    """List all files in MinIO bucket with prefix"""
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])
    return files

def delete_minio_files(s3_client, bucket: str, keys: List[str], dry_run: bool = True):
    """Delete files from MinIO (in batches of 1000)"""
    if not keys:
        print(f"No files to delete in {bucket}")
        return
    
    total = len(keys)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Deleting {total} files from {bucket}...")
    
    if dry_run:
        print(f"[DRY RUN] Would delete {total} files")
        print("Sample files to delete:")
        for key in keys[:10]:
            print(f"  - {key}")
        if total > 10:
            print(f"  ... and {total - 10} more")
        return
    
    # Delete in batches of 1000 (S3 limit)
    batch_size = 1000
    deleted = 0
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i + batch_size]
        objects = [{'Key': key} for key in batch]
        response = s3_client.delete_objects(
            Bucket=bucket,
            Delete={'Objects': objects, 'Quiet': True}
        )
        deleted += len(batch)
        print(f"Deleted {deleted}/{total} files...")
    
    print(f"✅ Successfully deleted {deleted} files from {bucket}")

def main(dry_run: bool = True):
    """Main cleanup function"""
    print("=" * 80)
    print("MinIO Orphan File Cleanup Script")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'LIVE (files WILL be deleted)'}")
    print()
    
    # Connect to database
    print("Connecting to database...")
    conn = get_db_connection()
    
    # Connect to MinIO
    print("Connecting to MinIO...")
    s3 = get_minio_client()
    
    try:
        # === AUDIO CHUNKS ===
        print("\n" + "=" * 80)
        print("ANALYZING AUDIO CHUNKS")
        print("=" * 80)
        
        print("Fetching referenced audio chunks from DB...")
        referenced_chunks = get_referenced_audio_chunks(conn)
        print(f"✓ Found {len(referenced_chunks)} referenced chunks in DB")
        
        print("Listing all files in MinIO bucket 'heimdall-audio-chunks'...")
        minio_chunks = set(list_minio_files(s3, 'heimdall-audio-chunks'))
        print(f"✓ Found {len(minio_chunks)} files in MinIO")
        
        orphan_chunks = minio_chunks - referenced_chunks
        print(f"\n🗑️  Found {len(orphan_chunks)} orphan audio chunks")
        print(f"📊 Storage: {len(orphan_chunks)} × ~800KB ≈ {len(orphan_chunks) * 0.8 / 1024:.2f} GB")
        
        if orphan_chunks:
            delete_minio_files(s3, 'heimdall-audio-chunks', list(orphan_chunks), dry_run)
        
        # === SYNTHETIC IQ ===
        print("\n" + "=" * 80)
        print("ANALYZING SYNTHETIC IQ")
        print("=" * 80)
        
        print("Fetching referenced synthetic IQ paths from DB...")
        referenced_iq = get_referenced_synthetic_iq(conn)
        print(f"✓ Found {len(referenced_iq)} referenced IQ files in DB")
        
        print("Listing all files in MinIO bucket 'heimdall-synthetic-iq'...")
        minio_iq = set(list_minio_files(s3, 'heimdall-synthetic-iq'))
        print(f"✓ Found {len(minio_iq)} files in MinIO")
        
        orphan_iq = minio_iq - referenced_iq
        print(f"\n🗑️  Found {len(orphan_iq)} orphan synthetic IQ files")
        print(f"📊 Storage: {len(orphan_iq)} × ~1.5MB ≈ {len(orphan_iq) * 1.5 / 1024:.2f} GB")
        
        if orphan_iq:
            delete_minio_files(s3, 'heimdall-synthetic-iq', list(orphan_iq), dry_run)
        
        # === SUMMARY ===
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total_orphans = len(orphan_chunks) + len(orphan_iq)
        total_gb = (len(orphan_chunks) * 0.8 + len(orphan_iq) * 1.5) / 1024
        print(f"Total orphan files: {total_orphans:,}")
        print(f"Total space to reclaim: ~{total_gb:.2f} GB")
        
        if dry_run:
            print("\n⚠️  This was a DRY RUN. No files were deleted.")
            print("To actually delete files, run with: python3 cleanup_orphan_minio_files.py --execute")
        else:
            print("\n✅ Cleanup complete!")
        
    finally:
        conn.close()

if __name__ == '__main__':
    import sys
    dry_run = '--execute' not in sys.argv
    main(dry_run=dry_run)
