"""
MinIO Lifecycle Management - Periodic Cleanup Task

This module implements automatic cleanup of orphaned MinIO files that are not
referenced in the database. This prevents disk space leaks from failed operations
or improper deletion workflows.

Runs daily at 3 AM via Celery Beat schedule.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from celery import shared_task
from sqlalchemy import text

from ..config import settings
from ..storage.db_manager import get_db_manager
from ..storage.minio_client import MinIOClient

logger = logging.getLogger(__name__)


# Configuration for lifecycle policies
LIFECYCLE_CONFIG = {
    "heimdall-synthetic-iq": {
        "enabled": True,
        "min_age_days": 30,  # Only delete orphans older than 30 days
        "batch_size": 1000,
        "description": "Synthetic IQ data files for training datasets"
    },
    "heimdall-audio-chunks": {
        "enabled": True,
        "min_age_days": 30,
        "batch_size": 1000,
        "description": "Preprocessed audio chunks for training"
    },
    "heimdall-raw-iq": {
        "enabled": True,
        "min_age_days": 60,  # Keep raw IQ longer for debugging
        "batch_size": 1000,
        "description": "Raw IQ data from WebSDR acquisition sessions"
    },
}


@shared_task(
    name="tasks.minio_lifecycle.cleanup_orphan_files",
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
)
def cleanup_orphan_files(self, dry_run: bool = False) -> Dict[str, any]:
    """
    Periodic task to cleanup orphaned MinIO files not referenced in database.
    
    This task prevents disk space leaks by:
    1. Identifying files in MinIO that have no corresponding DB records
    2. Filtering out recently created files (within min_age_days threshold)
    3. Deleting orphans in batches to avoid memory issues
    
    Args:
        dry_run: If True, only report what would be deleted without actual deletion
    
    Returns:
        dict: Summary of cleanup operation
            {
                "status": "success" | "partial" | "failed",
                "buckets": {
                    "bucket_name": {
                        "orphans_found": int,
                        "orphans_deleted": int,
                        "deletion_failures": int,
                        "space_recovered_gb": float
                    }
                },
                "total_orphans_found": int,
                "total_deleted": int,
                "total_failed": int,
                "total_space_recovered_gb": float,
                "dry_run": bool,
                "execution_time_seconds": float
            }
    """
    start_time = datetime.now()
    
    logger.info(
        "Starting MinIO lifecycle cleanup task",
        extra={"dry_run": dry_run, "task_id": self.request.id}
    )
    
    result = {
        "status": "success",
        "buckets": {},
        "total_orphans_found": 0,
        "total_deleted": 0,
        "total_failed": 0,
        "total_space_recovered_gb": 0.0,
        "dry_run": dry_run,
        "execution_time_seconds": 0.0,
        "started_at": start_time.isoformat(),
    }
    
    db_manager = get_db_manager()
    
    try:
        # Process each configured bucket
        for bucket_name, config in LIFECYCLE_CONFIG.items():
            if not config["enabled"]:
                logger.info(f"Skipping disabled bucket: {bucket_name}")
                continue
            
            logger.info(
                f"Processing bucket: {bucket_name}",
                extra={
                    "min_age_days": config["min_age_days"],
                    "batch_size": config["batch_size"]
                }
            )
            
            # Initialize MinIO client for this bucket
            minio_client = MinIOClient(
                endpoint_url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name=bucket_name,
            )
            
            # Get orphan files for this bucket
            bucket_result = _cleanup_bucket_orphans(
                db_manager=db_manager,
                minio_client=minio_client,
                bucket_name=bucket_name,
                min_age_days=config["min_age_days"],
                batch_size=config["batch_size"],
                dry_run=dry_run,
            )
            
            result["buckets"][bucket_name] = bucket_result
            result["total_orphans_found"] += bucket_result["orphans_found"]
            result["total_deleted"] += bucket_result["orphans_deleted"]
            result["total_failed"] += bucket_result["deletion_failures"]
            result["total_space_recovered_gb"] += bucket_result["space_recovered_gb"]
        
        # Determine overall status
        if result["total_failed"] > 0:
            result["status"] = "partial"
        
        end_time = datetime.now()
        result["execution_time_seconds"] = (end_time - start_time).total_seconds()
        result["completed_at"] = end_time.isoformat()
        
        logger.info(
            "MinIO lifecycle cleanup completed",
            extra={
                "status": result["status"],
                "total_orphans_found": result["total_orphans_found"],
                "total_deleted": result["total_deleted"],
                "total_failed": result["total_failed"],
                "space_recovered_gb": result["total_space_recovered_gb"],
                "execution_time_seconds": result["execution_time_seconds"],
                "dry_run": dry_run,
            }
        )
        
        # Update storage metrics after cleanup
        try:
            from ..monitoring.storage_metrics import update_storage_metrics
            stats = get_storage_stats()
            update_storage_metrics(stats)
            logger.info("Storage metrics updated after cleanup")
        except Exception as e:
            logger.warning(f"Failed to update storage metrics: {e}")
        
        return result
    
    except Exception as e:
        logger.error(f"MinIO lifecycle cleanup failed: {e}", exc_info=True)
        result["status"] = "failed"
        result["error"] = str(e)
        
        # Retry task with exponential backoff
        raise self.retry(exc=e)


def _cleanup_bucket_orphans(
    db_manager,
    minio_client: MinIOClient,
    bucket_name: str,
    min_age_days: int,
    batch_size: int,
    dry_run: bool,
) -> Dict[str, any]:
    """
    Cleanup orphaned files in a specific bucket.
    
    Args:
        db_manager: Database manager instance
        minio_client: MinIO client for this bucket
        bucket_name: Name of the bucket to process
        min_age_days: Only delete files older than this many days
        batch_size: Number of files to delete per batch
        dry_run: If True, don't actually delete files
    
    Returns:
        dict: Cleanup summary for this bucket
    """
    result = {
        "orphans_found": 0,
        "orphans_deleted": 0,
        "deletion_failures": 0,
        "space_recovered_gb": 0.0,
    }
    
    try:
        # Get referenced files from database based on bucket type
        if bucket_name == "heimdall-synthetic-iq":
            referenced_files = _get_referenced_synthetic_iq_files(db_manager)
        elif bucket_name == "heimdall-audio-chunks":
            referenced_files = _get_referenced_audio_chunks(db_manager)
        elif bucket_name == "heimdall-raw-iq":
            referenced_files = _get_referenced_raw_iq_files(db_manager)
        else:
            logger.warning(f"Unknown bucket type: {bucket_name}, skipping")
            return result
        
        logger.info(f"Found {len(referenced_files)} referenced files in {bucket_name}")
        
        # Get all files in MinIO bucket
        all_minio_files = _list_all_minio_files(minio_client, bucket_name)
        logger.info(f"Found {len(all_minio_files)} total files in MinIO bucket {bucket_name}")
        
        # Find orphans (files in MinIO but not in DB)
        orphan_files = set(all_minio_files.keys()) - referenced_files
        result["orphans_found"] = len(orphan_files)
        
        if result["orphans_found"] == 0:
            logger.info(f"No orphan files found in {bucket_name}")
            return result
        
        # Filter by age (only delete old orphans)
        age_cutoff = datetime.now() - timedelta(days=min_age_days)
        old_orphans = []
        total_size_bytes = 0
        
        for key in orphan_files:
            file_info = all_minio_files[key]
            if file_info["last_modified"] < age_cutoff:
                old_orphans.append(key)
                total_size_bytes += file_info["size"]
        
        result["space_recovered_gb"] = total_size_bytes / (1024**3)
        
        logger.info(
            f"Found {len(old_orphans)} orphans older than {min_age_days} days in {bucket_name}",
            extra={
                "total_orphans": result["orphans_found"],
                "old_orphans": len(old_orphans),
                "estimated_space_gb": result["space_recovered_gb"]
            }
        )
        
        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(old_orphans)} files from {bucket_name}")
            result["orphans_deleted"] = len(old_orphans)
            return result
        
        # Delete orphans in batches
        for i in range(0, len(old_orphans), batch_size):
            batch = old_orphans[i : i + batch_size]
            
            try:
                response = minio_client.s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": [{"Key": key} for key in batch]}
                )
                
                if "Deleted" in response:
                    result["orphans_deleted"] += len(response["Deleted"])
                
                if "Errors" in response:
                    result["deletion_failures"] += len(response["Errors"])
                    for error in response["Errors"]:
                        logger.error(
                            f"Failed to delete {error['Key']}: {error['Message']}",
                            extra={"bucket": bucket_name}
                        )
                
                logger.info(
                    f"Deleted batch {i // batch_size + 1}: "
                    f"{result['orphans_deleted']}/{len(old_orphans)} files"
                )
            
            except Exception as e:
                logger.error(f"Failed to delete batch in {bucket_name}: {e}", exc_info=True)
                result["deletion_failures"] += len(batch)
        
        logger.info(
            f"Completed cleanup for {bucket_name}",
            extra={
                "deleted": result["orphans_deleted"],
                "failed": result["deletion_failures"],
                "space_recovered_gb": result["space_recovered_gb"]
            }
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error cleaning up bucket {bucket_name}: {e}", exc_info=True)
        return result


def _get_referenced_synthetic_iq_files(db_manager) -> set:
    """Get all synthetic IQ file paths referenced in database."""
    query = text("""
        SELECT DISTINCT jsonb_each_text.value as file_path
        FROM heimdall.synthetic_iq_samples,
             jsonb_each_text(iq_storage_paths)
    """)
    
    with db_manager.get_session() as session:
        result = session.execute(query)
        return {row[0] for row in result}


def _get_referenced_audio_chunks(db_manager) -> set:
    """Get all audio chunk file paths referenced in database."""
    query = text("""
        SELECT minio_path
        FROM heimdall.audio_chunks
    """)
    
    with db_manager.get_session() as session:
        result = session.execute(query)
        return {row[0] for row in result}


def _get_referenced_raw_iq_files(db_manager) -> set:
    """Get all raw IQ file paths referenced in database."""
    query = text("""
        SELECT DISTINCT iq_data_location
        FROM heimdall.measurements
        WHERE iq_data_location IS NOT NULL
    """)
    
    with db_manager.get_session() as session:
        result = session.execute(query)
        return {row[0] for row in result if row[0]}


def _list_all_minio_files(minio_client: MinIOClient, bucket_name: str) -> Dict[str, Dict]:
    """
    List all files in a MinIO bucket with metadata.
    
    Returns:
        dict: {file_key: {"size": bytes, "last_modified": datetime}}
    """
    files = {}
    
    try:
        paginator = minio_client.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name)
        
        for page in pages:
            if "Contents" not in page:
                continue
            
            for obj in page["Contents"]:
                files[obj["Key"]] = {
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                }
    
    except Exception as e:
        logger.error(f"Error listing files in {bucket_name}: {e}", exc_info=True)
    
    return files


@shared_task(name="tasks.minio_lifecycle.get_storage_stats")
def get_storage_stats() -> Dict[str, any]:
    """
    Get current storage statistics for all configured buckets.
    
    Used for monitoring and alerting.
    
    Returns:
        dict: Storage statistics
            {
                "buckets": {
                    "bucket_name": {
                        "total_objects": int,
                        "total_size_gb": float,
                        "referenced_objects": int,
                        "orphan_objects": int,
                        "orphan_size_gb": float
                    }
                },
                "timestamp": str
            }
    """
    result = {
        "buckets": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    db_manager = get_db_manager()
    
    for bucket_name, config in LIFECYCLE_CONFIG.items():
        if not config["enabled"]:
            continue
        
        try:
            minio_client = MinIOClient(
                endpoint_url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name=bucket_name,
            )
            
            # Get referenced files
            if bucket_name == "heimdall-synthetic-iq":
                referenced_files = _get_referenced_synthetic_iq_files(db_manager)
            elif bucket_name == "heimdall-audio-chunks":
                referenced_files = _get_referenced_audio_chunks(db_manager)
            elif bucket_name == "heimdall-raw-iq":
                referenced_files = _get_referenced_raw_iq_files(db_manager)
            else:
                continue
            
            # Get all MinIO files
            all_files = _list_all_minio_files(minio_client, bucket_name)
            
            # Calculate statistics
            total_size = sum(f["size"] for f in all_files.values())
            orphan_keys = set(all_files.keys()) - referenced_files
            orphan_size = sum(all_files[k]["size"] for k in orphan_keys)
            
            result["buckets"][bucket_name] = {
                "total_objects": len(all_files),
                "total_size_gb": total_size / (1024**3),
                "referenced_objects": len(referenced_files),
                "orphan_objects": len(orphan_keys),
                "orphan_size_gb": orphan_size / (1024**3),
            }
        
        except Exception as e:
            logger.error(f"Error getting stats for {bucket_name}: {e}", exc_info=True)
            result["buckets"][bucket_name] = {"error": str(e)}
    
    # Update Prometheus metrics
    try:
        from ..monitoring.storage_metrics import update_storage_metrics
        update_storage_metrics(result)
        logger.debug("Storage metrics updated from get_storage_stats task")
    except Exception as e:
        logger.warning(f"Failed to update storage metrics: {e}")
    
    return result
