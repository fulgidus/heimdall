"""
Storage Metrics for Prometheus Monitoring

This module provides Prometheus metrics for MinIO storage monitoring:
- Disk usage tracking
- Per-bucket size monitoring  
- Orphan file detection and tracking
- Alert thresholds for disk space issues

Metrics are updated by the lifecycle cleanup task and exposed via /metrics endpoint.
"""

import logging
from typing import Dict

from prometheus_client import Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
STORAGE_DISK_USAGE_GB = Gauge(
    "heimdall_storage_disk_usage_gb",
    "Total disk space used by MinIO in GB",
    ["bucket"],
)

STORAGE_BUCKET_SIZE_GB = Gauge(
    "heimdall_storage_bucket_size_gb",
    "Size of each MinIO bucket in GB",
    ["bucket"],
)

STORAGE_ORPHAN_COUNT = Gauge(
    "heimdall_storage_orphan_files",
    "Number of orphaned files (not referenced in DB)",
    ["bucket"],
)

STORAGE_ORPHAN_SIZE_GB = Gauge(
    "heimdall_storage_orphan_size_gb",
    "Size of orphaned files in GB",
    ["bucket"],
)

STORAGE_TOTAL_OBJECTS = Gauge(
    "heimdall_storage_total_objects",
    "Total number of objects in bucket",
    ["bucket"],
)

STORAGE_REFERENCED_OBJECTS = Gauge(
    "heimdall_storage_referenced_objects",
    "Number of objects referenced in database",
    ["bucket"],
)


def init_storage_metrics():
    """
    Initialize storage metrics with zero values.
    
    Call this on application startup to ensure all metrics exist.
    """
    logger.info("Initializing storage metrics...")
    
    buckets = ["heimdall-synthetic-iq", "heimdall-audio-chunks", "heimdall-raw-iq"]
    
    for bucket in buckets:
        STORAGE_DISK_USAGE_GB.labels(bucket=bucket).set(0)
        STORAGE_BUCKET_SIZE_GB.labels(bucket=bucket).set(0)
        STORAGE_ORPHAN_COUNT.labels(bucket=bucket).set(0)
        STORAGE_ORPHAN_SIZE_GB.labels(bucket=bucket).set(0)
        STORAGE_TOTAL_OBJECTS.labels(bucket=bucket).set(0)
        STORAGE_REFERENCED_OBJECTS.labels(bucket=bucket).set(0)
    
    logger.info("Storage metrics initialized")


def update_storage_metrics(stats: Dict[str, any]):
    """
    Update Prometheus metrics from storage statistics.
    
    Args:
        stats: Dictionary from get_storage_stats() task
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
    if "buckets" not in stats:
        logger.warning("No bucket data in storage stats")
        return
    
    for bucket_name, bucket_stats in stats["buckets"].items():
        if "error" in bucket_stats:
            logger.error(
                f"Error in storage stats for {bucket_name}: {bucket_stats['error']}"
            )
            continue
        
        try:
            # Update metrics
            STORAGE_BUCKET_SIZE_GB.labels(bucket=bucket_name).set(
                bucket_stats.get("total_size_gb", 0)
            )
            STORAGE_TOTAL_OBJECTS.labels(bucket=bucket_name).set(
                bucket_stats.get("total_objects", 0)
            )
            STORAGE_REFERENCED_OBJECTS.labels(bucket=bucket_name).set(
                bucket_stats.get("referenced_objects", 0)
            )
            STORAGE_ORPHAN_COUNT.labels(bucket=bucket_name).set(
                bucket_stats.get("orphan_objects", 0)
            )
            STORAGE_ORPHAN_SIZE_GB.labels(bucket=bucket_name).set(
                bucket_stats.get("orphan_size_gb", 0)
            )
            
            # Disk usage is same as total size for MinIO buckets
            STORAGE_DISK_USAGE_GB.labels(bucket=bucket_name).set(
                bucket_stats.get("total_size_gb", 0)
            )
            
            logger.debug(
                f"Updated metrics for {bucket_name}: "
                f"{bucket_stats.get('total_size_gb', 0):.2f} GB, "
                f"{bucket_stats.get('orphan_objects', 0)} orphans"
            )
        
        except Exception as e:
            logger.error(
                f"Failed to update metrics for {bucket_name}: {e}",
                exc_info=True
            )
    
    logger.info(
        f"Storage metrics updated at {stats.get('timestamp', 'unknown time')}"
    )


def get_storage_health_status() -> Dict[str, any]:
    """
    Get current storage health status for health checks.
    
    Returns:
        dict: {
            "status": "healthy" | "warning" | "critical",
            "total_size_gb": float,
            "total_orphans": int,
            "orphan_size_gb": float,
            "buckets": {
                "bucket_name": {
                    "size_gb": float,
                    "orphans": int,
                    "orphan_size_gb": float
                }
            }
        }
    """
    buckets_info = {}
    total_size_gb = 0.0
    total_orphans = 0
    total_orphan_size_gb = 0.0
    
    buckets = ["heimdall-synthetic-iq", "heimdall-audio-chunks", "heimdall-raw-iq"]
    
    for bucket in buckets:
        size_gb = STORAGE_BUCKET_SIZE_GB.labels(bucket=bucket)._value.get()
        orphans = STORAGE_ORPHAN_COUNT.labels(bucket=bucket)._value.get()
        orphan_size_gb = STORAGE_ORPHAN_SIZE_GB.labels(bucket=bucket)._value.get()
        
        buckets_info[bucket] = {
            "size_gb": size_gb,
            "orphans": int(orphans),
            "orphan_size_gb": orphan_size_gb,
        }
        
        total_size_gb += size_gb
        total_orphans += int(orphans)
        total_orphan_size_gb += orphan_size_gb
    
    # Determine health status based on orphan percentage
    orphan_percentage = (
        (total_orphan_size_gb / total_size_gb * 100) if total_size_gb > 0 else 0
    )
    
    if orphan_percentage > 25:
        status = "critical"  # More than 25% orphaned data
    elif orphan_percentage > 10:
        status = "warning"  # More than 10% orphaned data
    else:
        status = "healthy"
    
    return {
        "status": status,
        "total_size_gb": total_size_gb,
        "total_orphans": total_orphans,
        "orphan_size_gb": total_orphan_size_gb,
        "orphan_percentage": orphan_percentage,
        "buckets": buckets_info,
    }
