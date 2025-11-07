"""
Test Storage Cleanup and Monitoring System

This test verifies:
1. Lifecycle cleanup task is registered in Celery Beat
2. Storage metrics initialization works
3. Metrics endpoint responds correctly
4. Storage stats task updates metrics
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_lifecycle_cleanup_registered_in_celery_beat():
    """Verify that cleanup task is registered in Celery Beat schedule."""
    from services.backend.src.main import celery_app
    
    beat_schedule = celery_app.conf.beat_schedule
    
    # Check cleanup task is scheduled
    assert "minio-lifecycle-cleanup" in beat_schedule
    cleanup_task = beat_schedule["minio-lifecycle-cleanup"]
    assert cleanup_task["task"] == "tasks.minio_lifecycle.cleanup_orphan_files"
    assert cleanup_task["schedule"] == 86400.0  # Daily
    assert "dry_run" in cleanup_task["kwargs"]
    
    # Check stats task is scheduled
    assert "minio-storage-stats" in beat_schedule
    stats_task = beat_schedule["minio-storage-stats"]
    assert stats_task["task"] == "tasks.minio_lifecycle.get_storage_stats"
    assert stats_task["schedule"] == 3600.0  # Hourly


def test_storage_metrics_initialization():
    """Verify storage metrics initialize with zero values."""
    from services.backend.src.monitoring.storage_metrics import (
        init_storage_metrics,
        STORAGE_BUCKET_SIZE_GB,
        STORAGE_ORPHAN_COUNT,
    )
    
    # Initialize metrics
    init_storage_metrics()
    
    # Verify metrics exist for all buckets
    buckets = ["heimdall-synthetic-iq", "heimdall-audio-chunks", "heimdall-raw-iq"]
    for bucket in buckets:
        size = STORAGE_BUCKET_SIZE_GB.labels(bucket=bucket)._value.get()
        orphans = STORAGE_ORPHAN_COUNT.labels(bucket=bucket)._value.get()
        assert size == 0
        assert orphans == 0


def test_storage_metrics_update():
    """Verify storage metrics update correctly from stats."""
    from services.backend.src.monitoring.storage_metrics import (
        update_storage_metrics,
        STORAGE_BUCKET_SIZE_GB,
        STORAGE_ORPHAN_COUNT,
        STORAGE_ORPHAN_SIZE_GB,
    )
    
    # Mock storage stats
    stats = {
        "buckets": {
            "heimdall-synthetic-iq": {
                "total_objects": 1000,
                "total_size_gb": 50.5,
                "referenced_objects": 900,
                "orphan_objects": 100,
                "orphan_size_gb": 5.5,
            }
        },
        "timestamp": "2025-11-07T23:00:00",
    }
    
    # Update metrics
    update_storage_metrics(stats)
    
    # Verify metrics were updated
    bucket = "heimdall-synthetic-iq"
    size = STORAGE_BUCKET_SIZE_GB.labels(bucket=bucket)._value.get()
    orphans = STORAGE_ORPHAN_COUNT.labels(bucket=bucket)._value.get()
    orphan_size = STORAGE_ORPHAN_SIZE_GB.labels(bucket=bucket)._value.get()
    
    assert size == 50.5
    assert orphans == 100
    assert orphan_size == 5.5


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Verify Prometheus /metrics endpoint responds correctly."""
    from services.backend.src.routers.metrics import metrics
    
    # Call metrics endpoint
    response = await metrics()
    
    # Verify response
    assert response.status_code == 200
    assert "text/plain" in response.media_type
    assert b"heimdall_storage" in response.body


@pytest.mark.asyncio
async def test_storage_health_endpoint():
    """Verify /metrics/storage endpoint returns health status."""
    from services.backend.src.routers.metrics import storage_health
    from services.backend.src.monitoring.storage_metrics import init_storage_metrics
    
    # Initialize metrics
    init_storage_metrics()
    
    # Call storage health endpoint
    result = await storage_health()
    
    # Verify response structure
    assert "status" in result
    assert "total_size_gb" in result
    assert "total_orphans" in result
    assert "orphan_size_gb" in result
    assert "orphan_percentage" in result
    assert "buckets" in result
    assert result["status"] in ["healthy", "warning", "critical"]


def test_lifecycle_config():
    """Verify lifecycle configuration is correct."""
    from services.backend.src.tasks.minio_lifecycle import LIFECYCLE_CONFIG
    
    # Check all expected buckets are configured
    expected_buckets = [
        "heimdall-synthetic-iq",
        "heimdall-audio-chunks",
        "heimdall-raw-iq",
    ]
    
    for bucket in expected_buckets:
        assert bucket in LIFECYCLE_CONFIG
        config = LIFECYCLE_CONFIG[bucket]
        assert config["enabled"] is True
        assert config["min_age_days"] >= 30
        assert config["batch_size"] == 1000
        assert "description" in config


def test_dataset_deletion_calls_minio_cleanup():
    """Verify that delete_synthetic_dataset now calls MinIO cleanup."""
    import inspect
    from services.backend.src.routers.training import delete_synthetic_dataset
    
    # Get source code of the function
    source = inspect.getsource(delete_synthetic_dataset)
    
    # Verify it calls delete_dataset_iq_data
    assert "delete_dataset_iq_data" in source
    assert "minio_client" in source or "MinIOClient" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
