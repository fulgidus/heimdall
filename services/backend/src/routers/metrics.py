"""Prometheus metrics endpoint router."""

import logging

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..monitoring.storage_metrics import get_storage_health_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all Heimdall metrics in Prometheus format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/storage")
async def storage_health():
    """
    Get current storage health status with fresh data.
    
    Calls the storage stats task synchronously to get up-to-date metrics.
    Returns human-readable storage health information.
    Useful for debugging and manual monitoring.
    
    Returns:
        dict: Storage statistics with size and orphan information
    """
    # Import here to avoid circular dependency
    from ..main import celery_app
    
    # Call the task synchronously to get fresh data
    task = celery_app.send_task("tasks.minio_lifecycle.get_storage_stats")
    result = task.get(timeout=60)  # Wait up to 60 seconds
    
    return result
