"""Admin endpoints for system maintenance."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..db import get_pool
from ..tasks.batch_feature_extraction import (
    backfill_all_features,
    batch_feature_extraction_task,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/features/batch-extract")
async def trigger_batch_extraction(
    batch_size: int = 50,
    max_batches: int = 5,
):
    """
    Manually trigger batch feature extraction.

    Args:
        batch_size: Recordings per batch
        max_batches: Maximum batches to process

    Returns:
        Task result
    """
    try:
        task = batch_feature_extraction_task.apply_async(
            kwargs={'batch_size': batch_size, 'max_batches': max_batches},
            queue='celery'
        )

        return {
            "task_id": task.id,
            "message": f"Batch extraction started (batch_size={batch_size}, max_batches={max_batches})",
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Error triggering batch extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/backfill-all")
async def trigger_full_backfill():
    """
    Trigger full backfill of features for ALL recordings.

    WARNING: This may queue thousands of tasks. Use with caution.

    Returns:
        Task result
    """
    try:
        task = backfill_all_features.apply_async(queue='celery')

        return {
            "task_id": task.id,
            "message": "Full backfill started - this may take hours",
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Error triggering full backfill: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/stats")
async def get_feature_extraction_stats(
    pool=Depends(get_pool)
):
    """
    Get statistics about feature extraction coverage.

    Returns:
        dict with coverage statistics
    """
    try:
        query = """
            WITH recording_stats AS (
                SELECT
                    COUNT(DISTINCT rs.id) as total_recordings,
                    COUNT(DISTINCT mf.recording_session_id) as recordings_with_features,
                    COUNT(m.id) as total_measurements
                FROM heimdall.recording_sessions rs
                LEFT JOIN heimdall.measurements m
                    ON m.created_at >= rs.session_start
                    AND (rs.session_end IS NULL OR m.created_at <= rs.session_end)
                    AND m.iq_data_location IS NOT NULL
                LEFT JOIN heimdall.measurement_features mf
                    ON mf.recording_session_id = rs.id
                WHERE rs.status = 'completed'
            )
            SELECT * FROM recording_stats
        """

        async with pool.acquire() as conn:
            result = await conn.fetchrow(query)

            if not result:
                return {
                    'total_recordings': 0,
                    'recordings_with_features': 0,
                    'recordings_without_features': 0,
                    'total_measurements': 0,
                    'coverage_percent': 0.0
                }

            total_recordings = result['total_recordings'] or 0
            recordings_with_features = result['recordings_with_features'] or 0
            total_measurements = result['total_measurements'] or 0

            coverage_percent = (
                recordings_with_features / total_recordings * 100
                if total_recordings > 0 else 0.0
            )

            return {
                'total_recordings': total_recordings,
                'recordings_with_features': recordings_with_features,
                'recordings_without_features': total_recordings - recordings_with_features,
                'total_measurements': total_measurements,
                'coverage_percent': round(coverage_percent, 2)
            }

    except Exception as e:
        logger.error(f"Error fetching feature extraction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
