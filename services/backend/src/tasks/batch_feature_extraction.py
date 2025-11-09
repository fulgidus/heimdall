"""
Background Celery task for batch feature extraction.

Processes recordings that don't have features yet.
"""

import logging

from celery import shared_task

from ..db import get_pool

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="backend.tasks.batch_feature_extraction")
def batch_feature_extraction_task(
    self, batch_size: int = 50, max_batches: int | None = None
) -> dict:
    """
    Find recordings without features and queue extraction tasks.

    Args:
        batch_size: Number of recordings to process per batch
        max_batches: Maximum number of batches to process (None = unlimited)

    Returns:
        dict with statistics
    """
    logger.info(
        f"Starting batch feature extraction (batch_size={batch_size}, max_batches={max_batches})"
    )

    try:
        # Get database pool and run async operations
        import asyncio
        from ..celery_worker import get_worker_loop

        async def run_batch():
            pool = get_pool()

            # Find recordings without features
            recordings_without_features = await _find_recordings_without_features(
                pool, batch_size=batch_size, max_batches=max_batches
            )

            logger.info(f"Found {len(recordings_without_features)} recordings without features")

            if not recordings_without_features:
                return {
                    "total_found": 0,
                    "tasks_queued": 0,
                    "message": "No recordings without features",
                }

            # Queue extraction tasks
            task_ids = []
            for recording in recordings_without_features:
                try:
                    # Import celery_app from main module to access shared task
                    from ..main import celery_app

                    task = celery_app.send_task(
                        "backend.tasks.extract_recording_features",
                        args=[str(recording["session_id"])],
                        queue="celery",  # Use default queue for background jobs
                    )
                    task_ids.append(task.id)
                except Exception as e:
                    logger.error(f"Error queuing task for session {recording['session_id']}: {e}")

            logger.info(f"Queued {len(task_ids)} feature extraction tasks")

            return {
                "total_found": len(recordings_without_features),
                "tasks_queued": len(task_ids),
                "task_ids": task_ids[:10],  # Return first 10 task IDs
            }

        # Use the worker's event loop instead of creating a new one
        try:
            loop = get_worker_loop()
        except RuntimeError:
            # Fallback to asyncio.run() if not in worker context (e.g., testing)
            return asyncio.run(run_batch())
        
        return loop.run_until_complete(run_batch())

    except Exception as e:
        logger.exception(f"Error in batch feature extraction: {e}")
        return {"error": str(e), "total_found": 0, "tasks_queued": 0}


async def _find_recordings_without_features(
    pool, batch_size: int = 50, max_batches: int | None = None
) -> list[dict]:
    """
    Find recording sessions that don't have features.

    Uses LEFT JOIN to find sessions without corresponding feature entries.

    Args:
        pool: Database connection pool
        batch_size: Max recordings to return
        max_batches: Max batches to process (for rate limiting)

    Returns:
        List of recording session dicts
    """
    # Calculate limit
    limit = batch_size * max_batches if max_batches else batch_size

    query = """
        SELECT
            rs.id as session_id,
            rs.created_at,
            rs.status,
            COUNT(m.id) as num_measurements
        FROM heimdall.recording_sessions rs
        LEFT JOIN heimdall.measurements m
            ON m.created_at >= rs.session_start
            AND (rs.session_end IS NULL OR m.created_at <= rs.session_end)
            AND m.iq_data_location IS NOT NULL
        LEFT JOIN heimdall.measurement_features mf
            ON mf.recording_session_id = rs.id
        WHERE rs.status = 'completed'
          AND mf.recording_session_id IS NULL  -- No features extracted yet
        GROUP BY rs.id, rs.created_at, rs.status
        HAVING COUNT(m.id) > 0  -- Has at least one measurement with IQ data
        ORDER BY rs.created_at DESC
        LIMIT $1
    """

    async with pool.acquire() as conn:
        results = await conn.fetch(query, limit)

        return [
            {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "status": row["status"],
                "num_measurements": row["num_measurements"],
            }
            for row in results
        ]


@shared_task(bind=True, name="backend.tasks.backfill_all_features")
def backfill_all_features(self) -> dict:
    """
    Backfill features for ALL recordings without features.

    This is a one-time migration task.
    Run manually via: celery -A src.main:celery_app call backend.tasks.backfill_all_features

    Returns:
        dict with backfill statistics
    """
    logger.info("Starting full backfill of feature extraction")

    total_processed = 0
    total_queued = 0
    batch_num = 0

    while True:
        # Process in batches of 50
        result = batch_feature_extraction_task(batch_size=50, max_batches=1)

        total_processed += result["total_found"]
        total_queued += result["tasks_queued"]
        batch_num += 1

        logger.info(f"Backfill batch {batch_num}: {result['tasks_queued']} tasks queued")

        # Stop if no more recordings found
        if result["total_found"] == 0:
            break

        # Safety limit: max 1000 batches (50,000 recordings)
        if batch_num >= 1000:
            logger.warning("Backfill safety limit reached (1000 batches)")
            break

    logger.info(
        f"Backfill complete: {total_processed} recordings found, {total_queued} tasks queued"
    )

    return {
        "total_recordings": total_processed,
        "total_tasks_queued": total_queued,
        "num_batches": batch_num,
    }
