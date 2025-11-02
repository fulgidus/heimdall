# Step 6: Background Feature Extraction Jobs

## Objective

Implement background Celery tasks to process existing recordings that don't have features yet:
1. Find recordings without features (LEFT JOIN query)
2. Queue extraction tasks in batches
3. Schedule periodic runs (Celery beat)
4. Monitor progress and report statistics

## Context

When deploying the feature extraction system, there may be existing recordings in the database that were created before feature extraction was implemented. We need to:
- Backfill features for all historical recordings
- Run automatically every 5 minutes to catch any missed recordings
- Process in batches to avoid overloading the system
- Skip recordings with existing features

## Implementation

### 1. Create Batch Extraction Task

**File**: `services/backend/src/tasks/batch_feature_extraction.py`

```python
"""
Background Celery task for batch feature extraction.

Processes recordings that don't have features yet.
"""

import logging
from datetime import datetime
from typing import Optional

from celery import Task
from sqlalchemy import text

from ..celery_app import celery_app
from ..db import get_pool
from .feature_extraction_task import ExtractRecordingFeaturesTask

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='backend.tasks.batch_feature_extraction')
class BatchFeatureExtractionTask(Task):
    """Process recordings without features in batches."""

    def run(self, batch_size: int = 50, max_batches: Optional[int] = None) -> dict:
        """
        Find recordings without features and queue extraction tasks.

        Args:
            batch_size: Number of recordings to process per batch
            max_batches: Maximum number of batches to process (None = unlimited)

        Returns:
            dict with statistics
        """
        logger.info(f"Starting batch feature extraction (batch_size={batch_size}, max_batches={max_batches})")

        try:
            pool = get_pool()

            # Find recordings without features
            recordings_without_features = self._find_recordings_without_features(
                pool,
                batch_size=batch_size,
                max_batches=max_batches
            )

            logger.info(f"Found {len(recordings_without_features)} recordings without features")

            if not recordings_without_features:
                return {
                    'total_found': 0,
                    'tasks_queued': 0,
                    'message': 'No recordings without features'
                }

            # Queue extraction tasks
            task_ids = []
            for recording in recordings_without_features:
                try:
                    task = ExtractRecordingFeaturesTask().apply_async(
                        args=[str(recording['session_id'])],
                        queue='celery'  # Use default queue for background jobs
                    )
                    task_ids.append(task.id)
                except Exception as e:
                    logger.error(f"Error queuing task for session {recording['session_id']}: {e}")

            logger.info(f"Queued {len(task_ids)} feature extraction tasks")

            return {
                'total_found': len(recordings_without_features),
                'tasks_queued': len(task_ids),
                'task_ids': task_ids[:10]  # Return first 10 task IDs
            }

        except Exception as e:
            logger.exception(f"Error in batch feature extraction: {e}")
            return {
                'error': str(e),
                'total_found': 0,
                'tasks_queued': 0
            }

    def _find_recordings_without_features(
        self,
        pool,
        batch_size: int = 50,
        max_batches: Optional[int] = None
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

        query = text("""
            SELECT
                rs.id as session_id,
                rs.created_at,
                rs.status,
                COUNT(m.id) as num_measurements
            FROM heimdall.recording_sessions rs
            LEFT JOIN heimdall.measurements m
                ON m.created_at >= rs.created_at
                AND m.created_at <= rs.created_at + (rs.duration_seconds * INTERVAL '1 second')
                AND m.iq_data_location IS NOT NULL
            LEFT JOIN heimdall.measurement_features mf
                ON mf.recording_session_id = rs.id
            WHERE rs.status = 'completed'
              AND mf.recording_session_id IS NULL  -- No features extracted yet
            GROUP BY rs.id, rs.created_at, rs.status
            HAVING COUNT(m.id) > 0  -- Has at least one measurement with IQ data
            ORDER BY rs.created_at DESC
            LIMIT :limit
        """)

        with pool.connect() as conn:
            results = conn.execute(query, {'limit': limit}).fetchall()

            return [
                {
                    'session_id': row[0],
                    'created_at': row[1],
                    'status': row[2],
                    'num_measurements': row[3]
                }
                for row in results
            ]


@celery_app.task(name='backend.tasks.backfill_all_features')
def backfill_all_features() -> dict:
    """
    Backfill features for ALL recordings without features.

    This is a one-time migration task.
    Run manually via: celery -A celery_app call backend.tasks.backfill_all_features

    Returns:
        dict with backfill statistics
    """
    logger.info("Starting full backfill of feature extraction")

    total_processed = 0
    total_queued = 0
    batch_num = 0

    while True:
        # Process in batches of 50
        task = BatchFeatureExtractionTask()
        result = task.run(batch_size=50, max_batches=1)

        total_processed += result['total_found']
        total_queued += result['tasks_queued']
        batch_num += 1

        logger.info(f"Backfill batch {batch_num}: {result['tasks_queued']} tasks queued")

        # Stop if no more recordings found
        if result['total_found'] == 0:
            break

        # Safety limit: max 1000 batches (50,000 recordings)
        if batch_num >= 1000:
            logger.warning("Backfill safety limit reached (1000 batches)")
            break

    logger.info(f"Backfill complete: {total_processed} recordings found, {total_queued} tasks queued")

    return {
        'total_recordings': total_processed,
        'total_tasks_queued': total_queued,
        'num_batches': batch_num
    }
```

### 2. Add Celery Beat Schedule

**File**: `services/backend/src/celery_app.py`

Update Celery beat schedule to run batch extraction every 5 minutes:

```python
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    # Existing tasks...
    'monitor-websdr-uptime': {
        'task': 'backend.tasks.monitor_uptime',
        'schedule': 60.0,  # Every 60 seconds
    },

    # NEW: Batch feature extraction
    'batch-feature-extraction': {
        'task': 'backend.tasks.batch_feature_extraction',
        'schedule': 300.0,  # Every 5 minutes
        'kwargs': {
            'batch_size': 50,
            'max_batches': 5  # Max 250 recordings per run
        }
    },
}

celery_app.conf.timezone = 'UTC'
```

### 3. Create Admin Endpoint for Manual Trigger

**File**: `services/backend/src/routers/admin.py`

```python
"""Admin endpoints for system maintenance."""

from fastapi import APIRouter, Depends, HTTPException
from ..auth import require_admin
from ..tasks.batch_feature_extraction import BatchFeatureExtractionTask, backfill_all_features

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/features/batch-extract")
async def trigger_batch_extraction(
    batch_size: int = 50,
    max_batches: int = 5,
    current_user: dict = Depends(require_admin)
):
    """
    Manually trigger batch feature extraction.

    Args:
        batch_size: Recordings per batch
        max_batches: Maximum batches to process

    Returns:
        Task result
    """
    task = BatchFeatureExtractionTask()
    result = task.apply_async(
        kwargs={'batch_size': batch_size, 'max_batches': max_batches},
        queue='celery'
    )

    return {
        "task_id": result.id,
        "message": f"Batch extraction started (batch_size={batch_size}, max_batches={max_batches})",
        "status": "queued"
    }


@router.post("/features/backfill-all")
async def trigger_full_backfill(
    current_user: dict = Depends(require_admin)
):
    """
    Trigger full backfill of features for ALL recordings.

    WARNING: This may queue thousands of tasks. Use with caution.

    Returns:
        Task result
    """
    task = backfill_all_features.apply_async(queue='celery')

    return {
        "task_id": task.id,
        "message": "Full backfill started - this may take hours",
        "status": "queued"
    }


@router.get("/features/stats")
async def get_feature_extraction_stats(
    current_user: dict = Depends(require_admin),
    pool = Depends(get_pool)
):
    """
    Get statistics about feature extraction coverage.

    Returns:
        dict with coverage statistics
    """
    query = text("""
        WITH recording_stats AS (
            SELECT
                COUNT(DISTINCT rs.id) as total_recordings,
                COUNT(DISTINCT mf.recording_session_id) as recordings_with_features,
                COUNT(m.id) as total_measurements
            FROM heimdall.recording_sessions rs
            LEFT JOIN heimdall.measurements m
                ON m.created_at >= rs.created_at
                AND m.created_at <= rs.created_at + (rs.duration_seconds * INTERVAL '1 second')
                AND m.iq_data_location IS NOT NULL
            LEFT JOIN heimdall.measurement_features mf
                ON mf.recording_session_id = rs.id
            WHERE rs.status = 'completed'
        )
        SELECT * FROM recording_stats
    """)

    async with pool.acquire() as conn:
        result = await conn.fetchrow(query)

        if not result:
            return {
                'total_recordings': 0,
                'recordings_with_features': 0,
                'total_measurements': 0,
                'measurements_with_features': 0,
                'coverage_percent': 0.0
            }

        total_recordings = result[0]
        recordings_with_features = result[1]
        total_measurements = result[2]

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
```

Add router to main app:

```python
# services/backend/src/main.py
from .routers import admin

app.include_router(admin.router)
```

## Verification

### 1. Test Batch Extraction Manually

```bash
# Trigger manual batch extraction
curl -X POST http://localhost:8001/api/v1/admin/features/batch-extract \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 10, "max_batches": 2}'
```

Expected response:
```json
{
  "task_id": "abc123...",
  "message": "Batch extraction started (batch_size=10, max_batches=2)",
  "status": "queued"
}
```

### 2. Check Coverage Statistics

```bash
curl http://localhost:8001/api/v1/admin/features/stats \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

Expected response:
```json
{
  "total_recordings": 150,
  "recordings_with_features": 145,
  "recordings_without_features": 5,
  "total_measurements": 1050,
  "measurements_with_features": 1015,
  "measurements_without_features": 35,
  "coverage_percent": 96.67
}
```

### 3. Verify Celery Beat Schedule

```bash
# Check beat schedule
DOCKER_HOST="" docker exec heimdall-backend celery -A celery_app inspect scheduled
```

Expected: `batch-feature-extraction` task appears with 300-second interval.

### 4. Monitor Batch Processing

```bash
# Watch logs for batch extraction
DOCKER_HOST="" docker compose logs -f backend | grep "batch feature extraction"
```

Expected (every 5 minutes):
```
Starting batch feature extraction (batch_size=50, max_batches=5)
Found 23 recordings without features
Queued 23 feature extraction tasks
```

### 5. Test Full Backfill (WARNING: Use on test data only)

```bash
# Trigger full backfill (admin only)
curl -X POST http://localhost:8001/api/v1/admin/features/backfill-all \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

Monitor progress:
```bash
DOCKER_HOST="" docker compose logs -f backend | grep "Backfill"
```

Expected:
```
Starting full backfill of feature extraction
Backfill batch 1: 50 tasks queued
Backfill batch 2: 50 tasks queued
...
Backfill complete: 237 recordings found, 237 tasks queued
```

## Performance Considerations

### Rate Limiting

- **Batch size**: 50 recordings per batch (configurable)
- **Max batches per run**: 5 (250 recordings per 5 minutes)
- **Celery concurrency**: 10 workers (set in docker-compose.yml)
- **Processing rate**: ~250 recordings/5 min = ~50 recordings/minute

### Resource Usage

- **Database queries**: ~1 query per batch (LEFT JOIN)
- **Celery queue**: Up to 250 tasks queued per run
- **Memory**: ~100 MB per worker
- **CPU**: Minimal (feature extraction is the heavy part)

### Scaling

For large backlogs (10k+ recordings):
1. Increase Celery worker count: `--concurrency=20`
2. Run multiple manual batches in parallel
3. Adjust batch size to 100-200
4. Monitor RabbitMQ queue depth

## Success Criteria

- ✅ Batch extraction task implemented
- ✅ Celery beat schedule configured (every 5 minutes)
- ✅ Admin endpoints for manual trigger and stats
- ✅ Coverage statistics endpoint working
- ✅ Full backfill task tested
- ✅ Logs show periodic batch processing
- ✅ No duplicate feature extraction (LEFT JOIN prevents it)

## Next Step

Proceed to **`07-tests.md`** to implement comprehensive test suite for the entire feature extraction pipeline.
