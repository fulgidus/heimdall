# Step 5: Feature Extraction for Real Recording Sessions

## Objective

Implement automatic feature extraction for real IQ recordings from WebSDR stations:
1. Load IQ data from MinIO when recording completes
2. Extract features using same `RFFeatureExtractor` as synthetic data
3. Save to `measurement_features` table
4. Handle errors gracefully (save error state, don't fail silently)
5. Trigger automatically when recording session completes

## Context

Real recordings are stored in MinIO at:
```
heimdall-raw-iq/{year}/{month}/{day}/{station_id}/{frequency_hz}/{timestamp}.bin
```

We need to:
- Extract features from all receivers that recorded the signal
- Handle variable-length recordings (chunk into 1000ms segments)
- Save error messages if extraction fails (antenna issue, corrupted file, etc.)
- Use same chunking/aggregation as synthetic (1000ms → 5×200ms)

## Implementation

### 1. Create Feature Extraction Task

**File**: `services/backend/src/tasks/feature_extraction_task.py`

```python
"""
Celery task for extracting features from real IQ recordings.

Automatically triggered when a recording session completes.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional
import json

import numpy as np
from celery import Task

from ..celery_app import celery_app
from ..db import get_pool
from ..storage.minio_client import get_minio_client
from sqlalchemy import text

# Import feature extractor (add to sys.path if needed)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from common.feature_extraction import RFFeatureExtractor, IQSample

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='backend.tasks.extract_recording_features')
class ExtractRecordingFeaturesTask(Task):
    """Extract features from a completed recording session."""

    def run(self, recording_session_id: str) -> dict:
        """
        Extract features from all recordings in a session.

        Args:
            recording_session_id: UUID of recording session

        Returns:
            dict with extraction results
        """
        logger.info(f"Starting feature extraction for recording session {recording_session_id}")

        try:
            # Get database pool
            pool = get_pool()

            # Get recording session details
            session = self._get_session_details(pool, recording_session_id)

            if not session:
                logger.error(f"Recording session {recording_session_id} not found")
                return {'success': False, 'error': 'Session not found'}

            # Get all measurements for this session
            measurements = self._get_session_measurements(pool, recording_session_id)

            logger.info(f"Found {len(measurements)} measurements for session {recording_session_id}")

            # Extract features for each measurement
            results = {
                'total': len(measurements),
                'success': 0,
                'failed': 0,
                'errors': []
            }

            for measurement in measurements:
                try:
                    self._extract_measurement_features(
                        pool,
                        measurement,
                        session
                    )
                    results['success'] += 1
                except Exception as e:
                    logger.error(f"Error extracting features for measurement {measurement['id']}: {e}")
                    results['failed'] += 1
                    results['errors'].append(str(e))

                    # Save error to database
                    self._save_extraction_error(pool, measurement['id'], str(e))

            logger.info(f"Feature extraction complete: {results['success']} success, {results['failed']} failed")

            return results

        except Exception as e:
            logger.exception(f"Fatal error in feature extraction task: {e}")
            return {'success': False, 'error': str(e)}

    def _get_session_details(self, pool, session_id: str) -> Optional[dict]:
        """Get recording session details."""
        query = text("""
            SELECT
                id, user_id, status, frequency_hz, bandwidth_hz,
                duration_seconds, created_at
            FROM heimdall.recording_sessions
            WHERE id = :session_id
        """)

        with pool.connect() as conn:
            result = conn.execute(query, {'session_id': session_id}).fetchone()

            if not result:
                return None

            return {
                'id': result[0],
                'user_id': result[1],
                'status': result[2],
                'frequency_hz': result[3],
                'bandwidth_hz': result[4],
                'duration_seconds': result[5],
                'created_at': result[6]
            }

    def _get_session_measurements(self, pool, session_id: str) -> list[dict]:
        """Get all measurements for a recording session."""
        query = text("""
            SELECT
                m.id, m.websdr_station_id, m.frequency_hz,
                m.signal_strength_db, m.iq_data_location, m.created_at
            FROM heimdall.measurements m
            JOIN heimdall.recording_sessions rs ON rs.id = :session_id
            WHERE m.created_at >= rs.created_at
              AND m.created_at <= rs.created_at + (rs.duration_seconds * INTERVAL '1 second')
              AND m.iq_data_location IS NOT NULL
        """)

        with pool.connect() as conn:
            results = conn.execute(query, {'session_id': session_id}).fetchall()

            return [
                {
                    'id': row[0],
                    'websdr_station_id': row[1],
                    'frequency_hz': row[2],
                    'signal_strength_db': row[3],
                    'iq_data_location': row[4],
                    'created_at': row[5]
                }
                for row in results
            ]

    def _extract_measurement_features(
        self,
        pool,
        measurement: dict,
        session: dict
    ) -> None:
        """
        Extract features from a single measurement.

        Args:
            pool: Database connection pool
            measurement: Measurement dict
            session: Recording session dict
        """
        # Load IQ data from MinIO
        iq_sample = self._load_iq_from_minio(
            measurement['iq_data_location'],
            measurement['frequency_hz'],
            measurement['websdr_station_id'],
            measurement['created_at']
        )

        # Initialize feature extractor
        feature_extractor = RFFeatureExtractor(sample_rate_hz=iq_sample.sample_rate_hz)

        # Extract features (chunked)
        features_dict = feature_extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        # Wrap in receiver array (single receiver for real recordings)
        receiver_features = [features_dict]

        # Calculate quality metrics
        overall_confidence = features_dict.get('delay_spread_confidence', {}).get('mean', 0.8)
        mean_snr_db = features_dict.get('snr', {}).get('mean', 0.0)
        num_receivers_detected = 1 if features_dict.get('signal_present', False) else 0

        # Extraction metadata
        extraction_metadata = {
            'extraction_method': 'recorded',
            'iq_duration_ms': iq_sample.duration_ms,
            'sample_rate_hz': iq_sample.sample_rate_hz,
            'num_chunks': 5,
            'chunk_duration_ms': 200.0,
            'recording_session_id': str(session['id'])
        }

        # Save to database
        self._save_features_to_db(
            pool,
            measurement['id'],
            measurement['created_at'],
            receiver_features,
            extraction_metadata,
            overall_confidence,
            mean_snr_db,
            num_receivers_detected
        )

        logger.info(f"Extracted features for measurement {measurement['id']}: "
                   f"SNR={mean_snr_db:.1f} dB, confidence={overall_confidence:.2f}")

    def _load_iq_from_minio(
        self,
        iq_location: str,
        frequency_hz: int,
        station_id: str,
        timestamp: datetime
    ) -> IQSample:
        """
        Load IQ data from MinIO.

        Args:
            iq_location: S3 object path
            frequency_hz: Center frequency
            station_id: WebSDR station ID
            timestamp: Recording timestamp

        Returns:
            IQSample object
        """
        minio_client = get_minio_client()
        bucket_name = "heimdall-raw-iq"

        # Download IQ data
        response = minio_client.get_object(bucket_name, iq_location)
        iq_bytes = response.read()
        response.close()
        response.release_conn()

        # Parse binary format (complex64 numpy array)
        iq_samples = np.frombuffer(iq_bytes, dtype=np.complex64)

        # Infer sample rate from file size and duration (assume 200 kHz default)
        sample_rate_hz = 200_000
        duration_ms = len(iq_samples) / sample_rate_hz * 1000.0

        # Get receiver coordinates (lookup from database)
        # For now, use placeholder coordinates
        rx_lat, rx_lon = 0.0, 0.0  # TODO: lookup from websdr_stations table

        return IQSample(
            samples=iq_samples,
            sample_rate_hz=sample_rate_hz,
            duration_ms=duration_ms,
            center_frequency_hz=frequency_hz,
            rx_id=station_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            timestamp=timestamp.timestamp()
        )

    def _save_features_to_db(
        self,
        pool,
        measurement_id: uuid.UUID,
        timestamp: datetime,
        receiver_features: list[dict],
        extraction_metadata: dict,
        overall_confidence: float,
        mean_snr_db: float,
        num_receivers_detected: int
    ) -> None:
        """Save extracted features to database."""
        query = text("""
            INSERT INTO heimdall.measurement_features (
                timestamp, measurement_id, receiver_features, extraction_metadata,
                overall_confidence, mean_snr_db, num_receivers_detected,
                extraction_failed, created_at
            )
            VALUES (
                :timestamp, :measurement_id, CAST(:receiver_features AS jsonb),
                CAST(:extraction_metadata AS jsonb),
                :overall_confidence, :mean_snr_db, :num_receivers_detected,
                FALSE, NOW()
            )
        """)

        with pool.connect() as conn:
            conn.execute(
                query,
                {
                    'timestamp': timestamp,
                    'measurement_id': measurement_id,
                    'receiver_features': json.dumps(receiver_features),
                    'extraction_metadata': json.dumps(extraction_metadata),
                    'overall_confidence': overall_confidence,
                    'mean_snr_db': mean_snr_db,
                    'num_receivers_detected': num_receivers_detected
                }
            )
            conn.commit()

    def _save_extraction_error(
        self,
        pool,
        measurement_id: uuid.UUID,
        error_message: str
    ) -> None:
        """Save extraction error to database."""
        query = text("""
            INSERT INTO heimdall.measurement_features (
                timestamp, measurement_id, receiver_features, extraction_metadata,
                overall_confidence, mean_snr_db, num_receivers_detected,
                extraction_failed, error_message, created_at
            )
            VALUES (
                NOW(), :measurement_id, ARRAY[]::jsonb[], '{}'::jsonb,
                0.0, 0.0, 0,
                TRUE, :error_message, NOW()
            )
            ON CONFLICT (timestamp, measurement_id) DO UPDATE
            SET extraction_failed = TRUE,
                error_message = :error_message
        """)

        with pool.connect() as conn:
            conn.execute(
                query,
                {
                    'measurement_id': measurement_id,
                    'error_message': error_message
                }
            )
            conn.commit()
```

### 2. Trigger Extraction After Recording Completion

**File**: `services/backend/src/routers/acquisition.py`

Add trigger after recording session completes:

```python
from ..tasks.feature_extraction_task import ExtractRecordingFeaturesTask

@router.post("/sessions/{session_id}/complete")
async def complete_recording_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    pool = Depends(get_pool)
):
    """
    Mark recording session as completed and trigger feature extraction.

    Args:
        session_id: Recording session UUID
    """
    # Update session status to 'completed'
    query = text("""
        UPDATE heimdall.recording_sessions
        SET status = 'completed', updated_at = NOW()
        WHERE id = :session_id AND user_id = :user_id
        RETURNING id
    """)

    async with pool.acquire() as conn:
        result = await conn.fetchrow(query, session_id=session_id, user_id=current_user['id'])

        if not result:
            raise HTTPException(status_code=404, detail="Recording session not found")

    # Trigger feature extraction task (async Celery task)
    task = ExtractRecordingFeaturesTask().apply_async(args=[session_id])

    logger.info(f"Triggered feature extraction for session {session_id}, task_id={task.id}")

    return {
        "session_id": session_id,
        "status": "completed",
        "feature_extraction_task_id": task.id
    }
```

### 3. Move Feature Extractor to Common Module

**File**: `services/common/feature_extraction/__init__.py`

Move `feature_extractor.py` from training to common module so both backend and training can use it:

```python
"""Common feature extraction module."""

from .feature_extractor import RFFeatureExtractor, IQSample, ExtractedFeatures

__all__ = ['RFFeatureExtractor', 'IQSample', 'ExtractedFeatures']
```

Update imports in training service:
```python
# Before:
from data.feature_extractor import RFFeatureExtractor

# After:
from common.feature_extraction import RFFeatureExtractor
```

## Verification

### 1. Test Feature Extraction Task

```python
# Test manually in backend container
DOCKER_HOST="" docker exec -it heimdall-backend python

from tasks.feature_extraction_task import ExtractRecordingFeaturesTask
from uuid import UUID

# Use a real recording session ID from database
task = ExtractRecordingFeaturesTask()
result = task.run('your-recording-session-uuid-here')

print(result)
# Expected: {'total': N, 'success': N, 'failed': 0, 'errors': []}
```

### 2. Create Test Recording Session

From Recording Session UI:
1. Create new recording session
2. Select frequency and duration
3. Wait for completion
4. Check feature extraction triggered

### 3. Verify Database

```sql
-- Check features were extracted
SELECT COUNT(*) FROM heimdall.measurement_features
WHERE extraction_metadata->>'extraction_method' = 'recorded';

-- Check for errors
SELECT
    measurement_id,
    extraction_failed,
    error_message
FROM heimdall.measurement_features
WHERE extraction_failed = TRUE;
```

### 4. Check Celery Logs

```bash
DOCKER_HOST="" docker compose logs backend | grep "feature extraction"
```

Expected:
```
Triggered feature extraction for session abc123, task_id=xyz789
Starting feature extraction for recording session abc123
Extracted features for measurement def456: SNR=18.5 dB, confidence=0.85
Feature extraction complete: 7 success, 0 failed
```

## Error Handling

### Common Errors

1. **File Not Found in MinIO**:
   - Error message: "IQ file not found: {path}"
   - Action: Mark as failed, save error message
   - User action: Re-record session

2. **Corrupted IQ Data**:
   - Error message: "Invalid IQ data format"
   - Action: Mark as failed, save error message
   - User action: Check antenna/receiver hardware

3. **Insufficient Signal**:
   - Error message: "Signal below detection threshold"
   - Action: Extract features anyway (signal_present=False)
   - No user action needed (valid measurement)

## Success Criteria

- ✅ Feature extraction task implemented
- ✅ Automatic trigger on recording completion
- ✅ Features saved to `measurement_features` table
- ✅ Error handling saves error messages
- ✅ Task completes successfully for real recordings
- ✅ Same feature format as synthetic data
- ✅ Logs show extraction progress

## Next Step

Proceed to **`06-background-jobs.md`** to implement background processing for existing recordings without features.
