# Step 5: Feature Extraction for Real Recording Sessions

## Objective

Implement automatic feature extraction for real IQ recordings from WebSDR stations:
1. Load IQ data from **ALL receivers** in a recording session
2. Extract features from each receiver using same `RFFeatureExtractor` as synthetic data
3. Save **ONE record** per session with correlated multi-receiver features
4. Handle errors gracefully (save error state, don't fail silently)
5. Trigger automatically when recording session completes

## Context

**CRITICAL**: Real recordings must match synthetic data structure for ML training.

**Synthetic data**: 1 TX → features from 7 receivers → 1 database record
**Real recordings**: 1 recording session → features from N receivers (2-7) → 1 database record

Real recordings are stored in MinIO at:
```
heimdall-raw-iq/{year}/{month}/{day}/{station_id}/{frequency_hz}/{timestamp}.bin
```

We need to:
- Extract features from **ALL receivers** that recorded the signal **simultaneously**
- Group features by `recording_session_id` (NOT individual measurements)
- Save error messages if extraction fails (antenna issue, corrupted file, etc.)
- Use same chunking/aggregation as synthetic (1000ms → 5×200ms)

## Implementation

### 1. Create Feature Extraction Task

**File**: `services/backend/src/tasks/feature_extraction_task.py`

```python
"""
Celery task for extracting features from real IQ recordings.

Automatically triggered when a recording session completes.

IMPORTANT: Extracts features from ALL receivers in a session and saves
as ONE correlated sample (matching synthetic data structure).
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List
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
    """
    Extract features from a completed recording session.

    This creates ONE sample with features from ALL receivers (like synthetic data).
    """

    def run(self, recording_session_id: str) -> dict:
        """
        Extract features from ALL receivers in a recording session.

        Args:
            recording_session_id: UUID of recording session

        Returns:
            dict with extraction results
        """
        logger.info(f"Starting multi-receiver feature extraction for session {recording_session_id}")

        try:
            # Get database pool
            pool = get_pool()

            # Get recording session details
            session = self._get_session_details(pool, recording_session_id)

            if not session:
                logger.error(f"Recording session {recording_session_id} not found")
                return {'success': False, 'error': 'Session not found'}

            # Get ALL measurements for this session (all receivers)
            measurements = self._get_session_measurements(pool, recording_session_id)

            if not measurements:
                logger.warning(f"No measurements found for session {recording_session_id}")
                return {'success': False, 'error': 'No measurements found'}

            logger.info(f"Found {len(measurements)} receivers for session {recording_session_id}")

            # Extract features from ALL receivers
            try:
                self._extract_session_features(pool, session, measurements)

                logger.info(f"Successfully extracted features from {len(measurements)} receivers")

                return {
                    'success': True,
                    'num_receivers': len(measurements),
                    'session_id': recording_session_id
                }

            except Exception as e:
                logger.error(f"Error extracting features for session {recording_session_id}: {e}")

                # Save error to database
                self._save_extraction_error(pool, recording_session_id, str(e))

                return {
                    'success': False,
                    'error': str(e),
                    'session_id': recording_session_id
                }

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

    def _get_session_measurements(self, pool, session_id: str) -> List[dict]:
        """
        Get ALL measurements for a recording session.

        Returns one measurement per receiver that captured the session.
        """
        query = text("""
            SELECT
                m.id, m.websdr_station_id, m.frequency_hz,
                m.signal_strength_db, m.iq_data_location, m.created_at
            FROM heimdall.measurements m
            JOIN heimdall.recording_sessions rs ON rs.id = :session_id
            WHERE m.created_at >= rs.created_at
              AND m.created_at <= rs.created_at + (rs.duration_seconds * INTERVAL '1 second')
              AND m.iq_data_location IS NOT NULL
            ORDER BY m.websdr_station_id
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

    def _extract_session_features(
        self,
        pool,
        session: dict,
        measurements: List[dict]
    ) -> None:
        """
        Extract features from ALL receivers in a session.

        This creates ONE database record with features from multiple receivers
        (matching synthetic data structure for ML training).

        Args:
            pool: Database connection pool
            session: Recording session dict
            measurements: List of measurements (one per receiver)
        """
        logger.info(f"Extracting features from {len(measurements)} receivers")

        # Extract features for EACH receiver
        all_receiver_features = []
        snr_values = []
        confidence_scores = []

        for measurement in measurements:
            try:
                # Load IQ data from MinIO
                iq_sample = self._load_iq_from_minio(
                    measurement['iq_data_location'],
                    measurement['frequency_hz'],
                    measurement['websdr_station_id'],
                    measurement['created_at']
                )

                # Initialize feature extractor
                feature_extractor = RFFeatureExtractor(sample_rate_hz=iq_sample.sample_rate_hz)

                # Extract features (chunked: 1000ms → 5×200ms with aggregation)
                features_dict = feature_extractor.extract_features_chunked(
                    iq_sample,
                    chunk_duration_ms=200.0,
                    num_chunks=5
                )

                # Add to receiver features array
                all_receiver_features.append(features_dict)

                # Collect metrics for overall statistics
                if features_dict.get('signal_present', False):
                    snr_values.append(features_dict.get('snr', {}).get('mean', 0.0))
                    confidence_scores.append(
                        features_dict.get('delay_spread_confidence', {}).get('mean', 0.8)
                    )

                logger.info(f"Extracted features from {measurement['websdr_station_id']}: "
                           f"SNR={features_dict.get('snr', {}).get('mean', 0.0):.1f} dB, "
                           f"signal_present={features_dict.get('signal_present', False)}")

            except Exception as e:
                logger.error(f"Error extracting features from {measurement['websdr_station_id']}: {e}")
                # Continue with other receivers (don't fail entire session)
                continue

        if not all_receiver_features:
            raise Exception("No features extracted from any receiver")

        # Calculate overall quality metrics
        mean_snr_db = float(np.mean(snr_values)) if snr_values else 0.0
        overall_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        num_receivers_detected = len([f for f in all_receiver_features if f.get('signal_present', False)])

        # Calculate GDOP (if ≥3 receivers with signal)
        gdop = None
        if num_receivers_detected >= 3:
            gdop = self._calculate_gdop(all_receiver_features)

        # Extraction metadata
        extraction_metadata = {
            'extraction_method': 'recorded',
            'iq_duration_ms': 1000.0,  # Standard duration
            'sample_rate_hz': 200_000,
            'num_chunks': 5,
            'chunk_duration_ms': 200.0,
            'recording_session_id': str(session['id'])
        }

        # Check if this is a known beacon (ground truth available)
        tx_known, tx_lat, tx_lon, tx_power = self._check_known_beacon(
            pool,
            session['frequency_hz']
        )

        # Save to database (ONE record with ALL receivers)
        self._save_features_to_db(
            pool,
            recording_session_id=session['id'],
            timestamp=session['created_at'],
            receiver_features=all_receiver_features,
            extraction_metadata=extraction_metadata,
            overall_confidence=overall_confidence,
            mean_snr_db=mean_snr_db,
            num_receivers_detected=num_receivers_detected,
            gdop=gdop,
            tx_known=tx_known,
            tx_latitude=tx_lat,
            tx_longitude=tx_lon,
            tx_power_dbm=tx_power
        )

        logger.info(f"Saved features for session {session['id']}: "
                   f"{num_receivers_detected} receivers, SNR={mean_snr_db:.1f} dB, "
                   f"confidence={overall_confidence:.2f}, GDOP={gdop}")

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

        # Get receiver coordinates from database
        rx_lat, rx_lon = self._get_receiver_coordinates(station_id)

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

    def _get_receiver_coordinates(self, station_id: str) -> tuple[float, float]:
        """Get receiver coordinates from websdr_stations table."""
        # TODO: Implement actual database lookup
        # For now, return placeholder coordinates
        receiver_coords = {
            'Torino': (45.044, 7.672),
            'Milano': (45.464, 9.188),
            'Bologna': (44.494, 11.342),
            'Genova': (44.407, 8.934),
            'Roma': (41.902, 12.496),
            'Napoli': (40.852, 14.268),
            'Palermo': (38.116, 13.361)
        }
        return receiver_coords.get(station_id, (0.0, 0.0))

    def _check_known_beacon(
        self,
        pool,
        frequency_hz: int
    ) -> tuple[bool, Optional[float], Optional[float], Optional[float]]:
        """
        Check if this frequency matches a known beacon/transmitter.

        Returns:
            (tx_known, tx_latitude, tx_longitude, tx_power_dbm)
        """
        query = text("""
            SELECT latitude, longitude, power_dbm
            FROM heimdall.known_sources
            WHERE frequency_hz = :frequency_hz
            LIMIT 1
        """)

        with pool.connect() as conn:
            result = conn.execute(query, {'frequency_hz': frequency_hz}).fetchone()

            if result:
                return (True, result[0], result[1], result[2])
            else:
                return (False, None, None, None)

    def _calculate_gdop(self, receiver_features: List[dict]) -> Optional[float]:
        """
        Calculate Geometric Dilution of Precision.

        Simplified GDOP calculation based on receiver geometry.
        Lower GDOP = better geometry for localization.

        Args:
            receiver_features: List of receiver feature dicts

        Returns:
            GDOP value (or None if <3 receivers)
        """
        # Extract receiver positions with signal
        positions = []
        for features in receiver_features:
            if features.get('signal_present', False):
                positions.append((features['rx_lat'], features['rx_lon']))

        if len(positions) < 3:
            return None

        # Simple GDOP estimation: inverse of area covered by receivers
        # (Real GDOP requires full geometric calculation)
        lats = [p[0] for p in positions]
        lons = [p[1] for p in positions]

        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Area proxy (larger area = better geometry = lower GDOP)
        area = lat_range * lon_range

        if area < 0.01:  # Too clustered
            return 50.0  # High GDOP (bad geometry)

        gdop = 10.0 / area  # Simplified GDOP
        return min(gdop, 100.0)  # Cap at 100

    def _save_features_to_db(
        self,
        pool,
        recording_session_id: uuid.UUID,
        timestamp: datetime,
        receiver_features: List[dict],
        extraction_metadata: dict,
        overall_confidence: float,
        mean_snr_db: float,
        num_receivers_detected: int,
        gdop: Optional[float],
        tx_known: bool,
        tx_latitude: Optional[float],
        tx_longitude: Optional[float],
        tx_power_dbm: Optional[float]
    ) -> None:
        """
        Save extracted features to database.

        ONE record per recording session with features from ALL receivers.
        """
        query = text("""
            INSERT INTO heimdall.measurement_features (
                recording_session_id, timestamp, receiver_features,
                extraction_metadata, overall_confidence, mean_snr_db,
                num_receivers_detected, gdop, tx_known, tx_latitude,
                tx_longitude, tx_power_dbm, extraction_failed, created_at
            )
            VALUES (
                :recording_session_id, :timestamp, CAST(:receiver_features AS jsonb[]),
                CAST(:extraction_metadata AS jsonb), :overall_confidence,
                :mean_snr_db, :num_receivers_detected, :gdop, :tx_known,
                :tx_latitude, :tx_longitude, :tx_power_dbm, FALSE, NOW()
            )
        """)

        with pool.connect() as conn:
            conn.execute(
                query,
                {
                    'recording_session_id': recording_session_id,
                    'timestamp': timestamp,
                    'receiver_features': json.dumps(receiver_features),
                    'extraction_metadata': json.dumps(extraction_metadata),
                    'overall_confidence': overall_confidence,
                    'mean_snr_db': mean_snr_db,
                    'num_receivers_detected': num_receivers_detected,
                    'gdop': gdop,
                    'tx_known': tx_known,
                    'tx_latitude': tx_latitude,
                    'tx_longitude': tx_longitude,
                    'tx_power_dbm': tx_power_dbm
                }
            )
            conn.commit()

    def _save_extraction_error(
        self,
        pool,
        recording_session_id: uuid.UUID,
        error_message: str
    ) -> None:
        """Save extraction error to database."""
        query = text("""
            INSERT INTO heimdall.measurement_features (
                recording_session_id, timestamp, receiver_features,
                extraction_metadata, overall_confidence, mean_snr_db,
                num_receivers_detected, extraction_failed, error_message, created_at
            )
            VALUES (
                :recording_session_id, NOW(), ARRAY[]::jsonb[], '{}'::jsonb,
                0.0, 0.0, 0, TRUE, :error_message, NOW()
            )
            ON CONFLICT (recording_session_id) DO UPDATE
            SET extraction_failed = TRUE,
                error_message = :error_message
        """)

        with pool.connect() as conn:
            conn.execute(
                query,
                {
                    'recording_session_id': recording_session_id,
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

    logger.info(f"Triggered multi-receiver feature extraction for session {session_id}, task_id={task.id}")

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
# Expected: {'success': True, 'num_receivers': 4, 'session_id': '...'}
```

### 2. Create Test Recording Session

From Recording Session UI:
1. Create new recording session
2. Select frequency and duration
3. Wait for completion
4. Check feature extraction triggered

### 3. Verify Database

```sql
-- Check features were extracted (one record per session)
SELECT
    recording_session_id,
    jsonb_array_length(receiver_features) as num_receivers,
    num_receivers_detected,
    mean_snr_db,
    gdop,
    tx_known
FROM heimdall.measurement_features
WHERE extraction_metadata->>'extraction_method' = 'recorded'
LIMIT 5;

-- Expected: One row per session with 2-7 receivers in array

-- Check multi-receiver structure
SELECT
    recording_session_id,
    receiver_features[1]->>'rx_id' as rx1,
    receiver_features[2]->>'rx_id' as rx2,
    receiver_features[3]->>'rx_id' as rx3,
    receiver_features[1]->'snr'->>'mean' as snr1,
    receiver_features[2]->'snr'->>'mean' as snr2
FROM heimdall.measurement_features
WHERE extraction_metadata->>'extraction_method' = 'recorded'
LIMIT 1;

-- Expected: Multiple receivers (rx1=Torino, rx2=Milano, etc.)

-- Check for errors
SELECT
    recording_session_id,
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
Triggered multi-receiver feature extraction for session abc123, task_id=xyz789
Starting multi-receiver feature extraction for session abc123
Found 4 receivers for session abc123
Extracting features from 4 receivers
Extracted features from Torino: SNR=22.5 dB, signal_present=True
Extracted features from Milano: SNR=20.0 dB, signal_present=True
Extracted features from Bologna: SNR=18.5 dB, signal_present=True
Extracted features from Genova: SNR=16.0 dB, signal_present=True
Saved features for session abc123: 4 receivers, SNR=19.3 dB, confidence=0.85, GDOP=8.5
```

## Error Handling

### Common Errors

1. **File Not Found in MinIO**:
   - Error message: "IQ file not found: {path}"
   - Action: Mark session as failed, save error message
   - User action: Re-record session

2. **Corrupted IQ Data**:
   - Error message: "Invalid IQ data format"
   - Action: Skip corrupted receiver, continue with others
   - User action: Check antenna/receiver hardware for specific station

3. **No Receivers Detected**:
   - Error message: "No features extracted from any receiver"
   - Action: Mark session as failed
   - User action: Check signal strength, frequency, antenna connections

4. **Insufficient Receivers for Localization**:
   - Not an error (still save features)
   - GDOP will be NULL if <3 receivers
   - Can still be used for training with partial data

## Data Structure Comparison

### Synthetic Data (from Prompt 04)
```python
{
    'recording_session_id': <synthetic_uuid>,
    'receiver_features': [
        {rx_id: "Torino", snr: {...}, ...},
        {rx_id: "Milano", snr: {...}, ...},
        # ... up to 7 receivers
    ],
    'tx_latitude': 45.123,  # Known (synthetic)
    'tx_longitude': 7.456,
    'tx_known': True
}
```

### Real Recording (this prompt)
```python
{
    'recording_session_id': <session_uuid>,
    'receiver_features': [
        {rx_id: "Torino", snr: {...}, ...},
        {rx_id: "Milano", snr: {...}, ...},
        {rx_id: "Bologna", snr: {...}, ...},
        # ... 2-7 receivers (those that captured signal)
    ],
    'tx_latitude': None,  # Unknown (to be estimated)
    'tx_longitude': None,
    'tx_known': False
}
```

**Perfect match for ML training!** ✅

## Success Criteria

- ✅ Feature extraction processes **ALL receivers** in a session
- ✅ **ONE record** per session (not per measurement)
- ✅ Features saved to `measurement_features` table with `recording_session_id` as PRIMARY KEY
- ✅ Multi-receiver array structure matches synthetic data
- ✅ Error handling saves error messages
- ✅ Ground truth detection for known beacons
- ✅ GDOP calculation for geometry quality
- ✅ Same feature format as synthetic data
- ✅ Logs show multi-receiver extraction progress

## Next Step

Proceed to **`06-background-jobs.md`** to implement background processing for existing recording sessions without features.
