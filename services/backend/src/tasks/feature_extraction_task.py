"""
Celery task for extracting features from real IQ recordings.

Automatically triggered when a recording session completes.

IMPORTANT: Extracts features from ALL receivers in a session and saves
as ONE correlated sample (matching synthetic data structure).
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, NamedTuple
import json

import numpy as np
from celery import shared_task

from ..config import settings
from ..db import get_pool

logger = logging.getLogger(__name__)

# Import feature extractor from common module
# PYTHONPATH is set to /app in Dockerfile, so this will resolve to /app/common/feature_extraction
from common.feature_extraction import RFFeatureExtractor, IQSample

# Constants
DEFAULT_SAMPLE_RATE_HZ = 200_000  # 200 kHz default for IQ recordings


class BeaconInfo(NamedTuple):
    """Ground truth information for known beacons."""
    known: bool
    latitude: Optional[float]
    longitude: Optional[float]
    power_dbm: Optional[float]


@shared_task(bind=True, name='backend.tasks.extract_recording_features')
def extract_recording_features(self, recording_session_id: str) -> dict:
    """
    Extract features from ALL receivers in a recording session.

    Args:
        recording_session_id: UUID of recording session

    Returns:
        dict with extraction results
    """
    logger.info(f"Starting multi-receiver feature extraction for session {recording_session_id}")

    try:
        # Get database pool and run async operations
        import asyncio
        
        async def run_extraction():
            pool = await get_pool()

            # Get recording session details
            session = await _get_session_details(pool, recording_session_id)

            if not session:
                logger.error(f"Recording session {recording_session_id} not found")
                return {'success': False, 'error': 'Session not found'}

            # Get ALL measurements for this session (all receivers)
            measurements = await _get_session_measurements(pool, recording_session_id)

            if not measurements:
                logger.warning(f"No measurements found for session {recording_session_id}")
                return {'success': False, 'error': 'No measurements found'}

            logger.info(f"Found {len(measurements)} receivers for session {recording_session_id}")

            # Extract features from ALL receivers
            try:
                await _extract_session_features(pool, session, measurements)

                logger.info(f"Successfully extracted features from {len(measurements)} receivers")

                return {
                    'success': True,
                    'num_receivers': len(measurements),
                    'session_id': recording_session_id
                }

            except Exception as e:
                logger.error(f"Error extracting features for session {recording_session_id}: {e}")

                # Save error to database
                await _save_extraction_error(pool, recording_session_id, str(e))

                return {
                    'success': False,
                    'error': str(e),
                    'session_id': recording_session_id
                }

        # Use asyncio.run() for proper event loop management
        return asyncio.run(run_extraction())

    except Exception as e:
        logger.exception(f"Fatal error in feature extraction task: {e}")
        return {'success': False, 'error': str(e)}


async def _get_session_details(pool, session_id: str) -> Optional[dict]:
    """Get recording session details."""
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            "SELECT id, status, created_at FROM heimdall.recording_sessions WHERE id = $1",
            uuid.UUID(session_id)
        )

        if not result:
            return None

        return {
            'id': result['id'],
            'status': result['status'],
            'created_at': result['created_at']
        }


async def _get_session_measurements(pool, session_id: str) -> List[dict]:
    """
    Get ALL measurements for a recording session.

    Returns one measurement per receiver that captured the session.
    """
    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT
                m.id, m.websdr_station_id, m.frequency_hz,
                m.signal_strength_db, m.iq_data_location, m.created_at,
                ws.latitude, ws.longitude, ws.name
            FROM heimdall.measurements m
            JOIN heimdall.websdr_stations ws ON ws.id = m.websdr_station_id
            JOIN heimdall.recording_sessions rs ON rs.id = $1
            WHERE m.created_at >= rs.session_start
              AND (rs.session_end IS NULL OR m.created_at <= rs.session_end)
              AND m.iq_data_location IS NOT NULL
            ORDER BY m.websdr_station_id
        """, uuid.UUID(session_id))

        return [
            {
                'id': row['id'],
                'websdr_station_id': row['websdr_station_id'],
                'frequency_hz': row['frequency_hz'],
                'signal_strength_db': row['signal_strength_db'],
                'iq_data_location': row['iq_data_location'],
                'created_at': row['created_at'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'name': row['name']
            }
            for row in results
        ]


async def _extract_session_features(
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
    failed_receivers = []

    for measurement in measurements:
        try:
            # Load IQ data from MinIO
            iq_sample = await _load_iq_from_minio(
                measurement['iq_data_location'],
                measurement['frequency_hz'],
                measurement['name'],
                measurement['created_at'],
                measurement['latitude'],
                measurement['longitude']
            )

            # Initialize feature extractor
            feature_extractor = RFFeatureExtractor(sample_rate_hz=iq_sample.sample_rate_hz)

            # Extract features (chunked: 1000ms → 5×200ms with aggregation)
            features_dict = feature_extractor.extract_features_chunked(
                iq_sample,
                chunk_duration_ms=200.0,
                num_chunks=5
            )

            # Add receiver metadata
            features_dict['rx_id'] = measurement['name']
            features_dict['rx_lat'] = measurement['latitude']
            features_dict['rx_lon'] = measurement['longitude']

            # Add to receiver features array
            all_receiver_features.append(features_dict)

            # Collect metrics for overall statistics
            if features_dict.get('signal_present', {}).get('mean', 0) > 0.5:
                snr_values.append(features_dict.get('snr_db', {}).get('mean', 0.0))
                confidence_scores.append(
                    features_dict.get('delay_spread_confidence', {}).get('mean', 0.8)
                )

            logger.info(f"Extracted features from {measurement['name']}: "
                       f"SNR={features_dict.get('snr_db', {}).get('mean', 0.0):.1f} dB, "
                       f"signal_present={features_dict.get('signal_present', {}).get('mean', 0) > 0.5}")

        except Exception as e:
            logger.error(f"Error extracting features from {measurement['name']}: {e}")
            failed_receivers.append({'name': measurement['name'], 'error': str(e)})
            # Continue with other receivers (don't fail entire session)
            continue

    if not all_receiver_features:
        error_msg = f"No features extracted from any receiver. Failures: {failed_receivers}"
        raise Exception(error_msg)

    # Log partial failures if any
    if failed_receivers:
        logger.warning(f"Partial extraction success: {len(all_receiver_features)} succeeded, "
                      f"{len(failed_receivers)} failed. Failed receivers: {[f['name'] for f in failed_receivers]}")

    # Calculate overall quality metrics
    mean_snr_db = float(np.mean(snr_values)) if snr_values else 0.0
    overall_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
    num_receivers_detected = len([f for f in all_receiver_features if f.get('signal_present', {}).get('mean', 0) > 0.5])

    # Calculate GDOP (if ≥3 receivers with signal)
    gdop = None
    if num_receivers_detected >= 3:
        gdop = _calculate_gdop(all_receiver_features)

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
    beacon = await _check_known_beacon(pool, measurements[0]['frequency_hz'])

    # Save to database (ONE record with ALL receivers)
    await _save_features_to_db(
        pool,
        recording_session_id=session['id'],
        timestamp=session['created_at'],
        receiver_features=all_receiver_features,
        extraction_metadata=extraction_metadata,
        overall_confidence=overall_confidence,
        mean_snr_db=mean_snr_db,
        num_receivers_detected=num_receivers_detected,
        gdop=gdop,
        tx_known=beacon.known,
        tx_latitude=beacon.latitude,
        tx_longitude=beacon.longitude,
        tx_power_dbm=beacon.power_dbm
    )

    logger.info(f"Saved features for session {session['id']}: "
               f"{num_receivers_detected} receivers, SNR={mean_snr_db:.1f} dB, "
               f"confidence={overall_confidence:.2f}, GDOP={gdop}")


async def _load_iq_from_minio(
    iq_location: str,
    frequency_hz: int,
    station_name: str,
    timestamp: datetime,
    rx_lat: float,
    rx_lon: float
) -> IQSample:
    """
    Load IQ data from MinIO.

    Args:
        iq_location: S3 object path
        frequency_hz: Center frequency
        station_name: WebSDR station name
        timestamp: Recording timestamp
        rx_lat: Receiver latitude
        rx_lon: Receiver longitude

    Returns:
        IQSample object
    """
    from ..storage.minio_client import MinIOClient

    # Initialize MinIO client
    minio_client = MinIOClient(
        endpoint_url=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name="heimdall-raw-iq"
    )

    # Download IQ data
    try:
        iq_bytes = minio_client.download_iq_data(s3_path=iq_location)
    except Exception as e:
        logger.error(f"Failed to download IQ data from {iq_location}: {e}")
        raise

    # Parse binary format (complex64 numpy array)
    iq_samples = np.frombuffer(iq_bytes, dtype=np.complex64)

    # Use default sample rate for IQ recordings
    sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ

    return IQSample(
        samples=iq_samples,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=frequency_hz,
        rx_id=station_name,
        rx_lat=rx_lat,
        rx_lon=rx_lon,
        timestamp=timestamp
    )


async def _check_known_beacon(pool, frequency_hz: int) -> BeaconInfo:
    """
    Check if this frequency matches a known beacon/transmitter.

    Args:
        pool: Database connection pool
        frequency_hz: Frequency to check

    Returns:
        BeaconInfo with ground truth data if beacon is known
    """
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT latitude, longitude, power_dbm
            FROM heimdall.known_sources
            WHERE frequency_hz = $1
            LIMIT 1
        """, frequency_hz)

        if result:
            return BeaconInfo(
                known=True,
                latitude=result['latitude'],
                longitude=result['longitude'],
                power_dbm=result['power_dbm']
            )
        else:
            return BeaconInfo(known=False, latitude=None, longitude=None, power_dbm=None)


def _calculate_gdop(receiver_features: List[dict]) -> Optional[float]:
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
        if features.get('signal_present', {}).get('mean', 0) > 0.5:
            positions.append((features['rx_lat'], features['rx_lon']))

    if len(positions) < 3:
        return None

    # Simple GDOP estimation: inverse of area covered by receivers
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


async def _save_features_to_db(
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
    async with pool.acquire() as conn:
        # asyncpg handles JSON serialization automatically - pass Python objects directly
        # For jsonb[] type, we need to convert each element to JSON string then cast to jsonb
        receiver_features_jsonb = [json.dumps(f) for f in receiver_features]

        await conn.execute("""
            INSERT INTO heimdall.measurement_features (
                recording_session_id, timestamp, receiver_features,
                extraction_metadata, overall_confidence, mean_snr_db,
                num_receivers_detected, gdop, tx_known, tx_latitude,
                tx_longitude, tx_power_dbm, extraction_failed, created_at
            )
            VALUES (
                $1, $2, $3::jsonb[], $4, $5, $6, $7, $8, $9, $10, $11, $12, FALSE, NOW()
            )
            ON CONFLICT (recording_session_id) DO UPDATE
            SET receiver_features = EXCLUDED.receiver_features,
                extraction_metadata = EXCLUDED.extraction_metadata,
                overall_confidence = EXCLUDED.overall_confidence,
                mean_snr_db = EXCLUDED.mean_snr_db,
                num_receivers_detected = EXCLUDED.num_receivers_detected,
                gdop = EXCLUDED.gdop,
                extraction_failed = FALSE,
                error_message = NULL
        """,
            recording_session_id,
            timestamp,
            receiver_features_jsonb,
            extraction_metadata,  # asyncpg handles dict -> jsonb conversion
            overall_confidence,
            mean_snr_db,
            num_receivers_detected,
            gdop,
            tx_known,
            tx_latitude,
            tx_longitude,
            tx_power_dbm
        )


async def _save_extraction_error(
    pool,
    recording_session_id: str,
    error_message: str
) -> None:
    """Save extraction error to database."""
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO heimdall.measurement_features (
                recording_session_id, timestamp, receiver_features,
                extraction_metadata, overall_confidence, mean_snr_db,
                num_receivers_detected, extraction_failed, error_message, created_at
            )
            VALUES (
                $1, NOW(), ARRAY[]::jsonb[], '{}'::jsonb,
                0.0, 0.0, 0, TRUE, $2, NOW()
            )
            ON CONFLICT (recording_session_id) DO UPDATE
            SET extraction_failed = TRUE,
                error_message = EXCLUDED.error_message
        """, uuid.UUID(recording_session_id), error_message)
