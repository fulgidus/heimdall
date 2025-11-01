"""Celery tasks for IQ data acquisition."""

from datetime import datetime
from io import BytesIO

import numpy as np
from celery import Task, shared_task
from celery.utils.log import get_task_logger

from ..config import settings
from ..fetchers.openwebrx_fetcher import OpenWebRXFetcher
from ..models.websdrs import WebSDRConfig
from ..processors.iq_processor import IQProcessor
from ..storage.minio_client import MinIOClient

logger = get_task_logger(__name__)


class AcquisitionTask(Task):
    """Base task for acquisitions."""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True


@shared_task(bind=True, base=AcquisitionTask)
def acquire_iq(
    self,
    frequency_mhz: float,
    duration_seconds: float,
    start_time_iso: str,
    websdrs_config_list: list[dict],
    sample_rate_khz: float = 12.5,
):
    """
    Acquire IQ data from multiple WebSDR receivers simultaneously.

    This task:
    1. Fetches IQ from 7 WebSDR simultaneously
    2. Processes signal metrics
    3. Saves to MinIO (via separate storage task)
    4. Writes metadata to TimescaleDB (via separate DB task)

    Args:
        frequency_mhz: Target frequency in MHz
        duration_seconds: Acquisition duration in seconds
        start_time_iso: Start time in ISO format
        websdrs_config_list: List of WebSDR config dicts
        sample_rate_khz: Sample rate in kHz

    Returns:
        Dict with acquisition results
    """
    # IMMEDIATE DEBUG LOG AT FUNCTION START
    print(f"🔥🔥🔥 ACQUIRE_IQ STARTED: task_id={self.request.id} 🔥🔥🔥")
    
    try:
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": len(websdrs_config_list),
                "status": "Initializing fetcher...",
            },
        )

        # Parse WebSDR configs
        websdrs = [WebSDRConfig(**cfg) for cfg in websdrs_config_list]

        logger.info(
            "🚀🚀🚀 MODIFIED ACQUIRE_IQ: Starting acquisition task from %d WebSDRs at %.2f MHz for %.1f seconds",
            len(websdrs),
            frequency_mhz,
            duration_seconds,
        )

        # Fetch IQ data
        measurements = []
        errors = []

        import asyncio

        async def fetch_and_process():
            fetcher = OpenWebRXFetcher(
                websdrs=websdrs, timeout=30, retry_count=3, concurrent_limit=7
            )

            # Update state: fetching
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": 0,
                    "total": len(websdrs),
                    "status": f"Fetching IQ from {len(websdrs)} WebSDRs...",
                },
            )

            # Fetch all simultaneously
            iq_data_dict = await fetcher.fetch_iq_simultaneous(
                frequency_mhz=frequency_mhz,
                duration_seconds=duration_seconds,
                sample_rate_khz=sample_rate_khz,
            )

            # Process results
            successful = 0
            for idx, (websdr_id, (iq_data, error)) in enumerate(iq_data_dict.items()):
                if error:
                    errors.append(f"WebSDR {websdr_id}: {error}")
                    logger.warning("WebSDR %d acquisition failed: %s", websdr_id, error)
                elif iq_data is not None:
                    try:
                        # Compute metrics
                        metrics = IQProcessor.compute_metrics(
                            iq_data=iq_data,
                            sample_rate_hz=int(sample_rate_khz * 1e3),
                            target_frequency_hz=int(frequency_mhz * 1e6),
                            noise_bandwidth_hz=10000,
                        )

                        # Create measurement record
                        measurement = {
                            "websdr_id": websdr_id,
                            "frequency_mhz": frequency_mhz,
                            "sample_rate_khz": sample_rate_khz,
                            "samples_count": len(iq_data),
                            "timestamp_utc": datetime.utcnow().isoformat(),
                            "metrics": metrics.dict(),
                            "iq_data_path": f"s3://heimdall-raw-iq/sessions/{self.request.id}/websdr_{websdr_id}.npy",
                            "iq_data": iq_data.tolist(),  # For now, store in memory
                        }
                        measurements.append(measurement)
                        successful += 1

                        logger.info(
                            "Processed WebSDR %d - SNR: %.2f dB, Offset: %.2f Hz",
                            websdr_id,
                            metrics.snr_db,
                            metrics.frequency_offset_hz,
                        )
                    except Exception as e:
                        error_msg = f"Processing error for WebSDR {websdr_id}: {str(e)}"
                        errors.append(error_msg)
                        logger.exception(error_msg)

                # Update progress
                progress = ((idx + 1) / len(websdrs)) * 100
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": idx + 1,
                        "total": len(websdrs),
                        "successful": successful,
                        "status": f"Processing {idx + 1}/{len(websdrs)} measurements...",
                        "progress": progress,
                    },
                )

            return measurements, errors

        # Run async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        measurements, errors = loop.run_until_complete(fetch_and_process())

        # CRITICAL DEBUG LOG - SHOULD ALWAYS APPEAR
        print(f"🚨🚨🚨 FETCH COMPLETE: {len(measurements)} measurements, {len(errors)} errors 🚨🚨🚨")
        
        # Debug: Log measurements count immediately after fetch
        logger.info("DEBUG: Fetch complete. Measurements count: %d, Errors count: %d", len(measurements), len(errors))
        
        # Debug: Log measurements count before saving
        logger.info("DEBUG: About to save. Measurements count: %d", len(measurements))

        # Save measurements to TimescaleDB
        if measurements:
            logger.info("Saving %d measurements to TimescaleDB...", len(measurements))
            try:
                from ..storage.db_manager import get_db_manager
                db_manager = get_db_manager()
                
                # Prepare measurements for database
                db_measurements = []
                for measurement in measurements:
                    db_measurement = {
                        'websdr_station_id': None,  # Will be mapped by db_manager from websdr_id
                        'websdr_id': measurement['websdr_id'],  # Temporary field for mapping
                        'frequency_hz': int(measurement['frequency_mhz'] * 1_000_000),
                        'iq_sample_rate': int(measurement['sample_rate_khz'] * 1000),
                        'iq_samples_count': measurement['samples_count'],
                        'timestamp': measurement['timestamp_utc'],
                        'snr_db': measurement['metrics']['snr_db'],
                        'frequency_offset_hz': int(measurement['metrics']['frequency_offset_hz']),
                        'signal_strength_db': measurement['metrics']['signal_power_dbm'],
                        'iq_data_location': measurement['iq_data_path'],
                        'iq_data_format': 'npy',
                        'duration_seconds': float(duration_seconds),
                    }
                    db_measurements.append(db_measurement)
                
                # Bulk insert measurements
                successful, failed = db_manager.insert_measurements_bulk(
                    task_id=self.request.id,
                    measurements_list=db_measurements
                )
                logger.info("Database save complete: %d successful, %d failed", successful, failed)
            except Exception as e:
                logger.error("Failed to save measurements to database: %s", str(e))
                errors.append(f"Database save error: {str(e)}")

        # Remove iq_data from result to make it JSON serializable
        # (iq_data contains complex numbers which can't be JSON serialized)
        measurements_summary = [
            {k: v for k, v in m.items() if k != "iq_data"} for m in measurements
        ]

        result = {
            "task_id": self.request.id,
            "status": "SUCCESS" if measurements else "PARTIAL_FAILURE",
            "measurements": measurements_summary,  # Without iq_data
            "measurements_count": len(measurements),
            "errors": errors,
            "start_time": start_time_iso,
            "end_time": datetime.utcnow().isoformat(),
            "frequency_mhz": frequency_mhz,
            "duration_seconds": duration_seconds,
        }

        logger.info(
            "Acquisition complete: %d successful, %d errors", len(measurements), len(errors)
        )

        return result

    except Exception as e:
        logger.exception("Acquisition task failed: %s", str(e))
        raise


@shared_task(bind=True)
def save_measurements_to_minio(
    self,
    task_id: str,
    measurements: list[dict],
):
    """
    Save IQ data from measurements to MinIO.

    This task:
    1. Initializes MinIO client
    2. Stores each measurement's IQ data as .npy file
    3. Stores associated metadata as JSON
    4. Returns S3 paths for all stored files

    Args:
        task_id: Parent task ID (session ID)
        measurements: List of measurement records with 'iq_data' and 'websdr_id'

    Returns:
        Dict with storage results
    """
    try:
        logger.info("Saving %d measurements to MinIO...", len(measurements))

        # Initialize MinIO client
        minio_client = MinIOClient(
            endpoint_url=settings.minio_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket_name=settings.minio_bucket_raw_iq,
        )

        # Ensure bucket exists
        if not minio_client.ensure_bucket_exists():
            logger.error("Failed to access MinIO bucket")
            return {
                "status": "FAILED",
                "message": "Failed to access MinIO bucket",
                "measurements_count": 0,
                "successful": 0,
            }

        stored_measurements = []
        failed_measurements = []

        for idx, measurement in enumerate(measurements):
            try:
                websdr_id = measurement.get("websdr_id")
                iq_data = measurement.get("iq_data")

                # Validate required fields
                if websdr_id is None:
                    logger.error("Missing websdr_id in measurement at index %d", idx)
                    failed_measurements.append(
                        {
                            "websdr_id": None,
                            "error": "Missing websdr_id",
                            "status": "FAILED",
                        }
                    )
                    continue

                if iq_data is None:
                    logger.error("Missing iq_data for WebSDR %d", websdr_id)
                    failed_measurements.append(
                        {
                            "websdr_id": websdr_id,
                            "error": "Missing iq_data",
                            "status": "FAILED",
                        }
                    )
                    continue

                # Convert IQ data list to numpy array if needed
                if isinstance(iq_data, list):
                    iq_array = np.array(iq_data, dtype=np.complex64)
                else:
                    iq_array = np.asarray(iq_data, dtype=np.complex64)

                # Prepare metadata (exclude iq_data to avoid duplication)
                metadata = {
                    k: v for k, v in measurement.items() if k not in ["iq_data", "iq_data_path"]
                }

                # Upload to MinIO
                success, result = minio_client.upload_iq_data(
                    iq_data=iq_array,
                    task_id=task_id,
                    websdr_id=int(websdr_id),
                    metadata=metadata,
                )

                if success:
                    stored_measurements.append(
                        {
                            "websdr_id": websdr_id,
                            "s3_path": result,
                            "samples_count": len(iq_array),
                            "status": "SUCCESS",
                        }
                    )
                    logger.info("Saved WebSDR %d measurement to %s", websdr_id, result)
                else:
                    failed_measurements.append(
                        {
                            "websdr_id": websdr_id,
                            "error": result,
                            "status": "FAILED",
                        }
                    )
                    logger.error("Failed to save WebSDR %d measurement: %s", websdr_id, result)

                # Update progress
                progress = ((idx + 1) / len(measurements)) * 100
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": idx + 1,
                        "total": len(measurements),
                        "successful": len(stored_measurements),
                        "failed": len(failed_measurements),
                        "status": f"Storing {idx + 1}/{len(measurements)} measurements to MinIO...",
                        "progress": progress,
                    },
                )

            except Exception as e:
                error_msg = f"Exception storing WebSDR measurement: {str(e)}"
                logger.exception(error_msg)
                failed_measurements.append(
                    {
                        "websdr_id": measurement.get("websdr_id"),
                        "error": error_msg,
                        "status": "FAILED",
                    }
                )

        result = {
            "status": "SUCCESS" if not failed_measurements else "PARTIAL_FAILURE",
            "message": f"Stored {len(stored_measurements)} measurements",
            "measurements_count": len(measurements),
            "successful": len(stored_measurements),
            "failed": len(failed_measurements),
            "stored_measurements": stored_measurements,
            "failed_measurements": failed_measurements,
        }

        logger.info(
            "MinIO storage complete: %d successful, %d failed",
            len(stored_measurements),
            len(failed_measurements),
        )

        return result

    except Exception as e:
        logger.exception("MinIO storage task failed: %s", str(e))
        return {
            "status": "FAILED",
            "error": str(e),
            "measurements_count": len(measurements),
            "successful": 0,
        }


@shared_task(bind=True)
def save_measurements_to_timescaledb(
    self,
    task_id: str,
    measurements: list[dict],
    s3_paths: dict[int, str] | None = None,
):
    """
    Save measurement metadata to TimescaleDB.

    This task:
    1. Initializes database manager
    2. Bulk inserts measurements into measurements hypertable
    3. Handles partial failures gracefully
    4. Returns detailed result with counts

    Args:
        task_id: Parent task ID (session ID)
        measurements: List of measurement dictionaries
        s3_paths: Optional dict mapping websdr_id to S3 paths

    Returns:
        Dict with insertion results

    Raises:
        Will retry on database errors (max 3 times)
    """
    try:
        logger.info("Saving %d measurements to TimescaleDB for task %s", len(measurements), task_id)

        # Initialize database manager
        from ..storage.db_manager import get_db_manager

        db_manager = get_db_manager()

        # Verify database connection
        if not db_manager.check_connection():
            logger.error("Failed to connect to TimescaleDB")
            # Retry task - will trigger autoretry
            raise Exception("Database connection failed")

        # Update progress: starting
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": len(measurements),
                "status": f"Inserting {len(measurements)} measurements into TimescaleDB...",
                "progress": 0,
            },
        )

        # Bulk insert measurements
        successful, failed = db_manager.insert_measurements_bulk(
            task_id=task_id, measurements_list=measurements, s3_paths=s3_paths
        )

        # Update final progress
        progress = 100 if successful > 0 else 0
        self.update_state(
            state="PROGRESS",
            meta={
                "current": successful,
                "total": len(measurements),
                "successful": successful,
                "failed": failed,
                "status": f"Completed: {successful} inserted, {failed} failed",
                "progress": progress,
            },
        )

        result = {
            "status": "SUCCESS" if failed == 0 else "PARTIAL_FAILURE",
            "message": f"Inserted {successful}/{len(measurements)} measurements",
            "measurements_count": len(measurements),
            "successful": successful,
            "failed": failed,
            "task_id": task_id,
        }

        logger.info("TimescaleDB storage complete: %d successful, %d failed", successful, failed)

        return result

    except Exception as e:
        logger.exception("TimescaleDB storage task failed: %s", str(e))
        # Task will automatically retry due to base=AcquisitionTask
        raise


@shared_task(bind=True)
def health_check_websdrs(self):
    """
    Health check all configured WebSDRs.

    Returns:
        Dict mapping WebSDR ID to health status
    """
    import asyncio

    from ..routers.acquisition import get_websdrs_config

    # Load WebSDR configs from the configuration
    websdrs_config_list = get_websdrs_config()
    websdrs = [WebSDRConfig(**cfg) for cfg in websdrs_config_list]

    if not websdrs:
        logger.warning("No WebSDRs configured for health check")
        return {}

    async def check():
        fetcher = OpenWebRXFetcher(websdrs=websdrs)
        return await fetcher.health_check()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    health_status = loop.run_until_complete(check())
    logger.info("WebSDR health check: %s", health_status)

    return health_status


@shared_task(bind=True, base=AcquisitionTask)
def acquire_iq_chunked(
    self,
    frequency_hz: int,
    duration_seconds: int,
    session_id: str,
    websdrs_config_list: list[dict],
    sample_rate_khz: float = 12.5,
):
    """
    Acquire IQ data in 1-second chunks for a recording session.

    This task splits the total duration into 1-second samples,
    acquiring each as a separate measurement for better training data granularity.

    Each 1-second chunk is:
    1. Fetched from all WebSDRs simultaneously
    2. Processed for signal metrics
    3. Saved to MinIO individually
    4. Written to TimescaleDB as separate measurements

    Args:
        frequency_hz: Target frequency in Hz
        duration_seconds: Total duration in seconds (will be split into 1s chunks)
        session_id: Recording session UUID
        websdrs_config_list: List of WebSDR config dicts
        sample_rate_khz: Sample rate in kHz

    Returns:
        Dict with chunked acquisition results
    """
    try:
        frequency_mhz = frequency_hz / 1e6

        self.update_state(
            state="PROGRESS",
            meta={
                "current": 0,
                "total": duration_seconds,
                "status": f"Starting chunked acquisition: {duration_seconds} x 1s samples",
            },
        )

        logger.info(
            "Starting chunked acquisition: %d x 1s chunks at %.2f MHz for session %s",
            duration_seconds,
            frequency_mhz,
            session_id,
        )

        # Parse WebSDR configs
        websdrs = [WebSDRConfig(**cfg) for cfg in websdrs_config_list]

        all_measurements = []
        all_errors = []
        successful_chunks = 0

        import asyncio

        # Acquire each 1-second chunk
        for chunk_idx in range(duration_seconds):
            chunk_start_time = datetime.utcnow().isoformat()

            logger.info(f"Acquiring chunk {chunk_idx + 1}/{duration_seconds}")

            async def fetch_and_process_chunk():
                fetcher = OpenWebRXFetcher(
                    websdrs=websdrs, timeout=30, retry_count=3, concurrent_limit=7
                )

                # Fetch 1-second sample from all WebSDRs
                iq_data_dict = await fetcher.fetch_iq_simultaneous(
                    frequency_mhz=frequency_mhz,
                    duration_seconds=1.0,  # 1-second chunks
                    sample_rate_khz=sample_rate_khz,
                )

                chunk_measurements = []
                chunk_errors = []

                # Process results for this chunk
                for websdr_id, (iq_data, error) in iq_data_dict.items():
                    if error:
                        chunk_errors.append(f"Chunk {chunk_idx+1}, WebSDR {websdr_id}: {error}")
                        logger.warning(
                            "Chunk %d WebSDR %d failed: %s", chunk_idx + 1, websdr_id, error
                        )
                    elif iq_data is not None:
                        try:
                            # Compute metrics
                            metrics = IQProcessor.compute_metrics(
                                iq_data=iq_data,
                                sample_rate_hz=int(sample_rate_khz * 1e3),
                                target_frequency_hz=int(frequency_mhz * 1e6),
                                noise_bandwidth_hz=10000,
                            )

                            # Create measurement record with chunk info
                            measurement = {
                                "websdr_id": websdr_id,
                                "frequency_mhz": frequency_mhz,
                                "sample_rate_khz": sample_rate_khz,
                                "samples_count": len(iq_data),
                                "timestamp_utc": chunk_start_time,
                                "chunk_index": chunk_idx,
                                "chunk_duration": 1.0,
                                "session_id": session_id,
                                "metrics": metrics.dict(),
                                "iq_data_path": f"s3://heimdall-raw-iq/sessions/{session_id}/chunk_{chunk_idx:03d}_websdr_{websdr_id}.npy",
                                "iq_data": iq_data.tolist(),
                            }
                            chunk_measurements.append(measurement)

                            logger.debug(
                                "Chunk %d WebSDR %d - SNR: %.2f dB, Offset: %.2f Hz",
                                chunk_idx + 1,
                                websdr_id,
                                metrics.snr_db,
                                metrics.frequency_offset_hz,
                            )
                        except Exception as e:
                            error_msg = f"Chunk {chunk_idx+1} processing error for WebSDR {websdr_id}: {str(e)}"
                            chunk_errors.append(error_msg)
                            logger.exception(error_msg)

                return chunk_measurements, chunk_errors

            # Run async chunk acquisition
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            chunk_measurements, chunk_errors = loop.run_until_complete(fetch_and_process_chunk())

            all_measurements.extend(chunk_measurements)
            all_errors.extend(chunk_errors)

            if chunk_measurements:
                successful_chunks += 1

            # Upload chunk IQ data to MinIO before saving to database
            if chunk_measurements:
                try:
                    from ..config import settings
                    minio_client = MinIOClient(
                        endpoint_url=settings.minio_url,
                        access_key=settings.minio_access_key,
                        secret_key=settings.minio_secret_key,
                        bucket_name=settings.minio_bucket_raw_iq,
                    )

                    # Upload each measurement's IQ data to MinIO
                    for measurement in chunk_measurements:
                        if 'iq_data' in measurement and measurement['iq_data'] is not None:
                            iq_array = np.array(measurement['iq_data'])
                            chunk_key = f"sessions/{session_id}/chunk_{chunk_idx+1:03d}_websdr_{measurement['websdr_id']}.npy"

                            try:
                                buffer = BytesIO()
                                np.save(buffer, iq_array)
                                buffer.seek(0)

                                minio_client.s3_client.put_object(
                                    Bucket=settings.minio_bucket_raw_iq,
                                    Key=chunk_key,
                                    Body=buffer.getvalue(),
                                    ContentType="application/octet-stream",
                                    Metadata={
                                        'session_id': str(session_id),
                                        'websdr_id': str(measurement['websdr_id']),
                                        'chunk_index': str(chunk_idx + 1),
                                        'samples_count': str(len(iq_array)),
                                    }
                                )
                                logger.debug(
                                    "Uploaded chunk %d websdr %d to MinIO: %s",
                                    chunk_idx + 1,
                                    measurement['websdr_id'],
                                    chunk_key
                                )
                            except Exception as upload_error:
                                logger.warning(
                                    "Failed to upload chunk %d websdr %d to MinIO: %s",
                                    chunk_idx + 1,
                                    measurement['websdr_id'],
                                    upload_error
                                )
                except Exception as minio_error:
                    logger.error("MinIO setup error for chunk %d: %s", chunk_idx + 1, minio_error)

            # Save chunk measurements to TimescaleDB immediately after each chunk
            if chunk_measurements:
                try:
                    from ..storage.db_manager import get_db_manager
                    db_manager = get_db_manager()

                    # Prepare measurements for database
                    db_measurements = []
                    for measurement in chunk_measurements:
                        db_measurement = {
                            'websdr_station_id': None,  # Will be mapped by db_manager from websdr_id
                            'websdr_id': measurement['websdr_id'],  # Temporary field for mapping
                            'frequency_hz': int(frequency_hz),
                            'iq_sample_rate': int(sample_rate_khz * 1000),
                            'iq_samples_count': measurement['samples_count'],
                            'timestamp': measurement['timestamp_utc'],
                            'snr_db': measurement['metrics']['snr_db'],
                            'frequency_offset_hz': int(measurement['metrics']['frequency_offset_hz']),
                            'signal_strength_db': measurement['metrics']['signal_power_dbm'],
                            'iq_data_location': measurement['iq_data_path'],
                            'iq_data_format': 'npy',
                            'duration_seconds': float(measurement.get('chunk_duration', 1.0)),
                            'recording_session_id': session_id,  # Link to session
                        }
                        db_measurements.append(db_measurement)
                    
                    # Bulk insert measurements for this chunk
                    successful, failed = db_manager.insert_measurements_bulk(
                        task_id=self.request.id,
                        measurements_list=db_measurements
                    )
                    logger.debug(
                        "Chunk %d: Saved %d measurements to database (%d failed)",
                        chunk_idx + 1,
                        successful,
                        failed
                    )
                except Exception as e:
                    logger.error("Failed to save chunk %d measurements to database: %s", chunk_idx + 1, str(e))

            # Update progress
            progress = ((chunk_idx + 1) / duration_seconds) * 100
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": chunk_idx + 1,
                    "total": duration_seconds,
                    "successful_chunks": successful_chunks,
                    "total_measurements": len(all_measurements),
                    "status": f"Acquired chunk {chunk_idx + 1}/{duration_seconds}",
                    "progress": progress,
                },
            )

            # Broadcast progress via RabbitMQ (not direct WebSocket from Celery)
            # This will be picked up by the backend WebSocket handlers
            try:
                from kombu import Connection, Exchange

                from ..config import settings

                connection = Connection(settings.celery_broker_url)
                exchange = Exchange("heimdall", type="topic", durable=True)

                with connection.Producer() as producer:
                    producer.publish(
                        {
                            "event": "session:progress",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "session_id": session_id,
                                "chunk": chunk_idx + 1,
                                "total_chunks": duration_seconds,
                                "progress": progress,
                                "measurements_count": len(all_measurements),
                            },
                        },
                        exchange=exchange,
                        routing_key=f"session.{session_id}.progress",
                        serializer="json",
                    )
                    logger.debug(
                        f"Published progress event for session {session_id}: chunk {chunk_idx + 1}/{duration_seconds}"
                    )
            except Exception as e:
                logger.warning(f"Failed to broadcast progress: {e}")

        # Prepare result summary (without iq_data for JSON serialization)
        measurements_summary = [
            {k: v for k, v in m.items() if k != "iq_data"} for m in all_measurements
        ]

        result = {
            "task_id": self.request.id,
            "session_id": session_id,
            "status": "SUCCESS" if successful_chunks == duration_seconds else "PARTIAL_FAILURE",
            "measurements": measurements_summary,
            "total_chunks": duration_seconds,
            "successful_chunks": successful_chunks,
            "measurements_count": len(all_measurements),
            "errors": all_errors,
            "frequency_mhz": frequency_mhz,
            "duration_seconds": duration_seconds,
            "end_time": datetime.utcnow().isoformat(),
        }

        logger.info(
            "Chunked acquisition complete: %d/%d chunks successful, %d total measurements",
            successful_chunks,
            duration_seconds,
            len(all_measurements),
        )

        # Update session status in database
        try:

            from sqlalchemy import create_engine, text

            from ..config import settings

            # Create synchronous engine for worker context (not async)
            engine = create_engine(settings.database_url, echo=False, pool_size=5, max_overflow=10)

            with engine.connect() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE heimdall.recording_sessions
                        SET status = :status,
                            session_end = NOW(),
                            duration_seconds = :duration,
                            updated_at = NOW()
                        WHERE id = :session_id
                    """
                    ),
                    {
                        "status": (
                            "completed"
                            if successful_chunks == duration_seconds
                            else "partial_failure"
                        ),
                        "duration": duration_seconds,
                        "session_id": session_id,
                    },
                )
                conn.commit()

            logger.info(f"Updated session {session_id} status to completed")
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")

        return result

    except Exception as e:
        logger.exception("Chunked acquisition task failed: %s", str(e))

        # Update session to failed
        try:

            from sqlalchemy import create_engine, text

            from ..config import settings

            engine = create_engine(settings.database_url, echo=False, pool_size=5, max_overflow=10)

            with engine.connect() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE heimdall.recording_sessions
                        SET status = 'failed',
                            session_end = NOW(),
                            updated_at = NOW()
                        WHERE id = :session_id
                    """
                    ),
                    {"session_id": session_id},
                )
                conn.commit()

            logger.info(f"Updated session {session_id} status to failed")
        except Exception as db_error:
            logger.error(f"Failed to update session to failed status: {db_error}")

        raise
