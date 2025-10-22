"""Celery tasks for IQ data acquisition."""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from celery import shared_task, Task
from celery.utils.log import get_task_logger
import numpy as np

from ..models.websdrs import WebSDRConfig, MeasurementRecord, AcquisitionRequest
from ..fetchers.websdr_fetcher import WebSDRFetcher
from ..processors.iq_processor import IQProcessor
from ..storage.minio_client import MinIOClient
from ..config import settings

logger = get_task_logger(__name__)


class AcquisitionTask(Task):
    """Base task for acquisitions."""
    
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True


@shared_task(bind=True, base=AcquisitionTask)
def acquire_iq(
    self,
    frequency_mhz: float,
    duration_seconds: float,
    start_time_iso: str,
    websdrs_config_list: List[Dict],
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
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': len(websdrs_config_list),
                'status': 'Initializing fetcher...'
            }
        )
        
        # Parse WebSDR configs
        websdrs = [
            WebSDRConfig(**cfg) for cfg in websdrs_config_list
        ]
        
        logger.info(
            "Starting acquisition task from %d WebSDRs at %.2f MHz for %.1f seconds",
            len(websdrs),
            frequency_mhz,
            duration_seconds
        )
        
        # Fetch IQ data
        measurements = []
        errors = []
        
        import asyncio
        
        async def fetch_and_process():
            async with WebSDRFetcher(
                websdrs=websdrs,
                timeout=30,
                retry_count=3,
                concurrent_limit=7
            ) as fetcher:
                # Update state: fetching
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': 0,
                        'total': len(websdrs),
                        'status': f'Fetching IQ from {len(websdrs)} WebSDRs...'
                    }
                )
                
                # Fetch all simultaneously
                iq_data_dict = await fetcher.fetch_iq_simultaneous(
                    frequency_mhz=frequency_mhz,
                    duration_seconds=duration_seconds,
                    sample_rate_khz=sample_rate_khz
                )
                
                # Process results
                successful = 0
                for idx, (websdr_id, (iq_data, error)) in enumerate(iq_data_dict.items()):
                    if error:
                        errors.append(f"WebSDR {websdr_id}: {error}")
                        logger.warning("WebSDR %d acquisition failed: %s", websdr_id, error)
                    else:
                        try:
                            # Compute metrics
                            metrics = IQProcessor.compute_metrics(
                                iq_data=iq_data,
                                sample_rate_hz=int(sample_rate_khz * 1e3),
                                target_frequency_hz=int(frequency_mhz * 1e6),
                                noise_bandwidth_hz=10000
                            )
                            
                            # Create measurement record
                            measurement = {
                                'websdr_id': websdr_id,
                                'frequency_mhz': frequency_mhz,
                                'sample_rate_khz': sample_rate_khz,
                                'samples_count': len(iq_data),
                                'timestamp_utc': datetime.utcnow().isoformat(),
                                'metrics': metrics.dict(),
                                'iq_data_path': f's3://heimdall-raw-iq/sessions/{self.request.id}/websdr_{websdr_id}.npy',
                                'iq_data': iq_data.tolist(),  # For now, store in memory
                            }
                            measurements.append(measurement)
                            successful += 1
                            
                            logger.info(
                                "Processed WebSDR %d - SNR: %.2f dB, Offset: %.2f Hz",
                                websdr_id,
                                metrics.snr_db,
                                metrics.frequency_offset_hz
                            )
                        except Exception as e:
                            error_msg = f"Processing error for WebSDR {websdr_id}: {str(e)}"
                            errors.append(error_msg)
                            logger.exception(error_msg)
                    
                    # Update progress
                    progress = ((idx + 1) / len(websdrs)) * 100
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': idx + 1,
                            'total': len(websdrs),
                            'successful': successful,
                            'status': f'Processing {idx + 1}/{len(websdrs)} measurements...',
                            'progress': progress
                        }
                    )
                
                return measurements, errors
        
        # Run async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        measurements, errors = loop.run_until_complete(fetch_and_process())
        
        result = {
            'task_id': self.request.id,
            'status': 'SUCCESS' if measurements else 'PARTIAL_FAILURE',
            'measurements': measurements,
            'measurements_count': len(measurements),
            'errors': errors,
            'start_time': start_time_iso,
            'end_time': datetime.utcnow().isoformat(),
            'frequency_mhz': frequency_mhz,
            'duration_seconds': duration_seconds,
        }
        
        logger.info(
            "Acquisition complete: %d successful, %d errors",
            len(measurements),
            len(errors)
        )
        
        return result
    
    except Exception as e:
        logger.exception("Acquisition task failed: %s", str(e))
        raise


@shared_task(bind=True)
def save_measurements_to_minio(
    self,
    task_id: str,
    measurements: List[Dict],
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
                'status': 'FAILED',
                'message': 'Failed to access MinIO bucket',
                'measurements_count': 0,
                'successful': 0,
            }
        
        stored_measurements = []
        failed_measurements = []
        
        for idx, measurement in enumerate(measurements):
            try:
                websdr_id = measurement.get('websdr_id')
                iq_data = measurement.get('iq_data')
                
                # Validate required fields
                if websdr_id is None:
                    logger.error("Missing websdr_id in measurement at index %d", idx)
                    failed_measurements.append({
                        'websdr_id': None,
                        'error': 'Missing websdr_id',
                        'status': 'FAILED',
                    })
                    continue
                
                if iq_data is None:
                    logger.error("Missing iq_data for WebSDR %d", websdr_id)
                    failed_measurements.append({
                        'websdr_id': websdr_id,
                        'error': 'Missing iq_data',
                        'status': 'FAILED',
                    })
                    continue
                
                # Convert IQ data list to numpy array if needed
                if isinstance(iq_data, list):
                    iq_array = np.array(iq_data, dtype=np.complex64)
                else:
                    iq_array = np.asarray(iq_data, dtype=np.complex64)
                
                # Prepare metadata (exclude iq_data to avoid duplication)
                metadata = {
                    k: v for k, v in measurement.items()
                    if k not in ['iq_data', 'iq_data_path']
                }
                
                # Upload to MinIO
                success, result = minio_client.upload_iq_data(
                    iq_data=iq_array,
                    task_id=task_id,
                    websdr_id=int(websdr_id),
                    metadata=metadata,
                )
                
                if success:
                    stored_measurements.append({
                        'websdr_id': websdr_id,
                        's3_path': result,
                        'samples_count': len(iq_array),
                        'status': 'SUCCESS',
                    })
                    logger.info(
                        "Saved WebSDR %d measurement to %s",
                        websdr_id,
                        result
                    )
                else:
                    failed_measurements.append({
                        'websdr_id': websdr_id,
                        'error': result,
                        'status': 'FAILED',
                    })
                    logger.error(
                        "Failed to save WebSDR %d measurement: %s",
                        websdr_id,
                        result
                    )
                
                # Update progress
                progress = ((idx + 1) / len(measurements)) * 100
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx + 1,
                        'total': len(measurements),
                        'successful': len(stored_measurements),
                        'failed': len(failed_measurements),
                        'status': f'Storing {idx + 1}/{len(measurements)} measurements to MinIO...',
                        'progress': progress
                    }
                )
            
            except Exception as e:
                error_msg = f"Exception storing WebSDR measurement: {str(e)}"
                logger.exception(error_msg)
                failed_measurements.append({
                    'websdr_id': measurement.get('websdr_id'),
                    'error': error_msg,
                    'status': 'FAILED',
                })
        
        result = {
            'status': 'SUCCESS' if not failed_measurements else 'PARTIAL_FAILURE',
            'message': f'Stored {len(stored_measurements)} measurements',
            'measurements_count': len(measurements),
            'successful': len(stored_measurements),
            'failed': len(failed_measurements),
            'stored_measurements': stored_measurements,
            'failed_measurements': failed_measurements,
        }
        
        logger.info(
            "MinIO storage complete: %d successful, %d failed",
            len(stored_measurements),
            len(failed_measurements)
        )
        
        return result
    
    except Exception as e:
        logger.exception("MinIO storage task failed: %s", str(e))
        return {
            'status': 'FAILED',
            'error': str(e),
            'measurements_count': len(measurements),
            'successful': 0,
        }


@shared_task(bind=True)
def save_measurements_to_timescaledb(
    self,
    task_id: str,
    measurements: List[Dict],
    s3_paths: Optional[Dict[int, str]] = None,
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
        logger.info(
            "Saving %d measurements to TimescaleDB for task %s",
            len(measurements),
            task_id
        )
        
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
            state='PROGRESS',
            meta={
                'current': 0,
                'total': len(measurements),
                'status': f'Inserting {len(measurements)} measurements into TimescaleDB...',
                'progress': 0
            }
        )
        
        # Bulk insert measurements
        successful, failed = db_manager.insert_measurements_bulk(
            task_id=task_id,
            measurements_list=measurements,
            s3_paths=s3_paths
        )
        
        # Update final progress
        progress = 100 if successful > 0 else 0
        self.update_state(
            state='PROGRESS',
            meta={
                'current': successful,
                'total': len(measurements),
                'successful': successful,
                'failed': failed,
                'status': f'Completed: {successful} inserted, {failed} failed',
                'progress': progress
            }
        )
        
        result = {
            'status': 'SUCCESS' if failed == 0 else 'PARTIAL_FAILURE',
            'message': f'Inserted {successful}/{len(measurements)} measurements',
            'measurements_count': len(measurements),
            'successful': successful,
            'failed': failed,
            'task_id': task_id,
        }
        
        logger.info(
            "TimescaleDB storage complete: %d successful, %d failed",
            successful,
            failed
        )
        
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
    
    # TODO: Load WebSDR configs from database
    websdrs = []
    
    async def check():
        async with WebSDRFetcher(websdrs=websdrs) as fetcher:
            return await fetcher.health_check()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    health_status = loop.run_until_complete(check())
    logger.info("WebSDR health check: %s", health_status)
    
    return health_status
