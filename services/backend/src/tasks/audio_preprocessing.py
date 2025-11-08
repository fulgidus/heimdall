"""
Celery task for preprocessing audio library files.

Preprocessing pipeline:
1. Download original audio file from MinIO (heimdall-audio-library bucket)
2. Extract 1-second chunks sequentially
3. Resample each chunk to 200kHz (training sample rate)
4. Save chunks as .npy files to MinIO (heimdall-audio-chunks bucket)
5. Create database records in audio_chunks table
6. Update audio_library status (PENDING → PROCESSING → READY/FAILED)

This preprocessing happens ONCE at upload time.
Training then loads pre-chunked audio instantly (no processing, no cache).
"""

import logging
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
from celery import shared_task
from pydub import AudioSegment
from scipy import signal

from ..config import settings
from ..db import get_pool, init_pool
from ..storage.minio_client import MinIOClient

logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE_HZ = 200_000  # 200 kHz for training
CHUNK_DURATION_SECONDS = 1.0  # 1-second chunks
CHUNKS_BUCKET = "heimdall-audio-chunks"
LIBRARY_BUCKET = "heimdall-audio-library"


@shared_task(bind=True, name='backend.tasks.preprocess_audio_file')
def preprocess_audio_file(self, audio_id: str) -> dict:
    """
    Preprocess audio file by extracting and resampling 1-second chunks.
    
    Args:
        audio_id: UUID of audio file in audio_library table
    
    Returns:
        dict with preprocessing results
    """
    logger.info(f"Starting audio preprocessing for audio_id={audio_id}")
    
    try:
        # Get database pool and run async operations
        import asyncio
        
        async def run_preprocessing():
            # Initialize pool if not already initialized (worker process doesn't inherit pool)
            try:
                pool = get_pool()
            except RuntimeError:
                logger.info("Database pool not initialized in worker, initializing now...")
                pool = await init_pool()
            
            # Update status to PROCESSING
            await _update_audio_status(pool, audio_id, "PROCESSING")
            
            try:
                # Get audio metadata from database
                audio_metadata = await _get_audio_metadata(pool, audio_id)
                
                if not audio_metadata:
                    logger.error(f"Audio file {audio_id} not found in database")
                    return {'success': False, 'error': 'Audio file not found'}
                
                logger.info(
                    f"Preprocessing audio file: {audio_metadata['filename']} "
                    f"(duration={audio_metadata['duration_seconds']:.1f}s, "
                    f"sample_rate={audio_metadata['sample_rate']}Hz)"
                )
                
                # Download audio file from MinIO
                audio_bytes = await _download_audio_from_minio(
                    audio_metadata['minio_bucket'],
                    audio_metadata['minio_path']
                )
                
                # Load audio with pydub
                audio_segment = AudioSegment.from_file(
                    BytesIO(audio_bytes),
                    format=audio_metadata['audio_format']
                )
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1]
                samples = samples / (2 ** (8 * audio_segment.sample_width - 1))
                
                # Handle stereo: convert to mono if needed
                if audio_segment.channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                
                original_sample_rate = audio_segment.frame_rate
                
                logger.info(
                    f"Loaded audio: {len(samples)} samples at {original_sample_rate}Hz, "
                    f"duration={len(samples) / original_sample_rate:.2f}s"
                )
                
                # Calculate number of chunks (floor to ignore incomplete last chunk)
                total_duration = len(samples) / original_sample_rate
                num_chunks = int(total_duration / CHUNK_DURATION_SECONDS)
                
                if num_chunks == 0:
                    error_msg = f"Audio file too short: {total_duration:.2f}s (need ≥1s)"
                    logger.error(error_msg)
                    await _update_audio_status(
                        pool, audio_id, "FAILED",
                        processing_error=error_msg
                    )
                    return {'success': False, 'error': error_msg}
                
                logger.info(f"Extracting {num_chunks} chunks (ignoring incomplete final chunk)")
                
                # Initialize MinIO client for chunks bucket
                minio_client = MinIOClient(
                    endpoint_url=settings.minio_url,
                    access_key=settings.minio_access_key,
                    secret_key=settings.minio_secret_key,
                    bucket_name=CHUNKS_BUCKET
                )
                
                # Ensure chunks bucket exists
                if not minio_client.ensure_bucket_exists():
                    raise RuntimeError(f"Failed to ensure bucket {CHUNKS_BUCKET} exists")
                
                # Process each chunk sequentially
                chunks_created = 0
                
                for chunk_index in range(num_chunks):
                    try:
                        # Extract chunk from original audio
                        start_sample = int(chunk_index * CHUNK_DURATION_SECONDS * original_sample_rate)
                        end_sample = int((chunk_index + 1) * CHUNK_DURATION_SECONDS * original_sample_rate)
                        chunk_samples = samples[start_sample:end_sample]
                        
                        # Resample to 200kHz
                        if original_sample_rate != TARGET_SAMPLE_RATE_HZ:
                            # Calculate resampling ratio
                            num_target_samples = int(len(chunk_samples) * TARGET_SAMPLE_RATE_HZ / original_sample_rate)
                            
                            # Use scipy.signal.resample for high-quality resampling
                            chunk_resampled = signal.resample(chunk_samples, num_target_samples)
                        else:
                            chunk_resampled = chunk_samples
                        
                        # Convert to float32 numpy array
                        chunk_array = np.array(chunk_resampled, dtype=np.float32)
                        
                        # Calculate RMS amplitude for normalization tracking
                        rms_amplitude = float(np.sqrt(np.mean(chunk_array ** 2)))
                        
                        # Save chunk to MinIO as .npy file
                        chunk_path = f"{audio_id}/chunk_{chunk_index:04d}.npy"
                        chunk_bytes = BytesIO()
                        np.save(chunk_bytes, chunk_array)
                        chunk_bytes.seek(0)
                        chunk_data = chunk_bytes.read()
                        
                        minio_client.s3_client.put_object(
                            Bucket=CHUNKS_BUCKET,
                            Key=chunk_path,
                            Body=chunk_data,
                            ContentType="application/octet-stream"
                        )
                        
                        # Save chunk metadata to database
                        await _save_chunk_to_db(
                            pool,
                            audio_id=audio_id,
                            chunk_index=chunk_index,
                            duration_seconds=CHUNK_DURATION_SECONDS,
                            sample_rate=TARGET_SAMPLE_RATE_HZ,
                            num_samples=len(chunk_array),
                            minio_bucket=CHUNKS_BUCKET,
                            minio_path=chunk_path,
                            file_size_bytes=len(chunk_data),
                            original_offset_seconds=chunk_index * CHUNK_DURATION_SECONDS,
                            rms_amplitude=rms_amplitude
                        )
                        
                        chunks_created += 1
                        
                        # Update progress in database every 100 chunks (for real-time UI feedback)
                        if (chunk_index + 1) % 100 == 0:
                            await _update_audio_status(
                                pool, audio_id, "PROCESSING",
                                total_chunks=chunks_created
                            )
                        
                        # Log progress every 10 chunks
                        if (chunk_index + 1) % 10 == 0 or (chunk_index + 1) == num_chunks:
                            logger.info(
                                f"Processed chunk {chunk_index + 1}/{num_chunks} "
                                f"({(chunk_index + 1) / num_chunks * 100:.1f}%)"
                            )
                    
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_index}: {e}")
                        # Continue with other chunks (don't fail entire preprocessing)
                        continue
                
                if chunks_created == 0:
                    error_msg = "Failed to create any chunks"
                    logger.error(error_msg)
                    await _update_audio_status(
                        pool, audio_id, "FAILED",
                        processing_error=error_msg
                    )
                    return {'success': False, 'error': error_msg}
                
                # Update audio_library status to READY
                await _update_audio_status(
                    pool, audio_id, "READY",
                    total_chunks=chunks_created
                )
                
                logger.info(
                    f"Successfully preprocessed audio {audio_id}: "
                    f"{chunks_created} chunks created "
                    f"({chunks_created - num_chunks} incomplete chunks skipped)"
                )
                
                return {
                    'success': True,
                    'audio_id': audio_id,
                    'chunks_created': chunks_created,
                    'chunks_expected': num_chunks
                }
            
            except Exception as e:
                logger.exception(f"Error preprocessing audio {audio_id}: {e}")
                
                # Update status to FAILED
                await _update_audio_status(
                    pool, audio_id, "FAILED",
                    processing_error=str(e)
                )
                
                return {
                    'success': False,
                    'audio_id': audio_id,
                    'error': str(e)
                }
        
        # Get or create event loop for Celery worker
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Don't close the loop - let it be reused by worker
        return loop.run_until_complete(run_preprocessing())
    
    except Exception as e:
        logger.exception(f"Fatal error in audio preprocessing task: {e}")
        return {'success': False, 'error': str(e)}


async def _get_audio_metadata(pool, audio_id: str) -> Optional[dict]:
    """Get audio metadata from database."""
    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id, filename, audio_format, duration_seconds, sample_rate,
                   channels, minio_bucket, minio_path, processing_status
            FROM heimdall.audio_library
            WHERE id = $1
            """,
            uuid.UUID(audio_id)
        )
        
        if not result:
            return None
        
        return {
            'id': result['id'],
            'filename': result['filename'],
            'audio_format': result['audio_format'],
            'duration_seconds': result['duration_seconds'],
            'sample_rate': result['sample_rate'],
            'channels': result['channels'],
            'minio_bucket': result['minio_bucket'],
            'minio_path': result['minio_path'],
            'processing_status': result['processing_status']
        }


async def _download_audio_from_minio(bucket: str, path: str) -> bytes:
    """Download audio file from MinIO."""
    minio_client = MinIOClient(
        endpoint_url=settings.minio_url,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=bucket
    )
    
    try:
        response = minio_client.s3_client.get_object(
            Bucket=bucket,
            Key=path
        )
        audio_bytes = response['Body'].read()
        logger.debug(f"Downloaded audio from MinIO: {bucket}/{path} ({len(audio_bytes)} bytes)")
        return audio_bytes
    
    except Exception as e:
        logger.error(f"Failed to download audio from MinIO: {bucket}/{path}: {e}")
        raise


async def _save_chunk_to_db(
    pool,
    audio_id: str,
    chunk_index: int,
    duration_seconds: float,
    sample_rate: int,
    num_samples: int,
    minio_bucket: str,
    minio_path: str,
    file_size_bytes: int,
    original_offset_seconds: float,
    rms_amplitude: float
) -> None:
    """Save chunk metadata to database."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.audio_chunks (
                audio_id, chunk_index, duration_seconds, sample_rate,
                num_samples, minio_bucket, minio_path, file_size_bytes,
                original_offset_seconds, rms_amplitude, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
            ON CONFLICT (audio_id, chunk_index) DO UPDATE
            SET duration_seconds = EXCLUDED.duration_seconds,
                sample_rate = EXCLUDED.sample_rate,
                num_samples = EXCLUDED.num_samples,
                minio_path = EXCLUDED.minio_path,
                file_size_bytes = EXCLUDED.file_size_bytes,
                original_offset_seconds = EXCLUDED.original_offset_seconds,
                rms_amplitude = EXCLUDED.rms_amplitude
            """,
            uuid.UUID(audio_id),
            chunk_index,
            duration_seconds,
            sample_rate,
            num_samples,
            minio_bucket,
            minio_path,
            file_size_bytes,
            original_offset_seconds,
            rms_amplitude
        )


async def _update_audio_status(
    pool,
    audio_id: str,
    status: str,
    total_chunks: Optional[int] = None,
    processing_error: Optional[str] = None
) -> None:
    """
    Update audio_library processing status.
    
    Args:
        pool: Database connection pool
        audio_id: Audio file UUID
        status: New status (PENDING, PROCESSING, READY, FAILED)
        total_chunks: Total chunks created (for READY status)
        processing_error: Error message (for FAILED status)
    """
    async with pool.acquire() as conn:
        if status == "PROCESSING":
            await conn.execute(
                """
                UPDATE heimdall.audio_library
                SET processing_status = $1,
                    processing_started_at = NOW(),
                    processing_completed_at = NULL,
                    processing_error = NULL
                WHERE id = $2
                """,
                status,
                uuid.UUID(audio_id)
            )
        
        elif status == "READY":
            await conn.execute(
                """
                UPDATE heimdall.audio_library
                SET processing_status = $1,
                    processing_completed_at = NOW(),
                    processing_error = NULL,
                    total_chunks = $2
                WHERE id = $3
                """,
                status,
                total_chunks,
                uuid.UUID(audio_id)
            )
        
        elif status == "FAILED":
            await conn.execute(
                """
                UPDATE heimdall.audio_library
                SET processing_status = $1,
                    processing_completed_at = NOW(),
                    processing_error = $2
                WHERE id = $3
                """,
                status,
                processing_error,
                uuid.UUID(audio_id)
            )
        
        else:
            # Default update (e.g., PENDING)
            await conn.execute(
                """
                UPDATE heimdall.audio_library
                SET processing_status = $1
                WHERE id = $2
                """,
                status,
                uuid.UUID(audio_id)
            )
    
    logger.info(f"Updated audio_library status: audio_id={audio_id}, status={status}")
