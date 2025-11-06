"""
Audio library storage management with MinIO and JSON metadata catalog.

Handles:
- File upload/download with MinIO
- Metadata CRUD operations (JSON catalog)
- Category-based organization
- Random sampling for training
- Graceful degradation when library is empty
"""

import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, BinaryIO
from urllib.parse import urlparse

import redis
import structlog
from pydub import AudioSegment
from pydub.utils import mediainfo

from ..models.audio_library import (
    AudioMetadata,
    AudioCategory,
    AudioFormat,
    AudioLibraryEmptyError,
    AudioLibraryStats,
    CategoryWeights,
)
from .minio_client import MinIOClient
from ..config import settings

logger = structlog.get_logger(__name__)


class AudioLibraryStorage:
    """
    Audio library storage manager.
    
    Features:
    - Upload audio files to MinIO with automatic metadata extraction
    - Store metadata in JSON catalog (metadata.json in bucket)
    - Enable/disable files (soft delete)
    - Random sampling for training (respects enabled flag)
    - Category filtering
    """
    
    BUCKET_NAME = "heimdall-audio-library"
    METADATA_KEY = "metadata.json"  # Catalog file in bucket root
    REDIS_WEIGHTS_KEY = "audio_library:category_weights"  # Redis key for weights
    
    def __init__(self, minio_client: MinIOClient):
        """Initialize audio library storage."""
        self.minio_client = minio_client
        
        # Initialize Redis client
        parsed = urlparse(settings.redis_url)
        self.redis_client = redis.Redis(
            host=parsed.hostname or "redis",
            port=parsed.port or 6379,
            password=parsed.password,
            db=int(parsed.path.lstrip("/")) if parsed.path else 0,
            socket_connect_timeout=2,
            decode_responses=True  # Auto-decode bytes to strings
        )
        
        self._ensure_bucket_exists()
        logger.info("AudioLibraryStorage initialized", bucket=self.BUCKET_NAME)
    
    def _ensure_bucket_exists(self):
        """Ensure audio library bucket exists."""
        try:
            # Use MinIOClient's ensure_bucket_exists method
            # But MinIOClient uses self.bucket_name, so we need to temporarily override it
            original_bucket = self.minio_client.bucket_name
            self.minio_client.bucket_name = self.BUCKET_NAME
            
            if not self.minio_client.ensure_bucket_exists():
                raise RuntimeError(f"Failed to ensure bucket {self.BUCKET_NAME} exists")
            
            self.minio_client.bucket_name = original_bucket
            logger.info("Ensured audio library bucket exists", bucket=self.BUCKET_NAME)
            
            # Initialize empty metadata catalog if it doesn't exist
            try:
                self._load_metadata_catalog()
            except Exception:
                # Catalog doesn't exist, create it
                self._save_metadata_catalog({})
                
        except Exception as e:
            logger.error("Failed to ensure bucket exists", bucket=self.BUCKET_NAME, error=str(e))
            raise
    
    def _load_metadata_catalog(self) -> Dict[str, AudioMetadata]:
        """
        Load metadata catalog from MinIO.
        
        Returns:
            Dict mapping audio_id -> AudioMetadata
        """
        try:
            # Use boto3 S3 API
            response = self.minio_client.s3_client.get_object(
                Bucket=self.BUCKET_NAME,
                Key=self.METADATA_KEY
            )
            catalog_bytes = response['Body'].read()
            catalog_json = json.loads(catalog_bytes.decode('utf-8'))
            
            # Convert JSON dict to AudioMetadata objects
            catalog = {}
            for audio_id, metadata_dict in catalog_json.items():
                try:
                    catalog[audio_id] = AudioMetadata(**metadata_dict)
                except Exception as e:
                    logger.warning(
                        "Failed to parse metadata entry, skipping",
                        audio_id=audio_id,
                        error=str(e)
                    )
            
            logger.debug("Loaded metadata catalog", num_entries=len(catalog))
            return catalog
        
        except Exception as e:
            # If metadata.json doesn't exist, return empty catalog
            if "NoSuchKey" in str(e) or "Not Found" in str(e):
                logger.info("Metadata catalog not found, initializing empty catalog")
                return {}
            logger.error("Failed to load metadata catalog", error=str(e))
            raise
    
    def _save_metadata_catalog(self, catalog: Dict[str, AudioMetadata]):
        """
        Save metadata catalog to MinIO.
        
        Args:
            catalog: Dict mapping audio_id -> AudioMetadata
        """
        try:
            # Convert AudioMetadata objects to JSON-serializable dict
            catalog_json = {}
            for audio_id, metadata in catalog.items():
                catalog_json[audio_id] = metadata.model_dump(mode='json')
            
            catalog_bytes = json.dumps(catalog_json, indent=2, default=str).encode('utf-8')
            
            self.minio_client.s3_client.put_object(
                Bucket=self.BUCKET_NAME,
                Key=self.METADATA_KEY,
                Body=catalog_bytes,
                ContentType="application/json"
            )
            
            logger.debug("Saved metadata catalog", num_entries=len(catalog))
        
        except Exception as e:
            logger.error("Failed to save metadata catalog", error=str(e))
            raise
    
    def _extract_audio_metadata(
        self,
        audio_data: BinaryIO,
        filename: str,
        category: AudioCategory,
        tags: List[str]
    ) -> AudioMetadata:
        """
        Extract metadata from audio file using pydub.
        
        Args:
            audio_data: Audio file data (file-like object)
            filename: Original filename
            category: Audio category
            tags: User-defined tags
        
        Returns:
            AudioMetadata with extracted properties
        """
        try:
            # Detect format from filename
            audio_format = AudioFormat.from_filename(filename)
            if not audio_format:
                raise ValueError(f"Unsupported audio format: {filename}")
            
            # Load audio with pydub
            audio_segment = AudioSegment.from_file(audio_data, format=audio_format.value)
            
            # Extract metadata
            duration_seconds = len(audio_segment) / 1000.0  # pydub uses milliseconds
            sample_rate = audio_segment.frame_rate
            channels = audio_segment.channels
            
            # Get file size (rewind and measure)
            audio_data.seek(0, 2)  # Seek to end
            file_size_bytes = audio_data.tell()
            audio_data.seek(0)  # Rewind
            
            # Try to get bitrate (for compressed formats)
            bitrate_kbps = None
            try:
                # Reset stream for mediainfo
                audio_data.seek(0)
                info = mediainfo(audio_data)
                if 'bit_rate' in info:
                    bitrate_kbps = int(info['bit_rate']) // 1000
            except Exception as e:
                logger.debug("Could not extract bitrate", error=str(e))
            
            # Generate unique ID and storage key
            audio_id = str(uuid.uuid4())
            file_extension = Path(filename).suffix
            minio_key = f"{category.value}/{audio_id}{file_extension}"
            
            metadata = AudioMetadata(
                id=audio_id,
                filename=filename,
                category=category,
                format=audio_format,
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                bitrate_kbps=bitrate_kbps,
                file_size_bytes=file_size_bytes,
                minio_bucket=self.BUCKET_NAME,
                minio_key=minio_key,
                enabled=True,
                tags=tags,
                uploaded_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            logger.info(
                "Extracted audio metadata",
                audio_id=audio_id,
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                file_size_mb=file_size_bytes / 1024 / 1024
            )
            
            return metadata
        
        except Exception as e:
            logger.error("Failed to extract audio metadata", filename=filename, error=str(e))
            raise
    
    async def upload_audio(
        self,
        file_data: BinaryIO,
        filename: str,
        category: AudioCategory,
        tags: List[str]
    ) -> AudioMetadata:
        """
        Upload audio file to MinIO, create database record, and trigger preprocessing.
        
        Args:
            file_data: Audio file data
            filename: Original filename
            category: Audio category
            tags: User-defined tags
        
        Returns:
            AudioMetadata for uploaded file
        """
        try:
            # Extract metadata
            metadata = self._extract_audio_metadata(file_data, filename, category, tags)
            
            # Upload to MinIO
            file_data.seek(0)  # Rewind after metadata extraction
            self.minio_client.s3_client.put_object(
                Bucket=self.BUCKET_NAME,
                Key=metadata.minio_key,
                Body=file_data.read(),
                ContentType=f"audio/{metadata.format.value}"
            )
            
            # Update metadata catalog (legacy JSON catalog)
            catalog = self._load_metadata_catalog()
            catalog[metadata.id] = metadata
            self._save_metadata_catalog(catalog)
            
            # Create database record for preprocessing pipeline
            await self._create_database_record(metadata, tags)
            
            # Fetch enriched metadata from database (includes processing_status, total_chunks)
            enriched_metadata = await self.get_audio_metadata(metadata.id)
            if enriched_metadata is None:
                # Fallback to original metadata if database query fails
                enriched_metadata = metadata
            
            logger.info(
                "Audio file uploaded successfully",
                audio_id=metadata.id,
                filename=filename,
                category=category.value,
                size_mb=metadata.file_size_bytes / 1024 / 1024,
                processing_status=enriched_metadata.processing_status.value
            )
            
            return enriched_metadata
        
        except Exception as e:
            logger.error("Failed to upload audio file", filename=filename, error=str(e))
            raise
    
    async def get_audio_metadata(self, audio_id: str) -> Optional[AudioMetadata]:
        """
        Get metadata for a specific audio file.
        
        Enriches JSON catalog data with database fields (processing_status, total_chunks).
        
        Args:
            audio_id: Audio file ID
        
        Returns:
            AudioMetadata or None if not found
        """
        try:
            # Load base metadata from JSON catalog
            catalog = self._load_metadata_catalog()
            metadata = catalog.get(audio_id)
            
            if metadata is None:
                return None
            
            # Enrich with database fields (processing_status, total_chunks)
            from ..db import get_pool
            
            pool = get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT processing_status, total_chunks 
                    FROM heimdall.audio_library 
                    WHERE id = $1
                    """,
                    uuid.UUID(audio_id)
                )
                
                if row:
                    from ..models.audio_library import ProcessingStatus
                    metadata.processing_status = ProcessingStatus(row['processing_status'])
                    metadata.total_chunks = row['total_chunks']
            
            return metadata
        
        except Exception as e:
            logger.error("Failed to get audio metadata", audio_id=audio_id, error=str(e))
            raise
    
    async def list_audio_files(
        self,
        category: Optional[AudioCategory] = None,
        enabled: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> List[AudioMetadata]:
        """
        List audio files with optional filtering.
        
        Enriches JSON catalog data with database fields (processing_status, total_chunks).
        
        Args:
            category: Filter by category (None = all)
            enabled: Filter by enabled status (None = all, True = enabled only, False = disabled only)
            tags: Filter by tags (AND logic, None = no filtering)
        
        Returns:
            List of AudioMetadata matching filters
        """
        try:
            catalog = self._load_metadata_catalog()
            results = []
            
            for metadata in catalog.values():
                # Filter by enabled status
                if enabled is not None and metadata.enabled != enabled:
                    continue
                
                # Filter by category
                if category and metadata.category != category:
                    continue
                
                # Filter by tags (AND logic)
                if tags:
                    if not all(tag in metadata.tags for tag in tags):
                        continue
                
                results.append(metadata)
            
            # Enrich all results with database fields (processing_status, total_chunks)
            if results:
                from ..db import get_pool
                from ..models.audio_library import ProcessingStatus
                
                pool = get_pool()
                async with pool.acquire() as conn:
                    # Batch query all audio IDs
                    audio_ids = [uuid.UUID(m.id) for m in results]
                    rows = await conn.fetch(
                        """
                        SELECT id, processing_status, total_chunks 
                        FROM heimdall.audio_library 
                        WHERE id = ANY($1)
                        """,
                        audio_ids
                    )
                    
                    # Create lookup map for O(1) enrichment
                    db_data = {str(row['id']): row for row in rows}
                    
                    # Enrich metadata objects
                    for metadata in results:
                        if metadata.id in db_data:
                            row = db_data[metadata.id]
                            metadata.processing_status = ProcessingStatus(row['processing_status'])
                            metadata.total_chunks = row['total_chunks']
            
            logger.debug(
                "Listed audio files",
                total=len(results),
                category=category.value if category else "all",
                enabled=enabled
            )
            
            return results
        
        except Exception as e:
            logger.error("Failed to list audio files", error=str(e))
            raise
    
    async def download_audio(self, audio_id: str) -> bytes:
        """
        Download audio file data from MinIO.
        
        Args:
            audio_id: Audio file ID
        
        Returns:
            Audio file bytes
        """
        try:
            metadata = await self.get_audio_metadata(audio_id)
            if not metadata:
                raise ValueError(f"Audio file not found: {audio_id}")
            
            response = self.minio_client.s3_client.get_object(
                Bucket=self.BUCKET_NAME,
                Key=metadata.minio_key
            )
            audio_bytes = response['Body'].read()
            
            logger.debug("Downloaded audio file", audio_id=audio_id, size_mb=len(audio_bytes) / 1024 / 1024)
            return audio_bytes
        
        except Exception as e:
            logger.error("Failed to download audio file", audio_id=audio_id, error=str(e))
            raise
    
    async def update_audio_metadata(
        self,
        audio_id: str,
        filename: Optional[str] = None,
        category: Optional[AudioCategory] = None,
        tags: Optional[List[str]] = None
    ) -> AudioMetadata:
        """
        Update audio file metadata.
        
        Args:
            audio_id: Audio file ID
            filename: New filename (optional)
            category: New category (optional)
            tags: New tags (replaces existing, optional)
        
        Returns:
            Updated AudioMetadata
        """
        try:
            catalog = self._load_metadata_catalog()
            metadata = catalog.get(audio_id)
            
            if not metadata:
                raise ValueError(f"Audio file not found: {audio_id}")
            
            # Update fields
            if filename:
                # Validate format hasn't changed
                old_ext = Path(metadata.filename).suffix
                new_ext = Path(filename).suffix
                if old_ext.lower() != new_ext.lower():
                    raise ValueError(f"Cannot change file extension: {old_ext} -> {new_ext}")
                metadata.filename = filename
            
            if category:
                metadata.category = category
            
            if tags is not None:
                metadata.tags = tags
            
            metadata.updated_at = datetime.utcnow()
            
            # Save updated catalog
            catalog[audio_id] = metadata
            self._save_metadata_catalog(catalog)
            
            logger.info("Updated audio metadata", audio_id=audio_id)
            return metadata
        
        except Exception as e:
            logger.error("Failed to update audio metadata", audio_id=audio_id, error=str(e))
            raise
    
    async def set_audio_enabled(self, audio_id: str, enabled: bool) -> AudioMetadata:
        """
        Enable or disable audio file (soft delete).
        
        Args:
            audio_id: Audio file ID
            enabled: Enable (True) or disable (False)
        
        Returns:
            Updated AudioMetadata
        """
        try:
            catalog = self._load_metadata_catalog()
            metadata = catalog.get(audio_id)
            
            if not metadata:
                raise ValueError(f"Audio file not found: {audio_id}")
            
            metadata.enabled = enabled
            metadata.updated_at = datetime.utcnow()
            
            catalog[audio_id] = metadata
            self._save_metadata_catalog(catalog)
            
            logger.info(
                "Updated audio enabled status",
                audio_id=audio_id,
                enabled=enabled
            )
            
            return metadata
        
        except Exception as e:
            logger.error("Failed to set audio enabled", audio_id=audio_id, error=str(e))
            raise
    
    async def delete_audio(self, audio_id: str):
        """
        Hard delete audio file from storage and metadata catalog.
        
        Cleanup operations:
        1. Delete all chunks from MinIO (heimdall-audio-chunks bucket)
        2. Delete database record from audio_library (CASCADE deletes audio_chunks)
        3. Delete original file from MinIO (heimdall-audio-library bucket)
        4. Remove from JSON catalog (legacy)
        
        Args:
            audio_id: Audio file ID
        """
        try:
            catalog = self._load_metadata_catalog()
            metadata = catalog.get(audio_id)
            
            if not metadata:
                raise ValueError(f"Audio file not found: {audio_id}")
            
            # 1. Delete all chunks from MinIO (heimdall-audio-chunks bucket)
            await self._delete_audio_chunks_from_minio(audio_id)
            
            # 2. Delete from database (CASCADE handles audio_chunks table)
            await self._delete_database_record(audio_id)
            
            # 3. Delete original file from MinIO
            self.minio_client.delete_object(metadata.minio_key)
            
            # 4. Remove from catalog (legacy)
            del catalog[audio_id]
            self._save_metadata_catalog(catalog)
            
            logger.info(
                "Deleted audio file completely",
                audio_id=audio_id,
                filename=metadata.filename
            )
        
        except Exception as e:
            logger.error("Failed to delete audio file", audio_id=audio_id, error=str(e))
            raise
    
    async def _delete_audio_chunks_from_minio(self, audio_id: str):
        """
        Delete all chunks for an audio file from MinIO.
        
        Args:
            audio_id: Audio file ID (used as prefix in chunks bucket)
        """
        try:
            # Initialize MinIO client for chunks bucket
            chunks_minio_client = MinIOClient(
                endpoint_url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name="heimdall-audio-chunks"
            )
            
            # List all objects with prefix {audio_id}/
            prefix = f"{audio_id}/"
            
            try:
                response = chunks_minio_client.s3_client.list_objects_v2(
                    Bucket="heimdall-audio-chunks",
                    Prefix=prefix
                )
                
                if 'Contents' not in response:
                    logger.info(
                        "No chunks found in MinIO for audio",
                        audio_id=audio_id
                    )
                    return
                
                # Collect all object keys
                chunk_keys = [obj['Key'] for obj in response['Contents']]
                
                if not chunk_keys:
                    logger.info(
                        "No chunks found in MinIO for audio",
                        audio_id=audio_id
                    )
                    return
                
                # Delete objects in batches (S3 allows up to 1000 per batch)
                for i in range(0, len(chunk_keys), 1000):
                    batch = chunk_keys[i:i + 1000]
                    delete_objects = [{'Key': key} for key in batch]
                    
                    chunks_minio_client.s3_client.delete_objects(
                        Bucket="heimdall-audio-chunks",
                        Delete={'Objects': delete_objects}
                    )
                    
                    logger.debug(
                        "Deleted chunk batch from MinIO",
                        audio_id=audio_id,
                        batch_size=len(batch)
                    )
                
                logger.info(
                    "Deleted all chunks from MinIO",
                    audio_id=audio_id,
                    total_chunks=len(chunk_keys)
                )
            
            except Exception as e:
                # If bucket doesn't exist or listing fails, log but don't fail
                # (chunks might not have been created yet)
                if "NoSuchBucket" in str(e):
                    logger.info(
                        "Chunks bucket doesn't exist, skipping chunk deletion",
                        audio_id=audio_id
                    )
                else:
                    logger.warning(
                        "Failed to list chunks in MinIO, continuing with deletion",
                        audio_id=audio_id,
                        error=str(e)
                    )
        
        except Exception as e:
            logger.error(
                "Error deleting chunks from MinIO",
                audio_id=audio_id,
                error=str(e)
            )
            # Don't raise - continue with rest of deletion
    
    async def _delete_database_record(self, audio_id: str):
        """
        Delete audio_library record from database.
        
        CASCADE DELETE automatically removes audio_chunks records.
        
        Args:
            audio_id: Audio file ID
        """
        from ..db import get_pool
        
        try:
            pool = get_pool()
            
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM heimdall.audio_library
                    WHERE id = $1
                    """,
                    uuid.UUID(audio_id)
                )
                
                logger.info(
                    "Deleted database record for audio",
                    audio_id=audio_id,
                    result=result
                )
        
        except Exception as e:
            logger.error(
                "Failed to delete database record",
                audio_id=audio_id,
                error=str(e)
            )
            raise
    
    async def get_random_audio_sample(
        self,
        category: Optional[AudioCategory] = None,
        seed: Optional[int] = None
    ) -> AudioMetadata:
        """
        Get random enabled audio file for training.
        
        Uses category weights for weighted random selection when no specific
        category is requested.
        
        Args:
            category: Filter by category (None = weighted random across all)
            seed: Random seed for reproducibility
        
        Returns:
            Random AudioMetadata
        
        Raises:
            AudioLibraryEmptyError: If no enabled files available
        """
        try:
            import random
            
            if seed is not None:
                random.seed(seed)
            
            # If specific category requested, use simple random selection
            if category:
                files = await self.list_audio_files(category=category, enabled=True)
                
                if not files:
                    raise AudioLibraryEmptyError(category=category)
                
                selected = random.choice(files)
            else:
                # Use weighted random selection across categories
                weights = await self.get_category_weights()
                weights_dict = weights.to_dict()
                
                # Get files for each category with non-zero weight
                category_files = {}
                available_categories = []
                available_weights = []
                
                for cat, weight in weights_dict.items():
                    if weight > 0:
                        files = await self.list_audio_files(category=cat, enabled=True)
                        if files:
                            category_files[cat] = files
                            available_categories.append(cat)
                            available_weights.append(weight)
                
                if not category_files:
                    raise AudioLibraryEmptyError()
                
                # Normalize weights for available categories
                total_weight = sum(available_weights)
                normalized_weights = [w / total_weight for w in available_weights]
                
                # Select category using weights
                selected_category = random.choices(
                    available_categories,
                    weights=normalized_weights,
                    k=1
                )[0]
                
                # Select random file from chosen category
                selected = random.choice(category_files[selected_category])
            
            logger.debug(
                "Selected random audio sample",
                audio_id=selected.id,
                category=selected.category.value,
                filename=selected.filename
            )
            
            return selected
        
        except AudioLibraryEmptyError:
            raise
        except Exception as e:
            logger.error("Failed to get random audio sample", error=str(e))
            raise
    
    async def get_library_stats(self) -> AudioLibraryStats:
        """
        Get statistics about audio library.
        
        Returns:
            AudioLibraryStats with counts and totals
        """
        try:
            catalog = self._load_metadata_catalog()
            files = list(catalog.values())
            
            total_files = len(files)
            enabled_files = sum(1 for f in files if f.enabled)
            disabled_files = total_files - enabled_files
            
            total_duration_seconds = sum(f.duration_seconds for f in files)
            total_size_bytes = sum(f.file_size_bytes for f in files)
            
            # Count by category
            files_by_category = {}
            for category in AudioCategory:
                count = sum(1 for f in files if f.category == category)
                if count > 0:
                    files_by_category[category] = count
            
            # Count by format
            files_by_format = {}
            for fmt in AudioFormat:
                count = sum(1 for f in files if f.format == fmt)
                if count > 0:
                    files_by_format[fmt] = count
            
            stats = AudioLibraryStats(
                total_files=total_files,
                enabled_files=enabled_files,
                disabled_files=disabled_files,
                total_duration_seconds=total_duration_seconds,
                total_size_bytes=total_size_bytes,
                files_by_category=files_by_category,
                files_by_format=files_by_format
            )
            
            logger.debug(
                "Calculated library stats",
                total_files=total_files,
                enabled_files=enabled_files,
                total_duration_min=total_duration_seconds / 60,
                total_size_mb=total_size_bytes / 1024 / 1024
            )
            
            return stats
        
        except Exception as e:
            logger.error("Failed to get library stats", error=str(e))
            raise
    
    async def get_category_weights(self) -> CategoryWeights:
        """
        Get category weights from Redis.
        
        Returns:
            CategoryWeights with current values (defaults if not set)
        """
        try:
            weights_json = self.redis_client.get(self.REDIS_WEIGHTS_KEY)
            
            if weights_json:
                weights_dict = json.loads(weights_json)
                weights = CategoryWeights(**weights_dict)
                logger.debug("Loaded category weights from Redis", weights=weights_dict)
            else:
                # Return defaults if not set
                weights = CategoryWeights()
                logger.debug("Using default category weights")
            
            return weights
        
        except Exception as e:
            logger.error("Failed to get category weights, using defaults", error=str(e))
            # Return defaults on error
            return CategoryWeights()
    
    async def set_category_weights(self, weights: CategoryWeights):
        """
        Save category weights to Redis.
        
        Args:
            weights: CategoryWeights to persist
        """
        try:
            weights_json = json.dumps(weights.model_dump())
            self.redis_client.set(self.REDIS_WEIGHTS_KEY, weights_json)
            
            logger.info(
                "Saved category weights to Redis",
                voice=weights.voice,
                music=weights.music,
                documentary=weights.documentary,
                conference=weights.conference,
                custom=weights.custom
            )
        
        except Exception as e:
            logger.error("Failed to save category weights", error=str(e))
            raise
    
    async def _create_database_record(self, metadata: AudioMetadata, tags: List[str]):
        """
        Create database record in audio_library table for preprocessing pipeline.
        
        Args:
            metadata: AudioMetadata object with file information
            tags: List of user-defined tags
        """
        from ..db import get_pool
        
        try:
            pool = get_pool()
            
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO heimdall.audio_library (
                        id, filename, category, tags, file_size_bytes,
                        duration_seconds, sample_rate, channels, audio_format,
                        minio_bucket, minio_path, processing_status,
                        enabled, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), NOW())
                    """,
                    uuid.UUID(metadata.id),
                    metadata.filename,
                    metadata.category.value,
                    tags,
                    metadata.file_size_bytes,
                    metadata.duration_seconds,
                    metadata.sample_rate,
                    metadata.channels,
                    metadata.format.value,
                    metadata.minio_bucket,
                    metadata.minio_key,  # Use minio_key as minio_path
                    "PENDING",  # Initial status
                    metadata.enabled
                )
            
            logger.info(
                "Created database record for audio preprocessing",
                audio_id=metadata.id,
                filename=metadata.filename
            )
        
        except Exception as e:
            logger.error(
                "Failed to create database record for audio",
                audio_id=metadata.id,
                error=str(e)
            )
            raise


def get_audio_storage() -> AudioLibraryStorage:
    """
    Dependency injection for AudioLibraryStorage.
    
    Returns:
        AudioLibraryStorage instance
    """
    from .minio_client import MinIOClient
    
    minio_client = MinIOClient(
        endpoint_url=settings.minio_url,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name="heimdall-audio-library"  # Use dedicated bucket for audio
    )
    return AudioLibraryStorage(minio_client)
