"""
Audio Library Loader for Training Service.

Loads preprocessed audio chunks from the database and MinIO.
All audio is pre-chunked and pre-resampled to 200kHz during upload preprocessing.
No disk cache or runtime resampling needed - instant loading!
"""

import io
import json
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import redis
import structlog
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import MinIO client from backend
import sys
sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
try:
    from storage.minio_client import MinIOClient
except ImportError:
    MinIOClient = None

from ..config import settings

logger = structlog.get_logger(__name__)


class AudioCategory(str, Enum):
    """Audio sample categories for organization."""
    
    VOICE = "voice"
    MUSIC = "music"
    DOCUMENTARY = "documentary"
    CONFERENCE = "conference"
    CUSTOM = "custom"


class CategoryWeights(BaseModel):
    """
    Category weights for proportional sampling during training.
    
    Each weight is a float between 0.0 and 1.0 representing the probability
    of selecting that category when generating training samples.
    """
    
    voice: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for voice samples")
    music: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for music samples")
    documentary: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for documentary samples")
    conference: float = Field(default=0.1, ge=0.0, le=1.0, description="Weight for conference samples")
    custom: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight for custom samples")
    
    @field_validator('voice', 'music', 'documentary', 'conference', 'custom')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure weight is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v
    
    def normalize(self) -> "CategoryWeights":
        """
        Normalize weights to sum to 1.0 (probability distribution).
        
        Returns a new CategoryWeights instance with normalized values.
        If all weights are 0, returns uniform distribution.
        """
        total = self.voice + self.music + self.documentary + self.conference + self.custom
        
        if total == 0.0:
            # Uniform distribution
            return CategoryWeights(
                voice=0.2,
                music=0.2,
                documentary=0.2,
                conference=0.2,
                custom=0.2
            )
        
        return CategoryWeights(
            voice=self.voice / total,
            music=self.music / total,
            documentary=self.documentary / total,
            conference=self.conference / total,
            custom=self.custom / total
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary mapping category string to weight."""
        return {
            "voice": self.voice,
            "music": self.music,
            "documentary": self.documentary,
            "conference": self.conference,
            "custom": self.custom
        }
    
    def to_lists(self) -> Tuple[List[str], List[float]]:
        """
        Convert to parallel lists of categories and weights.
        Only includes categories with weight > 0.
        
        Returns:
            Tuple of (category_names, weights)
        """
        categories = []
        weights = []
        
        for cat, weight in self.to_dict().items():
            if weight > 0:
                categories.append(cat)
                weights.append(weight)
        
        return categories, weights


class AudioLibraryEmptyError(Exception):
    """Raised when audio library has no enabled samples."""
    pass


class AudioLibraryLoader:
    """
    Loads preprocessed audio chunks from database and MinIO.
    
    Architecture:
    1. Load category weights from Redis (configured by user in frontend)
    2. Select category using weighted random selection
    3. Query database for random chunk from selected category (enabled=TRUE, status=READY)
    4. Download .npy chunk from MinIO (heimdall-audio-chunks bucket)
    5. Load and return (already at 200kHz, no resampling needed)
    
    Benefits:
    - Instant loading (no resampling, no disk cache needed)
    - All chunks are 1 second at 200kHz (consistent)
    - Respects user-configured category weights for realistic training data distribution
    - Random chunk selection per call (great for training diversity)
    """
    
    REDIS_WEIGHTS_KEY = "audio:category:weights"

    def __init__(
        self,
        backend_url: Optional[str] = None,
        target_sample_rate: int = 200000,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize audio library loader.

        Args:
            backend_url: Backend API base URL (defaults to env var BACKEND_URL)
            target_sample_rate: Expected sample rate (should be 200000 Hz for preprocessed chunks)
            rng: NumPy random generator for reproducible sampling (if None, uses default)
        """
        self.backend_url = backend_url if backend_url is not None else os.getenv(
            "BACKEND_URL", "http://backend:8001"
        )
        self.target_sample_rate = target_sample_rate
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Database connection
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Redis connection for category weights
        self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        
        # MinIO client for chunk downloads
        if MinIOClient is None:
            logger.warning(
                "minio_client_unavailable",
                note="MinIOClient not available - audio loading will fail"
            )
            self.minio_client = None
        else:
            self.minio_client = MinIOClient(
                endpoint_url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name="heimdall-audio-chunks"
            )

        # Statistics tracking
        self._chunks_loaded: int = 0
        self._category_stats: Dict[str, int] = {}  # Track chunks loaded per category

        logger.info(
            "audio_library_loader_initialized",
            backend_url=self.backend_url,
            target_sample_rate=target_sample_rate,
            rng_seeded=rng is not None,
            database_url_host=settings.postgres_host,
            redis_url=settings.redis_url,
        )

    def _get_category_weights(self) -> CategoryWeights:
        """
        Load category weights from Redis.
        
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
            
            return weights.normalize()  # Ensure normalized
        
        except Exception as e:
            logger.error("Failed to get category weights, using defaults", error=str(e))
            # Return defaults on error
            return CategoryWeights().normalize()
    
    def _select_category_weighted(self) -> str:
        """
        Select a category using weighted random selection based on user preferences.
        
        Returns:
            Selected category name (e.g., "voice", "music")
            
        Raises:
            AudioLibraryEmptyError: If no categories have available chunks
        """
        weights = self._get_category_weights()
        categories, category_weights = weights.to_lists()
        
        if not categories:
            # All weights are 0, use uniform distribution
            categories = ["voice", "music", "documentary", "conference", "custom"]
            category_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Verify that selected categories have available chunks
        db = self.SessionLocal()
        try:
            available_categories = []
            available_weights = []
            
            for cat, weight in zip(categories, category_weights):
                # Check if this category has any READY chunks
                query = """
                    SELECT COUNT(*) as count
                    FROM heimdall.audio_chunks ac
                    JOIN heimdall.audio_library al ON ac.audio_id = al.id
                    WHERE al.enabled = TRUE 
                      AND al.processing_status = 'READY'
                      AND al.category = :category
                """
                result = db.execute(text(query), {"category": cat}).fetchone()
                
                if result and result[0] > 0:
                    available_categories.append(cat)
                    available_weights.append(weight)
            
            if not available_categories:
                raise AudioLibraryEmptyError("No READY audio chunks available in any category")
            
            # Normalize weights for available categories only
            total_weight = sum(available_weights)
            normalized_weights = [w / total_weight for w in available_weights]
            
            # Use NumPy RNG for weighted selection
            selected_category = self.rng.choice(available_categories, p=normalized_weights)
            
            logger.debug(
                "category_selected_weighted",
                selected=selected_category,
                available_categories=available_categories,
                weights=dict(zip(available_categories, normalized_weights))
            )
            
            return selected_category
        
        finally:
            db.close()

    def get_random_sample(
        self,
        category: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Get a random preprocessed audio chunk from the library.
        
        Uses weighted random selection based on category weights configured by user.
        If category is explicitly specified, weights are ignored.

        Args:
            category: Filter by specific category (overrides weighted selection)
            duration_ms: Ignored (all chunks are 1 second at 200kHz)

        Returns:
            Tuple of (audio_samples, sample_rate)
            - audio_samples: 1D numpy array of float32 samples (200,000 samples = 1 second at 200kHz)
            - sample_rate: Always 200000 Hz

        Raises:
            AudioLibraryEmptyError: If no enabled samples with READY status available
        """
        if duration_ms is not None:
            logger.warning(
                "duration_ms_ignored",
                duration_ms=duration_ms,
                note="All preprocessed chunks are 1 second at 200kHz"
            )
        
        # Select category using weighted random selection (unless explicitly specified)
        if category is None:
            category = self._select_category_weighted()
        
        # Query database for random chunk from selected category
        db = self.SessionLocal()
        try:
            # Build SQL query with category filter
            query = """
                SELECT ac.id, ac.minio_bucket, ac.minio_path, ac.sample_rate, ac.num_samples,
                       al.category, al.filename
                FROM heimdall.audio_chunks ac
                JOIN heimdall.audio_library al ON ac.audio_id = al.id
                WHERE al.enabled = TRUE 
                  AND al.processing_status = 'READY'
                  AND al.category = :category
                ORDER BY RANDOM() 
                LIMIT 1
            """
            
            result = db.execute(text(query), {"category": category}).fetchone()
            
            if not result:
                raise AudioLibraryEmptyError(
                    f"No READY audio chunks available (category={category})"
                )
            
            chunk_id, minio_bucket, minio_path, sample_rate, num_samples, chunk_category, filename = result
            
            logger.debug(
                "chunk_selected_from_db",
                chunk_id=str(chunk_id),
                category=chunk_category,
                filename=filename,
                minio_path=minio_path,
                sample_rate=sample_rate,
                num_samples=num_samples
            )
        
        finally:
            db.close()
        
        # Download chunk from MinIO
        if self.minio_client is None:
            raise RuntimeError("MinIO client not available")
        
        try:
            chunk_bytes = self.minio_client.s3_client.get_object(
                Bucket=minio_bucket,
                Key=minio_path
            )['Body'].read()
            
            # Load numpy array from bytes
            audio_samples = np.load(io.BytesIO(chunk_bytes))
            
            logger.debug(
                "chunk_loaded_from_minio",
                chunk_id=str(chunk_id),
                minio_path=minio_path,
                size_bytes=len(chunk_bytes),
                num_samples=len(audio_samples),
                sample_rate=sample_rate
            )
        
        except Exception as e:
            logger.error(
                "chunk_download_failed",
                chunk_id=str(chunk_id),
                minio_path=minio_path,
                error=str(e),
                exc_info=True
            )
            raise

        # Normalize audio RMS to match formant synthesis power
        # This ensures audio library samples have similar signal strength to synthetic audio
        current_rms = np.sqrt(np.mean(audio_samples ** 2))
        if current_rms > 1e-6:  # Avoid division by zero
            target_rms = 1.5
            audio_samples = audio_samples * (target_rms / current_rms)
            logger.debug(
                "audio_rms_normalized",
                chunk_id=str(chunk_id),
                original_rms=float(current_rms),
                target_rms=target_rms,
                scaling_factor=float(target_rms / current_rms),
            )

        self._chunks_loaded += 1
        
        # Update category statistics
        if chunk_category not in self._category_stats:
            self._category_stats[chunk_category] = 0
        self._category_stats[chunk_category] += 1
        
        logger.debug(
            "audio_chunk_loaded",
            chunk_id=str(chunk_id),
            category=chunk_category,
            duration_s=1.0,
            sample_rate=sample_rate,
            chunks_loaded_total=self._chunks_loaded,
            category_stats=self._category_stats
        )

        return audio_samples, sample_rate

    def get_stats(self) -> Dict:
        """
        Get loader statistics.

        Returns:
            Dictionary with statistics (chunks loaded, category distribution, etc.)
        """
        # Calculate category distribution percentages
        category_distribution = {}
        if self._chunks_loaded > 0:
            for cat, count in self._category_stats.items():
                category_distribution[cat] = {
                    "count": count,
                    "percentage": round((count / self._chunks_loaded) * 100, 2)
                }
        
        return {
            "chunks_loaded": self._chunks_loaded,
            "target_sample_rate": self.target_sample_rate,
            "category_distribution": category_distribution,
        }

    def clear_stats(self):
        """Clear statistics (useful for testing)."""
        self._chunks_loaded = 0
        self._category_stats = {}
        logger.info("audio_library_stats_cleared")


# Global instance for easy access (lazy-initialized)
_loader_instance: Optional[AudioLibraryLoader] = None


def get_audio_loader(
    backend_url: Optional[str] = None,
    target_sample_rate: int = 200000,
    rng: Optional[np.random.Generator] = None,
    force_new: bool = False,
) -> AudioLibraryLoader:
    """
    Get the global AudioLibraryLoader instance (singleton pattern).

    Args:
        backend_url: Backend API base URL (only used on first call)
        target_sample_rate: Target sample rate in Hz (only used on first call)
        rng: NumPy random generator for reproducible sampling (only used on first call)
        force_new: If True, create a new instance instead of using singleton

    Returns:
        AudioLibraryLoader instance
    """
    global _loader_instance
    
    # If force_new requested, create a new instance (useful for testing)
    if force_new:
        return AudioLibraryLoader(
            backend_url=backend_url,
            target_sample_rate=target_sample_rate,
            rng=rng,
        )
    
    # Otherwise use singleton pattern
    if _loader_instance is None:
        _loader_instance = AudioLibraryLoader(
            backend_url=backend_url,
            target_sample_rate=target_sample_rate,
            rng=rng,
        )
    return _loader_instance
