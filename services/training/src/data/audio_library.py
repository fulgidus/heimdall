"""
Audio Library Loader for Training Service.

Provides cached access to audio samples from the MinIO audio library
for use in synthetic IQ generation. Implements LRU caching to minimize
network overhead and improve training performance.
"""

import io
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import requests
import structlog
import soundfile as sf
from scipy import signal as scipy_signal

logger = structlog.get_logger(__name__)


class AudioLibraryEmptyError(Exception):
    """Raised when audio library has no enabled samples."""

    pass


class AudioLibraryLoader:
    """
    Loads audio samples from the audio library API with LRU caching.

    Features:
    - LRU cache for downloaded audio files (reduces network calls)
    - Category-based filtering
    - Random sampling of enabled files
    - Graceful error handling with informative logging
    - Automatic resampling to target sample rate
    """

    def __init__(
        self,
        backend_url: Optional[str] = None,
        cache_size: int = 100,
        target_sample_rate: int = 48000,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize audio library loader.

        Args:
            backend_url: Backend API base URL (defaults to env var BACKEND_URL)
            cache_size: Maximum number of audio files to cache in memory
            target_sample_rate: Target sample rate for resampling (Hz)
            rng: NumPy random generator for reproducible sampling (if None, uses default)
        """
        self.backend_url = backend_url if backend_url is not None else os.getenv(
            "BACKEND_URL", "http://backend:8001"
        )
        self.cache_size = cache_size
        self.target_sample_rate = target_sample_rate
        self.rng = rng if rng is not None else np.random.default_rng()

        # Manual cache instead of lru_cache (incompatible with librosa in Celery)
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._cache_order: List[str] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        logger.info(
            "audio_library_loader_initialized",
            backend_url=self.backend_url,
            cache_size=cache_size,
            target_sample_rate=target_sample_rate,
            rng_seeded=rng is not None,
        )

    def get_random_sample(
        self,
        category: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Get a random audio sample from the library.

        Args:
            category: Filter by category (voice, music, documentary, conference, custom)
            duration_ms: Desired duration in milliseconds (will trim or loop if needed)

        Returns:
            Tuple of (audio_samples, sample_rate)
            - audio_samples: 1D numpy array of float32 samples [-1.0, 1.0]
            - sample_rate: Sample rate in Hz

        Raises:
            AudioLibraryEmptyError: If no enabled samples available
            requests.RequestException: If API call fails
        """
        # Get ALL samples from API first, then select one using our RNG
        params = {}
        params["enabled"] = "true"  # Only get enabled samples (string for query param)
        if category:
            params["category"] = category

        # Get list of all enabled samples
        url = f"{self.backend_url}/api/v1/audio-library/list"
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            list_response = response.json()
            enabled_samples = list_response.get("files", [])
        except requests.RequestException as e:
            logger.error(
                "audio_library_api_error",
                url=url,
                error=str(e),
                exc_info=True,
            )
            raise
        
        # Check if library is empty
        if not enabled_samples:
            logger.warning(
                "audio_library_empty",
                category=category,
            )
            raise AudioLibraryEmptyError(
                f"No enabled audio samples available (category={category})"
            )

        # Use our seeded RNG to select a sample (reproducible)
        sample_idx = self.rng.integers(0, len(enabled_samples))
        sample_info = enabled_samples[sample_idx]

        # Load audio from cache or download
        audio_id = sample_info["id"]
        audio_samples, sample_rate = self._load_audio_cached(audio_id)

        # Adjust duration if requested (pass RNG for reproducible trimming)
        if duration_ms is not None:
            target_samples = int(sample_rate * duration_ms / 1000)
            audio_samples = self._adjust_duration(audio_samples, target_samples, use_rng=True)

        # Normalize audio RMS to match formant synthesis power
        # This ensures audio library samples have similar signal strength to synthetic audio,
        # preventing noise from dominating correlation metrics during testing/training
        current_rms = np.sqrt(np.mean(audio_samples ** 2))
        if current_rms > 1e-6:  # Avoid division by zero
            # Use aggressive normalization to match formant synthesis
            # Target RMS=1.5 ensures >0.95 correlation at SNR=50 dB
            target_rms = 1.5
            audio_samples = audio_samples * (target_rms / current_rms)
            logger.debug(
                "audio_rms_normalized",
                audio_id=audio_id,
                original_rms=float(current_rms),
                target_rms=target_rms,
                scaling_factor=float(target_rms / current_rms),
            )

        logger.debug(
            "audio_sample_loaded",
            audio_id=audio_id,
            category=sample_info.get("category"),
            duration_s=len(audio_samples) / sample_rate,
            sample_rate=sample_rate,
        )

        return audio_samples, sample_rate

    def _load_audio(self, audio_id: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file from API (cache-miss function).

        Args:
            audio_id: Audio file UUID

        Returns:
            Tuple of (audio_samples, sample_rate)

        Raises:
            requests.RequestException: If download fails
        """
        url = f"{self.backend_url}/api/v1/audio-library/{audio_id}/download"

        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()

            # Load audio with soundfile (bypasses librosa/numba caching issues)
            audio_bytes = io.BytesIO(response.content)
            
            # Load audio with soundfile (handles MP3/WAV/FLAC/OGG)
            # Returns float64 [-1.0, 1.0] and original sample rate
            samples, original_sr = sf.read(audio_bytes, dtype='float32')
            
            # Convert stereo to mono if needed
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            
            logger.debug(
                "audio_loaded_original",
                audio_id=audio_id,
                original_sample_rate=original_sr,
                original_duration_s=len(samples) / original_sr,
                original_samples=len(samples)
            )

            # Resample to target sample rate using scipy (high-quality polyphase filtering)
            # This avoids all librosa/numba caching issues while maintaining quality
            if original_sr != self.target_sample_rate:
                logger.debug(
                    "audio_resampling",
                    audio_id=audio_id,
                    from_sr=original_sr,
                    to_sr=self.target_sample_rate,
                    method="scipy_resample_poly"
                )
                # Use resample_poly which is much faster than resample for large files
                # Calculate greatest common divisor for optimal performance
                from math import gcd
                up = self.target_sample_rate // gcd(self.target_sample_rate, original_sr)
                down = original_sr // gcd(self.target_sample_rate, original_sr)
                
                # If ratio is too large, process in chunks to avoid memory issues
                if up * down > 1000:
                    logger.warning(
                        "large_resampling_ratio",
                        audio_id=audio_id,
                        up=up,
                        down=down,
                        ratio=up/down
                    )
                
                samples = scipy_signal.resample_poly(samples, up, down)

            logger.info(
                "audio_file_loaded",
                audio_id=audio_id,
                duration_s=len(samples) / self.target_sample_rate,
                sample_rate=self.target_sample_rate,
                original_sample_rate=original_sr,
                resampled=original_sr != self.target_sample_rate
            )

            return samples.astype(np.float32), self.target_sample_rate

        except requests.RequestException as e:
            logger.error(
                "audio_download_error",
                audio_id=audio_id,
                url=url,
                error=str(e),
                exc_info=True,
            )
            raise

    def _load_audio_cached(self, audio_id: str) -> Tuple[np.ndarray, int]:
        """
        Load audio with manual LRU cache.
        
        Args:
            audio_id: Audio file UUID
            
        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        # Check cache
        if audio_id in self._audio_cache:
            self._cache_hits += 1
            logger.debug(
                "audio_cache_hit",
                audio_id=audio_id,
                cache_size=len(self._audio_cache),
            )
            return self._audio_cache[audio_id]
        
        # Cache miss - load audio
        self._cache_misses += 1
        logger.debug(
            "audio_cache_miss",
            audio_id=audio_id,
            cache_size=len(self._audio_cache),
        )
        
        audio_data = self._load_audio(audio_id)
        
        # Add to cache
        self._audio_cache[audio_id] = audio_data
        self._cache_order.append(audio_id)
        
        # Evict oldest if over limit (LRU)
        if len(self._audio_cache) > self.cache_size:
            oldest_id = self._cache_order.pop(0)
            del self._audio_cache[oldest_id]
            logger.debug(
                "audio_cache_eviction",
                evicted_id=oldest_id,
                cache_size=len(self._audio_cache),
            )
        
        return audio_data

    def _adjust_duration(
        self, audio_samples: np.ndarray, target_samples: int, use_rng: bool = False
    ) -> np.ndarray:
        """
        Adjust audio duration by trimming or looping.

        Args:
            audio_samples: Input audio samples
            target_samples: Desired number of samples
            use_rng: If True, use the instance's RNG for reproducible offset selection

        Returns:
            Audio samples with adjusted length
        """
        current_samples = len(audio_samples)

        if current_samples == target_samples:
            return audio_samples
        elif current_samples > target_samples:
            # Trim: random offset for variety (with optional RNG for reproducibility)
            max_offset = current_samples - target_samples
            if use_rng:
                offset = self.rng.integers(0, max_offset + 1)
            else:
                offset = random.randint(0, max_offset)
            return audio_samples[offset : offset + target_samples]
        else:
            # Loop: repeat audio until target length
            num_repeats = (target_samples // current_samples) + 1
            repeated = np.tile(audio_samples, num_repeats)
            return repeated[:target_samples]

    def get_cache_info(self) -> Dict:
        """
        Get LRU cache statistics.

        Returns:
            Dictionary with cache hits, misses, size, and max size
        """
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "current_size": len(self._audio_cache),
            "max_size": self.cache_size,
            "hit_rate": (
                self._cache_hits / total
                if total > 0
                else 0.0
            ),
        }

    def clear_cache(self):
        """Clear the LRU cache (useful for testing)."""
        self._audio_cache.clear()
        self._cache_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("audio_library_cache_cleared")


# Global instance for easy access (lazy-initialized)
_loader_instance: Optional[AudioLibraryLoader] = None


def get_audio_loader(
    backend_url: Optional[str] = None,
    cache_size: int = 100,
    target_sample_rate: int = 48000,
    rng: Optional[np.random.Generator] = None,
    force_new: bool = False,
) -> AudioLibraryLoader:
    """
    Get the global AudioLibraryLoader instance (singleton pattern).

    Args:
        backend_url: Backend API base URL (only used on first call)
        cache_size: LRU cache size (only used on first call)
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
            cache_size=cache_size,
            target_sample_rate=target_sample_rate,
            rng=rng,
        )
    
    # Otherwise use singleton pattern
    if _loader_instance is None:
        _loader_instance = AudioLibraryLoader(
            backend_url=backend_url,
            cache_size=cache_size,
            target_sample_rate=target_sample_rate,
            rng=rng,
        )
    return _loader_instance
