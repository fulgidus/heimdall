"""
RF Feature Extraction Module

Extracts features from IQ samples for ML training.
Processes both synthetic and real recording session data.

GPU Acceleration:
- Automatically uses CuPy if available (10-30x speedup for FFT operations)
- Falls back to NumPy if CuPy not installed or no GPU

CPU Parallelization:
- Uses multiprocessing.Pool to parallelize feature extraction across all CPU cores
- Achieves near-linear speedup on multi-core systems (e.g., 24 cores)
"""

import numpy as np
import structlog
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union
from multiprocessing import Pool, cpu_count
from functools import partial
import threading

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
except (ImportError, Exception):
    cp = None
    GPU_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Thread-local storage for worker extractors (initialized once per worker process)
_worker_local = threading.local()


def _init_worker(sample_rate_hz: int):
    """
    Initialize worker process with a reusable RFFeatureExtractor.
    
    Called once per worker process at pool creation.
    Stores the extractor in thread-local storage to avoid repeated initialization.
    
    Args:
        sample_rate_hz: Sample rate in Hz
    """
    _worker_local.extractor = None  # Will be lazily created
    _worker_local.sample_rate_hz = sample_rate_hz


def _extract_features_worker(iq_sample: 'IQSample') -> 'ExtractedFeatures':
    """
    Worker function for multiprocessing feature extraction.
    
    Uses a cached RFFeatureExtractor instance stored in thread-local storage
    to avoid repeated initialization overhead.
    
    Args:
        iq_sample: IQSample to process
        
    Returns:
        ExtractedFeatures object
    """
    # Lazy initialization of extractor (only once per worker)
    if _worker_local.extractor is None:
        _worker_local.extractor = RFFeatureExtractor(
            sample_rate_hz=_worker_local.sample_rate_hz, 
            use_gpu=False
        )
    
    return _worker_local.extractor.extract_features(iq_sample)


@dataclass
class IQSample:
    """Container for IQ sample data."""

    samples: np.ndarray  # Complex64 array of IQ samples
    sample_rate_hz: int  # Sample rate in Hz
    center_frequency_hz: int  # Center frequency
    rx_id: str  # Receiver identifier
    rx_lat: float  # Receiver latitude
    rx_lon: float  # Receiver longitude
    timestamp: datetime  # Capture timestamp


@dataclass
class ExtractedFeatures:
    """Single chunk feature extraction result."""

    # === TIER 1: Essential ===
    rssi_dbm: float  # Received Signal Strength Indicator
    snr_db: float  # Signal-to-Noise Ratio
    noise_floor_dbm: float  # Baseline noise level

    # === TIER 2: Frequency ===
    frequency_offset_hz: float  # Doppler shift
    bandwidth_hz: float  # Occupied bandwidth (-3dB)
    psd_dbm_per_hz: float  # Power Spectral Density (peak)
    spectral_centroid_hz: float  # Spectral center of mass
    spectral_rolloff_hz: float  # 85% energy point

    # === TIER 3: Temporal Statistics ===
    envelope_mean: float  # Mean amplitude (normalized 0-1)
    envelope_std: float  # Fading indicator
    envelope_max: float  # Peak amplitude
    peak_to_avg_ratio_db: float  # Crest factor
    zero_crossing_rate: float  # Instantaneous frequency indicator

    # === TIER 4: Advanced ===
    multipath_delay_spread_us: float  # RMS delay spread (microseconds)
    coherence_bandwidth_khz: float  # ~1/(5*delay_spread)
    delay_spread_confidence: float  # Confidence in delay spread estimate (0-1)

    # === Metadata ===
    signal_present: bool  # Detection flag
    confidence_score: float  # Overall feature quality (0-1)


class RFFeatureExtractor:
    """Extract RF features from IQ samples with GPU acceleration."""

    def __init__(self, sample_rate_hz: int = 200000, use_gpu: bool = True):
        """
        Initialize feature extractor.

        Args:
            sample_rate_hz: Sample rate in Hz (default 200kHz)
            use_gpu: Use GPU acceleration if available (default: True)
        """
        self.sample_rate_hz = sample_rate_hz
        
        # GPU acceleration setup
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.xp = cp  # Use CuPy (GPU arrays)
            logger.info(f"RFFeatureExtractor: GPU acceleration ENABLED (CuPy)")
        else:
            self.xp = np  # Use NumPy (CPU arrays)
            
            # Force NumPy/OpenBLAS to use ALL available CPU cores for maximum performance
            import os
            cpu_count = os.cpu_count() or 1
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
        
        # CPU multiprocessing pool (created on-demand, reused across batches)
        self._cpu_pool = None
    
    def __del__(self):
        """Cleanup resources on deletion."""
        self._cleanup_cpu_pool()
    
    def _cleanup_cpu_pool(self):
        """Close and cleanup the CPU multiprocessing pool if it exists."""
        if self._cpu_pool is not None:
            try:
                self._cpu_pool.close()
                self._cpu_pool.join()
                logger.debug("CPU multiprocessing pool cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up CPU pool: {e}")
            finally:
                self._cpu_pool = None
    
    def _get_cpu_pool(self):
        """
        Get or create the CPU multiprocessing pool.
        
        The pool is created once and reused across all batch operations
        to avoid the overhead of creating/destroying processes.
        
        Returns:
            multiprocessing.Pool instance
        """
        if self._cpu_pool is None:
            num_workers = cpu_count()
            logger.info(f"Creating CPU multiprocessing pool with {num_workers} workers")
            self._cpu_pool = Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(self.sample_rate_hz,)
            )
        return self._cpu_pool

    def extract_features(self, iq_sample: IQSample) -> ExtractedFeatures:
        """
        Extract all features from a single IQ chunk.

        Args:
            iq_sample: IQ sample data (NumPy array, will be transferred to GPU if needed)

        Returns:
            ExtractedFeatures object with all 18 base features
        """
        # Transfer data to GPU if using GPU acceleration
        if self.use_gpu:
            samples = self.xp.asarray(iq_sample.samples)
        else:
            samples = iq_sample.samples
        
        # === TIER 1: Essential Features ===

        # Signal power (mean squared magnitude)
        signal_power = self.xp.mean(self.xp.abs(samples) ** 2)
        rssi_dbm = 10 * self.xp.log10(signal_power + 1e-12) + 30  # Convert to dBm (assuming 50Ω)

        # === TIER 2: Frequency Features (compute FFT early for noise estimation) ===

        # FFT for spectral analysis (GPU-accelerated if using CuPy)
        N = len(samples)
        fft = self.xp.fft.fftshift(self.xp.fft.fft(samples))
        # Power per frequency bin (Parseval-normalized)
        # By Parseval's theorem: sum(|x|^2) = sum(|FFT(x)|^2) / N
        psd = self.xp.abs(fft) ** 2 / N
        freqs = self.xp.fft.fftshift(self.xp.fft.fftfreq(N, 1 / self.sample_rate_hz))

        # Find peak and estimate noise
        peak_idx = self.xp.argmax(psd)
        
        # Noise estimation: use median of lowest 30% of PSD bins
        # This assumes signal occupies <70% of spectrum
        sorted_psd = self.xp.sort(psd)
        n_noise_bins = max(1, int(0.3 * N))
        noise_floor_psd_bin = self.xp.median(sorted_psd[:n_noise_bins])
        
        # By Parseval, sum(psd) = sum(|x|^2)
        # If noise floor is noise_floor_psd_bin per bin, and noise is white (in all bins),
        # then total noise power = noise_floor_psd_bin * N
        # But psd is already divided by N, so we need to scale back
        # Actually: if each of N bins has power noise_floor_psd_bin/N in time domain,
        # then total is noise_floor_psd_bin
        # No wait: psd[i] represents power in frequency bin i
        # For white noise with variance σ², each frequency bin has |FFT|²/N ≈ σ²
        # So noise_floor_psd_bin ≈ noise_variance, and that's what we want!
        noise_power = noise_floor_psd_bin
        
        noise_floor_dbm = 10 * self.xp.log10(noise_power + 1e-12) + 30

        # SNR
        snr_linear = signal_power / (noise_power + 1e-12)
        snr_db = 10 * self.xp.log10(snr_linear)

        # Signal detection threshold
        signal_present = snr_db > 0.0

        # PSD in dBm/Hz
        psd_dbm_per_hz = 10 * self.xp.log10(self.xp.max(psd) / self.sample_rate_hz + 1e-12) + 30

        # Frequency offset (peak location relative to center) - already computed above
        frequency_offset_hz = float(freqs[peak_idx])

        # Bandwidth (-3dB)
        threshold = self.xp.max(psd) / 2  # -3dB point
        above_threshold = psd > threshold
        bandwidth_hz = float(self.xp.sum(above_threshold) * (self.sample_rate_hz / len(psd)))

        # Spectral centroid (center of mass)
        spectral_centroid_hz = float(self.xp.sum(freqs * psd) / (self.xp.sum(psd) + 1e-12))

        # Spectral rolloff (85% energy point)
        cumsum = self.xp.cumsum(psd)
        rolloff_idx = self.xp.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff_hz = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0

        # === TIER 3: Temporal Statistics ===

        # Envelope (magnitude)
        envelope = self.xp.abs(samples)
        envelope_normalized = envelope / (self.xp.max(envelope) + 1e-12)

        envelope_mean = float(self.xp.mean(envelope_normalized))
        envelope_std = float(self.xp.std(envelope_normalized))
        envelope_max = float(self.xp.max(envelope_normalized))

        # Peak-to-average ratio (PAR)
        power_samples = self.xp.abs(samples) ** 2
        peak_power = self.xp.max(power_samples)
        avg_power = self.xp.mean(power_samples)
        peak_to_avg_ratio_db = float(10 * self.xp.log10((peak_power / (avg_power + 1e-12))))

        # Zero crossing rate
        zero_crossings = self.xp.sum(self.xp.diff(self.xp.sign(self.xp.real(samples))) != 0)
        zero_crossing_rate = float(zero_crossings / len(samples))

        # === TIER 4: Advanced Features ===

        # Multipath delay spread via autocorrelation
        # Use FFT-based autocorrelation: O(N log N) instead of O(N²)
        # autocorr(x) = ifft(fft(x) * conj(fft(x)))
        fft_x = self.xp.fft.fft(samples)
        power_spectrum = fft_x * self.xp.conj(fft_x)
        autocorr_full = self.xp.fft.ifft(power_spectrum).real
        autocorr = autocorr_full[:len(autocorr_full) // 2 + 1]  # Keep only positive lags
        pdp = self.xp.abs(autocorr) ** 2
        pdp_norm = pdp / (self.xp.max(pdp) + 1e-12)

        # Find significant delays (above -10dB threshold)
        threshold = 0.1
        significant_delays_idx = self.xp.where(pdp_norm > threshold)[0]

        if len(significant_delays_idx) > 1:
            # Transfer small arrays to CPU for weighted average calculation
            if self.use_gpu:
                delays_idx_cpu = cp.asnumpy(significant_delays_idx)
                powers_cpu = cp.asnumpy(pdp_norm[significant_delays_idx])
            else:
                delays_idx_cpu = significant_delays_idx
                powers_cpu = pdp_norm[significant_delays_idx]
            
            delays_sec = delays_idx_cpu / self.sample_rate_hz

            mean_delay = np.average(delays_sec, weights=powers_cpu)
            rms_delay_spread_sec = np.sqrt(
                np.average((delays_sec - mean_delay) ** 2, weights=powers_cpu)
            )
            multipath_delay_spread_us = float(rms_delay_spread_sec * 1e6)

            # Coherence bandwidth ~ 1 / (5 * delay_spread)
            coherence_bandwidth_khz = float(
                (1 / (5 * rms_delay_spread_sec + 1e-12)) / 1000
            )

            # Confidence based on SNR (transfer snr_db to CPU if needed)
            snr_db_val = float(snr_db) if self.use_gpu else snr_db
            if snr_db_val > 15:
                delay_spread_confidence = 1.0
            elif snr_db_val > 10:
                delay_spread_confidence = 0.5 + (snr_db_val - 10) / 10
            elif snr_db_val > 5:
                delay_spread_confidence = 0.3 + (snr_db_val - 5) / 10
            else:
                delay_spread_confidence = max(0.0, snr_db_val / 15)
        else:
            # No multipath detected
            multipath_delay_spread_us = 0.0
            coherence_bandwidth_khz = 0.0
            delay_spread_confidence = 0.0

        # Overall confidence (SNR-based, simple version)
        confidence_score = float(self.xp.clip(snr_db / 30.0, 0, 1))

        # Convert results back to Python scalars (handle GPU tensors)
        # CuPy scalars need explicit conversion to float/bool
        def to_scalar(val):
            """Convert CuPy/NumPy scalar to Python scalar."""
            if self.use_gpu and hasattr(val, 'get'):
                return float(val.get())  # CuPy scalar
            elif hasattr(val, 'item'):
                return val.item()  # NumPy scalar
            else:
                return float(val)  # Already Python scalar
        
        return ExtractedFeatures(
            rssi_dbm=to_scalar(rssi_dbm),
            snr_db=to_scalar(snr_db),
            noise_floor_dbm=to_scalar(noise_floor_dbm),
            frequency_offset_hz=frequency_offset_hz,
            bandwidth_hz=bandwidth_hz,
            psd_dbm_per_hz=to_scalar(psd_dbm_per_hz),
            spectral_centroid_hz=spectral_centroid_hz,
            spectral_rolloff_hz=spectral_rolloff_hz,
            envelope_mean=envelope_mean,
            envelope_std=envelope_std,
            envelope_max=envelope_max,
            peak_to_avg_ratio_db=to_scalar(peak_to_avg_ratio_db),
            zero_crossing_rate=zero_crossing_rate,
            multipath_delay_spread_us=multipath_delay_spread_us,
            coherence_bandwidth_khz=coherence_bandwidth_khz,
            delay_spread_confidence=float(delay_spread_confidence),
            signal_present=bool(to_scalar(signal_present)),
            confidence_score=to_scalar(confidence_score),
        )

    def extract_features_chunked(
        self, iq_sample: IQSample, chunk_duration_ms: float = 200.0, num_chunks: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract features from multiple chunks and aggregate with mean/std/min/max.

        Args:
            iq_sample: IQ sample (should be 1000ms for 5×200ms chunks)
            chunk_duration_ms: Duration of each chunk in milliseconds
            num_chunks: Number of chunks to process

        Returns:
            Dict mapping feature_name to {mean, std, min, max}
        """
        chunk_samples = int(self.sample_rate_hz * chunk_duration_ms / 1000)

        # Validate IQ sample length
        expected_samples = chunk_samples * num_chunks
        if len(iq_sample.samples) < expected_samples:
            logger.warning(
                f"IQ sample too short: {len(iq_sample.samples)} < {expected_samples}. "
                f"Processing available chunks only."
            )
            num_chunks = len(iq_sample.samples) // chunk_samples

        # Extract features from each chunk
        chunk_features = []
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples

            chunk_iq = IQSample(
                samples=iq_sample.samples[start_idx:end_idx],
                sample_rate_hz=iq_sample.sample_rate_hz,
                center_frequency_hz=iq_sample.center_frequency_hz,
                rx_id=iq_sample.rx_id,
                rx_lat=iq_sample.rx_lat,
                rx_lon=iq_sample.rx_lon,
                timestamp=iq_sample.timestamp,
            )

            features = self.extract_features(chunk_iq)
            chunk_features.append(features)

        # Aggregate with mean, std, min, max
        aggregated = {}

        feature_names = [
            'rssi_dbm', 'snr_db', 'noise_floor_dbm',
            'frequency_offset_hz', 'bandwidth_hz', 'psd_dbm_per_hz',
            'spectral_centroid_hz', 'spectral_rolloff_hz',
            'envelope_mean', 'envelope_std', 'envelope_max',
            'peak_to_avg_ratio_db', 'zero_crossing_rate',
            'multipath_delay_spread_us', 'coherence_bandwidth_khz',
            'delay_spread_confidence', 'confidence_score'
        ]

        for feature_name in feature_names:
            values = [getattr(f, feature_name) for f in chunk_features]

            aggregated[feature_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        # Boolean signal_present: majority vote
        signal_present_votes = [f.signal_present for f in chunk_features]
        aggregated['signal_present'] = {
            'mean': float(np.mean(signal_present_votes)),  # Fraction of chunks with signal
            'std': 0.0,
            'min': float(min(signal_present_votes)),
            'max': float(max(signal_present_votes)),
        }

        return aggregated

    def _calculate_overall_confidence(
        self, snr_values: List[float], detection_rate: float, spectral_clarity: float
    ) -> float:
        """
        Calculate overall confidence score for feature extraction quality.

        Args:
            snr_values: List of SNR values from all receivers
            detection_rate: Fraction of receivers with signal_present=True (0-1)
            spectral_clarity: Spectral peak definition score (0-1)

        Returns:
            Overall confidence (0-1)
        """
        # Normalize SNR to 0-1 (assuming max useful SNR ~30 dB)
        snr_quality = float(np.clip(np.mean(snr_values) / 30.0, 0, 1))

        # Weighted average
        overall = snr_quality * 0.5 + detection_rate * 0.3 + spectral_clarity * 0.2

        return float(np.clip(overall, 0, 1))

    def extract_features_batch_conservative(
        self, 
        iq_samples_list: List[IQSample], 
        chunk_duration_ms: float = 200.0, 
        num_chunks: int = 5
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Extract features from multiple receivers using conservative GPU batching.
        
        Processes ONE chunk at a time across ALL receivers to avoid OOM.
        Memory: 7 receivers × 40,000 samples × 8 bytes = ~2.2 MB per chunk (safe).
        
        Args:
            iq_samples_list: List of IQ samples (one per receiver)
            chunk_duration_ms: Duration of each chunk in milliseconds
            num_chunks: Number of chunks to process
            
        Returns:
            List of aggregated feature dicts (one per receiver)
        """
        import time
        t_start = time.perf_counter()
        
        num_receivers = len(iq_samples_list)
        chunk_samples = int(self.sample_rate_hz * chunk_duration_ms / 1000)
        
        logger.info(f"extract_features_batch_conservative called: {num_receivers} receivers, {num_chunks} chunks, GPU={self.use_gpu}")
        
        # Validate all samples have enough data
        for iq_sample in iq_samples_list:
            expected_samples = chunk_samples * num_chunks
            if len(iq_sample.samples) < expected_samples:
                logger.warning(
                    f"IQ sample {iq_sample.rx_id} too short: "
                    f"{len(iq_sample.samples)} < {expected_samples}. Proceeding anyway."
                )
        
        # Storage for all chunk features per receiver
        # Shape: [num_receivers][num_chunks][feature_dict]
        all_chunk_features = [[] for _ in range(num_receivers)]
        
        # For CPU mode: Get the pool once and reuse it for all chunks (massive speedup!)
        cpu_pool = None
        if not self.use_gpu:
            cpu_pool = self._get_cpu_pool()
            logger.info(f"Reusing CPU pool for all {num_chunks} chunks (eliminates pool creation overhead)")
        
        t_chunk_start = time.perf_counter()
        # Process one chunk at a time across all receivers
        for chunk_idx in range(num_chunks):
            t_iter_start = time.perf_counter()
            start_idx = chunk_idx * chunk_samples
            end_idx = start_idx + chunk_samples
            
            # Extract chunk data for all receivers
            chunk_data_list = []
            for iq_sample in iq_samples_list:
                if end_idx <= len(iq_sample.samples):
                    chunk_data_list.append(iq_sample.samples[start_idx:end_idx])
                else:
                    # Pad with zeros if needed
                    chunk = iq_sample.samples[start_idx:]
                    padded = np.zeros(chunk_samples, dtype=np.complex64)
                    padded[:len(chunk)] = chunk
                    chunk_data_list.append(padded)
            
            # Process this chunk for all receivers (GPU or CPU)
            if self.use_gpu:
                chunk_features_list = self._extract_features_chunk_batch_gpu(
                    chunk_data_list, iq_samples_list
                )
            else:
                chunk_features_list = self._extract_features_chunk_batch_cpu(
                    chunk_data_list, iq_samples_list, cpu_pool
                )
            
            # Store features for each receiver
            for rx_idx, features in enumerate(chunk_features_list):
                all_chunk_features[rx_idx].append(features)
            
            t_iter_end = time.perf_counter()
            logger.info(f"Chunk {chunk_idx+1}/{num_chunks} processed in {(t_iter_end - t_iter_start)*1000:.1f}ms")
        
        t_chunk_end = time.perf_counter()
        logger.info(f"All chunks processed in {(t_chunk_end - t_chunk_start)*1000:.1f}ms")
        
        # Aggregate features for each receiver
        t_agg_start = time.perf_counter()
        aggregated_list = []
        for rx_idx, chunk_features in enumerate(all_chunk_features):
            aggregated = self._aggregate_chunk_features(chunk_features)
            aggregated_list.append(aggregated)
        
        t_agg_end = time.perf_counter()
        t_total = time.perf_counter() - t_start
        logger.info(f"Feature aggregation: {(t_agg_end - t_agg_start)*1000:.1f}ms, Total: {t_total*1000:.1f}ms")
        
        return aggregated_list
    
    def _extract_features_chunk_batch_gpu(
        self, 
        chunk_data_list: List[np.ndarray],
        iq_samples_list: List[IQSample]
    ) -> List[ExtractedFeatures]:
        """
        Extract features from one chunk across all receivers using GPU.
        
        Args:
            chunk_data_list: List of chunk arrays [num_receivers, chunk_samples]
            iq_samples_list: List of IQSample metadata
            
        Returns:
            List of ExtractedFeatures (one per receiver)
        """
        import time
        t_start = time.perf_counter()
        
        # Handle empty input gracefully
        if not chunk_data_list:
            logger.warning("_extract_features_chunk_batch_gpu called with empty chunk_data_list")
            return []
        
        # Handle empty input gracefully
        if not chunk_data_list:
            logger.warning("_extract_features_chunk_batch_gpu called with empty chunk_data_list")
            return []
        
        # Stack all chunks into batch [num_receivers, chunk_samples]
        batch = np.stack(chunk_data_list, axis=0)
        t_stack = time.perf_counter()
        
        # Transfer to GPU
        batch_gpu = cp.asarray(batch)
        t_transfer_to_gpu = time.perf_counter()
        
        # Batch FFT (process all receivers simultaneously)
        fft_batch = cp.fft.fftshift(cp.fft.fft(batch_gpu, axis=1), axes=1)
        psd_batch = cp.abs(fft_batch) ** 2 / batch.shape[1]
        t_fft = time.perf_counter()
        
        # Keep PSD on GPU for feature extraction
        t_transfer_to_cpu = time.perf_counter()
        
        # Extract features on GPU (keep data on GPU)
        t_gpu_features_start = time.perf_counter()
        features_list = self._extract_features_batch_gpu(
            batch_gpu, psd_batch, iq_samples_list
        )
        t_gpu_features = time.perf_counter() - t_gpu_features_start
        
        # Free GPU memory after feature extraction
        del batch_gpu, fft_batch, psd_batch
        if self.use_gpu and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        
        t_total = time.perf_counter() - t_start
        
        logger.info(
            f"GPU batch timing: stack={((t_stack-t_start)*1000):.1f}ms, "
            f"to_gpu={((t_transfer_to_gpu-t_stack)*1000):.1f}ms, "
            f"fft={((t_fft-t_transfer_to_gpu)*1000):.1f}ms, "
            f"gpu_features={t_gpu_features*1000:.1f}ms, "
            f"total={t_total*1000:.1f}ms"
        )
        
        return features_list
    
    def _extract_features_chunk_batch_cpu(
        self, 
        chunk_data_list: List[np.ndarray],
        iq_samples_list: List[IQSample],
        pool=None
    ) -> List[ExtractedFeatures]:
        """
        Extract features from one chunk across all receivers using CPU with multiprocessing.
        
        Uses multiprocessing.Pool to parallelize across all available CPU cores.
        This provides near-linear speedup on multi-core systems (e.g., 24 cores).
        
        Args:
            chunk_data_list: List of chunk arrays
            iq_samples_list: List of IQSample metadata
            pool: Optional pre-existing Pool to reuse (massive speedup if provided!)
            
        Returns:
            List of ExtractedFeatures (one per receiver)
        """
        import time
        t_start = time.perf_counter()
        
        # Create list of IQSample objects for parallel processing
        chunk_iq_list = []
        for chunk_data, iq_sample in zip(chunk_data_list, iq_samples_list):
            chunk_iq = IQSample(
                samples=chunk_data,
                sample_rate_hz=iq_sample.sample_rate_hz,
                center_frequency_hz=iq_sample.center_frequency_hz,
                rx_id=iq_sample.rx_id,
                rx_lat=iq_sample.rx_lat,
                rx_lon=iq_sample.rx_lon,
                timestamp=iq_sample.timestamp,
            )
            chunk_iq_list.append(chunk_iq)
        
        # Extract features in parallel using all CPU cores
        num_workers = cpu_count()
        
        if pool is not None:
            # Reuse provided pool (FAST - no creation overhead!)
            logger.debug(f"CPU parallel processing: {len(chunk_iq_list)} samples using REUSED pool with {num_workers} workers")
            features_list = pool.map(_extract_features_worker, chunk_iq_list)
        else:
            # Create temporary pool (SLOW - for backward compatibility only)
            logger.info(f"CPU parallel processing: {len(chunk_iq_list)} samples using NEW pool with {num_workers} workers (not optimal)")
            with Pool(processes=num_workers, initializer=_init_worker, initargs=(self.sample_rate_hz,)) as temp_pool:
                features_list = temp_pool.map(_extract_features_worker, chunk_iq_list)
        
        t_end = time.perf_counter()
        logger.debug(f"CPU parallel extraction: {(t_end - t_start)*1000:.1f}ms for {len(chunk_iq_list)} samples")
        
        return features_list
    
    def _extract_features_batch_gpu(
        self,
        batch_gpu: 'cp.ndarray',
        psd_batch_gpu: 'cp.ndarray',
        iq_samples_list: List[IQSample]
    ) -> List[ExtractedFeatures]:
        """
        Extract features from batch data entirely on GPU.
        
        Args:
            batch_gpu: Complex IQ samples on GPU [num_receivers, chunk_samples]
            psd_batch_gpu: PSD on GPU [num_receivers, chunk_samples]
            iq_samples_list: Metadata for each receiver
            
        Returns:
            List of ExtractedFeatures (one per receiver)
        """
        import time
        t_start = time.perf_counter()
        
        num_receivers = batch_gpu.shape[0]
        N = batch_gpu.shape[1]
        
        # Compute frequency array on GPU
        freqs_gpu = cp.fft.fftshift(cp.fft.fftfreq(N, 1 / self.sample_rate_hz))
        
        # === TIER 1: Essential Features (vectorized across all receivers) ===
        t1 = time.perf_counter()
        signal_power_batch = cp.mean(cp.abs(batch_gpu) ** 2, axis=1)  # [num_receivers]
        rssi_dbm_batch = 10 * cp.log10(signal_power_batch + 1e-12) + 30
        
        # Noise estimation (vectorized)
        sorted_psd_batch = cp.sort(psd_batch_gpu, axis=1)  # [num_receivers, N]
        n_noise_bins = max(1, int(0.3 * N))
        noise_floor_psd_bin_batch = cp.median(sorted_psd_batch[:, :n_noise_bins], axis=1)  # [num_receivers]
        noise_power_batch = noise_floor_psd_bin_batch
        noise_floor_dbm_batch = 10 * cp.log10(noise_power_batch + 1e-12) + 30
        
        # SNR
        snr_linear_batch = signal_power_batch / (noise_power_batch + 1e-12)
        snr_db_batch = 10 * cp.log10(snr_linear_batch)
        signal_present_batch = snr_db_batch > 0.0
        t2 = time.perf_counter()
        
        # === TIER 2: Frequency Features (vectorized) ===
        peak_idx_batch = cp.argmax(psd_batch_gpu, axis=1)  # [num_receivers]
        psd_max_batch = cp.max(psd_batch_gpu, axis=1)  # [num_receivers]
        psd_dbm_per_hz_batch = 10 * cp.log10(psd_max_batch / self.sample_rate_hz + 1e-12) + 30
        
        # Frequency offset (gather operation)
        frequency_offset_hz_batch = freqs_gpu[peak_idx_batch]  # [num_receivers]
        
        # Bandwidth (-3dB method)
        threshold_batch = psd_max_batch[:, cp.newaxis] / 2  # [num_receivers, 1]
        above_threshold_batch = psd_batch_gpu > threshold_batch  # [num_receivers, N]
        bandwidth_hz_batch = cp.sum(above_threshold_batch, axis=1) * (self.sample_rate_hz / N)
        
        # Spectral centroid
        spectral_centroid_hz_batch = cp.sum(freqs_gpu[cp.newaxis, :] * psd_batch_gpu, axis=1) / (cp.sum(psd_batch_gpu, axis=1) + 1e-12)
        
        # Spectral rolloff
        cumsum_batch = cp.cumsum(psd_batch_gpu, axis=1)  # [num_receivers, N]
        rolloff_threshold = 0.85 * cumsum_batch[:, -1:]  # [num_receivers, 1]
        rolloff_mask = cumsum_batch >= rolloff_threshold  # [num_receivers, N]
        rolloff_idx_batch = cp.argmax(rolloff_mask, axis=1)  # [num_receivers]
        spectral_rolloff_hz_batch = freqs_gpu[rolloff_idx_batch]
        t3 = time.perf_counter()
        
        # === TIER 3: Temporal Statistics (vectorized) ===
        envelope_batch = cp.abs(batch_gpu)  # [num_receivers, N]
        envelope_max_batch = cp.max(envelope_batch, axis=1, keepdims=True)  # [num_receivers, 1]
        envelope_normalized_batch = envelope_batch / (envelope_max_batch + 1e-12)
        envelope_mean_batch = cp.mean(envelope_normalized_batch, axis=1)
        envelope_std_batch = cp.std(envelope_normalized_batch, axis=1)
        envelope_max_batch = envelope_max_batch.squeeze()
        
        power_samples_batch = cp.abs(batch_gpu) ** 2  # [num_receivers, N]
        peak_power_batch = cp.max(power_samples_batch, axis=1)
        avg_power_batch = cp.mean(power_samples_batch, axis=1)
        peak_to_avg_ratio_db_batch = 10 * cp.log10((peak_power_batch / (avg_power_batch + 1e-12)))
        
        # Zero crossing rate
        real_batch = cp.real(batch_gpu)  # [num_receivers, N]
        sign_batch = cp.sign(real_batch)  # [num_receivers, N]
        diff_sign_batch = cp.diff(sign_batch, axis=1)  # [num_receivers, N-1]
        zero_crossings_batch = cp.sum(diff_sign_batch != 0, axis=1)  # [num_receivers]
        zero_crossing_rate_batch = zero_crossings_batch / N
        t4 = time.perf_counter()
        
        # === TIER 4: Autocorrelation (FFT-based, vectorized) ===
        # Use FFT-based correlation: O(N log N) instead of O(N^2)
        # autocorr(x) = ifft(fft(x) * conj(fft(x)))
        fft_x = cp.fft.fft(batch_gpu, axis=1)  # [num_receivers, N]
        power_spectrum = fft_x * cp.conj(fft_x)  # [num_receivers, N]
        autocorr_full = cp.fft.ifft(power_spectrum, axis=1).real  # [num_receivers, N]
        
        # Normalize autocorrelation
        pdp_batch = cp.abs(autocorr_full) ** 2  # [num_receivers, N]
        pdp_max_batch = cp.max(pdp_batch, axis=1, keepdims=True)  # [num_receivers, 1]
        pdp_norm_batch = pdp_batch / (pdp_max_batch + 1e-12)  # [num_receivers, N]
        
        # Find significant delays (threshold = 0.1)
        threshold = 0.1
        significant_mask = pdp_norm_batch > threshold  # [num_receivers, N]
        
        # For multipath calculation, we need to process each receiver separately
        # (complex weighted average operation doesn't vectorize easily)
        multipath_delay_spread_us_list = []
        coherence_bandwidth_khz_list = []
        delay_spread_confidence_list = []
        
        # Transfer needed data to CPU for per-receiver multipath calculation
        significant_mask_cpu = cp.asnumpy(significant_mask)
        pdp_norm_cpu = cp.asnumpy(pdp_norm_batch)
        snr_db_cpu = cp.asnumpy(snr_db_batch)
        
        for rx_idx in range(num_receivers):
            significant_delays_idx = np.where(significant_mask_cpu[rx_idx])[0]
            
            if len(significant_delays_idx) > 1:
                delays_sec = significant_delays_idx / self.sample_rate_hz
                powers = pdp_norm_cpu[rx_idx, significant_delays_idx]
                mean_delay = np.average(delays_sec, weights=powers)
                rms_delay_spread_sec = np.sqrt(
                    np.average((delays_sec - mean_delay) ** 2, weights=powers)
                )
                multipath_delay_spread_us = float(rms_delay_spread_sec * 1e6)
                coherence_bandwidth_khz = float((1 / (5 * rms_delay_spread_sec + 1e-12)) / 1000)
                
                snr = snr_db_cpu[rx_idx]
                if snr > 15:
                    delay_spread_confidence = 1.0
                elif snr > 10:
                    delay_spread_confidence = 0.5 + (snr - 10) / 10
                elif snr > 5:
                    delay_spread_confidence = 0.3 + (snr - 5) / 10
                else:
                    delay_spread_confidence = max(0.0, snr / 15)
            else:
                multipath_delay_spread_us = 0.0
                coherence_bandwidth_khz = 0.0
                delay_spread_confidence = 0.0
            
            multipath_delay_spread_us_list.append(multipath_delay_spread_us)
            coherence_bandwidth_khz_list.append(coherence_bandwidth_khz)
            delay_spread_confidence_list.append(delay_spread_confidence)
        
        t5 = time.perf_counter()
        
        # Confidence score
        confidence_score_batch = cp.clip(snr_db_batch / 30.0, 0, 1)
        
        # Transfer all features to CPU
        rssi_dbm_cpu = cp.asnumpy(rssi_dbm_batch)
        noise_floor_dbm_cpu = cp.asnumpy(noise_floor_dbm_batch)
        frequency_offset_hz_cpu = cp.asnumpy(frequency_offset_hz_batch)
        bandwidth_hz_cpu = cp.asnumpy(bandwidth_hz_batch)
        psd_dbm_per_hz_cpu = cp.asnumpy(psd_dbm_per_hz_batch)
        spectral_centroid_hz_cpu = cp.asnumpy(spectral_centroid_hz_batch)
        spectral_rolloff_hz_cpu = cp.asnumpy(spectral_rolloff_hz_batch)
        envelope_mean_cpu = cp.asnumpy(envelope_mean_batch)
        envelope_std_cpu = cp.asnumpy(envelope_std_batch)
        envelope_max_cpu = cp.asnumpy(envelope_max_batch)
        peak_to_avg_ratio_db_cpu = cp.asnumpy(peak_to_avg_ratio_db_batch)
        zero_crossing_rate_cpu = cp.asnumpy(zero_crossing_rate_batch)
        signal_present_cpu = cp.asnumpy(signal_present_batch)
        confidence_score_cpu = cp.asnumpy(confidence_score_batch)
        t6 = time.perf_counter()
        
        # Build feature objects
        features_list = []
        for rx_idx in range(num_receivers):
            features = ExtractedFeatures(
                rssi_dbm=float(rssi_dbm_cpu[rx_idx]),
                snr_db=float(snr_db_cpu[rx_idx]),
                noise_floor_dbm=float(noise_floor_dbm_cpu[rx_idx]),
                frequency_offset_hz=float(frequency_offset_hz_cpu[rx_idx]),
                bandwidth_hz=float(bandwidth_hz_cpu[rx_idx]),
                psd_dbm_per_hz=float(psd_dbm_per_hz_cpu[rx_idx]),
                spectral_centroid_hz=float(spectral_centroid_hz_cpu[rx_idx]),
                spectral_rolloff_hz=float(spectral_rolloff_hz_cpu[rx_idx]),
                envelope_mean=float(envelope_mean_cpu[rx_idx]),
                envelope_std=float(envelope_std_cpu[rx_idx]),
                envelope_max=float(envelope_max_cpu[rx_idx]),
                peak_to_avg_ratio_db=float(peak_to_avg_ratio_db_cpu[rx_idx]),
                zero_crossing_rate=float(zero_crossing_rate_cpu[rx_idx]),
                multipath_delay_spread_us=multipath_delay_spread_us_list[rx_idx],
                coherence_bandwidth_khz=coherence_bandwidth_khz_list[rx_idx],
                delay_spread_confidence=delay_spread_confidence_list[rx_idx],
                signal_present=bool(signal_present_cpu[rx_idx]),
                confidence_score=float(confidence_score_cpu[rx_idx]),
            )
            features_list.append(features)
        
        t_total = time.perf_counter() - t_start
        logger.info(
            f"GPU feature extraction timing: "
            f"tier1={((t2-t1)*1000):.1f}ms, "
            f"tier2={((t3-t2)*1000):.1f}ms, "
            f"tier3={((t4-t3)*1000):.1f}ms, "
            f"autocorr={((t5-t4)*1000):.1f}ms, "
            f"transfer={((t6-t5)*1000):.1f}ms, "
            f"total={t_total*1000:.1f}ms"
        )
        
        return features_list
    
    def _extract_features_from_psd(
        self, 
        chunk_data: np.ndarray, 
        psd: np.ndarray, 
        iq_sample: IQSample
    ) -> ExtractedFeatures:
        """
        Extract features given precomputed PSD (from GPU FFT).
        
        This avoids recomputing FFT on CPU - we already have it from GPU.
        
        Args:
            chunk_data: Complex IQ samples
            psd: Power spectral density (already computed)
            iq_sample: Metadata
            
        Returns:
            ExtractedFeatures
        """
        N = len(chunk_data)
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / self.sample_rate_hz))
        
        # === TIER 1: Essential Features ===
        signal_power = np.mean(np.abs(chunk_data) ** 2)
        rssi_dbm = 10 * np.log10(signal_power + 1e-12) + 30
        
        # Noise estimation
        sorted_psd = np.sort(psd)
        n_noise_bins = max(1, int(0.3 * N))
        noise_floor_psd_bin = np.median(sorted_psd[:n_noise_bins])
        noise_power = noise_floor_psd_bin
        noise_floor_dbm = 10 * np.log10(noise_power + 1e-12) + 30
        
        # SNR
        snr_linear = signal_power / (noise_power + 1e-12)
        snr_db = 10 * np.log10(snr_linear)
        signal_present = snr_db > 0.0
        
        # === TIER 2: Frequency Features ===
        peak_idx = np.argmax(psd)
        psd_dbm_per_hz = 10 * np.log10(np.max(psd) / self.sample_rate_hz + 1e-12) + 30
        frequency_offset_hz = float(freqs[peak_idx])
        
        threshold = np.max(psd) / 2
        above_threshold = psd > threshold
        bandwidth_hz = float(np.sum(above_threshold) * (self.sample_rate_hz / len(psd)))
        
        spectral_centroid_hz = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
        
        cumsum = np.cumsum(psd)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff_hz = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0
        
        # === TIER 3: Temporal Statistics ===
        envelope = np.abs(chunk_data)
        envelope_normalized = envelope / (np.max(envelope) + 1e-12)
        envelope_mean = float(np.mean(envelope_normalized))
        envelope_std = float(np.std(envelope_normalized))
        envelope_max = float(np.max(envelope_normalized))
        
        power_samples = np.abs(chunk_data) ** 2
        peak_power = np.max(power_samples)
        avg_power = np.mean(power_samples)
        peak_to_avg_ratio_db = float(10 * np.log10((peak_power / (avg_power + 1e-12))))
        
        zero_crossings = np.sum(np.diff(np.sign(np.real(chunk_data))) != 0)
        zero_crossing_rate = float(zero_crossings / len(chunk_data))
        
        # === TIER 4: Advanced Features ===
        autocorr = np.correlate(chunk_data, chunk_data, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        pdp = np.abs(autocorr) ** 2
        pdp_norm = pdp / (np.max(pdp) + 1e-12)
        
        threshold = 0.1
        significant_delays_idx = np.where(pdp_norm > threshold)[0]
        
        if len(significant_delays_idx) > 1:
            delays_sec = significant_delays_idx / self.sample_rate_hz
            powers = pdp_norm[significant_delays_idx]
            mean_delay = np.average(delays_sec, weights=powers)
            rms_delay_spread_sec = np.sqrt(
                np.average((delays_sec - mean_delay) ** 2, weights=powers)
            )
            multipath_delay_spread_us = float(rms_delay_spread_sec * 1e6)
            coherence_bandwidth_khz = float((1 / (5 * rms_delay_spread_sec + 1e-12)) / 1000)
            
            if snr_db > 15:
                delay_spread_confidence = 1.0
            elif snr_db > 10:
                delay_spread_confidence = 0.5 + (snr_db - 10) / 10
            elif snr_db > 5:
                delay_spread_confidence = 0.3 + (snr_db - 5) / 10
            else:
                delay_spread_confidence = max(0.0, snr_db / 15)
        else:
            multipath_delay_spread_us = 0.0
            coherence_bandwidth_khz = 0.0
            delay_spread_confidence = 0.0
        
        confidence_score = float(np.clip(snr_db / 30.0, 0, 1))
        
        return ExtractedFeatures(
            rssi_dbm=float(rssi_dbm),
            snr_db=float(snr_db),
            noise_floor_dbm=float(noise_floor_dbm),
            frequency_offset_hz=frequency_offset_hz,
            bandwidth_hz=bandwidth_hz,
            psd_dbm_per_hz=float(psd_dbm_per_hz),
            spectral_centroid_hz=spectral_centroid_hz,
            spectral_rolloff_hz=spectral_rolloff_hz,
            envelope_mean=envelope_mean,
            envelope_std=envelope_std,
            envelope_max=envelope_max,
            peak_to_avg_ratio_db=float(peak_to_avg_ratio_db),
            zero_crossing_rate=zero_crossing_rate,
            multipath_delay_spread_us=multipath_delay_spread_us,
            coherence_bandwidth_khz=coherence_bandwidth_khz,
            delay_spread_confidence=float(delay_spread_confidence),
            signal_present=bool(signal_present),
            confidence_score=confidence_score,
        )
    
    def _aggregate_chunk_features(
        self, 
        chunk_features: List[ExtractedFeatures]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate chunk features with mean/std/min/max.
        
        Args:
            chunk_features: List of ExtractedFeatures from multiple chunks
            
        Returns:
            Dict mapping feature_name to {mean, std, min, max}
        """
        aggregated = {}
        
        feature_names = [
            'rssi_dbm', 'snr_db', 'noise_floor_dbm',
            'frequency_offset_hz', 'bandwidth_hz', 'psd_dbm_per_hz',
            'spectral_centroid_hz', 'spectral_rolloff_hz',
            'envelope_mean', 'envelope_std', 'envelope_max',
            'peak_to_avg_ratio_db', 'zero_crossing_rate',
            'multipath_delay_spread_us', 'coherence_bandwidth_khz',
            'delay_spread_confidence', 'confidence_score'
        ]
        
        for feature_name in feature_names:
            values = [getattr(f, feature_name) for f in chunk_features]
            aggregated[feature_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        # Boolean signal_present: majority vote
        signal_present_votes = [f.signal_present for f in chunk_features]
        aggregated['signal_present'] = {
            'mean': float(np.mean(signal_present_votes)),
            'std': 0.0,
            'min': float(min(signal_present_votes)),
            'max': float(max(signal_present_votes)),
        }
        
        return aggregated
