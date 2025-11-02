"""
RF Feature Extraction Module

Extracts features from IQ samples for ML training.
Processes both synthetic and real recording session data.
"""

import numpy as np
import structlog
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = structlog.get_logger(__name__)


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
    """Extract RF features from IQ samples."""

    def __init__(self, sample_rate_hz: int = 200000):
        """
        Initialize feature extractor.

        Args:
            sample_rate_hz: Sample rate in Hz (default 200kHz)
        """
        self.sample_rate_hz = sample_rate_hz
        logger.info(f"Initialized RFFeatureExtractor with sample_rate={sample_rate_hz} Hz")

    def extract_features(self, iq_sample: IQSample) -> ExtractedFeatures:
        """
        Extract all features from a single IQ chunk.

        Args:
            iq_sample: IQ sample data

        Returns:
            ExtractedFeatures object with all 18 base features
        """
        # === TIER 1: Essential Features ===

        # Signal power (mean squared magnitude)
        signal_power = np.mean(np.abs(iq_sample.samples) ** 2)
        rssi_dbm = 10 * np.log10(signal_power + 1e-12) + 30  # Convert to dBm (assuming 50Ω)

        # === TIER 2: Frequency Features (compute FFT early for noise estimation) ===

        # FFT for spectral analysis
        N = len(iq_sample.samples)
        fft = np.fft.fftshift(np.fft.fft(iq_sample.samples))
        # Power per frequency bin (Parseval-normalized)
        # By Parseval's theorem: sum(|x|^2) = sum(|FFT(x)|^2) / N
        psd = np.abs(fft) ** 2 / N
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / self.sample_rate_hz))

        # Find peak and estimate noise
        peak_idx = np.argmax(psd)
        
        # Noise estimation: use median of lowest 30% of PSD bins
        # This assumes signal occupies <70% of spectrum
        sorted_psd = np.sort(psd)
        n_noise_bins = max(1, int(0.3 * N))
        noise_floor_psd_bin = np.median(sorted_psd[:n_noise_bins])
        
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
        
        noise_floor_dbm = 10 * np.log10(noise_power + 1e-12) + 30

        # SNR
        snr_linear = signal_power / (noise_power + 1e-12)
        snr_db = 10 * np.log10(snr_linear)

        # Signal detection threshold
        signal_present = snr_db > 0.0

        # PSD in dBm/Hz
        psd_dbm_per_hz = 10 * np.log10(np.max(psd) / self.sample_rate_hz + 1e-12) + 30

        # Frequency offset (peak location relative to center) - already computed above
        frequency_offset_hz = float(freqs[peak_idx])

        # Bandwidth (-3dB)
        threshold = np.max(psd) / 2  # -3dB point
        above_threshold = psd > threshold
        bandwidth_hz = float(np.sum(above_threshold) * (self.sample_rate_hz / len(psd)))

        # Spectral centroid (center of mass)
        spectral_centroid_hz = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))

        # Spectral rolloff (85% energy point)
        cumsum = np.cumsum(psd)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff_hz = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0

        # === TIER 3: Temporal Statistics ===

        # Envelope (magnitude)
        envelope = np.abs(iq_sample.samples)
        envelope_normalized = envelope / (np.max(envelope) + 1e-12)

        envelope_mean = float(np.mean(envelope_normalized))
        envelope_std = float(np.std(envelope_normalized))
        envelope_max = float(np.max(envelope_normalized))

        # Peak-to-average ratio (PAR)
        power_samples = np.abs(iq_sample.samples) ** 2
        peak_power = np.max(power_samples)
        avg_power = np.mean(power_samples)
        peak_to_avg_ratio_db = float(10 * np.log10((peak_power / (avg_power + 1e-12))))

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(np.real(iq_sample.samples))) != 0)
        zero_crossing_rate = float(zero_crossings / len(iq_sample.samples))

        # === TIER 4: Advanced Features ===

        # Multipath delay spread via autocorrelation
        autocorr = np.correlate(iq_sample.samples, iq_sample.samples, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags
        pdp = np.abs(autocorr) ** 2
        pdp_norm = pdp / (np.max(pdp) + 1e-12)

        # Find significant delays (above -10dB threshold)
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

            # Coherence bandwidth ~ 1 / (5 * delay_spread)
            coherence_bandwidth_khz = float(
                (1 / (5 * rms_delay_spread_sec + 1e-12)) / 1000
            )

            # Confidence based on SNR
            if snr_db > 15:
                delay_spread_confidence = 1.0
            elif snr_db > 10:
                delay_spread_confidence = 0.5 + (snr_db - 10) / 10
            elif snr_db > 5:
                delay_spread_confidence = 0.3 + (snr_db - 5) / 10
            else:
                delay_spread_confidence = max(0.0, snr_db / 15)
        else:
            # No multipath detected
            multipath_delay_spread_us = 0.0
            coherence_bandwidth_khz = 0.0
            delay_spread_confidence = 0.0

        # Overall confidence (SNR-based, simple version)
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
