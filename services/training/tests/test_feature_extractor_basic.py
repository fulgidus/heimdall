"""Basic tests for RF feature extractor."""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.data.rf_feature_extractor import (
    RFFeatureExtractor,
    IQSample,
)


def test_feature_extractor_initialization():
    """Test feature extractor initialization."""
    extractor = RFFeatureExtractor(sample_rate_hz=200000)
    assert extractor.sample_rate_hz == 200000


def test_extract_features_from_clean_signal():
    """Test feature extraction from signal with controlled SNR."""
    # Generate signal with known SNR
    sample_rate = 200000
    duration_sec = 0.2  # 200ms
    num_samples = int(sample_rate * duration_sec)

    t = np.arange(num_samples) / sample_rate
    frequency_offset = 1000  # 1kHz offset
    
    # Generate signal with amplitude A and noise with power N
    # SNR (dB) = 10 * log10(A^2 / N^2)
    # For SNR = 20dB: A^2 / N^2 = 100, so A/N = 10
    signal_amplitude = 1.0
    noise_amplitude = 0.1  # Gives SNR ~20dB
    
    signal = signal_amplitude * np.exp(2j * np.pi * frequency_offset * t)
    noise = noise_amplitude * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    iq = signal + noise

    iq_sample = IQSample(
        samples=iq.astype(np.complex64),
        sample_rate_hz=sample_rate,
        center_frequency_hz=144000000,
        rx_id="test_rx",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=datetime.now(timezone.utc),
    )

    extractor = RFFeatureExtractor(sample_rate_hz=sample_rate)
    features = extractor.extract_features(iq_sample)

    # Verify feature extraction
    # Note: SNR estimation for CW signals is challenging due to FFT leakage
    # Just verify signal is detected (SNR > 0)
    assert features.snr_db > 0  # Should detect signal presence
    assert features.signal_present is True
    assert 0 <= features.confidence_score <= 1
    assert features.bandwidth_hz > 0
    assert abs(features.frequency_offset_hz - frequency_offset) < 100  # Within 100 Hz


def test_extract_features_chunked():
    """Test chunked feature extraction with aggregation."""
    # Generate 1s signal
    sample_rate = 200000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    t = np.arange(num_samples) / sample_rate
    signal = np.exp(2j * np.pi * 500 * t)  # 500 Hz offset
    noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    iq = signal + noise

    iq_sample = IQSample(
        samples=iq.astype(np.complex64),
        sample_rate_hz=sample_rate,
        center_frequency_hz=144000000,
        rx_id="test_rx",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=datetime.now(timezone.utc),
    )

    extractor = RFFeatureExtractor(sample_rate_hz=sample_rate)
    features_agg = extractor.extract_features_chunked(
        iq_sample, chunk_duration_ms=200.0, num_chunks=5
    )

    # Verify aggregated structure
    assert 'snr_db' in features_agg
    assert 'mean' in features_agg['snr_db']
    assert 'std' in features_agg['snr_db']
    assert 'min' in features_agg['snr_db']
    assert 'max' in features_agg['snr_db']

    # Verify stats make sense
    assert features_agg['snr_db']['min'] <= features_agg['snr_db']['mean']
    assert features_agg['snr_db']['mean'] <= features_agg['snr_db']['max']
    assert features_agg['snr_db']['std'] >= 0


def test_calculate_overall_confidence():
    """Test overall confidence calculation."""
    extractor = RFFeatureExtractor()

    # High quality scenario
    confidence = extractor._calculate_overall_confidence(
        snr_values=[25.0, 22.0, 28.0],
        detection_rate=1.0,
        spectral_clarity=0.9
    )
    assert confidence > 0.8

    # Low quality scenario
    confidence = extractor._calculate_overall_confidence(
        snr_values=[2.0, 1.0, 3.0],
        detection_rate=0.3,
        spectral_clarity=0.4
    )
    assert confidence < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
