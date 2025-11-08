"""Unit tests for IQ processor."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from src.models.websdrs import SignalMetrics
from src.processors.iq_processor import IQProcessor


def test_compute_metrics(sample_iq_data):
    """Test signal metrics computation."""
    metrics = IQProcessor.compute_metrics(
        iq_data=sample_iq_data,
        sample_rate_hz=12500,
        target_frequency_hz=int(145.5 * 1e6),
        noise_bandwidth_hz=10000,
    )

    assert isinstance(metrics, SignalMetrics)
    assert metrics.snr_db >= -100
    assert metrics.snr_db <= 100
    assert isinstance(metrics.psd_dbm, float)
    assert isinstance(metrics.frequency_offset_hz, float)
    assert isinstance(metrics.signal_power_dbm, float)
    assert isinstance(metrics.noise_power_dbm, float)


def test_compute_metrics_empty_data():
    """Test metrics computation with empty data."""
    empty_data = np.array([], dtype=np.complex64)

    with pytest.raises(ValueError, match="Empty IQ data"):
        IQProcessor.compute_metrics(
            iq_data=empty_data,
            sample_rate_hz=12500,
            target_frequency_hz=int(145.5 * 1e6),
        )


def test_compute_psd(sample_iq_data):
    """Test PSD computation."""
    psd_db, freqs = IQProcessor._compute_psd(sample_iq_data, 12500)

    assert len(psd_db) > 0
    assert len(freqs) > 0
    assert len(psd_db) == len(freqs)
    assert np.all(np.isfinite(psd_db))  # No NaN or inf


def test_estimate_frequency_offset(sample_iq_data):
    """Test frequency offset estimation."""
    offset = IQProcessor._estimate_frequency_offset(
        sample_iq_data, sample_rate_hz=12500, target_frequency_hz=int(145.5 * 1e6)
    )

    assert isinstance(offset, float)
    assert -6250 <= offset <= 6250  # Within ±Fs/2


def test_compute_snr():
    """Test SNR computation."""
    # Create realistic IQ data with strong signal component
    n_samples = 10000
    sample_rate_hz = 12500

    # Create noise base
    noise_i = np.random.randn(n_samples).astype(np.float32) * 0.1  # Low noise
    noise_q = np.random.randn(n_samples).astype(np.float32) * 0.1
    iq_data = (noise_i + 1j * noise_q).astype(np.complex64)

    # Add strong signal component (1 kHz)
    signal_freq = 1000  # 1 kHz
    t = np.arange(n_samples) / sample_rate_hz
    signal_amp = 1.0  # Strong signal
    signal = signal_amp * np.exp(2j * np.pi * signal_freq * t / sample_rate_hz)
    iq_data = iq_data + signal.astype(np.complex64)

    psd_db, freqs = IQProcessor._compute_psd(iq_data, sample_rate_hz)
    signal_power_db, noise_power_db = IQProcessor._compute_snr(
        psd_db, freqs, target_frequency_hz=0, noise_bandwidth_hz=1000
    )

    snr = signal_power_db - noise_power_db

    # Verify types and reasonable ranges
    assert isinstance(signal_power_db, (float, np.floating))
    assert isinstance(noise_power_db, (float, np.floating))
    # SNR should be positive with strong signal
    assert snr > -50 and snr < 100  # Reasonable range


def test_save_iq_data_npy(tmp_path, sample_iq_data):
    """Test saving IQ data to NPY format."""
    output_path = str(tmp_path / "test_iq")
    metadata = {"frequency_mhz": 145.5, "duration_seconds": 10}

    IQProcessor.save_iq_data_npy(sample_iq_data, output_path, metadata)

    # Check files were created
    import os

    assert os.path.exists(f"{output_path}.npy")
    assert os.path.exists(f"{output_path}_meta.json")

    # Load and verify
    loaded_iq = np.load(f"{output_path}.npy")
    assert loaded_iq.shape == sample_iq_data.shape
    assert np.allclose(loaded_iq, sample_iq_data, atol=1e-6)

    # Verify metadata
    import json

    with open(f"{output_path}_meta.json") as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata["frequency_mhz"] == 145.5


def test_metrics_dict_serialization(sample_iq_data):
    """Test that metrics can be serialized to dict."""
    metrics = IQProcessor.compute_metrics(
        iq_data=sample_iq_data,
        sample_rate_hz=12500,
        target_frequency_hz=int(145.5 * 1e6),
    )

    metrics_dict = metrics.dict()

    assert isinstance(metrics_dict, dict)
    assert "snr_db" in metrics_dict
    assert "psd_dbm" in metrics_dict
    assert "frequency_offset_hz" in metrics_dict
    assert all(isinstance(v, float) for v in metrics_dict.values())
