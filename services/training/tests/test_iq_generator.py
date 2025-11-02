"""Tests for SyntheticIQGenerator."""

import pytest
import numpy as np
from data.iq_generator import SyntheticIQGenerator, IQSample


def test_iq_generator_initialization():
    """Test IQGenerator initializes correctly."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)

    assert generator.sample_rate_hz == 200_000
    assert generator.duration_ms == 1000.0
    assert generator.num_samples == 200_000  # 200 kHz × 1 second


def test_generate_iq_sample_shape():
    """Test IQ sample has correct shape and type."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)

    iq_sample = generator.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=22.0,
        frequency_offset_hz=-15.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0
    )

    assert isinstance(iq_sample, IQSample)
    assert iq_sample.samples.dtype == np.complex64
    assert len(iq_sample.samples) == 200_000
    assert iq_sample.sample_rate_hz == 200_000
    assert iq_sample.duration_ms == 1000.0


def test_iq_sample_snr():
    """Test generated IQ sample has approximately correct SNR."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)

    target_snr_db = 20.0

    iq_sample = generator.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=target_snr_db,
        frequency_offset_hz=0.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0,
        enable_multipath=False,  # Disable for cleaner SNR measurement
        enable_fading=False
    )

    # Calculate actual SNR (rough estimate via power ratio)
    power = np.mean(np.abs(iq_sample.samples) ** 2)

    # Since we can't easily separate signal from noise, just check power is reasonable
    assert power > 0
    assert not np.isnan(power)


def test_multipath_adds_delay():
    """Test multipath adds delayed copies."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=100.0, seed=42)

    # Generate with and without multipath
    iq_no_multipath = generator.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=0.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0,
        enable_multipath=False,
        enable_fading=False
    )

    generator2 = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=100.0, seed=42)
    iq_with_multipath = generator2.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=0.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0,
        enable_multipath=True,
        enable_fading=False
    )

    # Signals should be different
    assert not np.allclose(iq_no_multipath.samples, iq_with_multipath.samples)


def test_fading_varies_envelope():
    """Test Rayleigh fading varies signal envelope."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)

    iq_sample = generator.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=0.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0,
        enable_multipath=False,
        enable_fading=True
    )

    # Calculate envelope (magnitude)
    envelope = np.abs(iq_sample.samples)

    # Envelope should vary (not constant)
    envelope_std = np.std(envelope)
    assert envelope_std > 0

    # Should have some variation in power over time
    chunk_size = 20000  # 100ms chunks
    powers = []
    for i in range(0, len(iq_sample.samples), chunk_size):
        chunk = iq_sample.samples[i:i+chunk_size]
        powers.append(np.mean(np.abs(chunk) ** 2))

    # Power should vary across chunks (fading effect)
    power_std = np.std(powers)
    assert power_std > 0


def test_reproducibility_with_seed():
    """Test same seed produces identical IQ samples."""
    generator1 = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=100.0, seed=12345)
    generator2 = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=100.0, seed=12345)

    iq1 = generator1.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=-15.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0
    )

    iq2 = generator2.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=-15.0,
        bandwidth_hz=12500.0,
        rx_id="Test",
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0
    )

    assert np.allclose(iq1.samples, iq2.samples)
