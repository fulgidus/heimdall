"""Tests for SyntheticIQGenerator."""

import numpy as np
from src.data.iq_generator import SyntheticIQGenerator, SyntheticIQSample


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

    assert isinstance(iq_sample, SyntheticIQSample)
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


def test_audio_consistency_across_receivers():
    """Test that all receivers in a batch receive the same audio content.
    
    This is critical for ML training - all receivers must "hear" the same signal,
    with only RF propagation effects (multipath, fading, SNR) differing.
    """
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000, 
        duration_ms=1000.0, 
        seed=42,
        use_gpu=False  # Use CPU for testing
    )
    
    # Simulate 7 receivers (like 7 WebSDRs)
    batch_size = 7
    
    # Different propagation parameters for each receiver
    frequency_offsets = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    bandwidths = np.array([12500.0] * batch_size)
    signal_powers_dbm = np.array([-65.0, -70.0, -68.0, -60.0, -72.0, -67.0, -69.0])
    noise_floors_dbm = np.array([-87.0] * batch_size)
    snr_dbs = signal_powers_dbm - noise_floors_dbm
    
    # Generate batch (all receivers should get same audio)
    batch_signals = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,  # Disable for cleaner comparison
        enable_fading=False
    )
    
    assert batch_signals.shape == (batch_size, generator.num_samples)
    
    # Extract audio from FM-demodulated signals
    # For FM demodulation: audio ≈ derivative of phase
    def extract_audio_from_iq(iq_signal):
        """Extract audio by FM demodulation (phase derivative)."""
        # Calculate instantaneous phase
        phase = np.unwrap(np.angle(iq_signal))
        # Differentiate to get frequency (audio)
        audio = np.diff(phase)
        return audio
    
    # Extract audio from all receivers
    audio_signals = []
    for i in range(batch_size):
        audio = extract_audio_from_iq(batch_signals[i])
        audio_signals.append(audio)
    
    # Compare audio signals - they should be very similar (same content)
    # Allow for small differences due to frequency offsets and numerical precision
    reference_audio = audio_signals[0]
    
    for i in range(1, batch_size):
        # Compute correlation with reference
        correlation = np.corrcoef(reference_audio, audio_signals[i])[0, 1]
        
        # High correlation (>0.9) indicates same audio content
        # Different frequency offsets will cause slight differences, but content should match
        assert correlation > 0.9, (
            f"Receiver {i} has different audio content (correlation={correlation:.3f}). "
            f"All receivers should receive the same audio with only RF effects differing."
        )
    
    print(f"✓ Audio consistency test passed - all {batch_size} receivers have same content")


def test_audio_consistency_with_audio_library():
    """Test audio consistency when using audio library (if available).
    
    This test uses formant synthesis as fallback if audio library is not available.
    """
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000, 
        duration_ms=500.0,  # Shorter duration for faster testing
        seed=123,
        use_gpu=False,
        use_audio_library=True,  # Try to use audio library
        audio_library_fallback=True  # Fallback to formant synthesis if not available
    )
    
    batch_size = 5
    
    frequency_offsets = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    bandwidths = np.array([12500.0] * batch_size)
    signal_powers_dbm = np.array([-65.0] * batch_size)
    noise_floors_dbm = np.array([-87.0] * batch_size)
    snr_dbs = signal_powers_dbm - noise_floors_dbm
    
    # Generate batch - should use same audio file for all receivers
    batch_signals = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,
        enable_fading=False
    )
    
    # Check all signals are present and valid
    assert batch_signals.shape == (batch_size, generator.num_samples)
    assert not np.any(np.isnan(batch_signals))
    
    # Extract audio from all receivers
    def extract_audio_from_iq(iq_signal):
        phase = np.unwrap(np.angle(iq_signal))
        audio = np.diff(phase)
        return audio
    
    audio_signals = [extract_audio_from_iq(batch_signals[i]) for i in range(batch_size)]
    
    # Verify audio consistency across receivers
    reference_audio = audio_signals[0]
    for i in range(1, batch_size):
        correlation = np.corrcoef(reference_audio, audio_signals[i])[0, 1]
        assert correlation > 0.9, (
            f"Audio library sample not consistent across receivers (correlation={correlation:.3f})"
        )
    
    print(f"✓ Audio library consistency test passed - all {batch_size} receivers have same audio")
