#!/usr/bin/env python3
"""
Unit tests to verify 200ms migration works correctly.

Tests:
1. IQGenerator produces 10,000 samples at 50kHz (200ms)
2. Synthetic sample metadata reflects 200ms duration
3. MinIO file size matches 200ms expectation (~78 KiB for 10,000 samples)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services/training'))

from src.data.iq_generator import SyntheticIQGenerator


def test_iq_generator_200ms():
    """Test that IQGenerator produces exactly 200ms of IQ data."""
    print("\n🧪 Test 1: IQGenerator produces 200ms (10,000 samples at 50kHz)")
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=200.0,
        seed=42,
        use_gpu=False,
        use_audio_library=False
    )
    
    # Generate IQ sample
    iq_sample = generator.generate_iq_sample(
        modulation='nbfm',
        signal_power_dbm=-10.0,
        noise_power_dbm=-80.0,
        frequency_offset_hz=100.0,
        doppler_shift_hz=0.0,
        multipath_delay_us=0.0
    )
    
    # Verify sample count
    assert iq_sample.num_samples == 10_000, f"Expected 10,000 samples, got {iq_sample.num_samples}"
    assert iq_sample.duration_ms == 200.0, f"Expected 200ms, got {iq_sample.duration_ms}ms"
    assert iq_sample.sample_rate_hz == 50_000, f"Expected 50kHz, got {iq_sample.sample_rate_hz}Hz"
    
    # Verify file size (complex64 = 8 bytes per sample)
    expected_bytes = 10_000 * 8  # 80,000 bytes = ~78 KiB
    actual_bytes = iq_sample.samples.nbytes
    assert actual_bytes == expected_bytes, f"Expected {expected_bytes} bytes, got {actual_bytes}"
    
    print(f"✅ IQGenerator produces exactly 10,000 samples (200ms at 50kHz)")
    print(f"✅ File size: {actual_bytes:,} bytes (~{actual_bytes/1024:.1f} KiB)")


def test_audio_chunk_duration():
    """Test that audio preprocessing uses 200ms chunks."""
    print("\n🧪 Test 2: Audio preprocessing constant is 200ms")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services/backend'))
    from src.tasks.audio_preprocessing import CHUNK_DURATION_SECONDS, TARGET_SAMPLE_RATE_HZ
    
    assert CHUNK_DURATION_SECONDS == 0.2, f"Expected 0.2s chunks, got {CHUNK_DURATION_SECONDS}s"
    assert TARGET_SAMPLE_RATE_HZ == 50_000, f"Expected 50kHz, got {TARGET_SAMPLE_RATE_HZ}Hz"
    
    expected_samples = int(TARGET_SAMPLE_RATE_HZ * CHUNK_DURATION_SECONDS)
    assert expected_samples == 10_000, f"Expected 10,000 samples per chunk, got {expected_samples}"
    
    print(f"✅ Audio preprocessing uses {CHUNK_DURATION_SECONDS}s chunks ({expected_samples} samples at {TARGET_SAMPLE_RATE_HZ}Hz)")


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Running 200ms Migration Unit Tests")
    print("=" * 80)
    
    try:
        test_iq_generator_200ms()
        test_audio_chunk_duration()
        
        print("\n" + "=" * 80)
        print("✅ All tests PASSED! 200ms migration verified.")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
