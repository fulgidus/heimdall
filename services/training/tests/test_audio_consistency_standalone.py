#!/usr/bin/env python3
"""
Correct test for audio consistency - tests signals BEFORE noise is added.
"""
import sys
sys.path.insert(0, '/home/fulgidus/Documents/Projects/heimdall/services/training')

import numpy as np
from src.data.iq_generator import SyntheticIQGenerator


def test_clean_signal_consistency():
    """
    Test audio consistency by comparing CLEAN signals (before noise).
    
    The issue with the previous test was comparing noisy signals - different
    noise realizations decorrelate the FM-demodulated audio even when the
    underlying audio content is identical.
    
    This test uses _generate_clean_signal_batch() directly to verify that
    all receivers get the same FM-modulated signal (before noise).
    """
    print("="*70)
    print("CORRECT AUDIO CONSISTENCY TEST")
    print("="*70)
    print("\nTesting clean signal consistency (before noise addition)...")
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000, 
        duration_ms=1000.0, 
        seed=42,
        use_gpu=False
    )
    
    batch_size = 7
    
    # Different propagation parameters for each receiver
    # IMPORTANT: Use ZERO frequency offsets to eliminate carrier phase differences
    frequency_offsets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bandwidths = np.array([12500.0] * batch_size)
    
    # Generate CLEAN signals (before noise)
    print(f"Generating {batch_size} clean FM signals (same audio)...")
    
    clean_signals = generator._generate_clean_signal_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        batch_size=batch_size
    )
    
    print(f"✓ Generated clean batch with shape: {clean_signals.shape}")
    
    # Extract audio by FM demodulation
    def extract_audio_from_iq(iq_signal):
        """Extract audio by FM demodulation (phase derivative)."""
        phase = np.unwrap(np.angle(iq_signal))
        audio = np.diff(phase)
        return audio
    
    # Extract audio from all receivers
    print(f"\nExtracting audio from {batch_size} receivers...")
    audio_signals = []
    for i in range(batch_size):
        audio = extract_audio_from_iq(clean_signals[i])
        audio_signals.append(audio)
        print(f"  Receiver {i+1}: mean={np.mean(audio):.6f}, std={np.std(audio):.6f}")
    
    # Compare audio signals
    print(f"\nComputing audio correlations (clean signals)...")
    reference_audio = audio_signals[0]
    
    correlations = []
    max_diffs = []
    
    for i in range(1, batch_size):
        # Correlation
        correlation = np.corrcoef(reference_audio, audio_signals[i])[0, 1]
        correlations.append(correlation)
        
        # Direct array comparison (after normalizing)
        ref_norm = reference_audio / (np.std(reference_audio) + 1e-10)
        test_norm = audio_signals[i] / (np.std(audio_signals[i]) + 1e-10)
        max_diff = np.max(np.abs(ref_norm - test_norm))
        max_diffs.append(max_diff)
        
        status = "✓" if correlation > 0.95 else "❌"
        print(f"  {status} Receiver 1 vs Receiver {i+1}: corr={correlation:.6f}, max_diff={max_diff:.6f}")
    
    # Summary
    min_corr = min(correlations)
    max_corr = max(correlations)
    avg_corr = np.mean(correlations)
    avg_diff = np.mean(max_diffs)
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Correlation range: {min_corr:.6f} - {max_corr:.6f}")
    print(f"Average correlation: {avg_corr:.6f}")
    print(f"Average max difference (normalized): {avg_diff:.6f}")
    
    if min_corr > 0.95:
        print(f"\n✅ SUCCESS: All receivers have consistent audio content!")
        print(f"   All correlations > 0.95 - Audio broadcasting is working correctly.")
        return True
    else:
        print(f"\n❌ FAIL: Audio not consistent across receivers")
        print(f"   Minimum correlation: {min_corr:.6f} (expected >0.95)")
        print(f"   This indicates audio is NOT being broadcast correctly.")
        return False


def test_with_different_frequency_offsets():
    """
    Test with different frequency offsets (realistic scenario).
    
    With different frequency offsets, we expect LOWER but still HIGH correlation
    because the carrier phase difference affects the FM demodulation.
    """
    print("\n" + "="*70)
    print("TEST WITH DIFFERENT FREQUENCY OFFSETS")
    print("="*70)
    print("\nTesting with realistic frequency offsets...")
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000, 
        duration_ms=1000.0, 
        seed=42,
        use_gpu=False
    )
    
    batch_size = 7
    
    # Different frequency offsets (realistic WebSDR scenario)
    frequency_offsets = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    bandwidths = np.array([12500.0] * batch_size)
    
    # Generate CLEAN signals (before noise)
    clean_signals = generator._generate_clean_signal_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        batch_size=batch_size
    )
    
    # Extract audio
    def extract_audio(iq):
        return np.diff(np.unwrap(np.angle(iq)))
    
    audio_signals = [extract_audio(clean_signals[i]) for i in range(batch_size)]
    
    # Compare with receiver that has zero offset
    zero_offset_idx = 3  # frequency_offset = 0.0
    reference_audio = audio_signals[zero_offset_idx]
    
    print(f"\nComparing all receivers to Receiver {zero_offset_idx+1} (zero freq offset)...")
    
    correlations = []
    for i in range(batch_size):
        if i == zero_offset_idx:
            continue
        correlation = np.corrcoef(reference_audio, audio_signals[i])[0, 1]
        correlations.append(correlation)
        
        freq_offset = frequency_offsets[i]
        print(f"  Receiver {i+1} (offset={freq_offset:+.1f} Hz): corr={correlation:.6f}")
    
    avg_corr = np.mean(correlations)
    min_corr = min(correlations)
    
    print(f"\nAverage correlation: {avg_corr:.6f}")
    print(f"Minimum correlation: {min_corr:.6f}")
    
    # With frequency offsets, we expect slightly lower correlation
    # but still should be reasonable (>0.7) for same audio content
    if min_corr > 0.7:
        print(f"\n✅ Good: Even with frequency offsets, correlation > 0.7")
        print(f"   This confirms same audio content across receivers.")
        return True
    else:
        print(f"\n⚠️  Warning: Low correlation ({min_corr:.3f}) with frequency offsets")
        print(f"   This might indicate frequency offset compensation issues.")
        return False


def test_direct_iq_comparison():
    """
    Ultimate test: Directly compare IQ samples (before FM demodulation).
    
    If audio broadcasting is working, the IQ samples should be IDENTICAL
    when frequency offsets are all zero.
    """
    print("\n" + "="*70)
    print("DIRECT IQ SAMPLE COMPARISON (no FM demodulation)")
    print("="*70)
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000, 
        duration_ms=100.0,  # Shorter for faster test
        seed=42,
        use_gpu=False
    )
    
    batch_size = 5
    frequency_offsets = np.array([0.0] * batch_size)  # All zero
    bandwidths = np.array([12500.0] * batch_size)
    
    # Generate clean signals
    clean_signals = generator._generate_clean_signal_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        batch_size=batch_size
    )
    
    print(f"\nComparing IQ samples directly (all receivers have zero freq offset)...")
    
    reference_iq = clean_signals[0]
    
    all_identical = True
    for i in range(1, batch_size):
        # Direct comparison
        identical = np.allclose(reference_iq, clean_signals[i], rtol=1e-9, atol=1e-9)
        max_diff = np.max(np.abs(reference_iq - clean_signals[i]))
        
        status = "✓" if identical else "❌"
        print(f"  {status} Receiver 1 vs Receiver {i+1}: identical={identical}, max_diff={max_diff:.2e}")
        
        if not identical:
            all_identical = False
    
    if all_identical:
        print(f"\n✅ PERFECT: All IQ samples are identical!")
        print(f"   Audio broadcasting is working 100% correctly.")
        return True
    else:
        print(f"\n❌ FAIL: IQ samples differ between receivers")
        print(f"   With zero frequency offsets, IQ should be identical.")
        return False


if __name__ == "__main__":
    print("AUDIO CONSISTENCY TEST - CORRECT VERSION")
    print("Tests clean signals (before noise) to verify audio broadcasting")
    print("="*70)
    
    # Run tests
    test1_pass = test_clean_signal_consistency()
    test2_pass = test_with_different_frequency_offsets()
    test3_pass = test_direct_iq_comparison()
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Test 1 (clean signals, zero offset): {'PASS ✓' if test1_pass else 'FAIL ❌'}")
    print(f"Test 2 (different freq offsets): {'PASS ✓' if test2_pass else 'FAIL ❌'}")
    print(f"Test 3 (direct IQ comparison): {'PASS ✓' if test3_pass else 'FAIL ❌'}")
    
    if test1_pass and test3_pass:
        print("\n🎉 AUDIO BROADCASTING IS WORKING CORRECTLY!")
        print("   All receivers get the same audio content.")
        sys.exit(0)
    else:
        print("\n❌ AUDIO BROADCASTING NEEDS FIXES")
        sys.exit(1)
