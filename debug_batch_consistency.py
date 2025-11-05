#!/usr/bin/env python3
"""
Minimal debugging script to isolate batch consistency issue.
Tests if second generate_iq_batch() call produces different audio.
"""

import sys
import os
import numpy as np

# Add path for imports
training_path = os.path.join(os.path.dirname(__file__), 'services', 'training')
sys.path.insert(0, training_path)

try:
    # Try container path first
    from src.data.iq_generator import SyntheticIQGenerator  # type: ignore
except ImportError:
    # Fall back to local path
    sys.path.insert(0, os.path.join(training_path, 'src'))
    from data.iq_generator import SyntheticIQGenerator  # type: ignore

def extract_audio_from_iq(iq_signal):
    """Extract audio by FM demodulation."""
    phase = np.unwrap(np.angle(iq_signal))
    audio = np.diff(phase)
    return audio

def compute_correlation(audio1, audio2):
    """Compute normalized correlation between two audio signals."""
    # Normalize
    a1_norm = (audio1 - np.mean(audio1)) / (np.std(audio1) + 1e-8)
    a2_norm = (audio2 - np.mean(audio2)) / (np.std(audio2) + 1e-8)
    
    return np.corrcoef(a1_norm, a2_norm)[0, 1]

def main():
    print("\n" + "="*70)
    print("DEBUGGING BATCH CONSISTENCY ISSUE")
    print("="*70)
    
    # Create generator with fixed seed
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False,
        use_audio_library=False,  # Use formant synthesis to eliminate audio library caching
        audio_library_fallback=True
    )
    
    batch_size = 3  # Small batch for clarity
    frequency_offsets = np.zeros(batch_size)  # Zero offsets for identical modulation
    bandwidths = np.array([12500.0] * batch_size)
    signal_powers_dbm = np.array([-37.0] * batch_size)
    noise_floors_dbm = np.array([-87.0] * batch_size)  # 50 dB SNR
    snr_dbs = signal_powers_dbm - noise_floors_dbm
    
    print("\nTest 1: Direct call to _generate_clean_signal_batch")
    print("-" * 70)
    
    # Direct call - should be identical
    batch1_clean = generator._generate_clean_signal_batch(
        frequency_offsets, bandwidths, batch_size
    )
    
    audio1_0 = extract_audio_from_iq(batch1_clean[0])
    audio1_1 = extract_audio_from_iq(batch1_clean[1])
    audio1_2 = extract_audio_from_iq(batch1_clean[2])
    
    corr_1_0vs1 = compute_correlation(audio1_0, audio1_1)
    corr_1_0vs2 = compute_correlation(audio1_0, audio1_2)
    
    print(f"  Receiver 0 vs 1: correlation = {corr_1_0vs1:.6f}")
    print(f"  Receiver 0 vs 2: correlation = {corr_1_0vs2:.6f}")
    print(f"  Result: {'✅ IDENTICAL' if corr_1_0vs1 > 0.99 else '❌ DIFFERENT'}")
    
    print("\nTest 2: Single generate_iq_batch() call")
    print("-" * 70)
    
    # Reset generator state
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False,
        use_audio_library=False,
        audio_library_fallback=True
    )
    
    batch2 = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,
        enable_fading=False
    )
    
    audio2_0 = extract_audio_from_iq(batch2[0])
    audio2_1 = extract_audio_from_iq(batch2[1])
    audio2_2 = extract_audio_from_iq(batch2[2])
    
    corr_2_0vs1 = compute_correlation(audio2_0, audio2_1)
    corr_2_0vs2 = compute_correlation(audio2_0, audio2_2)
    
    print(f"  Receiver 0 vs 1: correlation = {corr_2_0vs1:.6f}")
    print(f"  Receiver 0 vs 2: correlation = {corr_2_0vs2:.6f}")
    print(f"  Result: {'✅ IDENTICAL' if corr_2_0vs1 > 0.99 else '❌ DIFFERENT'}")
    
    print("\nTest 3: Second generate_iq_batch() call (same generator)")
    print("-" * 70)
    
    # Second call - THIS IS WHERE THE BUG APPEARS
    batch3 = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,
        enable_fading=False
    )
    
    audio3_0 = extract_audio_from_iq(batch3[0])
    audio3_1 = extract_audio_from_iq(batch3[1])
    audio3_2 = extract_audio_from_iq(batch3[2])
    
    corr_3_0vs1 = compute_correlation(audio3_0, audio3_1)
    corr_3_0vs2 = compute_correlation(audio3_0, audio3_2)
    
    print(f"  Receiver 0 vs 1: correlation = {corr_3_0vs1:.6f}")
    print(f"  Receiver 0 vs 2: correlation = {corr_3_0vs2:.6f}")
    print(f"  Result: {'✅ IDENTICAL' if corr_3_0vs1 > 0.99 else '❌ DIFFERENT'}")
    
    print("\nTest 4: Compare batch2 vs batch3")
    print("-" * 70)
    
    corr_batch2_vs_batch3 = compute_correlation(audio2_0, audio3_0)
    print(f"  Batch2[0] vs Batch3[0]: correlation = {corr_batch2_vs_batch3:.6f}")
    print(f"  Expected: LOW (different audio content between batches)")
    print(f"  Result: {'✅ CORRECT' if corr_batch2_vs_batch3 < 0.8 else '❌ UNEXPECTED'}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = True
    
    if corr_1_0vs1 > 0.99:
        print("✅ Test 1: Direct _generate_clean_signal_batch works")
    else:
        print("❌ Test 1: Direct _generate_clean_signal_batch FAILS")
        all_pass = False
    
    if corr_2_0vs1 > 0.99:
        print("✅ Test 2: First generate_iq_batch call works")
    else:
        print("❌ Test 2: First generate_iq_batch call FAILS")
        all_pass = False
    
    if corr_3_0vs1 > 0.99:
        print("✅ Test 3: Second generate_iq_batch call works")
    else:
        print("❌ Test 3: Second generate_iq_batch call FAILS ⚠️ BUG DETECTED")
        all_pass = False
    
    if all_pass:
        print("\n🎉 ALL TESTS PASSED - No bug detected!")
        return 0
    else:
        print("\n🔴 BUG CONFIRMED - Second batch call breaks audio consistency")
        return 1

if __name__ == "__main__":
    sys.exit(main())
