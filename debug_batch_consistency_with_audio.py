#!/usr/bin/env python3
"""
Debug script testing with audio library enabled.
This should reproduce the bug if it's audio library related.
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
    print("DEBUGGING BATCH CONSISTENCY ISSUE - WITH AUDIO LIBRARY")
    print("="*70)
    
    # Create generator WITH audio library
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False,
        use_audio_library=True,  # ⚠️ ENABLE AUDIO LIBRARY
        audio_library_fallback=True
    )
    
    batch_size = 3
    frequency_offsets = np.zeros(batch_size)
    bandwidths = np.array([12500.0] * batch_size)
    signal_powers_dbm = np.array([-37.0] * batch_size)
    noise_floors_dbm = np.array([-87.0] * batch_size)
    snr_dbs = signal_powers_dbm - noise_floors_dbm
    
    print("\nTest 1: First generate_iq_batch() call (with audio library)")
    print("-" * 70)
    
    batch1 = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,
        enable_fading=False
    )
    
    audio1_0 = extract_audio_from_iq(batch1[0])
    audio1_1 = extract_audio_from_iq(batch1[1])
    audio1_2 = extract_audio_from_iq(batch1[2])
    
    corr_1_0vs1 = compute_correlation(audio1_0, audio1_1)
    corr_1_0vs2 = compute_correlation(audio1_0, audio1_2)
    
    print(f"  Receiver 0 vs 1: correlation = {corr_1_0vs1:.6f}")
    print(f"  Receiver 0 vs 2: correlation = {corr_1_0vs2:.6f}")
    print(f"  Result: {'✅ IDENTICAL' if corr_1_0vs1 > 0.99 else '❌ DIFFERENT'}")
    
    print("\nTest 2: Second generate_iq_batch() call (same generator)")
    print("-" * 70)
    print("This is where the bug should appear if audio library caching is broken")
    
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
    print(f"  Result: {'✅ IDENTICAL' if corr_2_0vs1 > 0.99 else '❌ DIFFERENT - BUG DETECTED!'}")
    
    print("\nTest 3: Compare batch1 vs batch2")
    print("-" * 70)
    
    corr_batch1_vs_batch2 = compute_correlation(audio1_0, audio2_0)
    print(f"  Batch1[0] vs Batch2[0]: correlation = {corr_batch1_vs_batch2:.6f}")
    print(f"  Expected: LOW (different audio content)")
    print(f"  Result: {'✅ CORRECT' if corr_batch1_vs_batch2 < 0.8 else '⚠️ UNEXPECTED'}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = True
    
    if corr_1_0vs1 > 0.99:
        print("✅ Test 1: First batch works (audio library)")
    else:
        print("❌ Test 1: First batch FAILS")
        all_pass = False
    
    if corr_2_0vs1 > 0.99:
        print("✅ Test 2: Second batch works")
    else:
        print("❌ Test 2: Second batch FAILS ⚠️ AUDIO LIBRARY CACHING BUG")
        all_pass = False
    
    if all_pass:
        print("\n🎉 ALL TESTS PASSED - Audio library working correctly!")
        return 0
    else:
        print("\n🔴 BUG CONFIRMED - Audio library caching issue detected")
        print("\nDiagnosis:")
        print("  - Formant synthesis works (previous test passed)")
        print("  - Audio library breaks on second batch")
        print("  - Likely cause: Audio loader caching stale samples")
        return 1

if __name__ == "__main__":
    sys.exit(main())
