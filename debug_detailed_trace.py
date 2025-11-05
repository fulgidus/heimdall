#!/usr/bin/env python3
"""
Detailed debug script to trace exactly where audio diverges in the batch.
"""

import sys
import os
import numpy as np

# Add path for imports
training_path = os.path.join(os.path.dirname(__file__), 'services', 'training')
sys.path.insert(0, training_path)

try:
    from src.data.iq_generator import SyntheticIQGenerator  # type: ignore
except ImportError:
    sys.path.insert(0, os.path.join(training_path, 'src'))
    from data.iq_generator import SyntheticIQGenerator  # type: ignore

def extract_audio_from_iq(iq_signal):
    """Extract audio by FM demodulation."""
    phase = np.unwrap(np.angle(iq_signal))
    audio = np.diff(phase)
    return audio

def compute_correlation(audio1, audio2):
    """Compute normalized correlation between two audio signals."""
    a1_norm = (audio1 - np.mean(audio1)) / (np.std(audio1) + 1e-8)
    a2_norm = (audio2 - np.mean(audio2)) / (np.std(audio2) + 1e-8)
    return np.corrcoef(a1_norm, a2_norm)[0, 1]

def main():
    print("\n" + "="*70)
    print("DETAILED BATCH GENERATION DEBUG")
    print("="*70)
    
    # Match integration test parameters exactly
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False,
        use_audio_library=True,
        audio_library_fallback=True
    )
    
    batch_size = 7  # Match integration test
    frequency_offsets = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    bandwidths = np.array([12500.0] * batch_size)
    signal_powers_dbm = np.array([-37.0] * batch_size)
    noise_floors_dbm = np.array([-87.0] * batch_size)
    snr_dbs = signal_powers_dbm - noise_floors_dbm
    
    print("\nStep 1: Generate first batch WITH effects")
    print("-" * 70)
    
    batch1 = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=True,
        enable_fading=True
    )
    
    print(f"Batch 1 shape: {batch1.shape}")
    
    print("\nStep 2: Generate second batch WITHOUT effects (ZERO frequency offsets)")
    print("-" * 70)
    print("This matches the integration test 'clean batch' configuration")
    
    batch2 = generator.generate_iq_batch(
        frequency_offsets=np.zeros(batch_size),  # ⚠️ ZERO offsets
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,
        enable_fading=False
    )
    
    print(f"Batch 2 shape: {batch2.shape}")
    
    print("\nStep 3: Extract audio from batch 2 (should be identical across receivers)")
    print("-" * 70)
    
    audios = []
    for i in range(batch_size):
        audio = extract_audio_from_iq(batch2[i])
        audios.append(audio)
        print(f"  Receiver {i}: mean={np.mean(audio):.6f}, std={np.std(audio):.6f}")
    
    print("\nStep 4: Compute correlations (should be >0.99)")
    print("-" * 70)
    
    ref_audio = audios[0]
    for i in range(1, batch_size):
        corr = compute_correlation(ref_audio, audios[i])
        status = "✅ PASS" if corr > 0.99 else "❌ FAIL"
        print(f"  Receiver 0 vs {i}: correlation = {corr:.6f} {status}")
    
    print("\nStep 5: Direct test of _generate_clean_signal_batch")
    print("-" * 70)
    print("This bypasses generate_iq_batch and tests the core method directly")
    
    batch3 = generator._generate_clean_signal_batch(
        np.zeros(batch_size),
        bandwidths,
        batch_size
    )
    
    print(f"Batch 3 shape: {batch3.shape}")
    
    audios3 = []
    for i in range(batch_size):
        audio = extract_audio_from_iq(batch3[i])
        audios3.append(audio)
    
    print("\nCorrelations from direct call:")
    ref_audio3 = audios3[0]
    for i in range(1, batch_size):
        corr = compute_correlation(ref_audio3, audios3[i])
        status = "✅ PASS" if corr > 0.99 else "❌ FAIL"
        print(f"  Receiver 0 vs {i}: correlation = {corr:.6f} {status}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    # Check if noise was added
    print("\nChecking if noise is the issue:")
    print(f"  Batch 2 includes AWGN: Yes (SNR = 50 dB)")
    print(f"  Batch 3 includes AWGN: No (direct clean signal)")
    
    print("\nConclusion:")
    print("  If Batch 3 has high correlation but Batch 2 has low correlation,")
    print("  then the issue is in the generate_iq_batch wrapper,")
    print("  likely in how effects (multipath, fading, normalization, AWGN) are applied.")

if __name__ == "__main__":
    sys.exit(main())
