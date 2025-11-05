#!/usr/bin/env python3
"""Debug script to trace audio loading behavior."""

import sys
import os
import numpy as np

# Container import path
sys.path.insert(0, '/app')
from src.data.iq_generator import SyntheticIQGenerator

def test_audio_trace():
    """Trace exactly what happens during audio loading."""
    print("="*70)
    print("AUDIO LOADING TRACE")
    print("="*70)
    
    # Create generator with audio library
    print("\nCreating IQ generator...")
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False,
        use_audio_library=True,
        audio_library_fallback=True
    )
    
    batch_size = 7
    frequency_offsets = np.zeros(batch_size)  # Zero offsets
    bandwidths = np.array([12500.0] * batch_size)
    signal_powers_dbm = np.array([-37.0] * batch_size)
    noise_floors_dbm = np.array([-87.0] * batch_size)
    snr_dbs = signal_powers_dbm - noise_floors_dbm
    
    print(f"\n{'='*70}")
    print("BATCH 1: First call to generate_iq_batch")
    print('='*70)
    
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
    
    print(f"\nBatch 1 shape: {batch1.shape}")
    print(f"Batch 1 sample 0 mean: {np.mean(np.abs(batch1[0])):.6f}")
    print(f"Batch 1 sample 1 mean: {np.mean(np.abs(batch1[1])):.6f}")
    
    # Extract audio from each receiver
    def extract_audio(iq_signal):
        phase = np.unwrap(np.angle(iq_signal))
        audio = np.diff(phase)
        return audio
    
    print("\nExtracting audio from batch 1...")
    audios1 = [extract_audio(batch1[i]) for i in range(batch_size)]
    
    # Compute correlations
    print("\nBatch 1 internal correlations:")
    ref = audios1[0]
    ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
    
    for i in range(1, batch_size):
        audio = audios1[i]
        audio_norm = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
        corr = np.corrcoef(ref_norm, audio_norm)[0, 1]
        print(f"  Receiver {i+1} vs Receiver 1: {corr:.6f}")

if __name__ == "__main__":
    test_audio_trace()
