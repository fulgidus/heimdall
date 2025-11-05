#!/usr/bin/env python3
"""
Test if generate_iq_batch regenerates different audio on each call.
"""

import numpy as np
import sys

# Add services paths
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

from data.iq_generator import SyntheticIQGenerator

def extract_audio(iq_signal, sample_rate=200000):
    """Extract audio from FM-modulated IQ signal."""
    inst_phase = np.unwrap(np.angle(iq_signal))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * sample_rate
    audio = inst_freq / (np.max(np.abs(inst_freq)) + 1e-10)
    return audio

def compute_correlation(audio1, audio2):
    """Compute correlation between two audio signals."""
    corr = np.corrcoef(audio1, audio2)[0, 1]
    return corr

print("=" * 70)
print("TEST: Does generate_iq_batch regenerate audio on each call?")
print("=" * 70)

# Initialize generator with FIXED SEED
generator = SyntheticIQGenerator(
    sample_rate_hz=200000,
    duration_ms=1000,
    use_gpu=False,
    use_audio_library=True,
    audio_library_fallback=True,
    seed=42  # Fixed seed for reproducibility
)

batch_size = 7
freq_offsets = np.zeros(batch_size, dtype=np.float32)
bandwidths = np.full(batch_size, 12500.0, dtype=np.float32)
signal_powers = np.full(batch_size, -20.0, dtype=np.float32)
noise_floors = np.full(batch_size, -100.0, dtype=np.float32)
snr_dbs = np.full(batch_size, 50.0, dtype=np.float32)

print("\n1. Generate first batch")
print("-" * 70)
batch1 = generator.generate_iq_batch(
    frequency_offsets=freq_offsets,
    bandwidths=bandwidths,
    signal_powers_dbm=signal_powers,
    noise_floors_dbm=noise_floors,
    snr_dbs=snr_dbs,
    batch_size=batch_size,
    enable_multipath=False,
    enable_fading=False
)

audio1_0 = extract_audio(batch1[0])
audio1_1 = extract_audio(batch1[1])
corr_within_batch1 = compute_correlation(audio1_0, audio1_1)
print(f"Batch 1 - Correlation between receivers 0 and 1: {corr_within_batch1:.6f}")

print("\n2. Generate second batch (same parameters, same seed)")
print("-" * 70)
batch2 = generator.generate_iq_batch(
    frequency_offsets=freq_offsets,
    bandwidths=bandwidths,
    signal_powers_dbm=signal_powers,
    noise_floors_dbm=noise_floors,
    snr_dbs=snr_dbs,
    batch_size=batch_size,
    enable_multipath=False,
    enable_fading=False
)

audio2_0 = extract_audio(batch2[0])
audio2_1 = extract_audio(batch2[1])
corr_within_batch2 = compute_correlation(audio2_0, audio2_1)
print(f"Batch 2 - Correlation between receivers 0 and 1: {corr_within_batch2:.6f}")

print("\n3. Cross-batch comparison")
print("-" * 70)
corr_across_batches = compute_correlation(audio1_0, audio2_0)
print(f"Batch 1 receiver 0 vs Batch 2 receiver 0: {corr_across_batches:.6f}")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print(f"Within-batch correlation (batch 1): {corr_within_batch1:.6f}")
print(f"Within-batch correlation (batch 2): {corr_within_batch2:.6f}")
print(f"Cross-batch correlation: {corr_across_batches:.6f}")

if corr_within_batch1 > 0.95 and corr_within_batch2 > 0.95:
    print("\n✅ PASS: Both batches have high within-batch correlation")
else:
    print("\n❌ FAIL: Low within-batch correlation indicates audio inconsistency")
    
if corr_across_batches > 0.95:
    print("✅ PASS: Same audio generated across batches (good seed reproducibility)")
else:
    print("⚠️  INFO: Different audio across batches (expected with random audio library)")
