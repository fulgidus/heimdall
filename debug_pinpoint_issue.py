#!/usr/bin/env python3
"""
Pinpoint exactly where correlation breaks in generate_iq_batch.
Tests after each operation: normalize_power, add_awgn.
"""

import numpy as np
import sys
import os

# Add services paths
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

from data.iq_generator import SyntheticIQGenerator
import structlog

logger = structlog.get_logger()

def extract_audio(iq_signal, sample_rate=200000):
    """Extract audio from FM-modulated IQ signal."""
    # FM demodulation using instantaneous phase
    inst_phase = np.unwrap(np.angle(iq_signal))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * sample_rate
    
    # Normalize
    audio = inst_freq / (np.max(np.abs(inst_freq)) + 1e-10)
    return audio

def compute_correlation(audio1, audio2):
    """Compute correlation between two audio signals."""
    corr = np.corrcoef(audio1, audio2)[0, 1]
    return corr

print("=" * 70)
print("PINPOINT WHERE CORRELATION BREAKS")
print("=" * 70)

# Initialize generator
generator = SyntheticIQGenerator(
    sample_rate_hz=200000,
    duration_ms=1000,
    use_gpu=False,
    use_audio_library=True,
    audio_library_fallback=True,
    seed=42
)

batch_size = 7

# Test parameters (zero frequency offsets, high SNR)
freq_offsets = np.zeros(batch_size, dtype=np.float32)
bandwidths = np.full(batch_size, 12500.0, dtype=np.float32)
signal_powers = np.full(batch_size, -20.0, dtype=np.float32)
noise_floors = np.full(batch_size, -100.0, dtype=np.float32)
snr_dbs = np.full(batch_size, 50.0, dtype=np.float32)

print("\n1. Generate clean signals")
print("-" * 70)
batch_clean = generator._generate_clean_signal_batch(freq_offsets, bandwidths, batch_size)
print(f"Clean batch shape: {batch_clean.shape}")

# Check correlation after clean generation
audio_0 = extract_audio(batch_clean[0])
audio_1 = extract_audio(batch_clean[1])
corr_clean = compute_correlation(audio_0, audio_1)
print(f"Correlation after clean generation: {corr_clean:.6f} {'✅' if corr_clean > 0.99 else '❌'}")

print("\n2. Apply normalize_power to each receiver")
print("-" * 70)
batch_normalized = np.copy(batch_clean)
for i in range(batch_size):
    batch_normalized[i] = generator._normalize_power(batch_normalized[i], float(signal_powers[i]))

audio_0 = extract_audio(batch_normalized[0])
audio_1 = extract_audio(batch_normalized[1])
corr_normalized = compute_correlation(audio_0, audio_1)
print(f"Correlation after normalize_power: {corr_normalized:.6f} {'✅' if corr_normalized > 0.99 else '❌'}")

print("\n3. Apply add_awgn to each receiver")
print("-" * 70)
batch_noisy = np.copy(batch_normalized)
for i in range(batch_size):
    batch_noisy[i] = generator._add_awgn(
        batch_noisy[i],
        noise_floor_dbm=float(noise_floors[i]),
        snr_db=float(snr_dbs[i])
    )

audio_0 = extract_audio(batch_noisy[0])
audio_1 = extract_audio(batch_noisy[1])
corr_noisy = compute_correlation(audio_0, audio_1)
print(f"Correlation after add_awgn (50 dB SNR): {corr_noisy:.6f} {'✅' if corr_noisy > 0.95 else '❌'}")

print("\n4. Compare with generate_iq_batch output")
print("-" * 70)
batch_full = generator.generate_iq_batch(
    frequency_offsets=freq_offsets,
    bandwidths=bandwidths,
    signal_powers_dbm=signal_powers,
    noise_floors_dbm=noise_floors,
    snr_dbs=snr_dbs,
    batch_size=batch_size,
    enable_multipath=False,
    enable_fading=False
)

audio_0 = extract_audio(batch_full[0])
audio_1 = extract_audio(batch_full[1])
corr_full = compute_correlation(audio_0, audio_1)
print(f"Correlation from generate_iq_batch: {corr_full:.6f} {'✅' if corr_full > 0.95 else '❌'}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if corr_clean < 0.99:
    print("❌ BUG: Correlation breaks at clean signal generation")
elif corr_normalized < 0.99:
    print("❌ BUG: Correlation breaks at normalize_power")
elif corr_noisy < 0.95:
    print("⚠️  Correlation reduced by AWGN (expected with noise)")
    if corr_full < 0.95:
        print("   But generate_iq_batch has SAME correlation, so this is expected behavior")
    else:
        print("   But generate_iq_batch has DIFFERENT correlation - something else is wrong!")
else:
    print("✅ All manual steps preserve correlation")
    if corr_full < 0.95:
        print("❌ But generate_iq_batch breaks it - must be in the wrapper logic")
    else:
        print("✅ generate_iq_batch also preserves correlation - ALL GOOD!")
