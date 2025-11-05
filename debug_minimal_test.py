#!/usr/bin/env python3
"""
Minimal test to identify the bug.
"""

import numpy as np
import sys

sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

from data.iq_generator import SyntheticIQGenerator

def extract_audio(iq_signal):
    """Extract audio from FM-modulated IQ signal."""
    inst_phase = np.unwrap(np.angle(iq_signal))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * 200000
    audio = inst_freq / (np.max(np.abs(inst_freq)) + 1e-10)
    return audio

print("Testing audio consistency bug...")

# Initialize generator
gen = SyntheticIQGenerator(
    sample_rate_hz=200000,
    duration_ms=1000,
    use_gpu=False,
    use_audio_library=False,  # Use formant synthesis for reproducibility
    seed=42
)

batch_size = 7
freq_offsets = np.zeros(batch_size, dtype=np.float32)
bandwidths = np.full(batch_size, 12500.0, dtype=np.float32)
signal_powers = np.full(batch_size, -20.0, dtype=np.float32)
noise_floors = np.full(batch_size, -100.0, dtype=np.float32)
snr_dbs = np.full(batch_size, 50.0, dtype=np.float32)

print(f"\nTest 1: Direct call to _generate_clean_signal_batch")
batch1 = gen._generate_clean_signal_batch(freq_offsets, bandwidths, batch_size)
audio1_0 = extract_audio(batch1[0])
audio1_1 = extract_audio(batch1[1])
corr1 = np.corrcoef(audio1_0, audio1_1)[0, 1]
print(f"Correlation: {corr1:.6f} {'✅' if corr1 > 0.99 else '❌'}")

print(f"\nTest 2: Call through generate_iq_batch")
batch2 = gen.generate_iq_batch(
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
corr2 = np.corrcoef(audio2_0, audio2_1)[0, 1]
print(f"Correlation: {corr2:.6f} {'✅' if corr2 > 0.99 else '❌'}")

print(f"\nResult: {'✅ PASS' if corr2 > 0.99 else '❌ FAIL'}")
