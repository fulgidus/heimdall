"""Test with audio library disabled to isolate the issue"""
import numpy as np
import sys
sys.path.insert(0, '/app/training/src')
from test_200ms_unit_tests import *

print("\n=== Test WITHOUT Audio Library ===")

audio_rng = np.random.default_rng(seed=999)
audio_1s = audio_rng.standard_normal(50_000).astype(np.float32)

# Disable audio library during initialization
gen1 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=False)
gen2 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=False)

print("\nExtracting windows (NO audio library):")
for i in range(5):
    w1 = gen1._extract_random_200ms_window(audio_1s)
    w2 = gen2._extract_random_200ms_window(audio_1s)
    match = "✓ MATCH" if np.array_equal(w1, w2) else "✗ DIFFERENT"
    print(f"  Call {i}: gen1[0]={w1[0]:.4f}, gen2[0]={w2[0]:.4f} {match}")

print("\n=== Test WITH Audio Library ===")

gen3 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=True)
gen4 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=True)

print("\nExtracting windows (WITH audio library):")
for i in range(5):
    w3 = gen3._extract_random_200ms_window(audio_1s)
    w4 = gen4._extract_random_200ms_window(audio_1s)
    match = "✓ MATCH" if np.array_equal(w3, w4) else "✗ DIFFERENT"
    print(f"  Call {i}: gen3[0]={w3[0]:.4f}, gen4[0]={w4[0]:.4f} {match}")
