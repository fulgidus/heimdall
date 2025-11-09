"""Test full window extraction sequence"""
import numpy as np
import sys
sys.path.insert(0, '/app/training/src')
from test_200ms_unit_tests import *

print("\n=== Full Sequence Test ===")

# Use seeded audio (same as in test)
audio_rng = np.random.default_rng(seed=999)
audio_1s = audio_rng.standard_normal(50_000).astype(np.float32)

gen1 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)
gen2 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)

print("\nExtracting windows (showing first sample value):")
for i in range(5):
    w1 = gen1._extract_random_200ms_window(audio_1s)
    w2 = gen2._extract_random_200ms_window(audio_1s)
    match = "✓ MATCH" if np.array_equal(w1, w2) else "✗ DIFFERENT"
    print(f"  Call {i}: gen1[0]={w1[0]:.4f}, gen2[0]={w2[0]:.4f} {match}")
