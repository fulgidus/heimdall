"""Test if CuPy global seed is causing issues"""
import numpy as np
import sys
sys.path.insert(0, '/app/training/src')
from test_200ms_unit_tests import *

print("\n=== Testing CuPy Global Seed Hypothesis ===")

audio_rng = np.random.default_rng(seed=999)
audio_1s = audio_rng.standard_normal(50_000).astype(np.float32)

# Test 1: Both with GPU
print("\nTest 1: Both generators with GPU=True")
gen1_gpu = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=False, use_gpu=True)
gen2_gpu = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=False, use_gpu=True)
w1 = gen1_gpu._extract_random_200ms_window(audio_1s)
w2 = gen2_gpu._extract_random_200ms_window(audio_1s)
print(f"  gen1_gpu[0]={w1[0]:.4f}, gen2_gpu[0]={w2[0]:.4f}, match={np.array_equal(w1, w2)}")

# Test 2: Both with CPU
print("\nTest 2: Both generators with GPU=False")
gen1_cpu = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=False, use_gpu=False)
gen2_cpu = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42, use_audio_library=False, use_gpu=False)
w1 = gen1_cpu._extract_random_200ms_window(audio_1s)
w2 = gen2_cpu._extract_random_200ms_window(audio_1s)
print(f"  gen1_cpu[0]={w1[0]:.4f}, gen2_cpu[0]={w2[0]:.4f}, match={np.array_equal(w1, w2)}")
