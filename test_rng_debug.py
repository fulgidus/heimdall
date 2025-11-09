"""
Debug script to trace RNG state during SyntheticIQGenerator initialization.
"""
import numpy as np
import sys
sys.path.insert(0, '/app/src')

from data.iq_generator import SyntheticIQGenerator

print("\n=== RNG State Debug ===\n")

print("Creating gen1...")
gen1 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)

print("\nCreating gen2...")
gen2 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)

print("\nTesting window selection:")
audio = np.arange(50_000, dtype=np.float32)

print("Gen1 selections:")
for i in range(5):
    w = gen1._extract_random_200ms_window(audio)
    print(f"  Call {i}: window starts at {int(w[0])}")

print("\nGen2 selections:")
for i in range(5):
    w = gen2._extract_random_200ms_window(audio)
    print(f"  Call {i}: window starts at {int(w[0])}")

print("\n=== Testing if selections match ===")
print("If gen1 and gen2 produce identical sequences, reproducibility works!")
