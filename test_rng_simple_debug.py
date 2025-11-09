"""Quick RNG debug in test context"""
import sys
sys.path.insert(0, '/app/training/src')
from test_200ms_unit_tests import *

print("\n=== Simple RNG Debug ===")
gen1 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)
print(f"Gen1 created. Calling rng_integers(0, 5) directly: {gen1.rng_integers(0, 5)}")

gen2 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)
print(f"Gen2 created. Calling rng_integers(0, 5) directly: {gen2.rng_integers(0, 5)}")

print("\nIf both print same number, CPU RNG is properly isolated.")
print("If different, then something during __init__ advanced the RNG state.")
