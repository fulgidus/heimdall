#!/usr/bin/env python3
"""
Test script to validate 50 kHz sample rate migration.
Verifies that IQ generation produces correct sample counts and storage savings.
"""
import sys
import os

# Add services paths
sys.path.insert(0, '/app/services/training/src')
sys.path.insert(0, '/app/services/common')

from data.iq_generator import SyntheticIQGenerator

def test_50khz_generation():
    """Test that IQ generator works correctly at 50 kHz."""
    print("=" * 60)
    print("🔬 TESTING 50 kHz IQ GENERATION")
    print("=" * 60)
    
    # Initialize generator
    print("\n1. Initializing generator...")
    gen = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=1000.0,
        use_gpu=False,
        use_audio_library=False  # Disable for quick test
    )
    print(f"   ✅ Sample rate: {gen.sample_rate_hz:,} Hz")
    print(f"   ✅ Duration: {gen.duration_ms} ms")
    
    # Generate test sample
    print("\n2. Generating test IQ sample...")
    result = gen.generate(
        tx_lat=45.0,
        tx_lon=10.0,
        tx_power_dbm=30.0,
        rx_lat=45.1,
        rx_lon=10.1
    )
    
    # Validate results
    print("\n3. Validation Results:")
    expected_samples = int(50_000 * 1.0)  # 50k Hz * 1 sec
    actual_samples = len(result.samples)
    size_mb = result.samples.nbytes / (1024 * 1024)
    old_size_mb = 200_000 * 8 / (1024 * 1024)  # 200k samples * 8 bytes (complex64)
    reduction_pct = (1 - (actual_samples / 200_000)) * 100
    
    print(f"   Expected samples: {expected_samples:,}")
    print(f"   Actual samples: {actual_samples:,}")
    print(f"   Match: {'✅ YES' if actual_samples == expected_samples else '❌ NO'}")
    print(f"   Size per sample: {size_mb:.3f} MB")
    print(f"   Old size (200kHz): {old_size_mb:.3f} MB")
    print(f"   Reduction: {reduction_pct:.1f}%")
    
    # Storage projections
    print("\n4. Storage Projections (80,000 samples):")
    total_size_gb = (size_mb * 80_000) / 1024
    old_total_gb = (old_size_mb * 80_000) / 1024
    savings_gb = old_total_gb - total_size_gb
    
    print(f"   New (50 kHz): {total_size_gb:.1f} GB")
    print(f"   Old (200 kHz): {old_total_gb:.1f} GB")
    print(f"   Savings: {savings_gb:.1f} GB ({(savings_gb/old_total_gb)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    if actual_samples == expected_samples:
        print("🎉 50 kHz MIGRATION SUCCESSFUL!")
    else:
        print("⚠️  WARNING: Sample count mismatch")
    print("=" * 60)
    
    return actual_samples == expected_samples

if __name__ == "__main__":
    success = test_50khz_generation()
    sys.exit(0 if success else 1)
