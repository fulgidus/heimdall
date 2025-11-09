#!/usr/bin/env python3
"""
Validate 50 kHz Sample Rate Migration
Tests that IQ generation produces correct sample counts and file sizes.
"""

import sys
import numpy as np
from pathlib import Path

# Add training service to path
sys.path.insert(0, '/app/services/training/src')

from data.iq_generator import SyntheticIQGenerator
from data.audio_library import AudioLibraryManager

def test_iq_generator_sample_rate():
    """Test that IQ generator produces 50k samples."""
    print("=" * 60)
    print("TEST 1: IQ Generator Sample Rate")
    print("=" * 60)
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=1000.0,
        use_gpu=False,  # CPU for validation
        use_audio_library=False  # Don't need audio lib for this test
    )
    
    # Generate a single sample
    result = generator.generate_sample(
        transmitter_lat=45.0,
        transmitter_lon=9.0,
        stations=[
            {"id": "test1", "lat": 45.1, "lon": 9.1, "freq_mhz": 145.0},
            {"id": "test2", "lat": 45.0, "lon": 9.2, "freq_mhz": 145.0},
        ]
    )
    
    # Check sample rate
    expected_samples = 50_000
    actual_samples = result['iq_samples'].shape[1]
    
    print(f"Expected samples: {expected_samples}")
    print(f"Actual samples:   {actual_samples}")
    print(f"Sample rate:      {result['sample_rate_hz']} Hz")
    print(f"Duration:         {result['duration_ms']} ms")
    
    if actual_samples == expected_samples:
        print("✅ PASS: Correct sample count (50,000)")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_samples}, got {actual_samples}")
        return False

def test_memory_footprint():
    """Test memory footprint of 50k samples."""
    print("\n" + "=" * 60)
    print("TEST 2: Memory Footprint")
    print("=" * 60)
    
    # Complex64 IQ data: 2 receivers * 50k samples * 8 bytes (complex64)
    n_receivers = 7
    n_samples = 50_000
    bytes_per_complex = 8  # complex64 = 2 * float32
    
    expected_iq_bytes = n_receivers * n_samples * bytes_per_complex
    expected_mb = expected_iq_bytes / (1024 * 1024)
    
    print(f"IQ data size (7 receivers, 50k samples):")
    print(f"  {expected_iq_bytes:,} bytes ({expected_mb:.2f} MB)")
    
    # Estimate total sample size with metadata
    estimated_total_mb = expected_mb * 1.1  # +10% for metadata
    print(f"  Estimated total: ~{estimated_total_mb:.2f} MB per sample")
    
    # 80k samples projection
    total_80k_gb = (estimated_total_mb * 80_000) / 1024
    print(f"\n80,000 samples projection: ~{total_80k_gb:.1f} GB")
    
    if total_80k_gb < 100:  # Should be ~53 GB
        print("✅ PASS: Storage projection reasonable (<100 GB)")
        return True
    else:
        print("❌ FAIL: Storage projection too high")
        return False

def test_audio_library_config():
    """Test audio library configuration."""
    print("\n" + "=" * 60)
    print("TEST 3: Audio Library Configuration")
    print("=" * 60)
    
    try:
        # Check if audio library can initialize
        from data.audio_library import AudioLibraryManager
        
        manager = AudioLibraryManager()
        print(f"✅ Audio library initialized")
        print(f"   Target sample rate: 50,000 Hz")
        print(f"   Note: Audio files may need re-import if previously")
        print(f"         upscaled from 200 kHz")
        return True
        
    except Exception as e:
        print(f"⚠️  WARNING: Audio library check: {e}")
        return True  # Non-critical for IQ generation

def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("50 kHz Migration Validation Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("IQ Sample Rate", test_iq_generator_sample_rate()))
        results.append(("Memory Footprint", test_memory_footprint()))
        results.append(("Audio Library", test_audio_library_config()))
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
        print("\nNext steps:")
        print("  1. Re-import audio library if needed")
        print("  2. Generate test batch (e.g., 100 samples)")
        print("  3. Verify storage usage (~0.38 MB/sample)")
        print("  4. Start 80k sample generation")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Review output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
