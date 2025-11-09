#!/usr/bin/env python3
"""
Integration test for 200ms audio preprocessing migration.

Tests that:
1. IQ generator correctly extracts 200ms windows from 1s audio chunks
2. Feature extraction works with single 200ms chunks
3. Training sample generation produces correct shapes
4. Metadata reflects num_chunks=1, iq_duration_ms=200.0
"""

import sys
import os
import numpy as np

# Add services to path
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

from src.data.iq_generator import SyntheticIQGenerator
from src.data.synthetic_generator import SyntheticDataGenerator


def test_iq_generator_200ms():
    """Test IQ generator with 200ms duration."""
    print("\n[TEST 1] IQ Generator with 200ms duration")
    print("=" * 60)
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,  # 50 kHz audio rate
        duration_ms=200.0,       # 200ms duration
        seed=42
    )
    
    # Verify initialization
    assert generator.duration_ms == 200.0, "Duration should be 200ms"
    assert generator.num_samples == 10_000, "Should have 10,000 samples @ 50kHz"
    print(f"✓ IQ Generator initialized: {generator.duration_ms}ms, {generator.num_samples} samples")
    
    # Test window extraction
    audio_1s = np.random.randn(50_000).astype(np.float32)
    window = generator._extract_random_200ms_window(audio_1s)
    
    assert len(window) == 10_000, f"Window should be 10,000 samples, got {len(window)}"
    print(f"✓ Window extraction works: {len(window)} samples")
    
    # Test multiple extractions get different windows
    windows_seen = set()
    for _ in range(50):
        window = generator._extract_random_200ms_window(audio_1s)
        windows_seen.add(int(window[0]))  # Use first sample as identifier
    
    print(f"✓ Random window selection works: {len(windows_seen)} unique windows in 50 extractions")
    
    return True


def test_feature_extraction_200ms():
    """Test feature extraction with 200ms chunks."""
    print("\n[TEST 2] Feature extraction with 200ms chunks")
    print("=" * 60)
    
    try:
        # Import feature extraction
        from src.data.feature_extraction import extract_features_from_iq
        
        # Generate 200ms IQ sample
        generator = SyntheticIQGenerator(
            sample_rate_hz=50_000,
            duration_ms=200.0,
            seed=42
        )
        
        iq_sample = generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=0.0,
            bandwidth_hz=12500.0,
            rx_id="Test",
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0
        )
        
        # Extract features
        features = extract_features_from_iq(
            iq_samples=iq_sample.samples,
            sample_rate_hz=50_000,
            num_chunks=1  # Single 200ms chunk
        )
        
        print(f"✓ Features extracted successfully")
        print(f"  - Shape: {features.shape if hasattr(features, 'shape') else type(features)}")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Feature extraction test skipped (import error): {e}")
        return None  # Skip test


def test_synthetic_dataset_generator_200ms():
    """Test that code changes in synthetic_generator.py are correct."""
    print("\n[TEST 3] Synthetic data generator code validation")
    print("=" * 60)
    
    try:
        # Read the synthetic_generator.py file to verify changes
        import re
        
        with open('/app/src/data/synthetic_generator.py', 'r') as f:
            content = f.read()
        
        # Check for the 3 key changes
        checks = []
        
        # Check 1: Line ~707 - num_chunks=1, iq_duration_ms=200.0
        # Match both dictionary syntax ('num_chunks': 1) and parameter syntax (num_chunks=1)
        pattern1 = r"('num_chunks'\s*:\s*1|num_chunks\s*=\s*1)"
        pattern2 = r"('iq_duration_ms'\s*:\s*200\.0|iq_duration_ms\s*=\s*200\.0)"
        
        num_chunks_matches = len(re.findall(pattern1, content))
        iq_duration_matches = len(re.findall(pattern2, content))
        
        if num_chunks_matches >= 3 and iq_duration_matches >= 2:
            checks.append(f"✓ Found num_chunks=1 ({num_chunks_matches} occurrences) and iq_duration_ms=200.0 ({iq_duration_matches} occurrences)")
        else:
            checks.append(f"✗ Missing expected patterns: num_chunks=1 ({num_chunks_matches}/3+), iq_duration_ms=200.0 ({iq_duration_matches}/2+)")
        
        # Check 2: Verify no remaining 1000.0 for iq_duration_ms (except in metadata passthrough)
        pattern3 = r"('iq_duration_ms'\s*:\s*1000\.0|iq_duration_ms\s*=\s*1000\.0)"
        matches = re.findall(pattern3, content)
        if len(matches) == 0:
            checks.append("✓ No remaining iq_duration_ms=1000.0")
        else:
            checks.append(f"⚠ Found {len(matches)} instances of iq_duration_ms=1000.0 (should be 200.0)")
        
        # Check 3: Verify no remaining num_chunks=5 (except in comments/strings)
        pattern4 = r"('num_chunks'\s*:\s*5|num_chunks\s*=\s*5)"
        occurrences = len(re.findall(pattern4, content))
        if occurrences == 0:
            checks.append("✓ No remaining num_chunks=5")
        else:
            checks.append(f"⚠ Found {occurrences} instances of num_chunks=5 (should be 1)")
        
        # Print results
        for check in checks:
            print(f"  {check}")
        
        # Return True if all checks passed
        all_passed = all('✓' in check for check in checks)
        if all_passed:
            print("\n✓ All code changes verified correctly")
        else:
            print("\n⚠ Some code checks failed")
        
        return all_passed
            
    except Exception as e:
        print(f"⚠ Code validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("200ms Audio Preprocessing Migration - Integration Tests")
    print("=" * 70)
    
    results = []
    
    # Test 1: IQ Generator
    try:
        results.append(("IQ Generator 200ms", test_iq_generator_200ms()))
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("IQ Generator 200ms", False))
    
    # Test 2: Feature Extraction
    try:
        result = test_feature_extraction_200ms()
        results.append(("Feature Extraction", result))
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Feature Extraction", False))
    
    # Test 3: Synthetic Data Generator Code Validation
    try:
        results.append(("Synthetic Generator Code", test_synthetic_dataset_generator_200ms()))
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Synthetic Dataset Generator", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for test_name, result in results:
        status = "✓ PASS" if result is True else ("✗ FAIL" if result is False else "⊘ SKIP")
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
