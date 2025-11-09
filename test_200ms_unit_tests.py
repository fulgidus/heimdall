#!/usr/bin/env python3
"""
Standalone unit test runner for 200ms window extraction.
Runs the 5 unit tests without requiring pytest.
"""

import sys
import numpy as np

# Add services to path
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

from src.data.iq_generator import SyntheticIQGenerator


def test_extract_random_200ms_window_shape():
    """Test that extracted window has correct shape."""
    print("\n[TEST] test_extract_random_200ms_window_shape")
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=200.0,
        seed=42
    )
    
    audio_1s = np.random.randn(50_000).astype(np.float32)
    window = generator._extract_random_200ms_window(audio_1s)
    
    assert len(window) == 10_000, f"Expected 10,000 samples, got {len(window)}"
    assert window.dtype == np.float32, f"Expected float32, got {window.dtype}"
    print("✓ PASS - Window shape correct: 10,000 samples, float32")
    return True


def test_extract_random_200ms_window_all_windows_accessible():
    """Test that all 5 windows can be extracted."""
    print("\n[TEST] test_extract_random_200ms_window_all_windows_accessible")
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=200.0,
        seed=None  # No seed for true randomness
    )
    
    audio_1s = np.arange(50_000, dtype=np.float32)  # Use sequential values
    windows_seen = set()
    
    # Try 1000 times to see all 5 windows
    for _ in range(1000):
        window = generator._extract_random_200ms_window(audio_1s)
        # Identify window by its first sample
        first_sample = int(window[0])
        if first_sample == 0:
            windows_seen.add(0)
        elif first_sample == 10_000:
            windows_seen.add(1)
        elif first_sample == 20_000:
            windows_seen.add(2)
        elif first_sample == 30_000:
            windows_seen.add(3)
        elif first_sample == 40_000:
            windows_seen.add(4)
    
    assert len(windows_seen) == 5, f"Expected all 5 windows, got {len(windows_seen)}: {windows_seen}"
    print(f"✓ PASS - All 5 windows accessible: {sorted(windows_seen)}")
    return True


def test_extract_random_200ms_window_correct_indices():
    """Test that window start indices are correct."""
    print("\n[TEST] test_extract_random_200ms_window_correct_indices")
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=200.0,
        seed=None
    )
    
    # Use sequential values so we can identify windows
    audio_1s = np.arange(50_000, dtype=np.float32)
    
    # Expected first samples for each window
    expected_starts = {0, 10_000, 20_000, 30_000, 40_000}
    
    # Extract 100 windows and verify they start at expected indices
    for _ in range(100):
        window = generator._extract_random_200ms_window(audio_1s)
        first_sample = int(window[0])
        assert first_sample in expected_starts, f"Invalid start index: {first_sample}"
        
        # Verify window is contiguous
        for i in range(len(window)):
            assert window[i] == first_sample + i, f"Window not contiguous at index {i}"
    
    print("✓ PASS - All extracted windows start at correct indices")
    return True


def test_extract_random_200ms_window_seeded_reproducibility():
    """Test that seeded RNG produces reproducible windows."""
    print("\n[TEST] test_extract_random_200ms_window_seeded_reproducibility")
    
    # IMPORTANT: Use seeded RNG to create test audio for reproducibility
    # If we used global np.random.randn(), the audio would be different each time
    audio_rng = np.random.default_rng(seed=999)
    audio_1s = audio_rng.standard_normal(50_000).astype(np.float32)
    
    # Generate two sequences with same seed
    gen1 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)
    gen2 = SyntheticIQGenerator(sample_rate_hz=50_000, duration_ms=200.0, seed=42)
    
    windows1 = [gen1._extract_random_200ms_window(audio_1s) for _ in range(10)]
    windows2 = [gen2._extract_random_200ms_window(audio_1s) for _ in range(10)]
    
    for i, (w1, w2) in enumerate(zip(windows1, windows2)):
        assert np.array_equal(w1, w2), f"Window {i} differs between seeded generators"
    
    print("✓ PASS - Seeded generators produce identical sequences")
    return True


def test_extract_random_200ms_window_distribution():
    """Test that window selection has roughly uniform distribution."""
    print("\n[TEST] test_extract_random_200ms_window_distribution")
    generator = SyntheticIQGenerator(
        sample_rate_hz=50_000,
        duration_ms=200.0,
        seed=None
    )
    
    audio_1s = np.arange(50_000, dtype=np.float32)
    
    # Count how often each window is selected
    window_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    n_samples = 5000
    
    for _ in range(n_samples):
        window = generator._extract_random_200ms_window(audio_1s)
        first_sample = int(window[0])
        
        if first_sample == 0:
            window_counts[0] += 1
        elif first_sample == 10_000:
            window_counts[1] += 1
        elif first_sample == 20_000:
            window_counts[2] += 1
        elif first_sample == 30_000:
            window_counts[3] += 1
        elif first_sample == 40_000:
            window_counts[4] += 1
    
    # Each window should appear ~1000 times (20% of 5000)
    # Allow ±15% deviation (850-1150)
    expected = n_samples / 5
    tolerance = 0.15
    
    for window_idx, count in window_counts.items():
        deviation = abs(count - expected) / expected
        assert deviation < tolerance, f"Window {window_idx} distribution skewed: {count}/{n_samples} (expected ~{expected})"
    
    print(f"✓ PASS - Distribution uniform: {window_counts}")
    return True


def main():
    """Run all unit tests."""
    print("=" * 70)
    print("Unit Tests: 200ms Window Extraction")
    print("=" * 70)
    
    tests = [
        test_extract_random_200ms_window_shape,
        test_extract_random_200ms_window_all_windows_accessible,
        test_extract_random_200ms_window_correct_indices,
        test_extract_random_200ms_window_seeded_reproducibility,
        test_extract_random_200ms_window_distribution,
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    
    for test_name, result in results:
        status = "✓ PASS" if result is True else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
