"""
Performance benchmarks for feature extraction pipeline.

Tests performance targets:
- IQ generation: < 50ms per sample
- Feature extraction: < 100ms per sample  
- End-to-end: < 150ms per sample
"""

import pytest
import time
from statistics import mean, stdev

from src.data.iq_generator import SyntheticIQGenerator
from src.data.rf_feature_extractor import RFFeatureExtractor


@pytest.mark.performance
def test_iq_generation_performance():
    """Benchmark IQ sample generation speed."""
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42
    )

    times = []
    for i in range(100):
        start = time.perf_counter()

        generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=-15.0,
            bandwidth_hz=12500.0,
            rx_id='Test',
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0 + i  # Vary timestamp
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = mean(times) * 1000  # Convert to ms
    std_time = stdev(times) * 1000
    min_time = min(times) * 1000
    max_time = max(times) * 1000

    print(f"\nIQ Generation Performance:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  StdDev:  {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")

    # Performance requirement: < 50ms per sample
    assert avg_time < 50.0, f"IQ generation too slow: {avg_time:.2f}ms (target: <50ms)"


@pytest.mark.performance
def test_feature_extraction_performance():
    """Benchmark feature extraction speed."""
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42
    )
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    # Generate test IQ samples (reuse to isolate extraction performance)
    iq_samples = []
    for i in range(10):
        iq_sample = generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=-15.0 + i * 5,  # Vary offset
            bandwidth_hz=12500.0,
            rx_id='Test',
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0 + i
        )
        iq_samples.append(iq_sample)

    times = []
    for i in range(100):
        # Cycle through test samples
        iq_sample = iq_samples[i % len(iq_samples)]
        
        start = time.perf_counter()

        extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = mean(times) * 1000
    std_time = stdev(times) * 1000
    min_time = min(times) * 1000
    max_time = max(times) * 1000

    print(f"\nFeature Extraction Performance:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  StdDev:  {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")

    # Performance requirement: < 100ms per sample
    assert avg_time < 100.0, f"Feature extraction too slow: {avg_time:.2f}ms (target: <100ms)"


@pytest.mark.performance
def test_end_to_end_performance():
    """Benchmark full pipeline (IQ generation + feature extraction)."""
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42
    )
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    times = []
    for i in range(50):
        start = time.perf_counter()

        # Generate IQ
        iq_sample = generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=-15.0 + (i % 10) * 3,
            bandwidth_hz=12500.0,
            rx_id='Test',
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0 + i
        )

        # Extract features
        extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = mean(times) * 1000
    std_time = stdev(times) * 1000
    min_time = min(times) * 1000
    max_time = max(times) * 1000

    print(f"\nEnd-to-End Performance:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  StdDev:  {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")

    # Performance requirement: < 150ms per sample
    assert avg_time < 150.0, f"End-to-end too slow: {avg_time:.2f}ms (target: <150ms)"

    # Calculate throughput
    throughput = 1000.0 / avg_time  # Samples per second
    print(f"  Throughput: {throughput:.1f} samples/second")


@pytest.mark.performance
@pytest.mark.slow
def test_batch_generation_performance():
    """Benchmark performance of batch generation (simulating multiprocessing scenario)."""
    from src.data.synthetic_generator import _generate_single_sample
    
    receivers_list = [
        {'name': f'RX_{i}', 'latitude': 45.0 + i*0.1, 'longitude': 7.0 + i*0.1, 'altitude': 300.0}
        for i in range(3)
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.5, 'lat_max': 45.5,
            'lon_min': 6.5, 'lon_max': 7.5
        },
        'training_bbox': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 6.0, 'lon_max': 8.0
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 37.0,
        'min_snr_db': 3.0,
        'min_receivers': 2,
        'max_gdop': 50.0,
        'inside_ratio': 0.7
    }

    # Time generating 10 samples sequentially
    num_samples = 10
    start = time.perf_counter()
    
    for i in range(num_samples):
        args = (i, receivers_list, training_config_dict, config, 42 + i)
        _generate_single_sample(args)
    
    elapsed = time.perf_counter() - start
    avg_time_per_sample = (elapsed / num_samples) * 1000

    print(f"\nBatch Generation Performance (Sequential):")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Per sample: {avg_time_per_sample:.2f} ms")
    print(f"  Samples/sec: {num_samples/elapsed:.1f}")
    
    # With 3 receivers, should still be reasonably fast
    # Allow 500ms per sample (includes 3x IQ generation + 3x feature extraction)
    assert avg_time_per_sample < 500.0, f"Batch generation too slow: {avg_time_per_sample:.2f}ms per sample"


@pytest.mark.performance
def test_feature_extraction_chunked_vs_single():
    """Compare performance of chunked vs single-shot feature extraction."""
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42
    )
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    # Generate test IQ
    iq_sample = generator.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=-15.0,
        bandwidth_hz=12500.0,
        rx_id='Test',
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0
    )

    # Benchmark single-shot extraction
    single_times = []
    for _i in range(50):
        start = time.perf_counter()
        extractor.extract_features(iq_sample)
        elapsed = time.perf_counter() - start
        single_times.append(elapsed)

    # Benchmark chunked extraction (5 chunks of 200ms)
    chunked_times = []
    for _i in range(50):
        start = time.perf_counter()
        extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )
        elapsed = time.perf_counter() - start
        chunked_times.append(elapsed)

    single_avg = mean(single_times) * 1000
    chunked_avg = mean(chunked_times) * 1000

    print(f"\nSingle vs Chunked Extraction:")
    print(f"  Single-shot: {single_avg:.2f} ms")
    print(f"  Chunked (5x): {chunked_avg:.2f} ms")
    print(f"  Overhead: {chunked_avg - single_avg:.2f} ms ({((chunked_avg/single_avg - 1)*100):.1f}%)")

    # Chunked should be slower but not excessively (< 3x overhead)
    assert chunked_avg < single_avg * 3.0, "Chunked extraction has excessive overhead"


@pytest.mark.performance
def test_memory_usage():
    """Test memory usage stays reasonable during generation."""
    import sys
    
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42
    )
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    # Generate and extract features for multiple samples
    samples = []
    for i in range(10):
        iq_sample = generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=-15.0 + i * 2,
            bandwidth_hz=12500.0,
            rx_id='Test',
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0 + i
        )
        
        features = extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )
        
        samples.append((iq_sample, features))

    # Calculate memory usage
    total_size = 0
    for iq_sample, features in samples:
        # IQ samples: complex64 = 8 bytes per sample
        iq_size = iq_sample.samples.nbytes
        # Features: dict with stats (rough estimate)
        features_size = sys.getsizeof(str(features))
        total_size += iq_size + features_size

    size_mb = total_size / (1024 * 1024)
    per_sample_mb = size_mb / len(samples)

    print(f"\nMemory Usage:")
    print(f"  Total for {len(samples)} samples: {size_mb:.2f} MB")
    print(f"  Per sample: {per_sample_mb:.2f} MB")

    # Each sample should be ~1.6 MB (200k samples × 8 bytes + overhead)
    # Allow up to 3 MB per sample
    assert per_sample_mb < 3.0, f"Excessive memory usage: {per_sample_mb:.2f} MB per sample"
