#!/usr/bin/env python3
"""Test CPU parallelization scaling with different sample counts."""

import sys
import numpy as np
import time

sys.path.insert(0, 'services/common')
from feature_extraction.rf_feature_extractor import RFFeatureExtractor, IQSample


def generate_iq_samples(num_samples: int) -> list:
    """Generate test IQ samples."""
    samples = []
    for i in range(num_samples):
        iq_data = np.random.randn(40960) + 1j * np.random.randn(40960)
        iq_data = iq_data.astype(np.complex64)
        
        iq_sample = IQSample(
            samples=iq_data,
            sample_rate_hz=200000,
            center_frequency_hz=145000000,
            rx_id=f"RX{i:03d}",
            rx_lat=45.0 + (i * 0.01),
            rx_lon=9.0 + (i * 0.01),
            timestamp=None
        )
        samples.append(iq_sample)
    return samples


def benchmark_cpu(num_samples: int):
    """Benchmark CPU mode with specified number of samples."""
    print(f"\n{'='*70}")
    print(f"TESTING WITH {num_samples} SAMPLES")
    print(f"{'='*70}")
    
    # Generate samples
    print("Generating samples...")
    samples = generate_iq_samples(num_samples)
    
    # Initialize extractor
    extractor = RFFeatureExtractor(sample_rate_hz=200000, use_gpu=False)
    
    # Warm-up
    print("Warm-up...")
    _ = extractor.extract_features_batch_conservative(samples[:2], num_chunks=5)
    
    # Benchmark
    print("Running benchmark...")
    start = time.perf_counter()
    features = extractor.extract_features_batch_conservative(samples, num_chunks=5)
    elapsed = time.perf_counter() - start
    
    samples_per_sec = num_samples / elapsed
    ms_per_sample = (elapsed / num_samples) * 1000
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Samples/sec: {samples_per_sec:.2f}")
    print(f"  Time per sample: {ms_per_sample:.2f}ms")
    print(f"  Features extracted: {len(features)}")


if __name__ == "__main__":
    for num_samples in [50, 100, 200]:
        benchmark_cpu(num_samples)
