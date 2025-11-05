#!/usr/bin/env python3
"""
Test CPU parallel feature extraction with multiprocessing.

This script tests the new multiprocessing-based CPU feature extraction
and compares performance with GPU mode.

Expected results:
- CPU mode should use all 24 cores (visible in htop)
- CPU mode should be faster than GPU for small batches with the new parallelization
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from services.common.feature_extraction.rf_feature_extractor import (
    RFFeatureExtractor,
    IQSample
)
from datetime import datetime

def generate_test_iq_samples(num_samples: int, sample_rate_hz: int = 200_000) -> list:
    """
    Generate synthetic IQ samples for testing.
    
    Args:
        num_samples: Number of IQ samples to generate
        sample_rate_hz: Sample rate in Hz
        
    Returns:
        List of IQSample objects
    """
    iq_samples = []
    
    for i in range(num_samples):
        # Generate 1 second of IQ data (200k samples)
        samples = np.random.randn(sample_rate_hz) + 1j * np.random.randn(sample_rate_hz)
        samples = samples.astype(np.complex64)
        
        iq_sample = IQSample(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            center_frequency_hz=145_000_000,  # 145 MHz (2m band)
            rx_id=f"TEST_RX_{i:03d}",
            rx_lat=45.0 + i * 0.01,
            rx_lon=9.0 + i * 0.01,
            timestamp=datetime.now()
        )
        iq_samples.append(iq_sample)
    
    return iq_samples


def benchmark_feature_extraction(use_gpu: bool, num_samples: int = 50, num_chunks: int = 5):
    """
    Benchmark feature extraction with GPU or CPU mode.
    
    Args:
        use_gpu: Use GPU if True, CPU if False
        num_samples: Number of samples to process
        num_chunks: Number of chunks per sample
        
    Returns:
        Tuple of (total_time_sec, samples_per_sec)
    """
    mode = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*70}")
    print(f"Feature Extraction Benchmark - {mode} Mode")
    print(f"{'='*70}")
    print(f"Samples: {num_samples}")
    print(f"Chunks per sample: {num_chunks}")
    print(f"Sample rate: 200 kHz")
    print()
    
    # Generate test data
    print("Generating test IQ samples...")
    iq_samples = generate_test_iq_samples(num_samples)
    print(f"  Generated {len(iq_samples)} samples")
    print()
    
    # Initialize feature extractor
    print(f"Initializing {mode} feature extractor...")
    extractor = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=use_gpu)
    print(f"  Extractor initialized (GPU={extractor.use_gpu})")
    print()
    
    # Warm-up run (for GPU kernel compilation or CPU cache)
    print("Warm-up run...")
    _ = extractor.extract_features_batch_conservative(
        iq_samples_list=iq_samples[:2],
        chunk_duration_ms=200.0,
        num_chunks=num_chunks
    )
    print("  Warm-up complete")
    print()
    
    # Benchmark run
    print(f"Running benchmark with {mode} mode...")
    print(f"Monitor CPU usage with: watch -n 0.1 'ps -eo pid,comm,%cpu --sort=-%cpu | head -30'")
    print()
    
    start_time = time.perf_counter()
    
    features_list = extractor.extract_features_batch_conservative(
        iq_samples_list=iq_samples,
        chunk_duration_ms=200.0,
        num_chunks=num_chunks
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Calculate statistics
    samples_per_sec = num_samples / total_time
    ms_per_sample = (total_time / num_samples) * 1000
    
    print()
    print(f"{'='*70}")
    print(f"RESULTS - {mode} Mode")
    print(f"{'='*70}")
    print(f"Total samples: {num_samples}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Samples/sec: {samples_per_sec:.2f}")
    print(f"Time per sample: {ms_per_sample:.2f}ms")
    print(f"Features extracted: {len(features_list)}")
    print()
    
    return total_time, samples_per_sec


def main():
    """Run CPU vs GPU comparison benchmark."""
    print("="*70)
    print("CPU PARALLEL FEATURE EXTRACTION TEST")
    print("="*70)
    print()
    
    # Check system info
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    print(f"System info:")
    print(f"  CPU cores: {num_cpus}")
    print()
    
    # Check GPU availability
    try:
        import cupy as cp
        gpu_available = cp.cuda.is_available()
        if gpu_available:
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
            print(f"  GPU: {gpu_name}")
        else:
            print(f"  GPU: Not available")
    except ImportError:
        gpu_available = False
        print(f"  GPU: CuPy not installed")
    print()
    
    num_samples = 50
    
    # Test 1: CPU mode (with new parallelization)
    print("Test 1: CPU Mode (with multiprocessing)")
    cpu_time, cpu_sps = benchmark_feature_extraction(use_gpu=False, num_samples=num_samples)
    
    # Test 2: GPU mode (if available)
    if gpu_available:
        print("Test 2: GPU Mode")
        gpu_time, gpu_sps = benchmark_feature_extraction(use_gpu=True, num_samples=num_samples)
        
        # Comparison
        print()
        print(f"{'='*70}")
        print(f"COMPARISON")
        print(f"{'='*70}")
        print(f"CPU time: {cpu_time:.2f}s ({cpu_sps:.2f} samples/sec)")
        print(f"GPU time: {gpu_time:.2f}s ({gpu_sps:.2f} samples/sec)")
        print()
        
        speedup = gpu_time / cpu_time
        if speedup > 1:
            print(f"CPU is {speedup:.2f}x FASTER than GPU")
        else:
            print(f"GPU is {1/speedup:.2f}x FASTER than CPU")
        
        percent_diff = ((gpu_time - cpu_time) / gpu_time) * 100
        print(f"CPU is {percent_diff:+.1f}% vs GPU")
        print()
    else:
        print()
        print("GPU not available - skipping GPU benchmark")
        print()
    
    print("="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    main()
