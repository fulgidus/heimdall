#!/usr/bin/env python3
"""
GPU Acceleration Benchmark for Synthetic Sample Generation

Compares CPU vs GPU performance for:
1. IQ sample generation (SyntheticIQGenerator)
2. Feature extraction (RFFeatureExtractor)
3. End-to-end synthetic sample generation

Expected speedup:
- IQ generation: 5-10x (GPU-accelerated FFT convolution + RNG)
- Feature extraction: 10-30x (GPU-accelerated FFT operations)
- End-to-end: 5-15x (depends on bottleneck)
"""

import sys
import os
import time
import numpy as np
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.training.src.data.iq_generator import SyntheticIQGenerator
from services.common.feature_extraction.rf_feature_extractor import RFFeatureExtractor, IQSample

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        gpu_memory = cp.cuda.Device(0).mem_info[1] / 1024**3  # Total memory in GB
    else:
        gpu_name = "N/A"
        gpu_memory = 0
except ImportError:
    GPU_AVAILABLE = False
    gpu_name = "N/A"
    gpu_memory = 0


def benchmark_iq_generation(num_samples: int = 100, duration_ms: float = 1000.0):
    """
    Benchmark IQ sample generation (CPU vs GPU).
    
    Args:
        num_samples: Number of IQ samples to generate
        duration_ms: Duration of each sample
    
    Returns:
        Tuple of (cpu_time, gpu_time, speedup)
    """
    print(f"\n{'='*60}")
    print(f"Benchmark 1: IQ Sample Generation")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}, Duration: {duration_ms}ms")
    print()
    
    # CPU benchmark
    print("CPU: Generating samples...")
    generator_cpu = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=duration_ms,
        seed=42,
        use_gpu=False
    )
    
    start_cpu = time.time()
    for i in range(num_samples):
        _ = generator_cpu.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id=f"rx_{i}",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=float(i),
            enable_multipath=True,
            enable_fading=True
        )
    cpu_time = time.time() - start_cpu
    cpu_per_sample = cpu_time / num_samples * 1000  # ms
    
    print(f"  Total: {cpu_time:.2f}s")
    print(f"  Per sample: {cpu_per_sample:.1f}ms")
    print()
    
    if not GPU_AVAILABLE:
        print("GPU: Not available")
        return cpu_time, None, None
    
    # GPU benchmark
    print("GPU: Generating samples...")
    generator_gpu = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=duration_ms,
        seed=42,
        use_gpu=True
    )
    
    # Warm-up (first call may be slower due to kernel compilation)
    _ = generator_gpu.generate_iq_sample(
        center_frequency_hz=145_000_000,
        signal_power_dbm=-80,
        noise_floor_dbm=-120,
        snr_db=15.0,
        frequency_offset_hz=25.0,
        bandwidth_hz=12500,
        rx_id="warmup",
        rx_lat=45.0,
        rx_lon=9.0,
        timestamp=0.0,
        enable_multipath=True,
        enable_fading=True
    )
    
    start_gpu = time.time()
    for i in range(num_samples):
        _ = generator_gpu.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id=f"rx_{i}",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=float(i),
            enable_multipath=True,
            enable_fading=True
        )
    gpu_time = time.time() - start_gpu
    gpu_per_sample = gpu_time / num_samples * 1000  # ms
    
    print(f"  Total: {gpu_time:.2f}s")
    print(f"  Per sample: {gpu_per_sample:.1f}ms")
    print()
    
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x")
    
    return cpu_time, gpu_time, speedup


def benchmark_feature_extraction(num_samples: int = 100):
    """
    Benchmark feature extraction (CPU vs GPU).
    
    Args:
        num_samples: Number of feature extractions to perform
    
    Returns:
        Tuple of (cpu_time, gpu_time, speedup)
    """
    print(f"\n{'='*60}")
    print(f"Benchmark 2: Feature Extraction")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}")
    print()
    
    # Generate test IQ samples (CPU)
    print("Generating test IQ samples...")
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False
    )
    
    iq_samples_list = []
    for i in range(num_samples):
        synth_sample = generator.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id=f"rx_{i}",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=float(i)
        )
        
        iq_sample = IQSample(
            samples=synth_sample.samples,
            sample_rate_hz=int(synth_sample.sample_rate_hz),
            center_frequency_hz=int(synth_sample.center_frequency_hz),
            rx_id=synth_sample.rx_id,
            rx_lat=synth_sample.rx_lat,
            rx_lon=synth_sample.rx_lon,
            timestamp=datetime.fromtimestamp(synth_sample.timestamp, tz=timezone.utc)
        )
        iq_samples_list.append(iq_sample)
    
    print(f"  Generated {len(iq_samples_list)} IQ samples")
    print()
    
    # CPU benchmark
    print("CPU: Extracting features...")
    extractor_cpu = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=False)
    
    start_cpu = time.time()
    for iq_sample in iq_samples_list:
        _ = extractor_cpu.extract_features(iq_sample)
    cpu_time = time.time() - start_cpu
    cpu_per_sample = cpu_time / num_samples * 1000  # ms
    
    print(f"  Total: {cpu_time:.2f}s")
    print(f"  Per sample: {cpu_per_sample:.1f}ms")
    print()
    
    if not GPU_AVAILABLE:
        print("GPU: Not available")
        return cpu_time, None, None
    
    # GPU benchmark
    print("GPU: Extracting features...")
    extractor_gpu = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=True)
    
    # Warm-up
    _ = extractor_gpu.extract_features(iq_samples_list[0])
    
    start_gpu = time.time()
    for iq_sample in iq_samples_list:
        _ = extractor_gpu.extract_features(iq_sample)
    gpu_time = time.time() - start_gpu
    gpu_per_sample = gpu_time / num_samples * 1000  # ms
    
    print(f"  Total: {gpu_time:.2f}s")
    print(f"  Per sample: {gpu_per_sample:.1f}ms")
    print()
    
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x")
    
    return cpu_time, gpu_time, speedup


def benchmark_chunked_extraction(num_samples: int = 50):
    """
    Benchmark chunked feature extraction (5x200ms chunks per sample).
    
    Args:
        num_samples: Number of chunked extractions to perform
    
    Returns:
        Tuple of (cpu_time, gpu_time, speedup)
    """
    print(f"\n{'='*60}")
    print(f"Benchmark 3: Chunked Feature Extraction (5x200ms)")
    print(f"{'='*60}")
    print(f"Samples: {num_samples}")
    print()
    
    # Generate test IQ samples (CPU)
    print("Generating test IQ samples...")
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False
    )
    
    iq_samples_list = []
    for i in range(num_samples):
        synth_sample = generator.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id=f"rx_{i}",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=float(i)
        )
        
        iq_sample = IQSample(
            samples=synth_sample.samples,
            sample_rate_hz=int(synth_sample.sample_rate_hz),
            center_frequency_hz=int(synth_sample.center_frequency_hz),
            rx_id=synth_sample.rx_id,
            rx_lat=synth_sample.rx_lat,
            rx_lon=synth_sample.rx_lon,
            timestamp=datetime.fromtimestamp(synth_sample.timestamp, tz=timezone.utc)
        )
        iq_samples_list.append(iq_sample)
    
    print(f"  Generated {len(iq_samples_list)} IQ samples")
    print()
    
    # CPU benchmark
    print("CPU: Chunked feature extraction...")
    extractor_cpu = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=False)
    
    start_cpu = time.time()
    for iq_sample in iq_samples_list:
        _ = extractor_cpu.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )
    cpu_time = time.time() - start_cpu
    cpu_per_sample = cpu_time / num_samples * 1000  # ms
    
    print(f"  Total: {cpu_time:.2f}s")
    print(f"  Per sample: {cpu_per_sample:.1f}ms")
    print()
    
    if not GPU_AVAILABLE:
        print("GPU: Not available")
        return cpu_time, None, None
    
    # GPU benchmark
    print("GPU: Chunked feature extraction...")
    extractor_gpu = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=True)
    
    # Warm-up
    _ = extractor_gpu.extract_features_chunked(
        iq_samples_list[0],
        chunk_duration_ms=200.0,
        num_chunks=5
    )
    
    start_gpu = time.time()
    for iq_sample in iq_samples_list:
        _ = extractor_gpu.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )
    gpu_time = time.time() - start_gpu
    gpu_per_sample = gpu_time / num_samples * 1000  # ms
    
    print(f"  Total: {gpu_time:.2f}s")
    print(f"  Per sample: {gpu_per_sample:.1f}ms")
    print()
    
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x")
    
    return cpu_time, gpu_time, speedup


def print_summary(results: dict):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    if GPU_AVAILABLE:
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("GPU: Not available (CPU-only mode)")
    
    print()
    print(f"{'Benchmark':<40} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*10}")
    
    for name, (cpu_time, gpu_time, speedup) in results.items():
        cpu_str = f"{cpu_time:.2f}s" if cpu_time else "N/A"
        gpu_str = f"{gpu_time:.2f}s" if gpu_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        print(f"{name:<40} {cpu_str:<12} {gpu_str:<12} {speedup_str:<10}")
    
    print()
    
    if GPU_AVAILABLE:
        avg_speedup = np.mean([s for _, _, s in results.values() if s is not None])
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print()
        
        if avg_speedup < 2.0:
            print("⚠️  WARNING: Low GPU speedup detected!")
            print("   - Check that CuPy is properly installed")
            print("   - Verify GPU is not throttling (temperature, power)")
            print("   - Ensure CUDA drivers are up to date")
        elif avg_speedup >= 10.0:
            print("✅ EXCELLENT: GPU acceleration working optimally!")
        else:
            print("✅ GOOD: GPU acceleration providing significant speedup")


def main():
    """Run all benchmarks."""
    print("="*60)
    print("GPU ACCELERATION BENCHMARK - Synthetic Sample Generation")
    print("="*60)
    print()
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if GPU_AVAILABLE:
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("GPU: Not available (CPU-only mode)")
    
    # Run benchmarks
    results = {}
    
    # 1. IQ Generation (100 samples)
    cpu_time, gpu_time, speedup = benchmark_iq_generation(num_samples=100, duration_ms=1000.0)
    results["IQ Generation (100 samples x 1000ms)"] = (cpu_time, gpu_time, speedup)
    
    # 2. Feature Extraction (100 samples)
    cpu_time, gpu_time, speedup = benchmark_feature_extraction(num_samples=100)
    results["Feature Extraction (100 samples)"] = (cpu_time, gpu_time, speedup)
    
    # 3. Chunked Extraction (50 samples, 5 chunks each)
    cpu_time, gpu_time, speedup = benchmark_chunked_extraction(num_samples=50)
    results["Chunked Extraction (50 samples x 5 chunks)"] = (cpu_time, gpu_time, speedup)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
