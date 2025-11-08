#!/usr/bin/env python3
"""
Full Vectorization Benchmark - Batch IQ Generation

Tests the performance of the fully vectorized batch processing implementation
with all for-loops eliminated (convolution, fading, multipath).

Expected results:
- Batch size 1024: 2,000-10,000 samples/sec
- GPU utilization: >90%
- Time for 100k samples: 10-50 seconds
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.training.src.data.iq_generator import SyntheticIQGenerator

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


def benchmark_batch_processing(batch_size: int, num_batches: int = 10):
    """
    Benchmark fully vectorized batch IQ generation.
    
    Args:
        batch_size: Number of samples per batch
        num_batches: Number of batches to generate
    
    Returns:
        Tuple of (total_time, samples_per_sec, gpu_utilization)
    """
    if not GPU_AVAILABLE:
        print("ERROR: GPU not available. This benchmark requires CUDA.")
        return None, None, None
    
    print(f"\n{'='*70}")
    print(f"Batch Processing Benchmark - Fully Vectorized")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    print(f"Num batches: {num_batches}")
    print(f"Total samples: {batch_size * num_batches}")
    print()
    
    # Initialize GPU generator
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=True
    )
    
    # Prepare batch parameters (same distribution as real generation)
    np.random.seed(42)
    frequency_offsets = np.random.uniform(-50, 50, batch_size).astype(np.float32)
    bandwidths = np.random.uniform(10_000, 15_000, batch_size).astype(np.float32)
    signal_powers_dbm = np.random.uniform(-90, -70, batch_size).astype(np.float32)
    noise_floors_dbm = np.random.uniform(-130, -110, batch_size).astype(np.float32)
    snr_dbs = np.random.uniform(10, 20, batch_size).astype(np.float32)
    
    # Warm-up (kernel compilation)
    print("Warming up GPU kernels...")
    _ = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size
    )
    print("  Warm-up complete\n")
    
    # Benchmark
    print("Running benchmark...")
    batch_times = []
    
    for batch_idx in range(num_batches):
        # Randomize parameters for each batch
        frequency_offsets = np.random.uniform(-50, 50, batch_size).astype(np.float32)
        bandwidths = np.random.uniform(10_000, 15_000, batch_size).astype(np.float32)
        signal_powers_dbm = np.random.uniform(-90, -70, batch_size).astype(np.float32)
        noise_floors_dbm = np.random.uniform(-130, -110, batch_size).astype(np.float32)
        snr_dbs = np.random.uniform(10, 20, batch_size).astype(np.float32)
        
        start = time.time()
        batch_signals = generator.generate_iq_batch(
            frequency_offsets=frequency_offsets,
            bandwidths=bandwidths,
            signal_powers_dbm=signal_powers_dbm,
            noise_floors_dbm=noise_floors_dbm,
            snr_dbs=snr_dbs,
            batch_size=batch_size
        )
        
        # Synchronize GPU to get accurate timing
        if hasattr(batch_signals, 'get'):  # CuPy array
            cp.cuda.Stream.null.synchronize()
        
        batch_time = time.time() - start
        batch_times.append(batch_time)
        
        samples_per_sec = batch_size / batch_time
        print(f"  Batch {batch_idx+1}/{num_batches}: {batch_time:.3f}s ({samples_per_sec:.1f} samples/sec)")
    
    # Calculate statistics
    total_samples = batch_size * num_batches
    total_time = sum(batch_times)
    avg_samples_per_sec = total_samples / total_time
    
    print()
    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average: {avg_samples_per_sec:.1f} samples/sec")
    print(f"Per-sample latency: {1000.0 / avg_samples_per_sec:.2f}ms")
    print()
    
    # Estimate time for 100k samples
    time_for_100k = 100_000 / avg_samples_per_sec
    print(f"Estimated time for 100,000 samples: {time_for_100k:.1f}s ({time_for_100k/60:.1f} min)")
    print()
    
    # Check GPU memory usage
    if GPU_AVAILABLE:
        mem_used = (cp.cuda.Device(0).mem_info[1] - cp.cuda.Device(0).mem_info[0]) / 1024**3
        mem_percent = mem_used / gpu_memory * 100
        print(f"GPU Memory Usage: {mem_used:.2f} GB / {gpu_memory:.1f} GB ({mem_percent:.1f}%)")
        print()
    
    return total_time, avg_samples_per_sec, None


def compare_batch_sizes():
    """
    Compare performance across different batch sizes to find optimal setting.
    """
    print(f"\n{'='*70}")
    print(f"Batch Size Comparison")
    print(f"{'='*70}")
    print()
    
    batch_sizes = [256, 512, 1024, 2048]
    results = []
    
    for batch_size in batch_sizes:
        try:
            total_time, samples_per_sec, _ = benchmark_batch_processing(
                batch_size=batch_size,
                num_batches=5
            )
            results.append((batch_size, total_time, samples_per_sec))
        except Exception as e:
            print(f"ERROR: Batch size {batch_size} failed: {e}")
            results.append((batch_size, None, None))
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"BATCH SIZE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Batch Size':<15} {'Total Time':<15} {'Samples/Sec':<20} {'100k Time':<15}")
    print(f"{'-'*15} {'-'*15} {'-'*20} {'-'*15}")
    
    for batch_size, total_time, samples_per_sec in results:
        if total_time is None:
            print(f"{batch_size:<15} {'FAILED':<15} {'N/A':<20} {'N/A':<15}")
        else:
            time_100k = 100_000 / samples_per_sec
            print(f"{batch_size:<15} {total_time:.2f}s{'':<9} {samples_per_sec:.1f}{'':<12} {time_100k:.1f}s{'':<10}")
    
    # Find optimal batch size
    valid_results = [(bs, sps) for bs, tt, sps in results if sps is not None]
    if valid_results:
        optimal_batch, optimal_sps = max(valid_results, key=lambda x: x[1])
        print()
        print(f"Optimal batch size: {optimal_batch} ({optimal_sps:.1f} samples/sec)")
        print()


def main():
    """Run all benchmarks."""
    print("="*70)
    print("FULL VECTORIZATION BENCHMARK - Batch IQ Generation")
    print("="*70)
    print()
    
    if not GPU_AVAILABLE:
        print("ERROR: GPU not available. This benchmark requires CUDA.")
        print("Install CuPy: pip install cupy-cuda11x (replace 11x with your CUDA version)")
        return
    
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print()
    
    # Test 1: Single batch size (1024)
    print("Test 1: Large batch (1024 samples)")
    benchmark_batch_processing(batch_size=1024, num_batches=10)
    
    # Test 2: Compare different batch sizes
    compare_batch_sizes()
    
    print()
    print("="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    main()
