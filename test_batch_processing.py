#!/usr/bin/env python3
"""
Test fully vectorized batch processing performance.
Tests the implementation with FFT convolution, vectorized fading, and advanced indexing multipath.
"""
import time
import sys
import numpy as np
sys.path.insert(0, '/app')

from src.data.iq_generator import SyntheticIQGenerator

def benchmark_batch_generation(batch_size=1024, num_batches=10):
    """Benchmark the fully vectorized batch generation."""
    print(f"\n{'='*70}")
    print(f"FULL VECTORIZATION BENCHMARK")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    print(f"Num batches: {num_batches}")
    print(f"Total samples: {batch_size * num_batches}")
    print(f"{'='*70}\n")
    
    # Initialize generator with GPU
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=True
    )
    
    print(f"GPU enabled: {generator.use_gpu}")
    if generator.use_gpu:
        import cupy as cp
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        gpu_memory = cp.cuda.Device(0).mem_info[1] / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Warmup (initialize GPU kernels)
    print("\nWarming up GPU kernels...")
    warmup_freq = np.random.uniform(-50, 50, batch_size).astype(np.float32)
    warmup_bw = np.random.uniform(10_000, 15_000, batch_size).astype(np.float32)
    warmup_pow = np.random.uniform(-90, -70, batch_size).astype(np.float32)
    warmup_noise = np.random.uniform(-130, -110, batch_size).astype(np.float32)
    warmup_snr = np.random.uniform(10, 20, batch_size).astype(np.float32)
    
    _ = generator.generate_iq_batch(
        frequency_offsets=warmup_freq,
        bandwidths=warmup_bw,
        signal_powers_dbm=warmup_pow,
        noise_floors_dbm=warmup_noise,
        snr_dbs=warmup_snr,
        batch_size=batch_size
    )
    print("Warmup complete\n")
    
    # Benchmark
    print(f"Running {num_batches} batches...")
    batch_times = []
    
    for batch_idx in range(num_batches):
        # Generate random parameters for each batch
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
        
        # Synchronize GPU for accurate timing
        if generator.use_gpu and hasattr(batch_signals, 'get'):
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        
        batch_time = time.time() - start
        batch_times.append(batch_time)
        
        samples_per_sec = batch_size / batch_time
        print(f"  Batch {batch_idx+1:>2}/{num_batches}: {batch_time:>6.3f}s ({samples_per_sec:>8.1f} samples/sec)")
    
    # Calculate statistics
    total_samples = batch_size * num_batches
    total_time = sum(batch_times)
    avg_samples_per_sec = total_samples / total_time
    min_batch_time = min(batch_times)
    max_batch_time = max(batch_times)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total samples:        {total_samples}")
    print(f"Total time:           {total_time:.2f}s")
    print(f"Average throughput:   {avg_samples_per_sec:.1f} samples/sec")
    print(f"Per-sample latency:   {1000.0 / avg_samples_per_sec:.2f}ms")
    print(f"Batch time (min/max): {min_batch_time:.3f}s / {max_batch_time:.3f}s")
    
    # Estimate time for 100k samples
    time_for_100k = 100_000 / avg_samples_per_sec
    print(f"\nEstimated time for 100,000 samples: {time_for_100k:.1f}s ({time_for_100k/60:.1f} min)")
    
    # Check GPU memory usage
    if generator.use_gpu:
        import cupy as cp
        mem_used = (cp.cuda.Device(0).mem_info[1] - cp.cuda.Device(0).mem_info[0]) / 1024**3
        mem_percent = mem_used / gpu_memory * 100
        print(f"GPU Memory Usage: {mem_used:.2f} GB / {gpu_memory:.1f} GB ({mem_percent:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return total_time, avg_samples_per_sec



if __name__ == '__main__':
    # Run single large batch test (reduced batch size to fit in GPU memory)
    print("\n" + "="*70)
    print("Test 1: Single batch (256 samples, 20 batches)")
    print("="*70)
    benchmark_batch_generation(batch_size=256, num_batches=20)
    
    # Compare batch sizes (adjusted for GPU memory)
    print("\n" + "="*70)
    print("Test 2: Batch size optimization")
    print("="*70)
    
    batch_sizes = [64, 128, 256, 512]
    results = []
    
    for batch_size in batch_sizes:
        try:
            total_time, samples_per_sec = benchmark_batch_generation(batch_size=batch_size, num_batches=10)
            results.append((batch_size, total_time, samples_per_sec))
        except Exception as e:
            print(f"ERROR: Batch size {batch_size} failed: {e}\n")
            results.append((batch_size, None, None))
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch Size':<15} {'Total Time':<15} {'Samples/Sec':<20} {'100k Est.':<15}")
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
        print(f"\nOptimal batch size: {optimal_batch} ({optimal_sps:.1f} samples/sec)")
    
    print(f"{'='*70}\n")
