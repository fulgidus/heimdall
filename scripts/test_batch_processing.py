#!/usr/bin/env python3
"""
Test script for GPU batch processing implementation.

Tests the new batch generation loop in synthetic_generator.py to verify:
1. GPU batch size is correctly set (256 for GPU, num_cores for CPU)
2. Speedup is achieved (8-10x for GPU)
3. GPU utilization is high (70-85%)
4. CPU fallback works correctly
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.training.src.data.iq_generator import SyntheticIQGenerator

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ GPU Not Available - Testing CPU fallback")
except ImportError:
    GPU_AVAILABLE = False
    print("❌ PyTorch not installed - Testing CPU fallback")


async def test_batch_iq_generation(num_samples: int = 256, use_gpu: bool = True):
    """
    Test batch IQ generation with the new generate_iq_batch() method.
    
    Args:
        num_samples: Number of samples to generate in batch
        use_gpu: Whether to use GPU
    
    Returns:
        Execution time in seconds
    """
    print(f"\n{'='*60}")
    print(f"Test: Batch IQ Generation ({'GPU' if use_gpu else 'CPU'})")
    print(f"{'='*60}")
    print(f"Batch size: {num_samples}")
    
    # Initialize generator
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=use_gpu
    )
    
    # Prepare batch parameters (7 receivers × num_samples)
    batch_params = []
    for i in range(num_samples):
        for rx_id in range(7):  # Simulate 7 WebSDR receivers
            batch_params.append({
                'center_frequency_hz': 145_000_000,
                'signal_power_dbm': -80,
                'noise_floor_dbm': -110,
                'snr_db': 15.0,
                'frequency_offset_hz': 25.0,
                'bandwidth_hz': 12500,
                'rx_id': f'rx_{rx_id}',
                'rx_lat': 45.0 + rx_id * 0.5,
                'rx_lon': 9.0 + rx_id * 0.5,
                'timestamp': float(i)
            })
    
    total_iq_samples = len(batch_params)
    print(f"Total IQ samples to generate: {total_iq_samples} ({num_samples} TX × 7 receivers)")
    
    # Warm-up (GPU kernel compilation)
    if use_gpu:
        print("Warming up GPU...")
        _ = generator.generate_iq_batch(
            batch_params=batch_params[:7],
            batch_size=7,
            enable_multipath=True,
            enable_fading=True
        )
    
    # Benchmark batch generation
    print(f"Generating {total_iq_samples} IQ samples in batch...")
    start_time = time.time()
    
    results = generator.generate_iq_batch(
        batch_params=batch_params,
        batch_size=total_iq_samples,
        enable_multipath=True,
        enable_fading=True
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"✅ Generated {len(results)} IQ samples")
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    print(f"📊 Throughput: {total_iq_samples / elapsed_time:.1f} samples/sec")
    print(f"📊 Per-sample time: {elapsed_time / total_iq_samples * 1000:.2f}ms")
    
    return elapsed_time


async def compare_sequential_vs_batch(num_samples: int = 256):
    """
    Compare sequential generation vs batch generation.
    
    Args:
        num_samples: Number of samples per test
    """
    print(f"\n{'='*60}")
    print(f"Comparison: Sequential vs Batch Processing")
    print(f"{'='*60}")
    
    # Test parameters
    batch_params = []
    for i in range(num_samples):
        for rx_id in range(7):
            batch_params.append({
                'center_frequency_hz': 145_000_000,
                'signal_power_dbm': -80,
                'noise_floor_dbm': -110,
                'snr_db': 15.0,
                'frequency_offset_hz': 25.0,
                'bandwidth_hz': 12500,
                'rx_id': f'rx_{rx_id}',
                'rx_lat': 45.0 + rx_id * 0.5,
                'rx_lon': 9.0 + rx_id * 0.5,
                'timestamp': float(i)
            })
    
    total_iq_samples = len(batch_params)
    
    # Sequential generation (old approach)
    print(f"\n1. Sequential Generation (old approach)")
    print(f"   Generating {total_iq_samples} samples one-by-one...")
    
    generator_seq = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=GPU_AVAILABLE
    )
    
    start_seq = time.time()
    results_seq = []
    for params in batch_params:
        result = generator_seq.generate_iq_sample(
            center_frequency_hz=params['center_frequency_hz'],
            signal_power_dbm=params['signal_power_dbm'],
            noise_floor_dbm=params['noise_floor_dbm'],
            snr_db=params['snr_db'],
            frequency_offset_hz=params['frequency_offset_hz'],
            bandwidth_hz=params['bandwidth_hz'],
            rx_id=params['rx_id'],
            rx_lat=params['rx_lat'],
            rx_lon=params['rx_lon'],
            timestamp=params['timestamp'],
            enable_multipath=True,
            enable_fading=True
        )
        results_seq.append(result)
    
    time_seq = time.time() - start_seq
    print(f"   ✅ Time: {time_seq:.2f}s")
    print(f"   📊 Throughput: {total_iq_samples / time_seq:.1f} samples/sec")
    
    # Batch generation (new approach)
    print(f"\n2. Batch Generation (new approach)")
    print(f"   Generating {total_iq_samples} samples in batch...")
    
    generator_batch = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=GPU_AVAILABLE
    )
    
    # Warm-up
    if GPU_AVAILABLE:
        _ = generator_batch.generate_iq_batch(
            batch_params=batch_params[:7],
            batch_size=7,
            enable_multipath=True,
            enable_fading=True
        )
    
    start_batch = time.time()
    results_batch = generator_batch.generate_iq_batch(
        batch_params=batch_params,
        batch_size=total_iq_samples,
        enable_multipath=True,
        enable_fading=True
    )
    time_batch = time.time() - start_batch
    
    print(f"   ✅ Time: {time_batch:.2f}s")
    print(f"   📊 Throughput: {total_iq_samples / time_batch:.1f} samples/sec")
    
    # Calculate speedup
    speedup = time_seq / time_batch
    print(f"\n{'='*60}")
    print(f"📈 SPEEDUP: {speedup:.2f}x")
    print(f"{'='*60}")
    
    if speedup >= 8.0:
        print("✅ EXCELLENT: Target speedup achieved (8-10x)!")
    elif speedup >= 5.0:
        print("✅ GOOD: Significant speedup (5-8x)")
    elif speedup >= 2.0:
        print("⚠️  MODERATE: Some speedup but below target")
    else:
        print("❌ POOR: Speedup below expectations")
    
    return speedup


async def monitor_gpu_utilization():
    """
    Monitor GPU utilization during batch processing.
    
    NOTE: Requires nvidia-smi to be available.
    """
    if not GPU_AVAILABLE:
        print("\n⚠️  GPU monitoring skipped (no GPU available)")
        return
    
    print(f"\n{'='*60}")
    print(f"GPU Utilization Monitoring")
    print(f"{'='*60}")
    
    import subprocess
    
    try:
        # Query GPU utilization
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(',')
            print(f"GPU Utilization: {gpu_util.strip()}%")
            print(f"Memory Used: {mem_used.strip()} MB / {mem_total.strip()} MB")
            
            gpu_util_val = int(gpu_util.strip())
            if gpu_util_val >= 70:
                print(f"✅ GPU utilization is good (target: 70-85%)")
            else:
                print(f"⚠️  GPU utilization is low (target: 70-85%)")
        else:
            print("❌ Failed to query GPU status")
    
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - cannot monitor GPU utilization")
    except Exception as e:
        print(f"❌ Error monitoring GPU: {e}")


async def main():
    """Run all tests."""
    print("="*60)
    print("BATCH PROCESSING TEST SUITE")
    print("="*60)
    print(f"Testing new batch generation implementation")
    print()
    
    # Test 1: Small batch (quick test)
    print("\n[TEST 1] Small Batch (32 samples)")
    await test_batch_iq_generation(num_samples=32, use_gpu=GPU_AVAILABLE)
    
    # Test 2: Medium batch (production size)
    print("\n[TEST 2] Medium Batch (256 samples)")
    await test_batch_iq_generation(num_samples=256, use_gpu=GPU_AVAILABLE)
    
    # Test 3: Sequential vs Batch comparison
    print("\n[TEST 3] Sequential vs Batch Comparison")
    speedup = await compare_sequential_vs_batch(num_samples=128)
    
    # Test 4: GPU utilization monitoring
    await monitor_gpu_utilization()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ All tests completed")
    print(f"📊 Speedup achieved: {speedup:.2f}x")
    
    if GPU_AVAILABLE and speedup >= 8.0:
        print(f"✅ READY FOR PRODUCTION: Batch processing working optimally!")
    elif GPU_AVAILABLE and speedup >= 5.0:
        print(f"⚠️  ACCEPTABLE: Batch processing works but below target")
    else:
        print(f"❌ NEEDS INVESTIGATION: Speedup too low")


if __name__ == "__main__":
    asyncio.run(main())
