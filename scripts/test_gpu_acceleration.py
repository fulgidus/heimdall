#!/usr/bin/env python3
"""
Quick test to verify GPU acceleration is working correctly.

Tests:
1. CuPy availability
2. IQ generator GPU mode
3. Feature extractor GPU mode
4. Numerical accuracy (CPU vs GPU results should be close)
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.training.src.data.iq_generator import SyntheticIQGenerator
from services.common.feature_extraction.rf_feature_extractor import RFFeatureExtractor, IQSample


def test_cupy_availability():
    """Test if CuPy is available."""
    print("="*60)
    print("Test 1: CuPy Availability")
    print("="*60)
    
    try:
        import cupy as cp
        gpu_available = cp.cuda.is_available()
        
        if gpu_available:
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
            gpu_memory = cp.cuda.Device(0).mem_info[1] / 1024**3
            print(f"✅ CuPy installed and GPU available")
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print(f"❌ CuPy installed but no GPU detected")
            return False
    except ImportError:
        print(f"❌ CuPy not installed (CPU-only mode)")
        return False


def test_iq_generator_gpu():
    """Test IQ generator GPU mode."""
    print("\n" + "="*60)
    print("Test 2: IQ Generator GPU Mode")
    print("="*60)
    
    try:
        # CPU mode
        gen_cpu = SyntheticIQGenerator(
            sample_rate_hz=200_000,
            duration_ms=100.0,
            seed=42,
            use_gpu=False
        )
        
        sample_cpu = gen_cpu.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id="test_rx",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=0.0
        )
        
        print(f"✅ CPU mode working")
        print(f"   Samples: {len(sample_cpu.samples)}")
        print(f"   Type: {type(sample_cpu.samples)}")
        print(f"   Mean power: {np.mean(np.abs(sample_cpu.samples)**2):.2e}")
        
        # GPU mode
        gen_gpu = SyntheticIQGenerator(
            sample_rate_hz=200_000,
            duration_ms=100.0,
            seed=42,
            use_gpu=True
        )
        
        sample_gpu = gen_gpu.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id="test_rx",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=0.0
        )
        
        print(f"✅ GPU mode working")
        print(f"   Samples: {len(sample_gpu.samples)}")
        print(f"   Type: {type(sample_gpu.samples)}")
        print(f"   Mean power: {np.mean(np.abs(sample_gpu.samples)**2):.2e}")
        
        # Check if GPU flag is set correctly
        if gen_gpu.use_gpu:
            print(f"✅ GPU acceleration ENABLED in generator")
        else:
            print(f"⚠️  GPU requested but not enabled (fallback to CPU)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extractor_gpu():
    """Test feature extractor GPU mode."""
    print("\n" + "="*60)
    print("Test 3: Feature Extractor GPU Mode")
    print("="*60)
    
    try:
        # Generate test IQ sample
        gen = SyntheticIQGenerator(
            sample_rate_hz=200_000,
            duration_ms=1000.0,
            seed=42,
            use_gpu=False  # Use CPU for IQ generation (just for test data)
        )
        
        synth_sample = gen.generate_iq_sample(
            center_frequency_hz=145_000_000,
            signal_power_dbm=-80,
            noise_floor_dbm=-120,
            snr_db=15.0,
            frequency_offset_hz=25.0,
            bandwidth_hz=12500,
            rx_id="test_rx",
            rx_lat=45.0,
            rx_lon=9.0,
            timestamp=0.0
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
        
        # CPU mode
        extractor_cpu = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=False)
        features_cpu = extractor_cpu.extract_features(iq_sample)
        
        print(f"✅ CPU mode working")
        print(f"   RSSI: {features_cpu.rssi_dbm:.2f} dBm")
        print(f"   SNR: {features_cpu.snr_db:.2f} dB")
        print(f"   Frequency offset: {features_cpu.frequency_offset_hz:.2f} Hz")
        
        # GPU mode
        extractor_gpu = RFFeatureExtractor(sample_rate_hz=200_000, use_gpu=True)
        features_gpu = extractor_gpu.extract_features(iq_sample)
        
        print(f"✅ GPU mode working")
        print(f"   RSSI: {features_gpu.rssi_dbm:.2f} dBm")
        print(f"   SNR: {features_gpu.snr_db:.2f} dB")
        print(f"   Frequency offset: {features_gpu.frequency_offset_hz:.2f} Hz")
        
        # Check if GPU flag is set correctly
        if extractor_gpu.use_gpu:
            print(f"✅ GPU acceleration ENABLED in feature extractor")
        else:
            print(f"⚠️  GPU requested but not enabled (fallback to CPU)")
        
        # Compare results (should be very close)
        rssi_diff = abs(features_cpu.rssi_dbm - features_gpu.rssi_dbm)
        snr_diff = abs(features_cpu.snr_db - features_gpu.snr_db)
        freq_diff = abs(features_cpu.frequency_offset_hz - features_gpu.frequency_offset_hz)
        
        print(f"\nNumerical accuracy:")
        print(f"   RSSI difference: {rssi_diff:.6f} dB")
        print(f"   SNR difference: {snr_diff:.6f} dB")
        print(f"   Frequency offset difference: {freq_diff:.6f} Hz")
        
        # Tolerance: 1e-3 (0.001) for dB values, 1e-1 for frequency
        if rssi_diff < 1e-3 and snr_diff < 1e-3 and freq_diff < 1e-1:
            print(f"✅ CPU and GPU results match (within tolerance)")
            return True
        else:
            print(f"⚠️  CPU and GPU results differ more than expected")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("GPU Acceleration Test Suite")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Test 1: CuPy availability
    gpu_available = test_cupy_availability()
    results.append(("CuPy Availability", gpu_available))
    
    if not gpu_available:
        print("\n⚠️  GPU not available. Only CPU tests will run.")
    
    # Test 2: IQ generator
    iq_gen_ok = test_iq_generator_gpu()
    results.append(("IQ Generator GPU Mode", iq_gen_ok))
    
    # Test 3: Feature extractor
    feature_ext_ok = test_feature_extractor_gpu()
    results.append(("Feature Extractor GPU Mode", feature_ext_ok))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
