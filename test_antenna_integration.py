#!/usr/bin/env python3
"""
Test script to verify antenna pattern integration in synthetic data generation.
Run this inside the training Docker container.
"""
import sys
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app/../common/src')

from data.synthetic_generator import _generate_single_sample_no_features
from data.propagation import PropagationModel, AntennaType
from common.terrain import SRTMTerrainService
import numpy as np

def main():
    print("=" * 70)
    print("Testing Antenna Pattern Integration")
    print("=" * 70)
    
    # Initialize services
    print("\n[1/4] Initializing terrain and propagation services...")
    terrain = SRTMTerrainService()
    propagation = PropagationModel()
    
    # Sample receivers (3 WebSDRs from Northern Italy)
    receivers_list = [
        {'name': 'I0SNY-Milano', 'latitude': 45.464, 'longitude': 9.188, 'altitude': 122},
        {'name': 'I4YEO-Rapallo', 'latitude': 44.349, 'longitude': 9.229, 'altitude': 20},
        {'name': 'IW4BLG-Modena', 'latitude': 44.647, 'longitude': 10.925, 'altitude': 35}
    ]
    
    # Configuration
    config = {
        'min_snr_db': -5.0,
        'max_gdop': 150.0,
        'min_receivers': 3,
        'tx_power_dbm': 40.0,
        'frequency_mhz': 145.0,
        'bbox': {'lat_min': 44.0, 'lat_max': 46.0, 'lon_min': 8.0, 'lon_max': 12.0}
    }
    
    # Generate multiple samples to see antenna variation
    print("\n[2/4] Generating 5 samples with random antenna patterns...")
    rng = np.random.default_rng(42)
    
    successful_samples = 0
    for i in range(10):
        sample = _generate_single_sample_no_features(
            sample_idx=i,
            receivers_list=receivers_list,
            config=config,
            propagation=propagation,
            terrain=terrain,
            rng=rng
        )
        
        if sample is not None:
            successful_samples += 1
            if successful_samples <= 3:
                print(f"\n  Sample {successful_samples}:")
                print(f"    TX: ({sample.tx_lat:.3f}, {sample.tx_lon:.3f})")
                print(f"    Receivers: {sample.num_receivers}/3")
                print(f"    GDOP: {sample.gdop:.1f}")
                
                # Show SNR variation across receivers
                snrs = [rx['snr'] for rx in sample.receivers if rx['signal_present'] == 1]
                if snrs:
                    print(f"    SNR range: {min(snrs):.1f} to {max(snrs):.1f} dB (variation: {max(snrs)-min(snrs):.1f} dB)")
        
        if successful_samples >= 3:
            break
    
    print(f"\n[3/4] Success rate: {successful_samples}/{i+1} samples ({100*successful_samples/(i+1):.1f}%)")
    
    # Test antenna gain calculation directly
    print("\n[4/4] Testing antenna gain calculations...")
    from data.propagation import AntennaPattern
    
    # Create different antenna types
    omni = AntennaPattern(AntennaType.OMNI_VERTICAL)
    yagi = AntennaPattern(AntennaType.YAGI)
    whip = AntennaPattern(AntennaType.WHIP)
    
    # Test at different angles
    test_angles = [
        (0, 0, "Boresight (0°, 0°)"),
        (45, 0, "45° azimuth"),
        (180, 0, "Rear (180°)"),
        (0, 30, "30° elevation")
    ]
    
    print("\n  Antenna gain comparison (dB):")
    print("  Angle          | OMNI | YAGI | WHIP")
    print("  " + "-" * 45)
    for azimuth, elevation, desc in test_angles:
        omni_gain = omni.get_gain(azimuth, elevation)
        yagi_gain = yagi.get_gain(azimuth, elevation)
        whip_gain = whip.get_gain(azimuth, elevation)
        print(f"  {desc:14s} | {omni_gain:+4.1f} | {yagi_gain:+5.1f} | {whip_gain:+4.1f}")
    
    print("\n" + "=" * 70)
    print("✅ Antenna pattern integration test complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
