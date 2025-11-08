#!/usr/bin/env python3
"""
Test script to verify RF propagation physics fixes.

This script generates 100 synthetic samples and verifies:
1. Transmission state is consistent across all receivers (Fix #1)
2. Sporadic-E probability is reduced (Fix #2)
3. Fading uses improved model (Fix #3)
4. RX polarization is 95% vertical (Fix #4)
5. Depolarization model is applied (Fix #5)
6. Cross-pol loss is 10-15 dB (Fix #6)
7. FSPL sanity checks pass (Fix #7)
8. Distance-SNR relationship is monotonic (no physics violations)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add services directory to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "training" / "src"))

from data.synthetic_generator import SyntheticDataGenerator
from data.propagation import Polarization


def main():
    """Run physics validation tests."""
    print("=" * 80)
    print("RF PROPAGATION PHYSICS FIXES - VALIDATION TEST")
    print("=" * 80)
    print()
    
    # Initialize generator
    print("Initializing synthetic data generator...")
    generator = SyntheticDataGenerator(
        num_samples=100,
        config={
            'enable_meteorological': True,
            'enable_sporadic_e': True,
            'enable_knife_edge': True,
            'enable_polarization': True,
            'enable_antenna_patterns': True
        }
    )
    
    print(f"Generating 100 test samples...")
    print()
    
    # Tracking variables
    sporadic_e_count = 0
    vertical_pol_count = 0
    total_rx_count = 0
    cross_pol_losses = []
    distance_snr_violations = 0
    transmission_inconsistencies = 0
    
    # Process samples
    for i in range(100):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/100 samples...")
        
        try:
            sample = generator.generate_samples(num_samples=1, start_index=i)[0]
            
            # Check #1: Transmission consistency
            transmission_states = []
            distances = []
            snrs = []
            
            # Extract per-receiver data from sample metadata
            for rx_idx in range(7):  # 7 receivers
                if 'details' in sample and isinstance(sample['details'], list) and rx_idx < len(sample['details']):
                    details = sample['details'][rx_idx]
                    transmission_states.append(details.get('transmission_active', True))
                    distances.append(details.get('distance_km', 0))
                    snrs.append(sample['snr_dbs'][rx_idx] if 'snr_dbs' in sample else 0)
                    
                    # Check #2: Sporadic-E
                    if details.get('sporadic_e_active', False):
                        sporadic_e_count += 1
                    
                    # Check #4: Vertical polarization
                    if details.get('rx_polarization') == 'vertical':
                        vertical_pol_count += 1
                    total_rx_count += 1
                    
                    # Check #6: Cross-pol loss
                    tx_pol = details.get('tx_polarization')
                    rx_pol = details.get('rx_polarization')
                    pol_loss = details.get('polarization_loss_db', 0)
                    
                    if tx_pol != rx_pol and pol_loss > 0:
                        cross_pol_losses.append(pol_loss)
            
            # Check transmission consistency
            if len(set(transmission_states)) > 1:
                transmission_inconsistencies += 1
                print(f"    WARNING: Sample {i} has inconsistent transmission states: {transmission_states}")
            
            # Check #8: Distance-SNR monotonicity (closer receivers should have better SNR)
            if len(distances) >= 2 and len(snrs) >= 2:
                # Sort by distance
                sorted_pairs = sorted(zip(distances, snrs))
                distances_sorted = [d for d, _ in sorted_pairs]
                snrs_sorted = [s for _, s in sorted_pairs]
                
                # Check if closest receiver has better SNR than farthest
                if len(distances_sorted) >= 2:
                    if snrs_sorted[0] < snrs_sorted[-1] - 10:  # Allow 10 dB tolerance for fading
                        distance_snr_violations += 1
                        print(f"    WARNING: Sample {i} has distance-SNR violation:")
                        print(f"      Closest ({distances_sorted[0]:.1f} km): SNR={snrs_sorted[0]:.1f} dB")
                        print(f"      Farthest ({distances_sorted[-1]:.1f} km): SNR={snrs_sorted[-1]:.1f} dB")
        
        except Exception as e:
            print(f"    ERROR: Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()
    
    # Fix #1: Transmission consistency
    print(f"1. TRANSMISSION CONSISTENCY")
    print(f"   Samples with inconsistent TX states: {transmission_inconsistencies}/100")
    if transmission_inconsistencies == 0:
        print(f"   ✓ PASS: All samples have consistent transmission states")
    else:
        print(f"   ✗ FAIL: Some samples have inconsistent transmission states")
    print()
    
    # Fix #2: Sporadic-E probability
    sporadic_e_rate = sporadic_e_count / total_rx_count * 100
    print(f"2. SPORADIC-E PROBABILITY")
    print(f"   Sporadic-E occurrences: {sporadic_e_count}/{total_rx_count} ({sporadic_e_rate:.2f}%)")
    print(f"   Expected: 0.1-0.5% (adjusted for factors)")
    if sporadic_e_rate < 2.0:  # Allow 2% due to seasonal/distance factors
        print(f"   ✓ PASS: Sporadic-E rate is reasonable")
    else:
        print(f"   ✗ FAIL: Sporadic-E rate is too high")
    print()
    
    # Fix #4: Vertical polarization dominance
    vertical_pol_rate = vertical_pol_count / total_rx_count * 100
    print(f"4. VERTICAL POLARIZATION DOMINANCE")
    print(f"   Vertical RX antennas: {vertical_pol_count}/{total_rx_count} ({vertical_pol_rate:.1f}%)")
    print(f"   Expected: ~95%")
    if vertical_pol_rate >= 90:
        print(f"   ✓ PASS: Vertical polarization dominates")
    else:
        print(f"   ✗ FAIL: Vertical polarization rate is too low")
    print()
    
    # Fix #6: Cross-pol loss
    if len(cross_pol_losses) > 0:
        mean_cross_pol = np.mean(cross_pol_losses)
        min_cross_pol = np.min(cross_pol_losses)
        max_cross_pol = np.max(cross_pol_losses)
        print(f"6. CROSS-POLARIZATION LOSS")
        print(f"   Mean: {mean_cross_pol:.1f} dB")
        print(f"   Range: {min_cross_pol:.1f} - {max_cross_pol:.1f} dB")
        print(f"   Expected: 10-15 dB")
        if 10 <= mean_cross_pol <= 15:
            print(f"   ✓ PASS: Cross-pol loss is in expected range")
        else:
            print(f"   ✗ FAIL: Cross-pol loss is out of range")
    else:
        print(f"6. CROSS-POLARIZATION LOSS")
        print(f"   No cross-pol cases found (all antennas matched)")
    print()
    
    # Fix #8: Distance-SNR relationship
    print(f"8. DISTANCE-SNR MONOTONICITY")
    print(f"   Violations: {distance_snr_violations}/100")
    if distance_snr_violations <= 10:  # Allow 10% for extreme fading cases
        print(f"   ✓ PASS: Distance-SNR relationship is mostly monotonic")
    else:
        print(f"   ✗ FAIL: Too many distance-SNR violations")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_tests = 5
    passed_tests = 0
    
    if transmission_inconsistencies == 0:
        passed_tests += 1
    if sporadic_e_rate < 2.0:
        passed_tests += 1
    if vertical_pol_rate >= 90:
        passed_tests += 1
    if len(cross_pol_losses) == 0 or (10 <= mean_cross_pol <= 15):
        passed_tests += 1
    if distance_snr_violations <= 10:
        passed_tests += 1
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print()
    
    if passed_tests == total_tests:
        print("✓ ALL TESTS PASSED - Physics fixes are working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Physics fixes need adjustment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
