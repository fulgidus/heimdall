#!/usr/bin/env python3
"""
Debug propagation model violations.

Generates a few samples and prints detailed physics breakdowns for the worst violations.
"""

import sys
import numpy as np
from typing import List, Dict

# Import from src package
from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.propagation import Polarization
from src.data.config import TrainingConfig, get_italian_receivers


def print_violation_details(sample, rx1_idx: int, rx2_idx: int):
    """Print detailed physics breakdown for a violation."""
    rx1 = sample.receivers[rx1_idx]
    rx2 = sample.receivers[rx2_idx]
    
    rx1_details = rx1.get('details', {})
    rx2_details = rx2.get('details', {})
    
    print("\n" + "=" * 100)
    print(f"VIOLATION: RX{rx1_idx} (closer) vs RX{rx2_idx} (farther)")
    print("=" * 100)
    print()
    
    print(f"TX Position: {sample.tx_lat:.4f}°N, {sample.tx_lon:.4f}°E")
    print(f"TX Power: {sample.tx_power_dbm:.1f} dBm")
    print(f"Frequency: {sample.frequency_hz / 1e6:.2f} MHz")
    print()
    
    print("-" * 100)
    print(f"{'Parameter':<40} {'RX' + str(rx1_idx) + ' (closer)':<28} {'RX' + str(rx2_idx) + ' (farther)':<28}")
    print("-" * 100)
    
    # Basic metrics
    print(f"{'Position':<40} {rx1['lat']:.4f}°N, {rx1['lon']:.4f}°E   {rx2['lat']:.4f}°N, {rx2['lon']:.4f}°E")
    print(f"{'Distance (km)':<40} {rx1_details.get('distance_km', 0):<28.2f} {rx2_details.get('distance_km', 0):<28.2f}")
    print(f"{'SNR (dB)':<40} {rx1['snr']:<28.2f} {rx2['snr']:<28.2f}")
    print(f"{'RX Power (dBm)':<40} {rx1_details.get('rx_power_dbm', 0):<28.2f} {rx2_details.get('rx_power_dbm', 0):<28.2f}")
    
    print()
    print("PROPAGATION BREAKDOWN:")
    print("-" * 100)
    
    # TX Power
    rx1_tx_power = rx1_details.get('actual_tx_power_dbm', sample.tx_power_dbm)
    rx2_tx_power = rx2_details.get('actual_tx_power_dbm', sample.tx_power_dbm)
    print(f"{'Actual TX Power (dBm)':<40} {rx1_tx_power:<28.2f} {rx2_tx_power:<28.2f}")
    
    # Antenna gains
    rx1_tx_ant = rx1_details.get('tx_antenna_gain_db', 0)
    rx2_tx_ant = rx2_details.get('tx_antenna_gain_db', 0)
    print(f"{'TX Antenna Gain (dBi)':<40} {rx1_tx_ant:<28.2f} {rx2_tx_ant:<28.2f}")
    
    rx1_rx_ant = rx1_details.get('rx_antenna_gain_db', 0)
    rx2_rx_ant = rx2_details.get('rx_antenna_gain_db', 0)
    print(f"{'RX Antenna Gain (dBi)':<40} {rx1_rx_ant:<28.2f} {rx2_rx_ant:<28.2f}")
    
    # Path losses
    rx1_fspl = rx1_details.get('fspl_db', 0)
    rx2_fspl = rx2_details.get('fspl_db', 0)
    print(f"{'Free Space Path Loss (dB)':<40} {rx1_fspl:<28.2f} {rx2_fspl:<28.2f}")
    
    rx1_terrain = rx1_details.get('terrain_loss_db', 0)
    rx2_terrain = rx2_details.get('terrain_loss_db', 0)
    print(f"{'Terrain Loss (dB)':<40} {rx1_terrain:<28.2f} {rx2_terrain:<28.2f}")
    
    rx1_knife = rx1_details.get('knife_edge_loss_db', 0)
    rx2_knife = rx2_details.get('knife_edge_loss_db', 0)
    print(f"{'Knife-Edge Diffraction (dB)':<40} {rx1_knife:<28.2f} {rx2_knife:<28.2f}")
    
    rx1_env = rx1_details.get('env_loss_db', 0)
    rx2_env = rx2_details.get('env_loss_db', 0)
    print(f"{'Environment Loss (dB)':<40} {rx1_env:<28.2f} {rx2_env:<28.2f}")
    
    rx1_pol = rx1_details.get('polarization_loss_db', 0)
    rx2_pol = rx2_details.get('polarization_loss_db', 0)
    rx1_pol_tx = rx1_details.get('tx_polarization', 'N/A')
    rx2_pol_tx = rx2_details.get('tx_polarization', 'N/A')
    rx1_pol_rx = rx1_details.get('rx_polarization', 'N/A')
    rx2_pol_rx = rx2_details.get('rx_polarization', 'N/A')
    print(f"{'Polarization Loss (dB)':<40} {rx1_pol:<28.2f} {rx2_pol:<28.2f}")
    print(f"{'  TX/RX Polarization':<40} {rx1_pol_tx}/{rx1_pol_rx:<21} {rx2_pol_tx}/{rx2_pol_rx:<21}")
    
    # Atmospheric effects
    rx1_atmos = rx1_details.get('atmospheric_absorption_db', 0)
    rx2_atmos = rx2_details.get('atmospheric_absorption_db', 0)
    print(f"{'Atmospheric Absorption (dB)':<40} {rx1_atmos:<28.2f} {rx2_atmos:<28.2f}")
    
    rx1_tropo = rx1_details.get('tropospheric_effect_db', 0)
    rx2_tropo = rx2_details.get('tropospheric_effect_db', 0)
    print(f"{'Tropospheric Refraction (dB)':<40} {rx1_tropo:<28.2f} {rx2_tropo:<28.2f}")
    
    # Fading
    rx1_fading = rx1_details.get('fading_db', 0)
    rx2_fading = rx2_details.get('fading_db', 0)
    print(f"{'Multipath Fading (dB)':<40} {rx1_fading:<28.2f} {rx2_fading:<28.2f}")
    
    # Sporadic-E
    rx1_es_active = rx1_details.get('sporadic_e_active', False)
    rx2_es_active = rx2_details.get('sporadic_e_active', False)
    rx1_es = rx1_details.get('sporadic_e_enhancement_db', 0)
    rx2_es = rx2_details.get('sporadic_e_enhancement_db', 0)
    print(f"{'Sporadic-E Enhancement (dB)':<40} {str(rx1_es) + (' [ACTIVE]' if rx1_es_active else ''):<28} {str(rx2_es) + (' [ACTIVE]' if rx2_es_active else ''):<28}")
    
    print()
    print("CALCULATED RX POWER:")
    print("-" * 100)
    
    # Calculate expected RX power manually
    rx1_expected = (
        rx1_tx_power
        + rx1_tx_ant
        + rx1_rx_ant
        - rx1_fspl
        - rx1_terrain
        - rx1_knife
        - rx1_env
        - rx1_pol
        - rx1_atmos
        + rx1_tropo
        + rx1_fading
        + rx1_es
    )
    
    rx2_expected = (
        rx2_tx_power
        + rx2_tx_ant
        + rx2_rx_ant
        - rx2_fspl
        - rx2_terrain
        - rx2_knife
        - rx2_env
        - rx2_pol
        - rx2_atmos
        + rx2_tropo
        + rx2_fading
        + rx2_es
    )
    
    print(f"{'Expected RX Power (dBm)':<40} {rx1_expected:<28.2f} {rx2_expected:<28.2f}")
    print(f"{'Actual RX Power (dBm)':<40} {rx1_details.get('rx_power_dbm', 0):<28.2f} {rx2_details.get('rx_power_dbm', 0):<28.2f}")
    print(f"{'Discrepancy (dB)':<40} {rx1_expected - rx1_details.get('rx_power_dbm', 0):<28.2f} {rx2_expected - rx2_details.get('rx_power_dbm', 0):<28.2f}")
    
    print()
    print("ANALYSIS:")
    print("-" * 100)
    
    # Identify the main causes of the violation
    fspl_diff = rx2_fspl - rx1_fspl
    terrain_diff = rx2_terrain - rx1_terrain
    fading_diff = rx2_fading - rx1_fading
    pol_diff = rx2_pol - rx1_pol
    antenna_diff = (rx2_tx_ant + rx2_rx_ant) - (rx1_tx_ant + rx1_rx_ant)
    es_diff = rx2_es - rx1_es
    
    print(f"FSPL Difference (farther - closer): {fspl_diff:.2f} dB (expected: positive, farther should have more loss)")
    print(f"Terrain Loss Difference: {terrain_diff:.2f} dB (negative = closer has MORE terrain loss)")
    print(f"Fading Difference: {fading_diff:.2f} dB (negative = closer has worse fading)")
    print(f"Polarization Loss Difference: {pol_diff:.2f} dB (negative = closer has MORE pol loss)")
    print(f"Total Antenna Gain Difference: {antenna_diff:.2f} dB (positive = farther has better antenna)")
    print(f"Sporadic-E Difference: {es_diff:.2f} dB (positive = farther has Es enhancement)")
    
    # Expected gain/loss based on distance
    distance_ratio = rx2_details.get('distance_km', 0) / max(rx1_details.get('distance_km', 1), 0.1)
    expected_fspl_diff = 20 * np.log10(distance_ratio)
    
    print(f"\nExpected FSPL difference from distance ratio: {expected_fspl_diff:.2f} dB")
    print(f"Actual FSPL difference: {fspl_diff:.2f} dB")
    
    # Diagnose main causes
    print("\nMAIN CAUSES OF VIOLATION:")
    causes = []
    
    if fading_diff < -10:
        causes.append(f"  ❌ Deep fade on closer receiver ({fading_diff:.1f} dB worse)")
    if pol_diff < -5:
        causes.append(f"  ❌ Cross-pol loss on closer receiver ({pol_diff:.1f} dB worse)")
    if antenna_diff > 10:
        causes.append(f"  ❌ Antenna null on closer receiver ({antenna_diff:.1f} dB worse gain)")
    if terrain_diff < -10:
        causes.append(f"  ❌ Terrain blockage on closer receiver ({terrain_diff:.1f} dB worse)")
    if es_diff > 15:
        causes.append(f"  ❌ Sporadic-E enhancement on farther receiver ({es_diff:.1f} dB)")
    if rx1_env > rx2_env + 5:
        causes.append(f"  ❌ Higher environment loss on closer receiver ({rx1_env - rx2_env:.1f} dB)")
    
    if causes:
        for cause in causes:
            print(cause)
    else:
        print("  ⚠️  NO CLEAR PHYSICS EXPLANATION FOUND - POTENTIAL BUG IN PROPAGATION MODEL")
    
    print()


def main():
    """Run diagnostic test."""
    print("=" * 100)
    print("PROPAGATION VIOLATION DIAGNOSTIC")
    print("=" * 100)
    print()
    print("Generating 50 samples and examining the worst violations...")
    print()
    
    # Initialize training configuration with Italian receivers
    receivers = get_italian_receivers()
    training_config = TrainingConfig.from_receivers(receivers, margin_degrees=0.5)
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        training_config=training_config,
        use_srtm_terrain=False  # Disable terrain to avoid missing SRTM tiles issue
    )
    
    # Generate 1000 samples to match validation test
    print("Generating samples...")
    samples = generator.generate_samples(num_samples=1000)
    print(f"Generated {len(samples)} samples")
    print()
    
    # Debug: Print first sample details to verify structure
    if len(samples) > 0:
        print("DEBUG: First sample structure check:")
        first_sample = samples[0]
        print(f"  TX Position: {first_sample.tx_lat:.4f}°N, {first_sample.tx_lon:.4f}°E")
        print(f"  Number of receivers: {len(first_sample.receivers)}")
        print(f"  Type of receivers: {type(first_sample.receivers)}")
        print(f"  Type of first receiver: {type(first_sample.receivers[0])}")
        for i, rx in enumerate(first_sample.receivers[:3]):  # Show first 3 receivers
            print(f"  RX{i}: keys={list(rx.keys())}")
            print(f"        snr={rx.get('snr', 'N/A'):.2f}, has_details={('details' in rx)}")
            if 'details' in rx:
                details = rx['details']
                print(f"        distance_km={details.get('distance_km', 'N/A'):.2f}, " +
                      f"rx_power_dbm={details.get('rx_power_dbm', 'N/A'):.2f}, " +
                      f"fading_db={details.get('fading_db', 'N/A'):.2f}")
            else:
                # Try to access distance_km directly
                print(f"        distance_km (direct)={rx.get('distance_km', 'N/A')}")
        print()
    
    # Find violations
    violations = []
    total_pairs_checked = 0
    
    for sample_idx, sample in enumerate(samples):
        # Extract receivers with signal
        receivers_with_signal = [
            (i, rx) for i, rx in enumerate(sample.receivers)
            if rx.get('snr', -999.0) > -500.0
        ]
        
        if len(receivers_with_signal) < 2:
            continue
        
        # Check all pairs
        for i in range(len(receivers_with_signal)):
            for j in range(i + 1, len(receivers_with_signal)):
                total_pairs_checked += 1
                idx1, rx1 = receivers_with_signal[i]
                idx2, rx2 = receivers_with_signal[j]
                
                dist1 = rx1.get('details', {}).get('distance_km', 0)
                dist2 = rx2.get('details', {}).get('distance_km', 0)
                snr1 = rx1.get('snr', -999.0)
                snr2 = rx2.get('snr', -999.0)
                
                # Check if closer receiver has worse SNR
                if dist1 < dist2 and snr2 > snr1 + 12.0:
                    violations.append({
                        'sample': sample,
                        'closer_idx': idx1,
                        'farther_idx': idx2,
                        'snr_diff': snr2 - snr1,
                        'dist_ratio': dist2 / max(dist1, 0.1)
                    })
                elif dist2 < dist1 and snr1 > snr2 + 12.0:
                    violations.append({
                        'sample': sample,
                        'closer_idx': idx2,
                        'farther_idx': idx1,
                        'snr_diff': snr1 - snr2,
                        'dist_ratio': dist1 / max(dist2, 0.1)
                    })
    
    print(f"Found {len(violations)} violations out of {total_pairs_checked} receiver pairs checked")
    print(f"Violation rate: {len(violations) / total_pairs_checked * 100:.2f}%")
    print()
    
    # Show top 5 worst violations
    violations.sort(key=lambda v: v['snr_diff'], reverse=True)
    
    for i, v in enumerate(violations[:5]):
        print_violation_details(v['sample'], v['closer_idx'], v['farther_idx'])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
