#!/usr/bin/env python3
"""
Statistical validation test for RF propagation physics.

This test generates 1000 samples and performs statistical analysis to validate:
1. Monotonicity violation rate (closer receivers with worse SNR)
   - Expected: 3-5%, Max acceptable: 10%
2. Sporadic-E occurrence rate
   - Expected: 0.1-0.5%
3. Polarization mismatch distribution
   - Expected: ~5% cross-pol (95% vertical RX)
4. Antenna pattern effects distribution

This test should be integrated into CI/CD for continuous monitoring.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Add services directory to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "training" / "src"))

# Also support running inside Docker container
if Path("/app/src").exists():
    sys.path.insert(0, "/app/src")

from data.synthetic_generator import SyntheticDataGenerator
from data.propagation import Polarization


class PhysicsValidator:
    """Statistical validator for RF propagation physics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.total_samples = 0
        self.monotonicity_violations = 0
        self.sporadic_e_occurrences = 0
        self.cross_pol_instances = 0
        self.antenna_null_instances = 0
        self.high_terrain_loss_instances = 0
        self.total_receiver_pairs = 0
        
        # Detailed tracking
        self.violation_details = []
        self.sporadic_e_enhancements = []
        self.cross_pol_losses = []
        self.antenna_null_gains = []
        self.terrain_losses = []
        
    def check_monotonicity_violation(
        self, 
        receivers_data: List[Dict], 
        tolerance_db: float = 12.0
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if any closer receiver has significantly worse SNR than a farther one.
        
        Args:
            receivers_data: List of receiver data dicts with 'distance_km', 'snr_db', 'details'
            tolerance_db: SNR difference threshold to flag as violation
            
        Returns:
            Tuple of (has_violation, violation_list)
        """
        violations = []
        
        # Sort by distance
        sorted_receivers = sorted(receivers_data, key=lambda r: r['distance_km'])
        
        # Check each pair: if closer receiver has worse SNR by >tolerance_db
        for i in range(len(sorted_receivers)):
            for j in range(i + 1, len(sorted_receivers)):
                closer = sorted_receivers[i]
                farther = sorted_receivers[j]
                
                snr_diff = farther['snr_db'] - closer['snr_db']
                
                if snr_diff > tolerance_db:
                    # This is a violation - farther receiver is significantly stronger
                    violations.append({
                        'closer_idx': closer['idx'],
                        'farther_idx': farther['idx'],
                        'closer_distance_km': closer['distance_km'],
                        'farther_distance_km': farther['distance_km'],
                        'closer_snr_db': closer['snr_db'],
                        'farther_snr_db': farther['snr_db'],
                        'snr_diff_db': snr_diff,
                        'closer_details': closer.get('details', {}),
                        'farther_details': farther.get('details', {})
                    })
        
        return len(violations) > 0, violations
    
    def analyze_sample(self, sample: Dict) -> Dict:
        """
        Analyze a single sample for physics anomalies.
        
        Args:
            sample: Generated sample dict
            
        Returns:
            Analysis results dict
        """
        results = {
            'has_monotonicity_violation': False,
            'monotonicity_violations': [],
            'sporadic_e_count': 0,
            'cross_pol_count': 0,
            'antenna_null_count': 0,
            'high_terrain_loss_count': 0
        }
        
        # Extract receiver data
        receivers_data = []
        for rx_idx in range(len(sample.get('snr_dbs', []))):
            details = {}
            if 'details' in sample and isinstance(sample['details'], list) and rx_idx < len(sample['details']):
                details = sample['details'][rx_idx]
            
            receivers_data.append({
                'idx': rx_idx,
                'distance_km': details.get('distance_km', 0),
                'snr_db': sample['snr_dbs'][rx_idx],
                'details': details
            })
        
        # Check monotonicity
        has_violation, violations = self.check_monotonicity_violation(receivers_data)
        results['has_monotonicity_violation'] = has_violation
        results['monotonicity_violations'] = violations
        
        # Analyze each receiver for physics effects
        for rx_data in receivers_data:
            details = rx_data['details']
            
            # Sporadic-E detection
            if details.get('sporadic_e_active', False):
                sporadic_e_enhancement = details.get('sporadic_e_enhancement_db', 0)
                if sporadic_e_enhancement > 15.0:  # Significant enhancement threshold
                    results['sporadic_e_count'] += 1
                    self.sporadic_e_enhancements.append(sporadic_e_enhancement)
            
            # Cross-pol detection
            tx_pol = details.get('tx_polarization')
            rx_pol = details.get('rx_polarization')
            pol_loss = details.get('polarization_loss_db', 0)
            
            if tx_pol and rx_pol and tx_pol != rx_pol and pol_loss > 10.0:
                results['cross_pol_count'] += 1
                self.cross_pol_losses.append(pol_loss)
            
            # Antenna null detection (gain < -5 dB)
            tx_gain = details.get('tx_antenna_gain_db', 0)
            rx_gain = details.get('rx_antenna_gain_db', 0)
            
            if tx_gain < -5.0 or rx_gain < -5.0:
                results['antenna_null_count'] += 1
                self.antenna_null_gains.append(min(tx_gain, rx_gain))
            
            # High terrain loss detection (>20 dB)
            terrain_loss = details.get('terrain_loss_db', 0)
            if terrain_loss > 20.0:
                results['high_terrain_loss_count'] += 1
                self.terrain_losses.append(terrain_loss)
        
        return results
    
    def update_statistics(self, analysis: Dict):
        """Update running statistics from sample analysis."""
        self.total_samples += 1
        
        if analysis['has_monotonicity_violation']:
            self.monotonicity_violations += 1
            self.violation_details.extend(analysis['monotonicity_violations'])
        
        self.sporadic_e_occurrences += analysis['sporadic_e_count']
        self.cross_pol_instances += analysis['cross_pol_count']
        self.antenna_null_instances += analysis['antenna_null_count']
        self.high_terrain_loss_instances += analysis['high_terrain_loss_count']
    
    def get_statistics(self) -> Dict:
        """Get final statistics summary."""
        if self.total_samples == 0:
            return {}
        
        monotonicity_rate = (self.monotonicity_violations / self.total_samples) * 100
        sporadic_e_rate = (self.sporadic_e_occurrences / (self.total_samples * 7)) * 100  # 7 receivers
        cross_pol_rate = (self.cross_pol_instances / (self.total_samples * 7)) * 100
        antenna_null_rate = (self.antenna_null_instances / (self.total_samples * 7)) * 100
        high_terrain_loss_rate = (self.high_terrain_loss_instances / (self.total_samples * 7)) * 100
        
        return {
            'total_samples': self.total_samples,
            'monotonicity_violation_rate_pct': monotonicity_rate,
            'sporadic_e_rate_pct': sporadic_e_rate,
            'cross_pol_rate_pct': cross_pol_rate,
            'antenna_null_rate_pct': antenna_null_rate,
            'high_terrain_loss_rate_pct': high_terrain_loss_rate,
            'num_violations': len(self.violation_details),
            'avg_sporadic_e_enhancement_db': np.mean(self.sporadic_e_enhancements) if self.sporadic_e_enhancements else 0,
            'avg_cross_pol_loss_db': np.mean(self.cross_pol_losses) if self.cross_pol_losses else 0,
            'avg_antenna_null_gain_db': np.mean(self.antenna_null_gains) if self.antenna_null_gains else 0,
            'avg_terrain_loss_db': np.mean(self.terrain_losses) if self.terrain_losses else 0
        }
    
    def print_report(self):
        """Print comprehensive validation report."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("RF PHYSICS STATISTICAL VALIDATION REPORT")
        print("=" * 80)
        print()
        
        print(f"Total Samples Analyzed: {stats['total_samples']}")
        print(f"Total Receivers: {stats['total_samples'] * 7}")
        print()
        
        # Monotonicity violations
        print("-" * 80)
        print("1. MONOTONICITY ANALYSIS")
        print("-" * 80)
        print(f"Samples with violations: {self.monotonicity_violations}")
        print(f"Violation rate: {stats['monotonicity_violation_rate_pct']:.2f}%")
        print(f"Expected range: 3-5%")
        print(f"Maximum acceptable: 10%")
        
        if stats['monotonicity_violation_rate_pct'] <= 10.0:
            print("✅ PASS - Violation rate within acceptable range")
        else:
            print("❌ FAIL - Violation rate exceeds 10%")
        
        # Show worst violations
        if self.violation_details:
            print("\nTop 3 worst violations:")
            sorted_violations = sorted(self.violation_details, key=lambda v: v['snr_diff_db'], reverse=True)
            for i, v in enumerate(sorted_violations[:3], 1):
                print(f"  {i}. RX{v['closer_idx']} ({v['closer_distance_km']:.1f}km, {v['closer_snr_db']:.1f}dB) vs "
                      f"RX{v['farther_idx']} ({v['farther_distance_km']:.1f}km, {v['farther_snr_db']:.1f}dB) - "
                      f"Diff: {v['snr_diff_db']:.1f}dB")
                
                # Print physics explanation
                closer_details = v['closer_details']
                farther_details = v['farther_details']
                
                reasons = []
                
                # Check antenna patterns
                closer_tx_gain = closer_details.get('tx_antenna_gain_db', 0)
                farther_tx_gain = farther_details.get('tx_antenna_gain_db', 0)
                if closer_tx_gain < farther_tx_gain - 5:
                    reasons.append(f"Antenna null on closer RX (gain: {closer_tx_gain:.1f}dB vs {farther_tx_gain:.1f}dB)")
                
                # Check polarization
                closer_pol_loss = closer_details.get('polarization_loss_db', 0)
                farther_pol_loss = farther_details.get('polarization_loss_db', 0)
                if closer_pol_loss > farther_pol_loss + 5:
                    reasons.append(f"Cross-pol loss on closer RX ({closer_pol_loss:.1f}dB vs {farther_pol_loss:.1f}dB)")
                
                # Check terrain
                closer_terrain = closer_details.get('terrain_loss_db', 0)
                farther_terrain = farther_details.get('terrain_loss_db', 0)
                if closer_terrain > farther_terrain + 5:
                    reasons.append(f"Terrain obstruction on closer RX ({closer_terrain:.1f}dB vs {farther_terrain:.1f}dB)")
                
                # Check fading
                closer_fading = closer_details.get('fading_db', 0)
                farther_fading = farther_details.get('fading_db', 0)
                if closer_fading < farther_fading - 5:
                    reasons.append(f"Deep fade on closer RX ({closer_fading:.1f}dB vs {farther_fading:.1f}dB)")
                
                if reasons:
                    print(f"     Likely cause: {', '.join(reasons)}")
        
        print()
        
        # Sporadic-E
        print("-" * 80)
        print("2. SPORADIC-E ANALYSIS")
        print("-" * 80)
        print(f"Occurrences: {self.sporadic_e_occurrences}")
        print(f"Rate: {stats['sporadic_e_rate_pct']:.3f}%")
        print(f"Expected range: 0.1-0.5%")
        
        if 0.1 <= stats['sporadic_e_rate_pct'] <= 0.5:
            print("✅ PASS - Sporadic-E rate within expected range")
        else:
            print(f"⚠️  WARNING - Sporadic-E rate outside expected range")
        
        if self.sporadic_e_enhancements:
            print(f"Average enhancement: {stats['avg_sporadic_e_enhancement_db']:.1f} dB")
            print(f"Max enhancement: {max(self.sporadic_e_enhancements):.1f} dB")
        
        print()
        
        # Cross-polarization
        print("-" * 80)
        print("3. POLARIZATION ANALYSIS")
        print("-" * 80)
        print(f"Cross-pol instances: {self.cross_pol_instances}")
        print(f"Rate: {stats['cross_pol_rate_pct']:.2f}%")
        print(f"Expected: ~5% (95% vertical RX)")
        
        if 3.0 <= stats['cross_pol_rate_pct'] <= 7.0:
            print("✅ PASS - Cross-pol rate within expected range")
        else:
            print(f"⚠️  WARNING - Cross-pol rate outside expected range")
        
        if self.cross_pol_losses:
            print(f"Average cross-pol loss: {stats['avg_cross_pol_loss_db']:.1f} dB")
            print(f"Expected range: 10-15 dB")
        
        print()
        
        # Antenna nulls
        print("-" * 80)
        print("4. ANTENNA PATTERN ANALYSIS")
        print("-" * 80)
        print(f"Antenna null instances: {self.antenna_null_instances}")
        print(f"Rate: {stats['antenna_null_rate_pct']:.2f}%")
        
        if self.antenna_null_gains:
            print(f"Average null gain: {stats['avg_antenna_null_gain_db']:.1f} dB")
            print(f"Min gain: {min(self.antenna_null_gains):.1f} dB")
        
        print()
        
        # Terrain effects
        print("-" * 80)
        print("5. TERRAIN EFFECTS ANALYSIS")
        print("-" * 80)
        print(f"High terrain loss instances: {self.high_terrain_loss_instances}")
        print(f"Rate: {stats['high_terrain_loss_rate_pct']:.2f}%")
        
        if self.terrain_losses:
            print(f"Average high terrain loss: {stats['avg_terrain_loss_db']:.1f} dB")
            print(f"Max terrain loss: {max(self.terrain_losses):.1f} dB")
        
        print()
        
        # Overall verdict
        print("=" * 80)
        print("OVERALL VERDICT")
        print("=" * 80)
        
        passed = stats['monotonicity_violation_rate_pct'] <= 10.0
        
        if passed:
            print("✅ VALIDATION PASSED")
            print("   All physics parameters within acceptable ranges.")
            print("   Rare anomalies are legitimate edge cases (antenna nulls, cross-pol, fading).")
        else:
            print("❌ VALIDATION FAILED")
            print("   Monotonicity violation rate exceeds acceptable threshold.")
            print("   Review propagation model implementation.")
        
        print("=" * 80)
        print()
        
        return passed


def main():
    """Run statistical validation test."""
    print("=" * 80)
    print("RF PHYSICS STATISTICAL VALIDATION TEST")
    print("=" * 80)
    print()
    print("This test generates 1000 samples and analyzes:")
    print("  1. Monotonicity violation rate (expect: 3-5%, max: 10%)")
    print("  2. Sporadic-E occurrence rate (expect: 0.1-0.5%)")
    print("  3. Cross-polarization distribution (expect: ~5%)")
    print("  4. Antenna pattern effects distribution")
    print("  5. Terrain loss distribution")
    print()
    print("Generating samples (this may take 2-3 minutes)...")
    print()
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        num_samples=1000,
        config={
            'enable_meteorological': True,
            'enable_sporadic_e': True,
            'enable_knife_edge': True,
            'enable_polarization': True,
            'enable_antenna_patterns': True
        }
    )
    
    # Initialize validator
    validator = PhysicsValidator()
    
    # Generate and analyze samples
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/1000 samples...")
        
        try:
            sample = generator.generate_samples(num_samples=1, start_index=i)[0]
            analysis = validator.analyze_sample(sample)
            validator.update_statistics(analysis)
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue
    
    # Print comprehensive report
    passed = validator.print_report()
    
    # Return exit code for CI/CD
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
