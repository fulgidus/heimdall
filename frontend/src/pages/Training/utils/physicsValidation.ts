/**
 * Physics Validation Utilities
 * 
 * Provides validation functions for RF propagation physics sanity checks:
 * - Distance-SNR monotonicity checks
 * - Sporadic-E event detection
 * - Polarization mismatch detection
 * - Antenna pattern null detection
 */

import type { ReceiverMetadata } from '../types';

export interface MonotonicityWarning {
  type: 'monotonicity_violation';
  severity: 'warning' | 'info';
  rx_closer: string;
  rx_farther: string;
  distance_closer_km: number;
  distance_farther_km: number;
  snr_closer_db: number;
  snr_farther_db: number;
  snr_difference_db: number;
  message: string;
}

export interface PhysicsAnomaly {
  rx_id: string;
  type: 'sporadic_e' | 'cross_pol' | 'antenna_null' | 'extreme_fading';
  severity: 'info' | 'warning';
  value: number;
  label: string;
  tooltip: string;
}

/**
 * Check if receivers follow expected distance-SNR relationship
 * 
 * Physical expectation: SNR should decrease with distance (due to FSPL)
 * Tolerance: 12 dB (accounts for antenna patterns, polarization, fading)
 * 
 * @param receivers - Array of receiver measurements
 * @returns Array of monotonicity warnings
 */
export function checkMonotonicity(receivers: ReceiverMetadata[]): MonotonicityWarning[] {
  const warnings: MonotonicityWarning[] = [];
  
  // Sort receivers by distance (ascending)
  const sorted = [...receivers].sort((a, b) => a.distance_km - b.distance_km);
  
  // Check each pair of adjacent receivers
  for (let i = 0; i < sorted.length - 1; i++) {
    const closer = sorted[i];
    const farther = sorted[i + 1];
    
    // Calculate SNR difference
    const snr_diff = farther.snr_db - closer.snr_db;
    
    // Allow 12 dB tolerance for antenna patterns, polarization, and fading
    // If farther receiver has >12 dB better SNR, flag as potential anomaly
    if (snr_diff > 12) {
      warnings.push({
        type: 'monotonicity_violation',
        severity: snr_diff > 20 ? 'warning' : 'info',
        rx_closer: closer.rx_id,
        rx_farther: farther.rx_id,
        distance_closer_km: closer.distance_km,
        distance_farther_km: farther.distance_km,
        snr_closer_db: closer.snr_db,
        snr_farther_db: farther.snr_db,
        snr_difference_db: snr_diff,
        message: `${closer.rx_id} (${closer.distance_km.toFixed(1)} km) has ${Math.abs(snr_diff).toFixed(1)} dB worse SNR than ${farther.rx_id} (${farther.distance_km.toFixed(1)} km)`
      });
    }
  }
  
  return warnings;
}

/**
 * Check if receiver has significant sporadic-E enhancement
 * 
 * Sporadic-E probability should be 0.1-0.5% after physics fixes.
 * Enhancement typically 20-40 dB for VHF/UHF.
 * 
 * @param rx - Receiver metadata
 * @returns True if significant sporadic-E detected
 */
export function checkSporadicE(rx: ReceiverMetadata): boolean {
  return (rx.sporadic_e_enhancement_db ?? 0) > 15;
}

/**
 * Check if receiver has significant polarization mismatch loss
 * 
 * Cross-pol loss should be 10-15 dB with depolarization model.
 * Same-pol should be <3 dB.
 * 
 * @param rx - Receiver metadata
 * @returns True if significant cross-pol loss detected
 */
export function checkPolarizationMismatch(rx: ReceiverMetadata): boolean {
  return (rx.polarization_loss_db ?? 0) > 10;
}

/**
 * Check if receiver is in antenna pattern null
 * 
 * Antenna nulls can cause -10 to -20 dB loss.
 * 
 * @param rx - Receiver metadata
 * @returns True if antenna null detected (gain < -5 dB)
 */
export function checkAntennaNull(rx: ReceiverMetadata): boolean {
  const tx_gain = rx.tx_antenna_gain_db ?? 0;
  const rx_gain = rx.rx_antenna_gain_db ?? 0;
  return tx_gain < -5 || rx_gain < -5;
}

/**
 * Check if receiver has extreme fading
 * 
 * Fading range: -20 to +10 dB (clipped in physics model)
 * Extreme fading: < -15 dB or > +8 dB
 * 
 * @param rx - Receiver metadata
 * @returns True if extreme fading detected
 */
export function checkExtremeFading(rx: ReceiverMetadata): boolean {
  // Calculate fading from power budget if individual components available
  if (rx.fspl_db && rx.rx_power_dbm && rx.snr_db) {
    // Fading is what's left after accounting for known losses/gains
    // This is approximate - we'd need full breakdown
    // For now, use heuristic: if terrain_loss is very high, might be fading
    return (rx.terrain_loss_db ?? 0) > 15;
  }
  return false;
}

/**
 * Get all physics anomalies for a receiver
 * 
 * @param rx - Receiver metadata
 * @returns Array of detected anomalies
 */
export function getPhysicsAnomalies(rx: ReceiverMetadata): PhysicsAnomaly[] {
  const anomalies: PhysicsAnomaly[] = [];
  
  // Check sporadic-E
  if (checkSporadicE(rx)) {
    anomalies.push({
      rx_id: rx.rx_id,
      type: 'sporadic_e',
      severity: 'info',
      value: rx.sporadic_e_enhancement_db ?? 0,
      label: 'Sporadic-E',
      tooltip: `Sporadic-E ionospheric propagation detected (+${(rx.sporadic_e_enhancement_db ?? 0).toFixed(1)} dB). This is a rare event (~0.3% probability).`
    });
  }
  
  // Check polarization mismatch
  if (checkPolarizationMismatch(rx)) {
    anomalies.push({
      rx_id: rx.rx_id,
      type: 'cross_pol',
      severity: 'warning',
      value: rx.polarization_loss_db ?? 0,
      label: 'Cross-Pol',
      tooltip: `Polarization mismatch detected: ${rx.tx_polarization ?? 'unknown'} TX → ${rx.rx_polarization ?? 'unknown'} RX (-${(rx.polarization_loss_db ?? 0).toFixed(1)} dB loss)`
    });
  }
  
  // Check antenna null
  if (checkAntennaNull(rx)) {
    const tx_gain = rx.tx_antenna_gain_db ?? 0;
    const rx_gain = rx.rx_antenna_gain_db ?? 0;
    const min_gain = Math.min(tx_gain, rx_gain);
    
    anomalies.push({
      rx_id: rx.rx_id,
      type: 'antenna_null',
      severity: 'info',
      value: min_gain,
      label: 'Antenna Null',
      tooltip: `Receiver is in antenna pattern null (TX gain: ${tx_gain.toFixed(1)} dBi, RX gain: ${rx_gain.toFixed(1)} dBi). This causes significant signal attenuation.`
    });
  }
  
  // Check extreme fading
  if (checkExtremeFading(rx)) {
    anomalies.push({
      rx_id: rx.rx_id,
      type: 'extreme_fading',
      severity: 'info',
      value: rx.terrain_loss_db ?? 0,
      label: 'High Terrain Loss',
      tooltip: `High terrain loss detected (-${(rx.terrain_loss_db ?? 0).toFixed(1)} dB). Path may be obstructed by terrain features.`
    });
  }
  
  return anomalies;
}

/**
 * Get physics validation summary for a sample
 * 
 * @param receivers - Array of receiver measurements
 * @returns Summary object with validation results
 */
export function getPhysicsValidationSummary(receivers: ReceiverMetadata[]): {
  monotonicity_warnings: MonotonicityWarning[];
  sporadic_e_count: number;
  cross_pol_count: number;
  antenna_null_count: number;
  extreme_fading_count: number;
  has_issues: boolean;
} {
  const monotonicity_warnings = checkMonotonicity(receivers);
  const sporadic_e_count = receivers.filter(checkSporadicE).length;
  const cross_pol_count = receivers.filter(checkPolarizationMismatch).length;
  const antenna_null_count = receivers.filter(checkAntennaNull).length;
  const extreme_fading_count = receivers.filter(checkExtremeFading).length;
  
  return {
    monotonicity_warnings,
    sporadic_e_count,
    cross_pol_count,
    antenna_null_count,
    extreme_fading_count,
    has_issues: monotonicity_warnings.length > 0 || sporadic_e_count > 0 || cross_pol_count > 0
  };
}

/**
 * Calculate expected FSPL for validation
 * 
 * @param distance_km - Distance in kilometers
 * @param frequency_mhz - Frequency in MHz
 * @returns Expected FSPL in dB
 */
export function calculateExpectedFSPL(distance_km: number, frequency_mhz: number): number {
  return 32.45 + 20 * Math.log10(frequency_mhz) + 20 * Math.log10(distance_km);
}

/**
 * Get badge color for anomaly type
 * 
 * @param type - Anomaly type
 * @returns Bootstrap badge color
 */
export function getAnomalyBadgeColor(type: PhysicsAnomaly['type']): string {
  switch (type) {
    case 'sporadic_e':
      return 'info';
    case 'cross_pol':
      return 'warning';
    case 'antenna_null':
      return 'secondary';
    case 'extreme_fading':
      return 'dark';
    default:
      return 'light';
  }
}
