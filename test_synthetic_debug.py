#!/usr/bin/env python3
"""
Debug script to identify the NoneType error in synthetic data generation.
"""
import sys
sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

import numpy as np
from src.data.propagation import RFPropagationModel, AntennaType, AntennaPattern
from src.data.config import MeteorologicalParameters
from common.terrain import TerrainLookup

# Create a minimal test case
rng = np.random.default_rng(42)

# Create propagation model
propagation = RFPropagationModel()
terrain = TerrainLookup(use_srtm=False)

# Test parameters (from a typical sample)
tx_power_dbm = 37.0
tx_lat = 43.5
tx_lon = 11.5
tx_alt = 300.0
rx_lat = 44.0
rx_lon = 12.0
rx_alt = 100.0
frequency_mhz = 145.0

# Create antenna patterns
tx_antenna = AntennaPattern(AntennaType.WHIP, 0.0)
rx_antenna = AntennaPattern(AntennaType.OMNI_VERTICAL, 0.0)

# Create meteorological parameters
meteo_params = MeteorologicalParameters.random(seed=42)

print("Testing propagation calculation...")
print(f"TX Power: {tx_power_dbm} dBm")
print(f"Frequency: {frequency_mhz} MHz")
print(f"TX Antenna: {tx_antenna}")
print(f"RX Antenna: {rx_antenna}")
print(f"Meteo params: {meteo_params}")

try:
    rx_power_dbm, snr_db, details = propagation.calculate_received_power(
        tx_power_dbm=tx_power_dbm,
        tx_lat=tx_lat,
        tx_lon=tx_lon,
        tx_alt=tx_alt,
        rx_lat=rx_lat,
        rx_lon=rx_lon,
        rx_alt=rx_alt,
        frequency_mhz=frequency_mhz,
        terrain_lookup=terrain,
        tx_antenna=tx_antenna,
        rx_antenna=rx_antenna,
        meteo_params=meteo_params,
        transmitter_quality=rng.uniform(0.3, 0.95),
        transmission_duty_cycle=rng.uniform(0.8, 1.0),
        enable_sporadic_e=True,
        enable_knife_edge=True,
        enable_polarization_effects=True
    )
    print(f"\n✓ Success!")
    print(f"RX Power: {rx_power_dbm:.2f} dBm")
    print(f"SNR: {snr_db:.2f} dB")
    print(f"Distance: {details['distance_km']:.2f} km")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
