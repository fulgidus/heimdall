#!/usr/bin/env python3
"""
Simplified SRTM verification script that runs inside training container.
"""

import sys
import os

# Container paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/common')

import numpy as np
import requests
import json

print("\n" + "="*80)
print("SRTM INTEGRATION VERIFICATION")
print("="*80)

# Test 1: API availability
print("\n[1/4] Testing API availability...")
try:
    response = requests.get("http://backend:8000/api/v1/terrain/tiles", timeout=5)
    if response.status_code == 200:
        tiles = response.json()
        print(f"✅ PASS: API accessible, {len(tiles)} tiles available")
        for tile in tiles[:3]:  # Show first 3
            print(f"   - {tile['name']}: {tile['status']}")
    else:
        print(f"❌ FAIL: API returned status {response.status_code}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# Test 2: TerrainLookup
print("\n[2/4] Testing TerrainLookup...")
try:
    from terrain.terrain import TerrainLookup
    
    lookup = TerrainLookup(
        minio_endpoint="minio:9000",
        minio_access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        minio_secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        bucket_name="heimdall-terrain"
    )
    
    # Test elevation query
    elev = lookup.get_elevation(45.0, 7.5)
    print(f"✅ PASS: TerrainLookup initialized, elevation at (45.0, 7.5) = {elev}m")
except Exception as e:
    print(f"❌ FAIL: {e}")

# Test 3: Propagation with terrain
print("\n[3/4] Testing RF propagation with terrain...")
try:
    from src.data.propagation import RFPropagationModel
    
    model = RFPropagationModel()
    
    # Test WITH terrain
    try:
        power_with = model.calculate_received_power(
            tx_lat=45.0, tx_lon=7.5, tx_alt=300.0,
            rx_lat=45.3, rx_lon=8.0, rx_alt=200.0,
            tx_power_dbm=37.0, frequency_mhz=145.0,
            terrain_lookup=lookup
        )
        print(f"   WITH terrain: {power_with:.2f} dBm")
    except Exception as e:
        print(f"   WITH terrain: ERROR - {e}")
        power_with = None
    
    # Test WITHOUT terrain
    power_without = model.calculate_received_power(
        tx_lat=45.0, tx_lon=7.5, tx_alt=300.0,
        rx_lat=45.3, rx_lon=8.0, rx_alt=200.0,
        tx_power_dbm=37.0, frequency_mhz=145.0,
        terrain_lookup=None
    )
    print(f"   WITHOUT terrain: {power_without:.2f} dBm")
    
    if power_with is not None:
        diff = abs(power_with - power_without)
        if diff > 2.0:  # Should differ by at least 2 dB
            print(f"✅ PASS: Terrain effect detected ({diff:.2f} dB difference)")
        else:
            print(f"⚠️  WARNING: Small difference ({diff:.2f} dB), terrain may not be used")
    else:
        print(f"❌ FAIL: Could not test WITH terrain")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 4: SyntheticDataGenerator config
print("\n[4/4] Testing SyntheticDataGenerator configuration...")
try:
    from src.data.synthetic_generator import SyntheticDataGenerator
    
    # Check if use_srtm_terrain parameter exists
    import inspect
    sig = inspect.signature(SyntheticDataGenerator.__init__)
    params = list(sig.parameters.keys())
    
    if 'use_srtm_terrain' in params:
        print(f"✅ PASS: SyntheticDataGenerator has 'use_srtm_terrain' parameter")
        
        # Try to instantiate
        try:
            gen = SyntheticDataGenerator(
                num_samples=1,
                use_srtm_terrain=True
            )
            print(f"   Generator instantiated successfully")
            print(f"   terrain_lookup: {gen.terrain_lookup is not None}")
        except Exception as e:
            print(f"   ⚠️  WARNING: Could not instantiate: {e}")
    else:
        print(f"❌ FAIL: Parameter 'use_srtm_terrain' not found")
        print(f"   Available params: {params}")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
