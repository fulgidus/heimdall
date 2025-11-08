#!/usr/bin/env python3
"""
Verification script for SRTM terrain integration in synthetic data generation.

This script tests that:
1. Terrain tiles are available
2. Propagation model correctly uses SRTM data
3. Terrain loss calculations differ from simplified model
4. Line-of-sight checks work correctly

Usage:
    python scripts/verify_srtm_integration.py
"""

import sys
import os
import numpy as np
import requests
import json

# Add backend to path
sys.path.insert(0, '/app/backend/src')
sys.path.insert(0, '/app/common')

from training.src.data.propagation import RFPropagationModel
from common.terrain import TerrainLookup
from training.src.config import settings as backend_settings
from storage.minio_client import MinIOClient


def test_terrain_availability():
    """Test 1: Check if SRTM tiles are available."""
    print("\n" + "="*80)
    print("TEST 1: Terrain Tile Availability")
    print("="*80)
    
    response = requests.get("http://localhost:8001/api/v1/terrain/tiles")
    if response.status_code != 200:
        print(f"❌ FAILED: Cannot reach terrain API (status {response.status_code})")
        return False
    
    data = response.json()
    ready_tiles = [t for t in data['tiles'] if t['status'] == 'ready']
    
    print(f"✅ API reachable")
    print(f"   Total tiles: {data['total']}")
    print(f"   Ready tiles: {len(ready_tiles)}")
    print(f"   Ready tile names: {[t['tile_name'] for t in ready_tiles]}")
    
    if len(ready_tiles) == 0:
        print("❌ FAILED: No ready tiles found")
        print("   ACTION: Download tiles first via UI or API")
        return False
    
    print("✅ PASSED: Terrain tiles available")
    return True


def test_terrain_lookup():
    """Test 2: Test TerrainLookup can load and query elevations."""
    print("\n" + "="*80)
    print("TEST 2: Terrain Lookup Functionality")
    print("="*80)
    
    try:
        # Initialize MinIO client
        minio_client = MinIOClient(
            endpoint_url=os.getenv('MINIO_URL', 'http://minio:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            bucket_name="heimdall-terrain"
        )
        
        # Create TerrainLookup with SRTM support
        terrain = TerrainLookup(use_srtm=True, minio_client=minio_client)
        
        # Test locations in Italy (where WebSDR stations are)
        test_points = [
            (45.0, 7.5, "Torino area"),
            (45.5, 9.0, "Milano area"),
            (44.5, 8.5, "Genova area"),
        ]
        
        print(f"✅ TerrainLookup initialized")
        
        for lat, lon, description in test_points:
            elevation = terrain.get_elevation(lat, lon)
            print(f"   {description}: ({lat}, {lon}) → {elevation:.1f}m ASL")
            
            if elevation is None:
                print(f"   ⚠️  WARNING: Elevation is None (tile may not be available)")
        
        print("✅ PASSED: Terrain lookup working")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_propagation_with_terrain():
    """Test 3: Compare propagation with and without terrain."""
    print("\n" + "="*80)
    print("TEST 3: RF Propagation with Terrain vs Simplified Model")
    print("="*80)
    
    try:
        # Initialize MinIO and terrain
        minio_client = MinIOClient(
            endpoint_url=os.getenv('MINIO_URL', 'http://minio:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
            bucket_name="heimdall-terrain"
        )
        terrain = TerrainLookup(use_srtm=True, minio_client=minio_client)
        propagation = RFPropagationModel()
        
        # Test scenario: TX in Torino, RX in Milano
        # Known line blocked by Alps
        tx_lat, tx_lon, tx_alt = 45.0, 7.5, 300.0  # Torino
        rx_lat, rx_lon, rx_alt = 45.5, 9.0, 200.0  # Milano
        frequency_mhz = 145.0
        tx_power_dbm = 37.0
        
        print(f"Test scenario:")
        print(f"   TX: ({tx_lat}, {tx_lon}) @ {tx_alt}m ASL")
        print(f"   RX: ({rx_lat}, {rx_lon}) @ {rx_alt}m ASL")
        print(f"   Frequency: {frequency_mhz} MHz")
        
        # Calculate WITHOUT terrain (simplified model)
        rx_power_simple, snr_simple, details_simple = propagation.calculate_received_power(
            tx_power_dbm=tx_power_dbm,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=frequency_mhz,
            terrain_lookup=None
        )
        
        # Calculate WITH terrain (SRTM)
        rx_power_srtm, snr_srtm, details_srtm = propagation.calculate_received_power(
            tx_power_dbm=tx_power_dbm,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=frequency_mhz,
            terrain_lookup=terrain
        )
        
        print("\n📊 Results:")
        print(f"\n   WITHOUT terrain (simplified):")
        print(f"      RX Power: {rx_power_simple:.1f} dBm")
        print(f"      SNR: {snr_simple:.1f} dB")
        print(f"      FSPL: {details_simple['fspl_db']:.1f} dB")
        print(f"      Terrain Loss: {details_simple['terrain_loss_db']:.1f} dB")
        print(f"      Total Loss: {details_simple['total_loss_db']:.1f} dB")
        
        print(f"\n   WITH terrain (SRTM):")
        print(f"      RX Power: {rx_power_srtm:.1f} dBm")
        print(f"      SNR: {snr_srtm:.1f} dB")
        print(f"      FSPL: {details_srtm['fspl_db']:.1f} dB")
        print(f"      Terrain Loss: {details_srtm['terrain_loss_db']:.1f} dB")
        print(f"      Total Loss: {details_srtm['total_loss_db']:.1f} dB")
        
        print(f"\n   DIFFERENCE:")
        terrain_diff = details_srtm['terrain_loss_db'] - details_simple['terrain_loss_db']
        snr_diff = snr_srtm - snr_simple
        print(f"      Terrain Loss Δ: {terrain_diff:+.1f} dB")
        print(f"      SNR Δ: {snr_diff:+.1f} dB")
        
        # Verify that SRTM provides different results
        if abs(terrain_diff) < 0.1:
            print("\n⚠️  WARNING: Terrain loss difference is negligible")
            print("   This might indicate SRTM data is not being used")
            print("   Expected: Higher terrain loss with SRTM (due to Alps)")
            return False
        
        if details_srtm['terrain_loss_db'] > details_simple['terrain_loss_db']:
            print("\n✅ PASSED: SRTM provides MORE realistic terrain loss")
            print("   This is expected for TX-RX path blocked by mountains")
        else:
            print("\n✅ PASSED: SRTM provides terrain-aware propagation")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_generation_config():
    """Test 4: Check if synthetic generation config supports SRTM."""
    print("\n" + "="*80)
    print("TEST 4: Synthetic Generation Configuration")
    print("="*80)
    
    # Check if we can inspect the synthetic generator code
    try:
        from training.src.data.synthetic_generator import SyntheticDataGenerator
        import inspect
        
        # Get __init__ signature
        sig = inspect.signature(SyntheticDataGenerator.__init__)
        params = list(sig.parameters.keys())
        
        print(f"✅ SyntheticDataGenerator found")
        print(f"   __init__ parameters: {params}")
        
        if 'use_srtm_terrain' in params:
            print(f"   ✅ 'use_srtm_terrain' parameter present")
        else:
            print(f"   ❌ 'use_srtm_terrain' parameter MISSING")
            return False
        
        if 'terrain_lookup' in params:
            print(f"   ✅ 'terrain_lookup' parameter present")
        else:
            print(f"   ⚠️  'terrain_lookup' parameter missing (optional)")
        
        print("\n✅ PASSED: Synthetic generator supports SRTM configuration")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("SRTM INTEGRATION VERIFICATION")
    print("="*80)
    print("\nThis script verifies that SRTM terrain data is properly integrated")
    print("into the synthetic data generation pipeline.")
    
    tests = [
        ("Terrain Tile Availability", test_terrain_availability),
        ("Terrain Lookup Functionality", test_terrain_lookup),
        ("RF Propagation with Terrain", test_propagation_with_terrain),
        ("Synthetic Generation Config", test_synthetic_generation_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All tests passed! SRTM integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. See details above.")
        print("\nNext steps:")
        print("1. Ensure terrain tiles are downloaded (visit Terrain Management page)")
        print("2. Check that backend can access MinIO and database")
        print("3. Verify synthetic data generation passes use_srtm_terrain=True")
        return 1


if __name__ == "__main__":
    sys.exit(main())
