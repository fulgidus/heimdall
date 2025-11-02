#!/usr/bin/env python3
"""
Fixed SRTM verification script for training container.
"""

import sys
import os

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/common')

print("\n" + "="*80)
print("SRTM INTEGRATION VERIFICATION")
print("="*80)

# Test 1: Check backend reachability
print("\n[1/4] Testing backend reachability...")
try:
    import requests
    response = requests.get("http://backend:8000/health", timeout=5)
    print(f"✅ PASS: Backend reachable (status {response.status_code})")
except Exception as e:
    print(f"⚠️  WARNING: Backend not reachable: {e}")
    print("   (This is expected if backend is down, but terrain can still work)")

# Test 2: TerrainLookup with MinIO
print("\n[2/4] Testing TerrainLookup with MinIO...")
try:
    # Import from backend mount point
    sys.path.insert(0, '/app/backend/src')
    from storage.minio_client import MinIOClient
    from terrain.terrain import TerrainLookup
    
    # Initialize MinIO client (use actual env vars from container)
    minio_endpoint = os.getenv("MINIO_URL", "http://minio:9000")
    # Remove http:// if present
    if minio_endpoint.startswith("http://"):
        minio_endpoint = minio_endpoint
    elif minio_endpoint.startswith("https://"):
        pass  # Keep as is
    else:
        minio_endpoint = f"http://{minio_endpoint}"
        
    minio_client = MinIOClient(
        endpoint_url=minio_endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket_name="heimdall-terrain"
    )
    
    # Check if tiles exist using s3_client
    try:
        response = minio_client.s3_client.list_objects_v2(
            Bucket="heimdall-terrain",
            MaxKeys=10
        )
        tiles = response.get('Contents', [])
        print(f"   Found {len(tiles)} files in MinIO bucket (showing max 10)")
        
        if len(tiles) == 0:
            print("   ⚠️  WARNING: No SRTM tiles found in MinIO")
            print("   Download tiles via UI: Terrain Management -> Download WebSDR Region")
            terrain_available = False
        else:
            print(f"   First 3 tiles: {[t['Key'] for t in tiles[:3]]}")
            terrain_available = True
    except Exception as e:
        print(f"   ⚠️  WARNING: Could not list MinIO objects: {e}")
        terrain_available = False
        tiles = []
    
    # Initialize TerrainLookup (always, even if no tiles found)
    lookup = TerrainLookup(use_srtm=True, minio_client=minio_client)
    
    # Test elevation query
    elev = lookup.get_elevation(45.0, 7.5)
    print(f"✅ PASS: TerrainLookup working, elevation at (45.0, 7.5) = {elev:.1f}m")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    terrain_available = False
    lookup = None

# Test 3: RF Propagation WITH and WITHOUT terrain
print("\n[3/4] Testing RF propagation comparison...")
try:
    from src.data.propagation import RFPropagationModel
    
    model = RFPropagationModel()
    
    # Test WITHOUT terrain (baseline)
    result_without = model.calculate_received_power(
        tx_power_dbm=37.0,
        tx_lat=45.0, tx_lon=7.5, tx_alt=300.0,
        rx_lat=45.3, rx_lon=8.0, rx_alt=200.0,
        frequency_mhz=145.0,
        terrain_lookup=None
    )
    power_without, snr_without, details_without = result_without
    print(f"   WITHOUT terrain:")
    print(f"     Rx power: {power_without:.2f} dBm, SNR: {snr_without:.2f} dB")
    print(f"     Terrain loss: {details_without.get('terrain_loss_db', 0):.2f} dB")
    
    # Test WITH terrain (if available)
    if terrain_available and lookup is not None:
        result_with = model.calculate_received_power(
            tx_power_dbm=37.0,
            tx_lat=45.0, tx_lon=7.5, tx_alt=300.0,
            rx_lat=45.3, rx_lon=8.0, rx_alt=200.0,
            frequency_mhz=145.0,
            terrain_lookup=lookup
        )
        power_with, snr_with, details_with = result_with
        print(f"   WITH SRTM terrain:")
        print(f"     Rx power: {power_with:.2f} dBm, SNR: {snr_with:.2f} dB")
        print(f"     Terrain loss: {details_with.get('terrain_loss_db', 0):.2f} dB")
        
        # Compare
        power_diff = abs(power_with - power_without)
        terrain_loss_diff = details_with.get('terrain_loss_db', 0) - details_without.get('terrain_loss_db', 0)
        
        print(f"\n   COMPARISON:")
        print(f"     Power difference: {power_diff:.2f} dB")
        print(f"     Terrain loss increase: {terrain_loss_diff:.2f} dB")
        
        if terrain_loss_diff > 2.0:
            print(f"✅ PASS: SRTM terrain significantly affects propagation ({terrain_loss_diff:.1f} dB)")
        elif terrain_loss_diff > 0.5:
            print(f"⚠️  MARGINAL: Small terrain effect ({terrain_loss_diff:.1f} dB), may be flat area")
        else:
            print(f"❌ FAIL: No terrain effect detected, SRTM may not be working")
    else:
        print("   ⚠️  SKIPPED: No SRTM tiles available for WITH terrain test")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 4: SyntheticDataGenerator supports use_srtm_terrain
print("\n[4/4] Testing SyntheticDataGenerator SRTM support...")
try:
    from src.data.synthetic_generator import SyntheticDataGenerator
    import inspect
    
    # Check if use_srtm_terrain parameter exists
    sig = inspect.signature(SyntheticDataGenerator.__init__)
    params = list(sig.parameters.keys())
    
    if 'use_srtm_terrain' in params:
        print("✅ PASS: SyntheticDataGenerator has 'use_srtm_terrain' parameter")
        print(f"   Constructor params: {params}")
        
        # Check if terrain_lookup is actually used
        import ast
        import textwrap
        source = inspect.getsource(SyntheticDataGenerator.__init__)
        
        if 'terrain_lookup' in source:
            print("   ✅ Code references 'terrain_lookup' in __init__")
        else:
            print("   ⚠️  WARNING: 'terrain_lookup' not found in __init__")
            
        # Check _generate_single_sample
        try:
            gen_source = inspect.getsource(SyntheticDataGenerator._generate_single_sample)
            if 'terrain_lookup=None' in gen_source:
                print("   ❌ CRITICAL: _generate_single_sample() passes terrain_lookup=None!")
                print("      This means SRTM is NOT used even if use_srtm_terrain=True")
            elif 'terrain_lookup=' in gen_source:
                print("   ✅ _generate_single_sample() passes terrain_lookup correctly")
            else:
                print("   ⚠️  Could not determine terrain_lookup usage in generation")
        except:
            print("   ⚠️  Could not inspect _generate_single_sample")
    else:
        print(f"❌ FAIL: Parameter 'use_srtm_terrain' not found in constructor")
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\n📝 SUMMARY:")
print("1. Backend connectivity: Optional (can work standalone)")
print("2. TerrainLookup: CRITICAL - needs SRTM tiles in MinIO")
print("3. Propagation model: CRITICAL - must show terrain effect")
print("4. Generator config: CRITICAL - must pass terrain_lookup correctly")
print("\n💡 If SRTM tiles missing:")
print("   1. Go to http://localhost/terrain")
print("   2. Click 'Download WebSDR Region Tiles'")
print("   3. Wait for download to complete")
print("   4. Re-run this verification script")
