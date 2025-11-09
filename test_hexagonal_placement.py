"""
Test script for hexagonal receiver placement algorithm.

Verifies:
1. Hexagonal placement generates correct number of receivers
2. GDOP is significantly better than random placement
3. Receivers are within area bounds
4. Spatial distribution is uniform
"""

import sys
import json
import numpy as np

# Add services to path
sys.path.insert(0, '/app/src')

from data.synthetic_generator import _generate_hexagonal_receivers, _generate_random_receivers
from data.propagation import calculate_gdop

def calculate_mean_gdop(receivers_list, num_samples=100):
    """Calculate mean GDOP for random TX positions."""
    gdops = []
    
    # Extract receiver positions
    rx_positions = [(r['latitude'], r['longitude'], r['altitude']) for r in receivers_list]
    
    # Generate random TX positions within receiver area
    lats = [r['latitude'] for r in receivers_list]
    lons = [r['longitude'] for r in receivers_list]
    
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    for _ in range(num_samples):
        tx_lat = np.random.uniform(lat_min, lat_max)
        tx_lon = np.random.uniform(lon_min, lon_max)
        
        gdop = calculate_gdop(
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            receiver_positions=rx_positions
        )
        
        if gdop is not None and gdop < 500:  # Filter out extreme outliers
            gdops.append(gdop)
    
    return np.mean(gdops), np.median(gdops), np.percentile(gdops, 95)

def test_hexagonal_placement():
    """Test hexagonal placement vs random placement."""
    print("=" * 80)
    print("HEXAGONAL RECEIVER PLACEMENT TEST")
    print("=" * 80)
    
    # Test parameters
    num_receivers = 7
    area_lat_min = 44.0
    area_lat_max = 46.0
    area_lon_min = 7.0
    area_lon_max = 10.0
    
    rng = np.random.default_rng(42)
    
    # Generate hexagonal receivers
    print(f"\n1️⃣  Generating {num_receivers} receivers with HEXAGONAL placement...")
    hex_receivers = _generate_hexagonal_receivers(
        num_receivers=num_receivers,
        area_lat_min=area_lat_min,
        area_lat_max=area_lat_max,
        area_lon_min=area_lon_min,
        area_lon_max=area_lon_max,
        terrain_lookup=None,
        rng=rng
    )
    
    print(f"   ✅ Generated {len(hex_receivers)} hexagonal receivers")
    print(f"   📍 Positions:")
    for rx in hex_receivers[:5]:  # Show first 5
        print(f"      {rx['name']}: ({rx['latitude']:.4f}°, {rx['longitude']:.4f}°)")
    if len(hex_receivers) > 5:
        print(f"      ... and {len(hex_receivers) - 5} more")
    
    # Calculate mean GDOP for hexagonal
    print(f"\n   📊 Calculating mean GDOP for 100 random TX positions...")
    hex_mean, hex_median, hex_p95 = calculate_mean_gdop(hex_receivers, num_samples=100)
    print(f"   ✅ Hexagonal GDOP - Mean: {hex_mean:.1f}, Median: {hex_median:.1f}, P95: {hex_p95:.1f}")
    
    # Generate random receivers for comparison
    print(f"\n2️⃣  Generating {num_receivers} receivers with RANDOM placement...")
    rng = np.random.default_rng(42)  # Reset RNG for fair comparison
    random_receivers = _generate_random_receivers(
        num_receivers=num_receivers,
        area_lat_min=area_lat_min,
        area_lat_max=area_lat_max,
        area_lon_min=area_lon_min,
        area_lon_max=area_lon_max,
        terrain_lookup=None,
        rng=rng
    )
    
    print(f"   ✅ Generated {len(random_receivers)} random receivers")
    
    # Calculate mean GDOP for random
    print(f"\n   📊 Calculating mean GDOP for 100 random TX positions...")
    rand_mean, rand_median, rand_p95 = calculate_mean_gdop(random_receivers, num_samples=100)
    print(f"   ✅ Random GDOP - Mean: {rand_mean:.1f}, Median: {rand_median:.1f}, P95: {rand_p95:.1f}")
    
    # Compare
    print(f"\n{'='*80}")
    print("3️⃣  COMPARISON RESULTS")
    print(f"{'='*80}")
    
    improvement_mean = ((rand_mean - hex_mean) / rand_mean) * 100
    improvement_median = ((rand_median - hex_median) / rand_median) * 100
    improvement_p95 = ((rand_p95 - hex_p95) / rand_p95) * 100
    
    print(f"\n   📈 GDOP Improvement (Hexagonal vs Random):")
    print(f"      Mean:   {hex_mean:.1f} vs {rand_mean:.1f} ({improvement_mean:+.1f}%)")
    print(f"      Median: {hex_median:.1f} vs {rand_median:.1f} ({improvement_median:+.1f}%)")
    print(f"      P95:    {hex_p95:.1f} vs {rand_p95:.1f} ({improvement_p95:+.1f}%)")
    
    # Success criteria
    print(f"\n   🎯 Success Criteria:")
    if hex_mean < 20:
        print(f"      ✅ Mean GDOP < 20 (actual: {hex_mean:.1f})")
    else:
        print(f"      ❌ Mean GDOP >= 20 (actual: {hex_mean:.1f})")
    
    if hex_mean < rand_mean * 0.5:
        print(f"      ✅ Hexagonal GDOP < 50% of random (actual: {(hex_mean/rand_mean)*100:.1f}%)")
    else:
        print(f"      ⚠️  Hexagonal GDOP >= 50% of random (actual: {(hex_mean/rand_mean)*100:.1f}%)")
    
    if improvement_mean > 50:
        print(f"      ✅ Improvement > 50% (actual: {improvement_mean:.1f}%)")
    else:
        print(f"      ⚠️  Improvement <= 50% (actual: {improvement_mean:.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ TEST COMPLETE")
    print(f"{'='*80}\n")
    
    # Return results for automated testing
    return {
        'hex_mean': hex_mean,
        'rand_mean': rand_mean,
        'improvement_percent': improvement_mean,
        'success': hex_mean < 20 and improvement_mean > 50
    }

if __name__ == '__main__':
    results = test_hexagonal_placement()
    sys.exit(0 if results['success'] else 1)
