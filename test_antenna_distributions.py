"""
Test script to verify antenna distribution configuration works correctly.
"""
import sys
import numpy as np
from collections import Counter

# Add services to path
sys.path.insert(0, '/home/fulgidus/Documents/Projects/heimdall/services/training/src')
sys.path.insert(0, '/home/fulgidus/Documents/Projects/heimdall/services')

# Import the functions we want to test
from services.training.src.data.synthetic_generator import _select_tx_antenna, _select_rx_antenna
from services.training.src.data.config import TxAntennaDistribution, RxAntennaDistribution

def test_tx_antenna_distribution():
    """Test TX antenna selection with custom distribution."""
    print("Testing TX Antenna Distribution...")
    print("=" * 60)
    
    # Create custom distribution
    tx_dist = TxAntennaDistribution(
        whip=0.60,
        rubber_duck=0.30,
        portable_directional=0.10
    )
    
    # Generate 1000 samples
    rng = np.random.default_rng(seed=42)
    samples = []
    for _ in range(1000):
        antenna = _select_tx_antenna(rng, tx_dist)
        samples.append(antenna.antenna_type.name)
    
    # Count distribution
    counts = Counter(samples)
    total = len(samples)
    
    print(f"Generated {total} samples")
    print("\nActual Distribution:")
    for antenna_type in ['WHIP', 'RUBBER_DUCK', 'PORTABLE_DIRECTIONAL']:
        count = counts[antenna_type]
        percentage = (count / total) * 100
        print(f"  {antenna_type:25s}: {count:4d} ({percentage:5.2f}%)")
    
    print("\nExpected Distribution:")
    print(f"  {'WHIP':25s}:  600 (60.00%)")
    print(f"  {'RUBBER_DUCK':25s}:  300 (30.00%)")
    print(f"  {'PORTABLE_DIRECTIONAL':25s}:  100 (10.00%)")
    
    # Verify within 5% tolerance
    assert abs(counts['WHIP'] - 600) < 50, f"WHIP count {counts['WHIP']} not within tolerance"
    assert abs(counts['RUBBER_DUCK'] - 300) < 50, f"RUBBER_DUCK count {counts['RUBBER_DUCK']} not within tolerance"
    assert abs(counts['PORTABLE_DIRECTIONAL'] - 100) < 50, f"PORTABLE_DIRECTIONAL count {counts['PORTABLE_DIRECTIONAL']} not within tolerance"
    
    print("\n✓ TX antenna distribution test PASSED\n")

def test_rx_antenna_distribution():
    """Test RX antenna selection with custom distribution."""
    print("Testing RX Antenna Distribution...")
    print("=" * 60)
    
    # Create custom distribution
    rx_dist = RxAntennaDistribution(
        omni_vertical=0.70,
        yagi=0.20,
        collinear=0.10
    )
    
    # Generate 1000 samples
    rng = np.random.default_rng(seed=42)
    samples = []
    for _ in range(1000):
        antenna = _select_rx_antenna(rng, rx_dist)
        samples.append(antenna.antenna_type.name)
    
    # Count distribution
    counts = Counter(samples)
    total = len(samples)
    
    print(f"Generated {total} samples")
    print("\nActual Distribution:")
    for antenna_type in ['OMNI_VERTICAL', 'YAGI', 'COLLINEAR']:
        count = counts[antenna_type]
        percentage = (count / total) * 100
        print(f"  {antenna_type:25s}: {count:4d} ({percentage:5.2f}%)")
    
    print("\nExpected Distribution:")
    print(f"  {'OMNI_VERTICAL':25s}:  700 (70.00%)")
    print(f"  {'YAGI':25s}:  200 (20.00%)")
    print(f"  {'COLLINEAR':25s}:  100 (10.00%)")
    
    # Verify within 5% tolerance
    assert abs(counts['OMNI_VERTICAL'] - 700) < 50, f"OMNI_VERTICAL count {counts['OMNI_VERTICAL']} not within tolerance"
    assert abs(counts['YAGI'] - 200) < 50, f"YAGI count {counts['YAGI']} not within tolerance"
    assert abs(counts['COLLINEAR'] - 100) < 50, f"COLLINEAR count {counts['COLLINEAR']} not within tolerance"
    
    print("\n✓ RX antenna distribution test PASSED\n")

def test_dict_input():
    """Test that dict inputs are properly converted to dataclasses."""
    print("Testing Dict Input Conversion...")
    print("=" * 60)
    
    # Test with dict input
    tx_dist_dict = {
        'whip': 0.60,
        'rubber_duck': 0.30,
        'portable_directional': 0.10
    }
    
    rx_dist_dict = {
        'omni_vertical': 0.70,
        'yagi': 0.20,
        'collinear': 0.10
    }
    
    rng = np.random.default_rng(seed=42)
    
    # These should not raise AttributeError
    try:
        tx_antenna = _select_tx_antenna(rng, tx_dist_dict)
        print(f"✓ TX antenna selected with dict input: {tx_antenna.antenna_type.name}")
        
        rx_antenna = _select_rx_antenna(rng, rx_dist_dict)
        print(f"✓ RX antenna selected with dict input: {rx_antenna.antenna_type.name}")
        
        print("\n✓ Dict input conversion test PASSED\n")
    except AttributeError as e:
        print(f"✗ Dict input conversion FAILED: {e}")
        raise

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Antenna Distribution Configuration Tests")
    print("="*60 + "\n")
    
    try:
        test_tx_antenna_distribution()
        test_rx_antenna_distribution()
        test_dict_input()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"TEST FAILED ✗")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
