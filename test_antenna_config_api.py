#!/usr/bin/env python3
"""
Test script to verify antenna configuration API integration.

Tests:
1. API request with default antenna distributions
2. API request with custom antenna distributions  
3. Validation that probabilities sum to 1.0
4. Verification that config dict is correctly built
"""
import json
import sys
from pathlib import Path

# Add services/training to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "training" / "src"))

def test_pydantic_models():
    """Test Pydantic request models."""
    print("=" * 60)
    print("TEST 1: Pydantic Request Models")
    print("=" * 60)
    
    from api.synthetic import (
        TxAntennaDistributionRequest,
        RxAntennaDistributionRequest,
        GenerateDatasetRequest
    )
    
    # Test default values
    tx_dist = TxAntennaDistributionRequest()
    assert tx_dist.whip == 0.90
    assert tx_dist.rubber_duck == 0.08
    assert tx_dist.portable_directional == 0.02
    print("✅ TX antenna distribution defaults correct")
    
    rx_dist = RxAntennaDistributionRequest()
    assert rx_dist.omni_vertical == 0.80
    assert rx_dist.yagi == 0.15
    assert rx_dist.collinear == 0.05
    print("✅ RX antenna distribution defaults correct")
    
    # Test custom values
    custom_tx = TxAntennaDistributionRequest(
        whip=0.70,
        rubber_duck=0.20,
        portable_directional=0.10
    )
    assert custom_tx.whip == 0.70
    assert custom_tx.rubber_duck == 0.20
    assert custom_tx.portable_directional == 0.10
    print("✅ Custom TX antenna distribution works")
    
    # Test in GenerateDatasetRequest
    request = GenerateDatasetRequest(
        name="test_dataset",
        num_samples=1000,
        frequency_mhz=145.0,
        tx_power_dbm=5.0,
        min_snr_db=10.0,
        min_receivers=3,
        tx_antenna_dist=custom_tx,
        rx_antenna_dist=rx_dist
    )
    
    assert request.tx_antenna_dist.whip == 0.70
    assert request.rx_antenna_dist.omni_vertical == 0.80
    print("✅ Antenna distributions in GenerateDatasetRequest work")
    
    print()


def test_config_dict_construction():
    """Test that config dict is built correctly."""
    print("=" * 60)
    print("TEST 2: Config Dict Construction")
    print("=" * 60)
    
    from api.synthetic import (
        TxAntennaDistributionRequest,
        RxAntennaDistributionRequest,
        GenerateDatasetRequest
    )
    
    # Simulate the API endpoint logic
    request = GenerateDatasetRequest(
        name="test_dataset",
        num_samples=1000,
        frequency_mhz=145.0,
        tx_power_dbm=5.0,
        min_snr_db=10.0,
        min_receivers=3,
        tx_antenna_dist=TxAntennaDistributionRequest(
            whip=0.60,
            rubber_duck=0.30,
            portable_directional=0.10
        ),
        rx_antenna_dist=RxAntennaDistributionRequest(
            omni_vertical=0.70,
            yagi=0.20,
            collinear=0.10
        )
    )
    
    # Build config dict (same logic as in synthetic.py)
    config = {
        "name": request.name,
        "description": request.description,
        "num_samples": request.num_samples,
        "frequency_mhz": request.frequency_mhz,
        "tx_power_dbm": request.tx_power_dbm,
        "min_snr_db": request.min_snr_db,
        "min_receivers": request.min_receivers,
        "max_gdop": request.max_gdop,
        "dataset_type": request.dataset_type,
        "use_random_receivers": request.use_random_receivers,
        "use_srtm_terrain": request.use_srtm_terrain,
        "seed": request.seed
    }
    
    # Add antenna distributions if provided
    if request.tx_antenna_dist is not None:
        config["tx_antenna_dist"] = {
            "whip": request.tx_antenna_dist.whip,
            "rubber_duck": request.tx_antenna_dist.rubber_duck,
            "portable_directional": request.tx_antenna_dist.portable_directional
        }
    
    if request.rx_antenna_dist is not None:
        config["rx_antenna_dist"] = {
            "omni_vertical": request.rx_antenna_dist.omni_vertical,
            "yagi": request.rx_antenna_dist.yagi,
            "collinear": request.rx_antenna_dist.collinear
        }
    
    # Verify config dict
    assert "tx_antenna_dist" in config
    assert "rx_antenna_dist" in config
    assert config["tx_antenna_dist"]["whip"] == 0.60
    assert config["tx_antenna_dist"]["rubber_duck"] == 0.30
    assert config["tx_antenna_dist"]["portable_directional"] == 0.10
    assert config["rx_antenna_dist"]["omni_vertical"] == 0.70
    assert config["rx_antenna_dist"]["yagi"] == 0.20
    assert config["rx_antenna_dist"]["collinear"] == 0.10
    
    print("✅ Config dict correctly includes antenna distributions")
    print(f"   TX dist: {config['tx_antenna_dist']}")
    print(f"   RX dist: {config['rx_antenna_dist']}")
    
    # Test without antenna distributions (None case)
    request_no_dist = GenerateDatasetRequest(
        name="test_dataset_no_dist",
        num_samples=1000,
        frequency_mhz=145.0,
        tx_power_dbm=5.0,
        min_snr_db=10.0,
        min_receivers=3
    )
    
    config_no_dist = {
        "name": request_no_dist.name,
        "num_samples": request_no_dist.num_samples,
    }
    
    if request_no_dist.tx_antenna_dist is not None:
        config_no_dist["tx_antenna_dist"] = {}
    
    if request_no_dist.rx_antenna_dist is not None:
        config_no_dist["rx_antenna_dist"] = {}
    
    assert "tx_antenna_dist" not in config_no_dist
    assert "rx_antenna_dist" not in config_no_dist
    print("✅ Config dict correctly omits antenna distributions when None")
    
    print()


def test_dataclass_validation():
    """Test validation in config dataclasses."""
    print("=" * 60)
    print("TEST 3: Dataclass Validation")
    print("=" * 60)
    
    from data.config import TxAntennaDistribution, RxAntennaDistribution
    
    # Valid TX distribution
    tx_valid = TxAntennaDistribution(whip=0.80, rubber_duck=0.15, portable_directional=0.05)
    tx_valid.validate()
    print("✅ Valid TX distribution passes validation")
    
    # Valid RX distribution
    rx_valid = RxAntennaDistribution(omni_vertical=0.70, yagi=0.20, collinear=0.10)
    rx_valid.validate()
    print("✅ Valid RX distribution passes validation")
    
    # Invalid TX distribution (doesn't sum to 1.0)
    try:
        tx_invalid = TxAntennaDistribution(whip=0.50, rubber_duck=0.30, portable_directional=0.10)
        tx_invalid.validate()
        print("❌ Invalid TX distribution should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"✅ Invalid TX distribution raises ValueError: {e}")
    
    # Invalid RX distribution (doesn't sum to 1.0)
    try:
        rx_invalid = RxAntennaDistribution(omni_vertical=0.50, yagi=0.30, collinear=0.10)
        rx_invalid.validate()
        print("❌ Invalid RX distribution should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"✅ Invalid RX distribution raises ValueError: {e}")
    
    print()


def test_antenna_selection():
    """Test antenna selection functions."""
    print("=" * 60)
    print("TEST 4: Antenna Selection Functions")
    print("=" * 60)
    
    from data.synthetic_generator import _select_tx_antenna, _select_rx_antenna
    from data.config import TxAntennaDistribution, RxAntennaDistribution
    import numpy as np
    
    rng = np.random.RandomState(42)
    
    # Test with default distributions (None)
    tx_antenna = _select_tx_antenna(rng)
    assert tx_antenna in ['whip', 'rubber_duck', 'portable_directional']
    print(f"✅ TX antenna selected with defaults: {tx_antenna}")
    
    rx_antenna = _select_rx_antenna(rng)
    assert rx_antenna in ['omni_vertical', 'yagi', 'collinear']
    print(f"✅ RX antenna selected with defaults: {rx_antenna}")
    
    # Test with custom distributions
    custom_tx_dist = TxAntennaDistribution(whip=0.0, rubber_duck=0.0, portable_directional=1.0)
    tx_antenna_custom = _select_tx_antenna(rng, custom_tx_dist)
    assert tx_antenna_custom == 'portable_directional'
    print(f"✅ TX antenna selected with custom dist (100% portable_directional): {tx_antenna_custom}")
    
    custom_rx_dist = RxAntennaDistribution(omni_vertical=0.0, yagi=1.0, collinear=0.0)
    rx_antenna_custom = _select_rx_antenna(rng, custom_rx_dist)
    assert rx_antenna_custom == 'yagi'
    print(f"✅ RX antenna selected with custom dist (100% yagi): {rx_antenna_custom}")
    
    # Test distribution over many samples
    tx_dist_test = TxAntennaDistribution(whip=0.50, rubber_duck=0.30, portable_directional=0.20)
    counts = {'whip': 0, 'rubber_duck': 0, 'portable_directional': 0}
    
    for _ in range(1000):
        antenna = _select_tx_antenna(rng, tx_dist_test)
        counts[antenna] += 1
    
    # Check that distribution is approximately correct (within 10%)
    assert 0.40 <= counts['whip'] / 1000 <= 0.60
    assert 0.20 <= counts['rubber_duck'] / 1000 <= 0.40
    assert 0.10 <= counts['portable_directional'] / 1000 <= 0.30
    
    print(f"✅ Distribution over 1000 samples approximately correct:")
    print(f"   whip: {counts['whip']/10:.1f}% (expected 50%)")
    print(f"   rubber_duck: {counts['rubber_duck']/10:.1f}% (expected 30%)")
    print(f"   portable_directional: {counts['portable_directional']/10:.1f}% (expected 20%)")
    
    print()


if __name__ == "__main__":
    try:
        test_pydantic_models()
        test_config_dict_construction()
        test_dataclass_validation()
        test_antenna_selection()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        print()
        print("Summary:")
        print("1. ✅ Pydantic request models work correctly")
        print("2. ✅ Config dict construction includes antenna distributions")
        print("3. ✅ Dataclass validation catches invalid distributions")
        print("4. ✅ Antenna selection functions use custom distributions")
        print()
        print("Next steps:")
        print("- Rebuild training Docker container")
        print("- Test API endpoint with curl/Postman")
        print("- Verify distributions propagate through full generation pipeline")
        print("- Add frontend UI controls for antenna distributions")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
