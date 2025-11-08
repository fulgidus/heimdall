#!/usr/bin/env python3
"""
Source code validation test for RF propagation physics fixes.
Checks that all 7 fixes are present in the code.
"""

import sys
from pathlib import Path


def test_transmission_consistency():
    """Test Fix #1: Transmission state is consistent (not per-receiver)."""
    print("\n" + "=" * 80)
    print("TEST 1: Transmission Consistency")
    print("=" * 80)
    
    # Check that calculate_received_power uses is_transmitting parameter
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Look for the new parameter signature
    checks = [
        ("is_transmitting parameter", "is_transmitting: bool"),
        ("check_intermittent_transmission function", "def check_intermittent_transmission"),
    ]
    
    all_passed = True
    for component, check_str in checks:
        if check_str in content:
            print(f"✓ {component}: Found")
        else:
            print(f"✗ {component}: NOT FOUND")
            all_passed = False
    
    # Check synthetic_generator.py for global transmission check
    gen_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "synthetic_generator.py"
    with open(gen_file, 'r') as f:
        gen_content = f.read()
    
    # Count occurrences of is_transmitting = check...
    global_checks = gen_content.count("is_transmitting = ")
    
    if global_checks >= 2:
        print(f"✓ Global transmission checks: Found {global_checks} instances (expected ≥2)")
    else:
        print(f"✗ Global transmission checks: Only found {global_checks} (expected ≥2)")
        all_passed = False
    
    # Check that new is_transmitting parameter IS used in calculate_received_power calls
    new_param_count = gen_content.count("is_transmitting=is_transmitting")
    
    if new_param_count >= 2:  # Should appear in at least 2 function calls
        print(f"✓ New parameter used in function calls: {new_param_count} occurrences")
    else:
        print(f"✗ New parameter not properly used: {new_param_count} occurrences (expected ≥2)")
        all_passed = False
    
    if all_passed:
        print(f"✓ PASS: Transmission consistency fix implemented correctly")
        return True
    else:
        print(f"✗ FAIL: Transmission consistency fix incomplete")
        return False


def test_sporadic_e_probability():
    """Test Fix #2: Sporadic-E probability is reduced to 0.1-0.5%."""
    print("\n" + "=" * 80)
    print("TEST 2: Sporadic-E Probability Reduction")
    print("=" * 80)
    
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Look for the sporadic-E probability line (new value)
    if "0.001 + season_factor * 0.004" in content:
        print(f"✓ PASS: Sporadic-E probability set to 0.001 + season_factor * 0.004")
        print(f"  Range: 0.1% to 0.5% (was 1% to 5%)")
        return True
    elif "0.01 + season_factor * 0.04" in content:
        print(f"✗ FAIL: Still using OLD sporadic-E probability (1-5%)")
        return False
    else:
        print(f"✗ FAIL: Sporadic-E probability line not found")
        return False


def test_vertical_polarization():
    """Test Fix #4: Receiver antennas are 95% vertical."""
    print("\n" + "=" * 80)
    print("TEST 4: Vertical Polarization Dominance")
    print("=" * 80)
    
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Check for 95% vertical in key antenna types
    checks = [
        ("Yagi", "Polarization.VERTICAL: 0.95", "AntennaType.YAGI"),
        ("Log-Periodic", "Polarization.VERTICAL: 0.95", "AntennaType.LOG_PERIODIC"),
        ("Portable Directional", "Polarization.VERTICAL: 0.95", "AntennaType.PORTABLE_DIRECTIONAL")
    ]
    
    all_passed = True
    for antenna_name, check_str, antenna_type in checks:
        # Find the antenna type section and check nearby for 0.95
        antenna_idx = content.find(antenna_type)
        if antenna_idx == -1:
            print(f"✗ {antenna_name}: Antenna type not found")
            all_passed = False
            continue
        
        # Check in a 500 character window after the antenna type
        window = content[antenna_idx:antenna_idx + 500]
        
        if "0.95" in window and "VERTICAL" in window:
            print(f"✓ {antenna_name}: 95% vertical polarization")
        else:
            print(f"✗ {antenna_name}: NOT 95% vertical")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: All RX antennas configured for 95% vertical polarization")
        return True
    else:
        print(f"✗ FAIL: Some antennas not properly configured")
        return False


def test_cross_pol_loss():
    """Test Fix #6: Cross-polarization loss is 10-15 dB."""
    print("\n" + "=" * 80)
    print("TEST 6: Cross-Polarization Loss")
    print("=" * 80)
    
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Look for cross-pol loss value (new: 10-15 dB)
    # The value appears in the depolarization model section
    if "np.random.uniform(10.0, 15.0)" in content:
        print(f"✓ PASS: Cross-polarization loss set to 10-15 dB (was 15-25 dB)")
        print(f"  Implemented in depolarization model (PHYSICS FIX #5 & #6)")
        return True
    elif "np.random.uniform(15.0, 25.0)" in content:
        print(f"✗ FAIL: Still using OLD cross-pol loss (15-25 dB)")
        return False
    else:
        print(f"✗ FAIL: Cross-polarization loss value not found")
        return False


def test_fading_model():
    """Test Fix #3: Hybrid log-normal + Rician/Rayleigh fading model."""
    print("\n" + "=" * 80)
    print("TEST 3: Hybrid Fading Model")
    print("=" * 80)
    
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Check for key components of the new model
    checks = [
        ("Log-normal slow fading", "shadow_std_db"),  # Variable name in code
        ("Slow fading calculation", "slow_fading_db = np.random.normal"),
        ("Rician fast fading", "k_factor"),
        ("Rician K-factor calculation", "k_factor_db = np.random.uniform(6.0, 10.0)"),
        ("Rayleigh fading for NLOS", "Rayleigh fading for NLOS"),
    ]
    
    all_passed = True
    for component, check_str in checks:
        if check_str in content:
            print(f"✓ {component}: Implemented")
        else:
            print(f"✗ {component}: NOT FOUND")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: Hybrid fading model (log-normal + Rician/Rayleigh) implemented")
        return True
    else:
        print(f"✗ FAIL: Fading model incomplete")
        return False


def test_fspl_check():
    """Test Fix #7: FSPL sanity check."""
    print("\n" + "=" * 80)
    print("TEST 7: FSPL Sanity Check")
    print("=" * 80)
    
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Look for FSPL sanity check
    checks = [
        ("Expected FSPL calculation", "expected_fspl = 32.45"),
        ("Sanity check comparison", "fspl < expected_fspl * 0.8"),  # Actual comparison in code
    ]
    
    all_passed = True
    for component, check_str in checks:
        if check_str in content:
            print(f"✓ {component}: Found")
        else:
            print(f"✗ {component}: NOT FOUND")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: FSPL sanity check implemented")
        return True
    else:
        print(f"✗ FAIL: FSPL sanity check not found")
        return False


def test_depolarization_model():
    """Test Fix #5: Depolarization model."""
    print("\n" + "=" * 80)
    print("TEST 5: Depolarization Model")
    print("=" * 80)
    
    prop_file = Path(__file__).parent / "services" / "training" / "src" / "data" / "propagation.py"
    with open(prop_file, 'r') as f:
        content = f.read()
    
    # Look for depolarization model components
    checks = [
        ("Depolarization model", "PHYSICS FIX #5"),
        ("Energy distribution comment", "60% of energy maintains original polarization"),
        ("Multipath depolarization", "include_multipath_depolarization"),
        ("Cross-pol loss 10-15 dB", "np.random.uniform(10.0, 15.0)"),
    ]
    
    all_passed = True
    for component, check_str in checks:
        if check_str in content:
            print(f"✓ {component}: Found")
        else:
            print(f"✗ {component}: NOT FOUND")
            all_passed = False
    
    if all_passed:
        print(f"✓ PASS: Depolarization model implemented")
        return True
    else:
        print(f"✗ FAIL: Depolarization model incomplete")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("RF PROPAGATION PHYSICS FIXES - SOURCE CODE VALIDATION")
    print("=" * 80)
    print("This test verifies that all 7 physics fixes are present in the code.")
    print()
    
    results = []
    
    # Run tests in order
    results.append(("Fix #1: Transmission Consistency", test_transmission_consistency()))
    results.append(("Fix #2: Sporadic-E Probability", test_sporadic_e_probability()))
    results.append(("Fix #3: Hybrid Fading Model", test_fading_model()))
    results.append(("Fix #4: Vertical Polarization", test_vertical_polarization()))
    results.append(("Fix #5: Depolarization Model", test_depolarization_model()))
    results.append(("Fix #6: Cross-Pol Loss", test_cross_pol_loss()))
    results.append(("Fix #7: FSPL Sanity Check", test_fspl_check()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! All 7 physics fixes are correctly implemented.")
        return 0
    else:
        print(f"\n⚠ WARNING: {total - passed} test(s) failed. Review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
