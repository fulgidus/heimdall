"""
Test script for new RF propagation features.

Tests:
1. Knife-edge diffraction
2. Sporadic-E propagation
3. TX power fluctuations
4. Intermittent transmissions
"""

import numpy as np
import structlog
from src.data.propagation import RFPropagationModel, AntennaType, AntennaPattern
from src.data.config import MeteorologicalParameters
from common.terrain import TerrainLookup

logger = structlog.get_logger(__name__)


def test_knife_edge_diffraction():
    """Test knife-edge diffraction calculation."""
    print("\n" + "="*70)
    print("TEST 1: Knife-Edge Diffraction")
    print("="*70)
    
    model = RFPropagationModel()
    
    # Test case 1: No obstruction (obstacle below line of sight)
    print("\nTest 1a: No obstruction")
    loss = model.calculate_knife_edge_diffraction(
        tx_alt=100.0,
        rx_alt=100.0,
        obstacle_alt=50.0,  # Below line of sight
        distance_to_obstacle_km=10.0,
        total_distance_km=20.0,
        frequency_mhz=145.0
    )
    print(f"  TX alt: 100m, RX alt: 100m, Obstacle alt: 50m")
    print(f"  Distance to obstacle: 10km, Total distance: 20km")
    print(f"  Diffraction loss: {loss:.2f} dB (expected: ~0 dB)")
    assert loss >= 0, "Loss should be non-negative"
    
    # Test case 2: Significant obstruction
    print("\nTest 1b: Significant obstruction")
    loss = model.calculate_knife_edge_diffraction(
        tx_alt=100.0,
        rx_alt=100.0,
        obstacle_alt=200.0,  # 100m above line of sight
        distance_to_obstacle_km=10.0,
        total_distance_km=20.0,
        frequency_mhz=145.0
    )
    print(f"  TX alt: 100m, RX alt: 100m, Obstacle alt: 200m")
    print(f"  Distance to obstacle: 10km, Total distance: 20km")
    print(f"  Diffraction loss: {loss:.2f} dB (expected: 10-30 dB)")
    assert 5.0 <= loss <= 40.0, f"Loss {loss:.2f} dB seems unrealistic"
    
    # Test case 3: Grazing obstacle (just at line of sight)
    print("\nTest 1c: Grazing obstacle")
    loss = model.calculate_knife_edge_diffraction(
        tx_alt=100.0,
        rx_alt=100.0,
        obstacle_alt=100.0,  # Exactly at line of sight
        distance_to_obstacle_km=10.0,
        total_distance_km=20.0,
        frequency_mhz=145.0
    )
    print(f"  TX alt: 100m, RX alt: 100m, Obstacle alt: 100m")
    print(f"  Distance to obstacle: 10km, Total distance: 20km")
    print(f"  Diffraction loss: {loss:.2f} dB (expected: ~6 dB)")
    assert 4.0 <= loss <= 8.0, f"Loss {loss:.2f} dB seems unrealistic for grazing"
    
    print("\n✅ Knife-edge diffraction tests PASSED")


def test_sporadic_e_propagation():
    """Test sporadic-E propagation calculation."""
    print("\n" + "="*70)
    print("TEST 2: Sporadic-E Propagation")
    print("="*70)
    
    model = RFPropagationModel()  # VHF, good for Es
    
    # Test case 1: Summer, high solar flux, optimal distance
    print("\nTest 2a: Favorable conditions (summer, high solar flux)")
    
    # Run multiple times to get probability estimate
    active_count = 0
    total_trials = 100
    enhancement_values = []
    
    for _ in range(total_trials):
        is_active, enhancement = model.calculate_sporadic_e_propagation(
            distance_km=1000.0,  # Optimal range
            frequency_mhz=50.0,
            solar_flux=150.0,
            season_factor=1.0
        )
        if is_active:
            active_count += 1
            enhancement_values.append(enhancement)
    
    probability = active_count / total_trials
    avg_enhancement = np.mean(enhancement_values) if enhancement_values else 0.0
    print(f"  Distance: 1000 km, Solar flux: 150, Season factor: 1.0")
    print(f"  Average enhancement (when active): {avg_enhancement:.2f} dB")
    print(f"  Sporadic-E occurred: {probability*100:.1f}% of time")
    if enhancement_values:
        assert avg_enhancement >= 20.0, "Enhancement should be at least 20 dB when active"
        assert avg_enhancement <= 40.0, "Enhancement should not exceed 40 dB"
    assert 0.0 <= probability <= 1.0, "Probability should be 0-1"
    
    # Test case 2: Winter, low solar flux (unlikely)
    print("\nTest 2b: Unfavorable conditions (winter, low solar flux)")
    active_count = 0
    for _ in range(total_trials):
        is_active, _ = model.calculate_sporadic_e_propagation(
            distance_km=1000.0,
            frequency_mhz=50.0,
            solar_flux=70.0,
            season_factor=0.2
        )
        if is_active:
            active_count += 1
    
    probability = active_count / total_trials
    print(f"  Distance: 1000 km, Solar flux: 70, Season factor: 0.2")
    print(f"  Sporadic-E occurred: {probability*100:.1f}% of time")
    assert probability < 0.15, "Winter probability should be low"
    
    # Test case 3: Too close for Es (ground wave)
    print("\nTest 2c: Distance too short for Es")
    is_active, enhancement = model.calculate_sporadic_e_propagation(
        distance_km=300.0,  # Below minimum
        frequency_mhz=50.0,
        solar_flux=150.0,
        season_factor=1.0
    )
    print(f"  Distance: 300 km (below Es minimum)")
    print(f"  Sporadic-E active: {is_active}")
    assert not is_active, "Es should not occur at short distances"
    
    # Test case 4: Too far for single-hop Es
    print("\nTest 2d: Distance too far for single-hop Es")
    is_active, enhancement = model.calculate_sporadic_e_propagation(
        distance_km=3000.0,  # Above maximum
        frequency_mhz=50.0,
        solar_flux=150.0,
        season_factor=1.0
    )
    print(f"  Distance: 3000 km (above Es maximum)")
    print(f"  Sporadic-E active: {is_active}")
    assert not is_active, "Single-hop Es should not occur at very long distances"
    
    print("\n✅ Sporadic-E propagation tests PASSED")


def test_tx_power_fluctuations():
    """Test TX power fluctuation calculation."""
    print("\n" + "="*70)
    print("TEST 3: TX Power Fluctuations")
    print("="*70)
    
    model = RFPropagationModel()
    
    # Test case 1: High-quality transmitter (low fluctuation)
    print("\nTest 3a: High-quality transmitter")
    fluctuations = []
    nominal_power = 30.0  # 30 dBm (1W)
    for _ in range(1000):
        fluc = model.calculate_tx_power_fluctuation(nominal_power, transmitter_quality=0.9)
        fluctuations.append(fluc - nominal_power)  # Get deviation from nominal
    
    std_dev = np.std(fluctuations)
    mean = np.mean(fluctuations)
    print(f"  Transmitter quality: 0.9 (high)")
    print(f"  Mean fluctuation: {mean:.3f} dB (expected: ~0)")
    print(f"  Std dev: {std_dev:.3f} dB (expected: ~0.3)")
    assert abs(mean) < 0.1, "Mean should be near zero"
    assert 0.2 <= std_dev <= 0.6, f"Std dev {std_dev:.3f} outside expected range"
    
    # Test case 2: Poor-quality transmitter (high fluctuation)
    print("\nTest 3b: Poor-quality transmitter")
    fluctuations = []
    for _ in range(1000):
        fluc = model.calculate_tx_power_fluctuation(nominal_power, transmitter_quality=0.2)
        fluctuations.append(fluc - nominal_power)  # Get deviation from nominal
    
    std_dev = np.std(fluctuations)
    mean = np.mean(fluctuations)
    print(f"  Transmitter quality: 0.2 (poor)")
    print(f"  Mean fluctuation: {mean:.3f} dB (expected: ~0)")
    print(f"  Std dev: {std_dev:.3f} dB (expected: ~1.7)")
    assert abs(mean) < 0.2, "Mean should be near zero"
    assert 1.4 <= std_dev <= 2.2, f"Std dev {std_dev:.3f} outside expected range"
    
    # Test case 3: Medium-quality transmitter
    print("\nTest 3c: Medium-quality transmitter")
    fluctuations = []
    for _ in range(1000):
        fluc = model.calculate_tx_power_fluctuation(nominal_power, transmitter_quality=0.5)
        fluctuations.append(fluc - nominal_power)  # Get deviation from nominal
    
    std_dev = np.std(fluctuations)
    mean = np.mean(fluctuations)
    print(f"  Transmitter quality: 0.5 (medium)")
    print(f"  Mean fluctuation: {mean:.3f} dB (expected: ~0)")
    print(f"  Std dev: {std_dev:.3f} dB (expected: ~1.0)")
    assert abs(mean) < 0.15, "Mean should be near zero"
    assert 0.8 <= std_dev <= 1.4, f"Std dev {std_dev:.3f} outside expected range"
    
    print("\n✅ TX power fluctuation tests PASSED")


def test_intermittent_transmissions():
    """Test intermittent transmission check."""
    print("\n" + "="*70)
    print("TEST 4: Intermittent Transmissions")
    print("="*70)
    
    model = RFPropagationModel()
    
    # Test case 1: Always transmitting (100% duty cycle)
    print("\nTest 4a: Always transmitting (100% duty cycle)")
    on_count = 0
    trials = 1000
    for _ in range(trials):
        is_on = model.check_intermittent_transmission(transmission_duty_cycle=1.0)
        if is_on:
            on_count += 1
    
    on_percentage = (on_count / trials) * 100
    print(f"  Duty cycle: 100%")
    print(f"  TX on: {on_percentage:.1f}% of time")
    assert on_count == trials, "Should always be on with 100% duty cycle"
    
    # Test case 2: Never transmitting (0% duty cycle)
    print("\nTest 4b: Never transmitting (0% duty cycle)")
    on_count = 0
    for _ in range(trials):
        is_on = model.check_intermittent_transmission(transmission_duty_cycle=0.0)
        if is_on:
            on_count += 1
    
    on_percentage = (on_count / trials) * 100
    print(f"  Duty cycle: 0%")
    print(f"  TX on: {on_percentage:.1f}% of time")
    assert on_count == 0, "Should never be on with 0% duty cycle"
    
    # Test case 3: 50% duty cycle (typical repeater)
    print("\nTest 4c: 50% duty cycle (typical repeater timeout)")
    on_count = 0
    for _ in range(trials):
        is_on = model.check_intermittent_transmission(transmission_duty_cycle=0.5)
        if is_on:
            on_count += 1
    
    on_percentage = (on_count / trials) * 100
    print(f"  Duty cycle: 50%")
    print(f"  TX on: {on_percentage:.1f}% of time")
    assert 45.0 <= on_percentage <= 55.0, f"On percentage {on_percentage:.1f}% deviates too much from 50%"
    
    # Test case 4: 95% duty cycle (typical amateur station)
    print("\nTest 4d: 95% duty cycle (typical amateur station)")
    on_count = 0
    for _ in range(trials):
        is_on = model.check_intermittent_transmission(transmission_duty_cycle=0.95)
        if is_on:
            on_count += 1
    
    on_percentage = (on_count / trials) * 100
    print(f"  Duty cycle: 95%")
    print(f"  TX on: {on_percentage:.1f}% of time")
    assert 92.0 <= on_percentage <= 98.0, f"On percentage {on_percentage:.1f}% deviates too much from 95%"
    
    print("\n✅ Intermittent transmission tests PASSED")


def test_integration():
    """Test integration of all features in calculate_received_power()."""
    print("\n" + "="*70)
    print("TEST 5: Integration Test (calculate_received_power)")
    print("="*70)
    
    model = RFPropagationModel()
    
    # Mock terrain lookup
    class MockTerrainLookup:
        def get_elevation(self, lat, lon):
            return 100.0  # Simple flat terrain
        
        def get_elevation_profile(self, lat1, lon1, lat2, lon2, num_points=50):
            # Return profile with one obstacle in the middle
            lats = np.linspace(lat1, lat2, num_points)
            lons = np.linspace(lon1, lon2, num_points)
            elevations = [100.0] * num_points
            elevations[num_points // 2] = 200.0  # Obstacle in the middle
            return list(zip(lats, lons, elevations))
    
    terrain_lookup = MockTerrainLookup()
    
    # Test case 1: All features enabled, favorable conditions
    print("\nTest 5a: All features enabled")
    meteo = MeteorologicalParameters(
        ground_temperature=25.0,
        relative_humidity=60.0,
        solar_flux=120.0,
        season_factor=0.8
    )
    
    # Create antenna patterns
    tx_antenna = AntennaPattern(AntennaType.WHIP)
    rx_antenna = AntennaPattern(AntennaType.OMNI_VERTICAL)
    
    rx_power, snr, details = model.calculate_received_power(
        tx_power_dbm=30.0,
        tx_lat=45.0,
        tx_lon=10.0,
        tx_alt=100.0,
        rx_lat=45.5,
        rx_lon=10.5,
        rx_alt=100.0,
        frequency_mhz=145.0,
        tx_antenna=tx_antenna,
        rx_antenna=rx_antenna,
        terrain_lookup=terrain_lookup,
        meteo_params=meteo,
        transmitter_quality=0.7,
        transmission_duty_cycle=0.95,
        enable_sporadic_e=True,
        enable_knife_edge=True
    )
    
    print(f"  TX power: 30 dBm, Distance: ~78 km")
    print(f"  RX power: {rx_power:.2f} dBm")
    print(f"\n  Breakdown:")
    print(f"    - Free space path loss: {details['fspl_db']:.2f} dB")
    print(f"    - Atmospheric absorption: {details['atmospheric_absorption_db']:.2f} dB")
    print(f"    - Knife-edge diffraction loss: {details['knife_edge_loss_db']:.2f} dB")
    print(f"    - TX power variation: {details['tx_power_variation_db']:.2f} dB")
    print(f"    - Sporadic-E enhancement: {details['sporadic_e_enhancement_db']:.2f} dB")
    print(f"    - Sporadic-E active: {details['sporadic_e_active']}")
    print(f"    - Transmission active: {details['transmission_active']}")
    
    # Verify details dict has all new keys
    assert 'knife_edge_loss_db' in details
    assert 'tx_power_variation_db' in details
    assert 'sporadic_e_enhancement_db' in details
    assert 'sporadic_e_active' in details
    assert 'transmission_active' in details
    
    # Test case 2: Transmission inactive
    print("\nTest 5b: Transmission inactive (intermittent)")
    # Force transmission off by setting duty cycle to 0
    rx_power, snr, details = model.calculate_received_power(
        tx_power_dbm=30.0,
        tx_lat=45.0,
        tx_lon=10.0,
        tx_alt=100.0,
        rx_lat=45.5,
        rx_lon=10.5,
        rx_alt=100.0,
        frequency_mhz=145.0,
        transmission_duty_cycle=0.0,  # Force off
    )
    
    print(f"  Transmission active: {details['transmission_active']}")
    print(f"  RX power: {rx_power:.2f} dBm (should be noise floor)")
    assert details['transmission_active'] is False
    assert rx_power <= -120.0, "Should return noise floor when TX is off"
    
    print("\n✅ Integration tests PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING NEW RF PROPAGATION FEATURES")
    print("="*70)
    
    try:
        test_knife_edge_diffraction()
        test_sporadic_e_propagation()
        test_tx_power_fluctuations()
        test_intermittent_transmissions()
        test_integration()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise


if __name__ == "__main__":
    main()
