"""
Test meteorological effects implementation for RF propagation model.

This script validates:
1. MeteorologicalParameters random generation
2. Atmospheric absorption calculation
3. Tropospheric refraction modeling
4. Integration with propagation model
"""

import sys
import os

# Adjust path based on where we're running from
if os.path.exists('/app/src'):
    # Running inside Docker container
    sys.path.insert(0, '/app/src')
else:
    # Running from project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'training', 'src'))

import numpy as np
from data.config import MeteorologicalParameters
from data.propagation import RFPropagationModel


def test_meteo_parameters_random():
    """Test random meteorological parameter generation."""
    print("\n=== Test 1: MeteorologicalParameters.random() ===")
    
    # Generate 10 random samples
    for i in range(10):
        meteo = MeteorologicalParameters.random(seed=i)
        
        print(f"\nSample {i}:")
        print(f"  Temperature: {meteo.ground_temperature:.1f}°C")
        print(f"  Humidity: {meteo.relative_humidity:.1f}%")
        print(f"  Pressure: {meteo.pressure_hpa:.1f} hPa")
        print(f"  Time of day: {meteo.time_of_day:.1f}h")
        print(f"  Season: {['Spring', 'Summer', 'Autumn', 'Winter'][meteo.season]}")
        print(f"  Ducting probability: {meteo.ducting_probability:.3f}")
        
        # Validate ranges
        assert -20 <= meteo.ground_temperature <= 45, f"Temperature out of range: {meteo.ground_temperature}"
        assert 20 <= meteo.relative_humidity <= 95, f"Humidity out of range: {meteo.relative_humidity}"
        assert 980 <= meteo.pressure_hpa <= 1040, f"Pressure out of range: {meteo.pressure_hpa}"
        assert 0 <= meteo.time_of_day < 24, f"Time out of range: {meteo.time_of_day}"
        assert 0 <= meteo.season <= 3, f"Season out of range: {meteo.season}"
    
    print("\n✅ MeteorologicalParameters.random() validation passed!")


def test_atmospheric_absorption():
    """Test atmospheric absorption calculation."""
    print("\n=== Test 2: Atmospheric Absorption ===")
    
    propagation = RFPropagationModel()
    
    # Test cases: (distance_km, frequency_mhz, temp_c, humidity_%, description)
    test_cases = [
        (100, 145, 20, 50, "Normal conditions, VHF"),
        (500, 145, 30, 80, "Hot humid day, VHF"),
        (100, 430, 20, 50, "Normal conditions, UHF"),
        (500, 430, 30, 80, "Hot humid day, UHF"),
        (50, 145, 5, 30, "Cold dry day, VHF"),
    ]
    
    for distance_km, freq_mhz, temp_c, humidity, desc in test_cases:
        absorption = propagation.calculate_atmospheric_absorption(
            distance_km=distance_km,
            frequency_mhz=freq_mhz,
            temperature_c=temp_c,
            relative_humidity=humidity
        )
        
        print(f"\n{desc}:")
        print(f"  Distance: {distance_km} km, Frequency: {freq_mhz} MHz")
        print(f"  Temperature: {temp_c}°C, Humidity: {humidity}%")
        print(f"  Atmospheric absorption: {absorption:.3f} dB")
        
        # Validate: absorption should be positive
        # For VHF/UHF: ~0.01-0.02 dB/km typical, up to 0.05 dB/km in bad conditions
        # Max realistic: 500 km * 0.05 dB/km = 25 dB
        assert absorption >= 0, f"Absorption cannot be negative: {absorption}"
        assert absorption < 25.0, f"Absorption unrealistically high: {absorption}"
    
    print("\n✅ Atmospheric absorption calculation passed!")


def test_tropospheric_refraction():
    """Test tropospheric refraction modeling."""
    print("\n=== Test 3: Tropospheric Refraction ===")
    
    propagation = RFPropagationModel()
    
    # Test normal conditions (no ducting)
    print("\nNormal refraction (no ducting):")
    refraction_samples = []
    for i in range(20):
        refraction = propagation.calculate_tropospheric_refraction(
            distance_km=200,
            frequency_mhz=145,
            ground_temperature_c=20,
            relative_humidity=50,
            ducting_active=False
        )
        refraction_samples.append(refraction)
    
    mean_refraction = np.mean(refraction_samples)
    std_refraction = np.std(refraction_samples)
    print(f"  Mean: {mean_refraction:.2f} dB, Std: {std_refraction:.2f} dB")
    print(f"  Range: {min(refraction_samples):.2f} to {max(refraction_samples):.2f} dB")
    
    # Validate: typical range ±3-5 dB
    assert -10 <= mean_refraction <= 10, f"Mean refraction out of range: {mean_refraction}"
    
    # Test ducting conditions
    print("\nTropospheric ducting:")
    ducting_samples = []
    for i in range(20):
        refraction = propagation.calculate_tropospheric_refraction(
            distance_km=200,
            frequency_mhz=145,
            ground_temperature_c=30,
            relative_humidity=70,
            ducting_active=True
        )
        ducting_samples.append(refraction)
    
    mean_ducting = np.mean(ducting_samples)
    std_ducting = np.std(ducting_samples)
    print(f"  Mean: {mean_ducting:.2f} dB, Std: {std_ducting:.2f} dB")
    print(f"  Range: {min(ducting_samples):.2f} to {max(ducting_samples):.2f} dB")
    
    # Validate: ducting should be positive enhancement (5-20 dB)
    assert mean_ducting > 0, f"Ducting should enhance signal: {mean_ducting}"
    assert 3 <= mean_ducting <= 25, f"Ducting enhancement out of range: {mean_ducting}"
    
    print("\n✅ Tropospheric refraction modeling passed!")


def test_propagation_with_meteo():
    """Test full propagation calculation with meteorological effects."""
    print("\n=== Test 4: Propagation with Meteorological Effects ===")
    
    propagation = RFPropagationModel()
    
    # TX and RX positions (100 km apart)
    tx_lat, tx_lon, tx_alt = 45.0, 7.5, 300.0
    rx_lat, rx_lon, rx_alt = 45.9, 7.5, 500.0
    
    # Test without meteorological effects (but with new features disabled)
    rx_power_no_meteo, snr_no_meteo, details_no_meteo = propagation.calculate_received_power(
        tx_power_dbm=37.0,
        tx_lat=tx_lat,
        tx_lon=tx_lon,
        tx_alt=tx_alt,
        rx_lat=rx_lat,
        rx_lon=rx_lon,
        rx_alt=rx_alt,
        frequency_mhz=145.0,
        terrain_lookup=None,
        meteo_params=None,
        enable_sporadic_e=False,
        enable_knife_edge=False
    )
    
    print(f"\nWithout meteorological effects:")
    print(f"  RX power: {rx_power_no_meteo:.2f} dBm")
    print(f"  SNR: {snr_no_meteo:.2f} dB")
    print(f"  FSPL: {details_no_meteo['fspl_db']:.2f} dB")
    print(f"  Terrain loss: {details_no_meteo['terrain_loss_db']:.2f} dB")
    print(f"  Environment loss: {details_no_meteo['env_loss_db']:.2f} dB")
    print(f"  Knife-edge loss: {details_no_meteo['knife_edge_loss_db']:.2f} dB")
    
    # Test with meteorological effects (10 samples)
    print(f"\nWith meteorological effects (10 samples):")
    rx_powers = []
    snrs = []
    
    for i in range(10):
        meteo = MeteorologicalParameters.random(seed=i)
        
        rx_power, snr, details = propagation.calculate_received_power(
            tx_power_dbm=37.0,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=145.0,
            terrain_lookup=None,
            meteo_params=meteo,
            enable_sporadic_e=False,
            enable_knife_edge=False
        )
        
        rx_powers.append(rx_power)
        snrs.append(snr)
        
        if i < 3:  # Print first 3 samples
            print(f"\n  Sample {i}:")
            print(f"    Temperature: {meteo.ground_temperature:.1f}°C, Humidity: {meteo.relative_humidity:.1f}%")
            print(f"    RX power: {rx_power:.2f} dBm (Δ = {rx_power - rx_power_no_meteo:+.2f} dB)")
            print(f"    SNR: {snr:.2f} dB")
            if 'atmospheric_absorption_db' in details:
                print(f"    Atmospheric absorption: {details['atmospheric_absorption_db']:.3f} dB")
            if 'tropospheric_effect_db' in details:
                print(f"    Tropospheric effect: {details['tropospheric_effect_db']:+.2f} dB")
    
    # Statistics
    mean_rx_power = np.mean(rx_powers)
    std_rx_power = np.std(rx_powers)
    mean_snr = np.mean(snrs)
    std_snr = np.std(snrs)
    
    print(f"\nStatistics (10 samples):")
    print(f"  RX power: mean={mean_rx_power:.2f} dBm, std={std_rx_power:.2f} dB")
    print(f"  SNR: mean={mean_snr:.2f} dB, std={std_snr:.2f} dB")
    print(f"  Range: {min(rx_powers):.2f} to {max(rx_powers):.2f} dBm ({max(rx_powers) - min(rx_powers):.2f} dB)")
    
    # Validate: meteorological effects should add variation (±3-5 dB typical)
    variation = max(rx_powers) - min(rx_powers)
    print(f"  Signal variation from meteo: {variation:.2f} dB")
    
    # Expect at least 2 dB variation from meteo effects
    assert variation >= 2.0, f"Meteorological variation too small: {variation}"
    
    print("\n✅ Propagation with meteorological effects passed!")


def test_knife_edge_diffraction():
    """Test knife-edge diffraction calculation."""
    print("\n=== Test 5: Knife-Edge Diffraction ===")
    
    propagation = RFPropagationModel()
    
    # Test cases: (tx_alt, rx_alt, obstacle_alt, dist_to_obstacle, total_dist, freq, description)
    test_cases = [
        (100, 100, 50, 25, 50, 145, "Small obstacle (grazing)"),
        (100, 100, 150, 25, 50, 145, "Medium obstacle (significant diffraction)"),
        (100, 100, 200, 25, 50, 145, "Large obstacle (deep obstruction)"),
        (100, 100, 50, 10, 50, 145, "Obstacle closer to TX"),
        (100, 100, 100, 25, 50, 430, "UHF frequency (more diffraction loss)"),
    ]
    
    for tx_h, rx_h, obs_h, dist_to_obs, total_dist, freq, desc in test_cases:
        loss = propagation.calculate_knife_edge_diffraction(
            tx_alt=tx_h,
            rx_alt=rx_h,
            obstacle_alt=obs_h,
            distance_to_obstacle_km=dist_to_obs,
            total_distance_km=total_dist,
            frequency_mhz=freq
        )
        
        print(f"\n{desc}:")
        print(f"  TX: {tx_h}m, RX: {rx_h}m, Obstacle: {obs_h}m")
        print(f"  Distance to obstacle: {dist_to_obs}km, Total: {total_dist}km")
        print(f"  Frequency: {freq}MHz")
        print(f"  Diffraction loss: {loss:.2f} dB")
        
        # Validate ranges
        assert loss >= 0, f"Diffraction loss cannot be negative: {loss:.2f} dB"
        assert loss <= 30, f"Diffraction loss unrealistically high: {loss:.2f} dB"
    
    print("\n✅ Knife-edge diffraction test passed!")


def test_sporadic_e_propagation():
    """Test sporadic-E propagation modeling."""
    print("\n=== Test 6: Sporadic-E Propagation ===")
    
    propagation = RFPropagationModel()
    
    # Test cases: (solar_flux, season_factor, frequency_mhz, distance_km, description)
    test_cases = [
        (150, 0.8, 145, 1000, "High solar flux, summer, optimal distance"),
        (70, 0.2, 145, 1000, "Low solar flux, winter, optimal distance"),
        (150, 0.8, 145, 300, "Too short for sporadic-E"),
        (150, 0.8, 145, 2800, "Too far for sporadic-E"),
        (150, 0.8, 430, 1000, "UHF (less sporadic-E)"),
    ]
    
    for solar, season, freq, dist, desc in test_cases:
        # Run multiple times due to stochastic nature
        activations = []
        enhancements = []
        
        for i in range(20):
            np.random.seed(3000 + i)  # Set seed for reproducibility
            active, enhancement = propagation.calculate_sporadic_e_propagation(
                distance_km=dist,
                frequency_mhz=freq,
                solar_flux=solar,
                season_factor=season
            )
            activations.append(active)
            if active:
                enhancements.append(enhancement)
        
        activation_rate = sum(activations) / len(activations)
        mean_enhancement = np.mean(enhancements) if enhancements else 0.0
        
        print(f"\n{desc}:")
        print(f"  Solar flux: {solar}, Season: {season:.1f}")
        print(f"  Frequency: {freq}MHz, Distance: {dist}km")
        print(f"  Activation rate: {activation_rate*100:.1f}%")
        if enhancements:
            print(f"  Mean enhancement: {mean_enhancement:.1f} dB (range: {min(enhancements):.1f}-{max(enhancements):.1f} dB)")
        else:
            print(f"  No sporadic-E events detected")
        
        # Enhancement should be positive when active
        if enhancements:
            assert all(e > 0 for e in enhancements), "Sporadic-E should enhance signal"
            assert all(e <= 40 for e in enhancements), "Enhancement unrealistically high"
        
        # Validate activation rate is reasonable (sporadic-E is rare, so low rates are expected)
        assert 0 <= activation_rate <= 0.5, f"Activation rate out of expected range: {activation_rate:.2f}"
    
    print("\n✅ Sporadic-E propagation test passed!")


def test_tx_power_fluctuations():
    """Test transmitter power fluctuation modeling."""
    print("\n=== Test 7: TX Power Fluctuations ===")
    
    propagation = RFPropagationModel()
    nominal_power = 37.0  # dBm
    
    # Test cases: (transmitter_quality, description)
    test_cases = [
        (0.95, "High-quality transmitter"),
        (0.70, "Medium-quality transmitter"),
        (0.40, "Low-quality transmitter"),
    ]
    
    for quality, desc in test_cases:
        # Generate multiple samples
        actual_powers = []
        deviations = []
        
        for i in range(50):
            np.random.seed(4000 + i)  # Set seed for reproducibility
            actual_power = propagation.calculate_tx_power_fluctuation(
                nominal_power_dbm=nominal_power,
                transmitter_quality=quality
            )
            actual_powers.append(actual_power)
            deviations.append(actual_power - nominal_power)
        
        mean_power = np.mean(actual_powers)
        std_dev = np.std(deviations)
        
        print(f"\n{desc} (quality={quality}):")
        print(f"  Nominal power: {nominal_power:.1f} dBm")
        print(f"  Mean actual power: {mean_power:.2f} dBm (Δ = {mean_power - nominal_power:+.2f} dB)")
        print(f"  Std deviation: {std_dev:.3f} dB")
        print(f"  Range: {min(actual_powers):.2f} to {max(actual_powers):.2f} dBm")
        
        # Validate: higher quality = lower variation
        assert 0 < std_dev < 3.0, f"Power variation out of expected range: {std_dev:.3f} dB"
        
        # Check quality correlation
        if quality > 0.9:
            assert std_dev < 0.5, f"High-quality TX should have low variation: {std_dev:.3f} dB"
        elif quality < 0.5:
            assert std_dev > 0.5, f"Low-quality TX should have higher variation: {std_dev:.3f} dB"
    
    print("\n✅ TX power fluctuation test passed!")


def test_intermittent_transmissions():
    """Test intermittent transmission modeling."""
    print("\n=== Test 8: Intermittent Transmissions ===")
    
    propagation = RFPropagationModel()
    
    # Test cases: (duty_cycle, description)
    test_cases = [
        (1.0, "Continuous transmission"),
        (0.9, "High duty cycle (occasional drops)"),
        (0.7, "Medium duty cycle"),
        (0.5, "Low duty cycle (50% on-air time)"),
    ]
    
    for duty_cycle, desc in test_cases:
        # Generate multiple samples
        transmissions = []
        
        for i in range(100):
            np.random.seed(5000 + i)  # Set seed for reproducibility
            active = propagation.check_intermittent_transmission(
                transmission_duty_cycle=duty_cycle
            )
            transmissions.append(active)
        
        on_air_rate = sum(transmissions) / len(transmissions)
        
        print(f"\n{desc} (duty_cycle={duty_cycle}):")
        print(f"  Expected on-air: {duty_cycle*100:.0f}%")
        print(f"  Actual on-air: {on_air_rate*100:.1f}%")
        print(f"  Difference: {abs(on_air_rate - duty_cycle)*100:.1f}%")
        
        # Validate: actual rate should be close to duty cycle (within 15% for 100 samples)
        assert abs(on_air_rate - duty_cycle) < 0.15, \
            f"On-air rate {on_air_rate:.2f} too far from duty cycle {duty_cycle}"
    
    print("\n✅ Intermittent transmission test passed!")


def test_new_features_integration():
    """Test integration of all new propagation features."""
    print("\n=== Test 9: New Features Integration ===")
    
    propagation = RFPropagationModel()
    
    # TX and RX positions with obstacle (100 km apart)
    tx_lat, tx_lon, tx_alt = 45.0, 7.5, 300.0
    rx_lat, rx_lon, rx_alt = 45.9, 7.5, 500.0
    
    print("\nTesting with all new features enabled...")
    
    results = []
    active_transmissions = []
    sporadic_e_events = []
    
    for i in range(100):  # More samples for statistical significance
        meteo = MeteorologicalParameters.random(seed=2000 + i)
        
        rx_power, snr, details = propagation.calculate_received_power(
            tx_power_dbm=37.0,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=145.0,
            terrain_lookup=None,
            meteo_params=meteo,
            transmitter_quality=0.7,
            transmission_duty_cycle=0.9,
            enable_sporadic_e=True,
            enable_knife_edge=True
        )
        
        results.append(rx_power)
        active_transmissions.append(details['transmission_active'])
        
        # Track sporadic-E events (only when transmission is active)
        if details['transmission_active'] and details['sporadic_e_active']:
            sporadic_e_events.append(details['sporadic_e_enhancement_db'])
        
        if i < 3:  # Print first 3 samples
            print(f"\n  Sample {i}:")
            print(f"    Transmission active: {details['transmission_active']}")
            print(f"    TX power variation: {details['tx_power_variation_db']:+.2f} dB")
            print(f"    Knife-edge loss: {details['knife_edge_loss_db']:.2f} dB")
            print(f"    Sporadic-E: {'YES' if details['sporadic_e_active'] else 'NO'}", end='')
            if details['sporadic_e_active']:
                print(f" (+{details['sporadic_e_enhancement_db']:.1f} dB)")
            else:
                print()
            print(f"    RX power: {rx_power:.2f} dBm, SNR: {snr:.2f} dB")
    
    # Statistics
    on_air_rate = sum(active_transmissions) / len(active_transmissions)
    sporadic_e_rate = len(sporadic_e_events) / len(results)
    
    print(f"\nIntegration statistics ({len(results)} samples):")
    print(f"  On-air rate: {on_air_rate*100:.1f}% (expected ~90%)")
    print(f"  Sporadic-E rate: {sporadic_e_rate*100:.1f}%")
    if sporadic_e_events:
        print(f"  Sporadic-E enhancement: {np.mean(sporadic_e_events):.1f} dB (mean)")
    print(f"  RX power range: {min(results):.2f} to {max(results):.2f} dBm")
    print(f"  Total variation: {max(results) - min(results):.2f} dB")
    
    # Validate (allow 80-100% since it's probabilistic with small sample size)
    assert 0.80 < on_air_rate <= 1.0, f"On-air rate unexpected: {on_air_rate:.2f}"
    
    print("\n✅ New features integration test passed!")


def test_diversity_statistics():
    """Test that meteorological effects add sufficient training data diversity."""
    print("\n=== Test 10: Training Data Diversity ===")
    
    propagation = RFPropagationModel()
    
    # Fixed TX/RX positions (simulate 100 training samples at same location)
    tx_lat, tx_lon, tx_alt = 45.0, 7.5, 300.0
    rx_lat, rx_lon, rx_alt = 45.9, 7.5, 500.0
    
    num_samples = 100
    rx_powers = []
    
    print(f"\nGenerating {num_samples} samples with random meteorological conditions...")
    
    for i in range(num_samples):
        meteo = MeteorologicalParameters.random(seed=1000 + i)
        
        rx_power, _, _ = propagation.calculate_received_power(
            tx_power_dbm=37.0,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            tx_alt=tx_alt,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            rx_alt=rx_alt,
            frequency_mhz=145.0,
            terrain_lookup=None,
            meteo_params=meteo,
            transmitter_quality=0.7,
            transmission_duty_cycle=0.95,
            enable_sporadic_e=True,
            enable_knife_edge=True
        )
        
        rx_powers.append(rx_power)
    
    # Statistics
    mean = np.mean(rx_powers)
    std = np.std(rx_powers)
    min_power = min(rx_powers)
    max_power = max(rx_powers)
    
    print(f"\nDiversity statistics ({num_samples} samples):")
    print(f"  Mean RX power: {mean:.2f} dBm")
    print(f"  Standard deviation: {std:.2f} dB")
    print(f"  Range: {min_power:.2f} to {max_power:.2f} dBm")
    print(f"  Total variation: {max_power - min_power:.2f} dB")
    
    # Create histogram
    bins = np.linspace(min_power, max_power, 10)
    hist, _ = np.histogram(rx_powers, bins=bins)
    
    print(f"\nDistribution histogram:")
    for i, count in enumerate(hist):
        bar = "█" * int(count / max(hist) * 40)
        print(f"  {bins[i]:.1f} - {bins[i+1]:.1f} dBm: {bar} ({count})")
    
    # Validate: expect significant variation from meteorological effects
    # With all features: atmospheric absorption (~1-6 dB) + tropospheric refraction (±5-20 dB)
    # + knife-edge diffraction (0-15 dB) + sporadic-E (0-30 dB) + TX variations (±1-2 dB)
    # total variation can be 40-70 dB for extreme cases
    total_variation = max_power - min_power
    assert total_variation >= 8.0, f"Total variation too small: {total_variation:.2f} dB"
    assert total_variation <= 80.0, f"Total variation unrealistically large: {total_variation:.2f} dB"
    
    print(f"\n✅ Training data diversity test passed!")
    print(f"   All propagation features add ~{total_variation:.1f} dB variation to training data")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Meteorological Effects & New Propagation Features")
    print("=" * 70)
    
    # Original meteorological tests
    test_meteo_parameters_random()
    test_atmospheric_absorption()
    test_tropospheric_refraction()
    test_propagation_with_meteo()
    
    # New propagation feature tests
    test_knife_edge_diffraction()
    test_sporadic_e_propagation()
    test_tx_power_fluctuations()
    test_intermittent_transmissions()
    test_new_features_integration()
    
    # Combined diversity test
    test_diversity_statistics()
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print("- MeteorologicalParameters.random() generates realistic conditions")
    print("- Atmospheric absorption: ~0.01-0.5 dB for VHF/UHF over 100-500 km")
    print("- Tropospheric refraction: ±3-5 dB typical, 5-20 dB during ducting")
    print("- Knife-edge diffraction: 0-15 dB loss depending on obstacle geometry")
    print("- Sporadic-E propagation: 10-30 dB enhancement at 500-2500 km (VHF)")
    print("- TX power fluctuations: ±0.3-2.0 dB depending on transmitter quality")
    print("- Intermittent transmissions: 50-100% duty cycle modeling")
    print("- Combined effects: ±10-40 dB variation in training data")
    print("- With antenna patterns (±10-15 dB): total ±20-50 dB variation")
    print("\nImpact: Comprehensive RF propagation modeling with realistic phenomena!")
