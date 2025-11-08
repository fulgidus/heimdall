"""
Tests for polarization mismatch loss implementation.

Validates:
1. Same polarization (V-V, H-H) → 0-0.5 dB loss
2. Cross-polarization (V-H, H-V) → 15-25 dB loss
3. Circular polarization cases → 3-5 dB loss
4. Slant-45° polarization → 3-5 dB loss
5. Integration with full propagation model
6. Antenna type polarization assignment
"""

import pytest
import numpy as np
from src.data.propagation import (
    RFPropagationModel,
    AntennaPattern,
    AntennaType,
    Polarization
)


class TestPolarizationMismatchLoss:
    """Test suite for polarization mismatch loss calculations."""
    
    @pytest.fixture
    def propagation_model(self):
        """Create a propagation model instance."""
        return RFPropagationModel()
    
    def test_same_polarization_vertical(self, propagation_model):
        """Test that V-V polarization has minimal loss (0-0.5 dB)."""
        losses = []
        
        for _ in range(100):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.VERTICAL,
                Polarization.VERTICAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(0.0 <= loss <= 0.5 for loss in losses), \
            f"V-V loss out of range: {min(losses)}-{max(losses)} dB"
        
        # Check mean is reasonable
        mean_loss = np.mean(losses)
        assert 0.0 <= mean_loss <= 0.5, \
            f"V-V mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_same_polarization_horizontal(self, propagation_model):
        """Test that H-H polarization has minimal loss (0-0.5 dB)."""
        losses = []
        
        for _ in range(100):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.HORIZONTAL,
                Polarization.HORIZONTAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(0.0 <= loss <= 0.5 for loss in losses), \
            f"H-H loss out of range: {min(losses)}-{max(losses)} dB"
        
        mean_loss = np.mean(losses)
        assert 0.0 <= mean_loss <= 0.5, \
            f"H-H mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_cross_polarization_vertical_horizontal(self, propagation_model):
        """Test that V-H cross-polarization has 15-25 dB loss."""
        losses = []
        
        for _ in range(100):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.VERTICAL,
                Polarization.HORIZONTAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(15.0 <= loss <= 25.0 for loss in losses), \
            f"V-H loss out of range: {min(losses)}-{max(losses)} dB"
        
        mean_loss = np.mean(losses)
        assert 17.0 <= mean_loss <= 23.0, \
            f"V-H mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_cross_polarization_horizontal_vertical(self, propagation_model):
        """Test that H-V cross-polarization has 15-25 dB loss."""
        losses = []
        
        for _ in range(100):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.HORIZONTAL,
                Polarization.VERTICAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(15.0 <= loss <= 25.0 for loss in losses), \
            f"H-V loss out of range: {min(losses)}-{max(losses)} dB"
        
        mean_loss = np.mean(losses)
        assert 17.0 <= mean_loss <= 23.0, \
            f"H-V mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_cross_polarization_no_multipath(self, propagation_model):
        """Test that cross-pol loss is higher without multipath depolarization."""
        losses_with_multipath = []
        losses_without_multipath = []
        
        for _ in range(50):
            loss_with = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.VERTICAL,
                Polarization.HORIZONTAL,
                include_multipath_depolarization=True
            )
            loss_without = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.VERTICAL,
                Polarization.HORIZONTAL,
                include_multipath_depolarization=False
            )
            losses_with_multipath.append(loss_with)
            losses_without_multipath.append(loss_without)
        
        # Without multipath: 20-30 dB
        assert all(20.0 <= loss <= 30.0 for loss in losses_without_multipath), \
            f"V-H loss without multipath out of range: {min(losses_without_multipath)}-{max(losses_without_multipath)} dB"
        
        # Mean loss should be higher without multipath
        mean_with = np.mean(losses_with_multipath)
        mean_without = np.mean(losses_without_multipath)
        assert mean_without > mean_with, \
            f"Loss without multipath ({mean_without:.2f} dB) should be higher than with multipath ({mean_with:.2f} dB)"
    
    def test_circular_to_linear_polarization(self, propagation_model):
        """Test that circular-to-linear polarization has 3-5 dB loss."""
        losses = []
        
        # RHCP to V
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.CIRCULAR_RIGHT,
                Polarization.VERTICAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # LHCP to H
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.CIRCULAR_LEFT,
                Polarization.HORIZONTAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(3.0 <= loss <= 5.0 for loss in losses), \
            f"Circular-to-linear loss out of range: {min(losses)}-{max(losses)} dB"
        
        mean_loss = np.mean(losses)
        assert 3.5 <= mean_loss <= 4.5, \
            f"Circular-to-linear mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_linear_to_circular_polarization(self, propagation_model):
        """Test that linear-to-circular polarization has 3-5 dB loss."""
        losses = []
        
        # V to RHCP
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.VERTICAL,
                Polarization.CIRCULAR_RIGHT,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # H to LHCP
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.HORIZONTAL,
                Polarization.CIRCULAR_LEFT,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(3.0 <= loss <= 5.0 for loss in losses), \
            f"Linear-to-circular loss out of range: {min(losses)}-{max(losses)} dB"
        
        mean_loss = np.mean(losses)
        assert 3.5 <= mean_loss <= 4.5, \
            f"Linear-to-circular mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_opposite_circular_polarizations(self, propagation_model):
        """Test that opposite circular polarizations have 20-30 dB isolation."""
        losses = []
        
        # RHCP to LHCP
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.CIRCULAR_RIGHT,
                Polarization.CIRCULAR_LEFT,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # LHCP to RHCP
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.CIRCULAR_LEFT,
                Polarization.CIRCULAR_RIGHT,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(20.0 <= loss <= 30.0 for loss in losses), \
            f"Opposite circular loss out of range: {min(losses)}-{max(losses)} dB"
        
        mean_loss = np.mean(losses)
        assert 22.0 <= mean_loss <= 28.0, \
            f"Opposite circular mean loss {mean_loss:.2f} dB not in expected range"
    
    def test_slant_45_same_polarization(self, propagation_model):
        """Test that matching slant-45 polarization has minimal loss."""
        losses = []
        
        for _ in range(100):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.SLANT_45,
                Polarization.SLANT_45,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(0.0 <= loss <= 0.5 for loss in losses), \
            f"Slant-45 matching loss out of range: {min(losses)}-{max(losses)} dB"
    
    def test_slant_45_mismatch(self, propagation_model):
        """Test that slant-45 to V/H has 3-5 dB loss."""
        losses = []
        
        # Slant to V
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.SLANT_45,
                Polarization.VERTICAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Slant to H
        for _ in range(50):
            loss = propagation_model.calculate_polarization_mismatch_loss(
                Polarization.SLANT_45,
                Polarization.HORIZONTAL,
                include_multipath_depolarization=True
            )
            losses.append(loss)
        
        # Check all losses are in expected range
        assert all(3.0 <= loss <= 5.0 for loss in losses), \
            f"Slant-45 mismatch loss out of range: {min(losses)}-{max(losses)} dB"


class TestAntennaTypePolarizationAssignment:
    """Test that antenna types get correct polarization assignments."""
    
    def test_omni_vertical_always_vertical(self):
        """Test that OMNI_VERTICAL antennas are always vertically polarized."""
        for _ in range(50):
            antenna = AntennaPattern(AntennaType.OMNI_VERTICAL)
            assert antenna.polarization == Polarization.VERTICAL, \
                "OMNI_VERTICAL should always be vertically polarized"
    
    def test_collinear_always_vertical(self):
        """Test that COLLINEAR antennas are always vertically polarized."""
        for _ in range(50):
            antenna = AntennaPattern(AntennaType.COLLINEAR)
            assert antenna.polarization == Polarization.VERTICAL, \
                "COLLINEAR should always be vertically polarized"
    
    def test_whip_always_vertical(self):
        """Test that WHIP antennas are always vertically polarized."""
        for _ in range(50):
            antenna = AntennaPattern(AntennaType.WHIP)
            assert antenna.polarization == Polarization.VERTICAL, \
                "WHIP should always be vertically polarized"
    
    def test_rubber_duck_always_vertical(self):
        """Test that RUBBER_DUCK antennas are always vertically polarized."""
        for _ in range(50):
            antenna = AntennaPattern(AntennaType.RUBBER_DUCK)
            assert antenna.polarization == Polarization.VERTICAL, \
                "RUBBER_DUCK should always be vertically polarized"
    
    def test_yagi_polarization_distribution(self):
        """Test that YAGI antennas are 70% horizontal, 30% vertical."""
        polarizations = []
        
        for _ in range(1000):
            antenna = AntennaPattern(AntennaType.YAGI)
            polarizations.append(antenna.polarization)
        
        vertical_count = sum(1 for p in polarizations if p == Polarization.VERTICAL)
        horizontal_count = sum(1 for p in polarizations if p == Polarization.HORIZONTAL)
        
        vertical_ratio = vertical_count / len(polarizations)
        horizontal_ratio = horizontal_count / len(polarizations)
        
        # Allow ±5% tolerance due to randomness
        assert 0.25 <= vertical_ratio <= 0.35, \
            f"YAGI vertical ratio {vertical_ratio:.2%} not near 30%"
        assert 0.65 <= horizontal_ratio <= 0.75, \
            f"YAGI horizontal ratio {horizontal_ratio:.2%} not near 70%"
    
    def test_log_periodic_polarization_distribution(self):
        """Test that LOG_PERIODIC antennas are 80% horizontal, 20% vertical."""
        polarizations = []
        
        for _ in range(1000):
            antenna = AntennaPattern(AntennaType.LOG_PERIODIC)
            polarizations.append(antenna.polarization)
        
        vertical_count = sum(1 for p in polarizations if p == Polarization.VERTICAL)
        horizontal_count = sum(1 for p in polarizations if p == Polarization.HORIZONTAL)
        
        vertical_ratio = vertical_count / len(polarizations)
        horizontal_ratio = horizontal_count / len(polarizations)
        
        # Allow ±5% tolerance due to randomness
        assert 0.15 <= vertical_ratio <= 0.25, \
            f"LOG_PERIODIC vertical ratio {vertical_ratio:.2%} not near 20%"
        assert 0.75 <= horizontal_ratio <= 0.85, \
            f"LOG_PERIODIC horizontal ratio {horizontal_ratio:.2%} not near 80%"
    
    def test_portable_directional_polarization_distribution(self):
        """Test that PORTABLE_DIRECTIONAL antennas are 50% vertical, 50% horizontal."""
        polarizations = []
        
        for _ in range(1000):
            antenna = AntennaPattern(AntennaType.PORTABLE_DIRECTIONAL)
            polarizations.append(antenna.polarization)
        
        vertical_count = sum(1 for p in polarizations if p == Polarization.VERTICAL)
        horizontal_count = sum(1 for p in polarizations if p == Polarization.HORIZONTAL)
        
        vertical_ratio = vertical_count / len(polarizations)
        horizontal_ratio = horizontal_count / len(polarizations)
        
        # Allow ±5% tolerance due to randomness
        assert 0.45 <= vertical_ratio <= 0.55, \
            f"PORTABLE_DIRECTIONAL vertical ratio {vertical_ratio:.2%} not near 50%"
        assert 0.45 <= horizontal_ratio <= 0.55, \
            f"PORTABLE_DIRECTIONAL horizontal ratio {horizontal_ratio:.2%} not near 50%"


class TestPolarizationIntegrationWithPropagation:
    """Test polarization loss integration with full propagation model."""
    
    @pytest.fixture
    def propagation_model(self):
        """Create a propagation model instance."""
        return RFPropagationModel()
    
    def test_polarization_affects_received_power_same_pol(self, propagation_model):
        """Test that polarization mismatch affects received power correctly."""
        tx_power_dbm = 40.0
        tx_lat, tx_lon, tx_alt = 45.0, 10.0, 500.0
        rx_lat, rx_lon, rx_alt = 45.1, 10.1, 300.0
        frequency_mhz = 145.0
        
        # Same polarization (both vertical)
        tx_antenna_v1 = AntennaPattern(AntennaType.WHIP)
        rx_antenna_v = AntennaPattern(AntennaType.OMNI_VERTICAL)
        
        rx_power_same, _, details_same = propagation_model.calculate_received_power(
            tx_power_dbm, tx_lat, tx_lon, tx_alt,
            rx_lat, rx_lon, rx_alt, frequency_mhz,
            tx_antenna=tx_antenna_v1,
            rx_antenna=rx_antenna_v,
            enable_polarization_effects=True
        )
        
        # Cross polarization (vertical TX, horizontal RX)
        tx_antenna_v2 = AntennaPattern(AntennaType.WHIP)  # Vertical
        rx_antenna_h = AntennaPattern(AntennaType.YAGI)  # Might be horizontal
        
        # Ensure it's horizontal by checking and recreating if needed
        while rx_antenna_h.polarization != Polarization.HORIZONTAL:
            rx_antenna_h = AntennaPattern(AntennaType.YAGI)
        
        rx_power_cross, _, details_cross = propagation_model.calculate_received_power(
            tx_power_dbm, tx_lat, tx_lon, tx_alt,
            rx_lat, rx_lon, rx_alt, frequency_mhz,
            tx_antenna=tx_antenna_v2,
            rx_antenna=rx_antenna_h,
            enable_polarization_effects=True
        )
        
        # Check polarization loss values directly (more reliable than comparing rx_power)
        # Same pol should have minimal loss (0-0.5 dB)
        assert 0.0 <= details_same["polarization_loss_db"] <= 0.5, \
            f"Same pol loss ({details_same['polarization_loss_db']:.2f} dB) should be 0-0.5 dB"
        
        # Cross pol should have 15-25 dB loss
        assert 15.0 <= details_cross["polarization_loss_db"] <= 25.0, \
            f"Cross pol loss ({details_cross['polarization_loss_db']:.2f} dB) should be 15-25 dB"
        
        # Same pol should still have higher rx_power overall
        assert rx_power_same > rx_power_cross, \
            f"Same pol rx_power ({rx_power_same:.2f} dBm) should be higher than cross pol ({rx_power_cross:.2f} dBm)"
    
    def test_polarization_loss_in_details_dict(self, propagation_model):
        """Test that polarization loss is included in details dict."""
        tx_power_dbm = 40.0
        tx_lat, tx_lon, tx_alt = 45.0, 10.0, 500.0
        rx_lat, rx_lon, rx_alt = 45.1, 10.1, 300.0
        frequency_mhz = 145.0
        
        tx_antenna = AntennaPattern(AntennaType.WHIP)
        rx_antenna = AntennaPattern(AntennaType.OMNI_VERTICAL)
        
        _, _, details = propagation_model.calculate_received_power(
            tx_power_dbm, tx_lat, tx_lon, tx_alt,
            rx_lat, rx_lon, rx_alt, frequency_mhz,
            tx_antenna=tx_antenna,
            rx_antenna=rx_antenna,
            enable_polarization_effects=True
        )
        
        # Check that polarization fields are present
        assert "polarization_loss_db" in details, \
            "polarization_loss_db should be in details dict"
        assert "tx_polarization" in details, \
            "tx_polarization should be in details dict"
        assert "rx_polarization" in details, \
            "rx_polarization should be in details dict"
        
        # Check values are reasonable
        assert details["polarization_loss_db"] >= 0.0, \
            "polarization_loss_db should be non-negative"
        assert details["tx_polarization"] == "vertical", \
            "TX polarization should be vertical (WHIP)"
        assert details["rx_polarization"] == "vertical", \
            "RX polarization should be vertical (OMNI_VERTICAL)"
    
    def test_polarization_disabled(self, propagation_model):
        """Test that polarization effects can be disabled."""
        tx_power_dbm = 40.0
        tx_lat, tx_lon, tx_alt = 45.0, 10.0, 500.0
        rx_lat, rx_lon, rx_alt = 45.1, 10.1, 300.0
        frequency_mhz = 145.0
        
        # Cross-pol antennas
        tx_antenna = AntennaPattern(AntennaType.WHIP)  # Vertical
        rx_antenna = AntennaPattern(AntennaType.YAGI)
        
        # Ensure horizontal
        while rx_antenna.polarization != Polarization.HORIZONTAL:
            rx_antenna = AntennaPattern(AntennaType.YAGI)
        
        # With polarization effects enabled (for comparison)
        _, _, details_enabled = propagation_model.calculate_received_power(
            tx_power_dbm, tx_lat, tx_lon, tx_alt,
            rx_lat, rx_lon, rx_alt, frequency_mhz,
            tx_antenna=tx_antenna,
            rx_antenna=rx_antenna,
            enable_polarization_effects=True
        )
        
        # With polarization effects disabled
        _, _, details_disabled = propagation_model.calculate_received_power(
            tx_power_dbm, tx_lat, tx_lon, tx_alt,
            rx_lat, rx_lon, rx_alt, frequency_mhz,
            tx_antenna=tx_antenna,
            rx_antenna=rx_antenna,
            enable_polarization_effects=False
        )
        
        # Enabled should have loss > 0
        assert details_enabled["polarization_loss_db"] > 0.0, \
            "Enabled polarization effects should have loss > 0 for cross-pol"
        
        # Disabled should have loss = 0
        assert details_disabled["polarization_loss_db"] == 0.0, \
            "polarization_loss_db should be 0 when disabled"
        assert details_disabled["tx_polarization"] is None, \
            "tx_polarization should be None when disabled"
        assert details_disabled["rx_polarization"] is None, \
            "rx_polarization should be None when disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
