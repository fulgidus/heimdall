"""
Unit tests for random receiver generation.

Tests:
- generate_random_receivers() creates correct number of receivers
- Receivers are within specified bounding box
- Reproducibility with seed
- Altitude assignment (with and without terrain lookup)
"""

import pytest
import numpy as np
from unittest.mock import Mock

from src.data.config import (
    BoundingBox,
    ReceiverLocation,
    generate_random_receivers
)


class TestGenerateRandomReceivers:
    """Test suite for generate_random_receivers function."""

    def test_generates_correct_number(self):
        """Verify correct number of receivers generated."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)

        # Test different receiver counts
        for num_rx in [1, 5, 10, 15]:
            receivers = generate_random_receivers(
                bbox=bbox,
                num_receivers=num_rx,
                terrain_lookup=None,
                seed=42
            )
            assert len(receivers) == num_rx

    def test_receivers_within_bbox(self):
        """Verify all receivers are within bounding box."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=20,
            terrain_lookup=None,
            seed=42
        )

        for rx in receivers:
            assert bbox.lat_min <= rx.latitude <= bbox.lat_max
            assert bbox.lon_min <= rx.longitude <= bbox.lon_max

    def test_reproducibility_with_seed(self):
        """Verify same seed produces same receivers."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)

        receivers1 = generate_random_receivers(
            bbox=bbox,
            num_receivers=7,
            terrain_lookup=None,
            seed=12345
        )

        receivers2 = generate_random_receivers(
            bbox=bbox,
            num_receivers=7,
            terrain_lookup=None,
            seed=12345
        )

        # Should produce identical receivers
        for rx1, rx2 in zip(receivers1, receivers2):
            assert rx1.name == rx2.name
            assert rx1.latitude == rx2.latitude
            assert rx1.longitude == rx2.longitude
            assert rx1.altitude == rx2.altitude

    def test_different_seeds_produce_different_receivers(self):
        """Verify different seeds produce different receivers."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)

        receivers1 = generate_random_receivers(
            bbox=bbox,
            num_receivers=5,
            terrain_lookup=None,
            seed=111
        )

        receivers2 = generate_random_receivers(
            bbox=bbox,
            num_receivers=5,
            terrain_lookup=None,
            seed=999
        )

        # Should produce different positions (very unlikely to match by chance)
        different_positions = False
        for rx1, rx2 in zip(receivers1, receivers2):
            if rx1.latitude != rx2.latitude or rx1.longitude != rx2.longitude:
                different_positions = True
                break

        assert different_positions, "Different seeds should produce different positions"

    def test_altitude_assignment_without_terrain(self):
        """Verify altitude fallback when no terrain lookup provided."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=10,
            terrain_lookup=None,
            seed=42
        )

        # All altitudes should be within fallback range
        for rx in receivers:
            assert 50 <= rx.altitude <= 800

    def test_altitude_from_terrain_lookup(self):
        """Verify altitude from terrain lookup when available."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)

        # Mock terrain lookup
        mock_terrain = Mock()
        mock_terrain.get_elevation = Mock(return_value=350.5)

        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=5,
            terrain_lookup=mock_terrain,
            seed=42
        )

        # All receivers should have the mocked elevation
        for rx in receivers:
            assert rx.altitude == 350.5

        # Verify terrain lookup was called for each receiver
        assert mock_terrain.get_elevation.call_count == 5

    def test_terrain_lookup_fallback_on_error(self):
        """Verify fallback to random altitude when terrain lookup fails."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)

        # Mock terrain lookup that raises exception
        mock_terrain = Mock()
        mock_terrain.get_elevation = Mock(side_effect=Exception("Terrain data unavailable"))

        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=5,
            terrain_lookup=mock_terrain,
            seed=42
        )

        # Should still generate receivers with fallback altitudes
        assert len(receivers) == 5
        for rx in receivers:
            assert 50 <= rx.altitude <= 800

    def test_receiver_naming(self):
        """Verify receivers are named correctly (RX_01, RX_02, etc.)."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=12,
            terrain_lookup=None,
            seed=42
        )

        expected_names = [f"RX_{i+1:02d}" for i in range(12)]
        actual_names = [rx.name for rx in receivers]

        assert actual_names == expected_names

    def test_invalid_receiver_count(self):
        """Verify error on invalid receiver count."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)

        with pytest.raises(ValueError, match="Must generate at least 1 receiver"):
            generate_random_receivers(
                bbox=bbox,
                num_receivers=0,
                terrain_lookup=None,
                seed=42
            )

        with pytest.raises(ValueError, match="Must generate at least 1 receiver"):
            generate_random_receivers(
                bbox=bbox,
                num_receivers=-5,
                terrain_lookup=None,
                seed=42
            )

    def test_edge_case_single_receiver(self):
        """Verify single receiver generation works."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=1,
            terrain_lookup=None,
            seed=42
        )

        assert len(receivers) == 1
        assert receivers[0].name == "RX_01"
        assert bbox.lat_min <= receivers[0].latitude <= bbox.lat_max
        assert bbox.lon_min <= receivers[0].longitude <= bbox.lon_max

    def test_large_number_of_receivers(self):
        """Verify generation of large number of receivers."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=100,
            terrain_lookup=None,
            seed=42
        )

        assert len(receivers) == 100

        # Check all are within bounds
        for rx in receivers:
            assert bbox.lat_min <= rx.latitude <= bbox.lat_max
            assert bbox.lon_min <= rx.longitude <= bbox.lon_max
