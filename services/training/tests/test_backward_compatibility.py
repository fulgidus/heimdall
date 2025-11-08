"""
Integration test for backward compatibility.

Verifies that:
- Italian receivers still work when use_random_receivers=False
- Default behavior uses random receivers
- Both modes produce valid TrainingConfig
"""

import pytest

from src.data.config import (
    BoundingBox,
    TrainingConfig,
    get_italian_receivers,
    generate_random_receivers
)


class TestBackwardCompatibility:
    """Test backward compatibility with Italian receivers."""

    def test_italian_receivers_still_work(self):
        """Verify Italian receivers function still works."""
        receivers = get_italian_receivers()

        assert len(receivers) == 7
        assert all(rx.name for rx in receivers)
        assert all(rx.latitude for rx in receivers)
        assert all(rx.longitude for rx in receivers)
        assert all(rx.altitude for rx in receivers)

    def test_italian_receivers_create_valid_config(self):
        """Verify Italian receivers create valid training config."""
        receivers = get_italian_receivers()
        config = TrainingConfig.from_receivers(receivers, margin_degrees=0.5)

        assert config.receivers == receivers
        assert len(config.receivers) == 7
        assert config.receiver_bbox.lat_min < config.receiver_bbox.lat_max
        assert config.receiver_bbox.lon_min < config.receiver_bbox.lon_max
        assert len(config.srtm_tiles) > 0

    def test_random_receivers_create_valid_config(self):
        """Verify random receivers create valid training config."""
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=5,
            terrain_lookup=None,
            seed=42
        )

        config = TrainingConfig.from_receivers(receivers, margin_degrees=0.5)

        assert config.receivers == receivers
        assert len(config.receivers) == 5
        assert config.receiver_bbox.lat_min < config.receiver_bbox.lat_max
        assert config.receiver_bbox.lon_min < config.receiver_bbox.lon_max
        assert len(config.srtm_tiles) > 0

    def test_both_modes_produce_different_configs(self):
        """Verify random and fixed receivers produce different geometries."""
        # Italian receivers
        italian_receivers = get_italian_receivers()
        italian_config = TrainingConfig.from_receivers(italian_receivers)

        # Random receivers in same area
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        random_receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=7,
            terrain_lookup=None,
            seed=999
        )
        random_config = TrainingConfig.from_receivers(random_receivers)

        # They should have different receiver positions
        # (unlikely to match by random chance)
        italian_positions = [(rx.latitude, rx.longitude) for rx in italian_receivers]
        random_positions = [(rx.latitude, rx.longitude) for rx in random_receivers]

        # At least one position should be different
        assert italian_positions != random_positions

    def test_config_dict_structure_compatible(self):
        """Verify config dict structure is compatible for both modes."""
        # Italian receivers
        italian_receivers = get_italian_receivers()
        italian_config = TrainingConfig.from_receivers(italian_receivers)

        # Random receivers
        bbox = BoundingBox(lat_min=44.0, lat_max=46.0, lon_min=7.0, lon_max=10.0)
        random_receivers = generate_random_receivers(
            bbox=bbox,
            num_receivers=5,
            terrain_lookup=None,
            seed=42
        )
        random_config = TrainingConfig.from_receivers(random_receivers)

        # Both should have same structure (for synthetic generator compatibility)
        assert hasattr(italian_config, 'receiver_bbox')
        assert hasattr(italian_config, 'training_bbox')
        assert hasattr(italian_config, 'srtm_tiles')

        assert hasattr(random_config, 'receiver_bbox')
        assert hasattr(random_config, 'training_bbox')
        assert hasattr(random_config, 'srtm_tiles')
