"""
Integration test for synthetic IQ generation pipeline.

Tests the full pipeline:
1. IQ sample generation
2. Feature extraction
3. Quality validation
"""

import pytest
import numpy as np

from src.data.synthetic_generator import _generate_single_sample
from src.data.config import TrainingConfig


def test_generate_single_sample_basic():
    """Test that _generate_single_sample produces valid output."""
    # Setup receiver configuration
    receivers_list = [
        {
            'name': 'I4YOU',
            'latitude': 45.4642,
            'longitude': 9.1900,
            'altitude': 120.0
        },
        {
            'name': 'IU1AZZ',
            'latitude': 44.4056,
            'longitude': 8.9463,
            'altitude': 4.0
        },
        {
            'name': 'IK4VET',
            'latitude': 44.5075,
            'longitude': 11.3514,
            'altitude': 55.0
        }
    ]

    # Setup training config (simplified bounding box)
    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.0,
            'lat_max': 46.0,
            'lon_min': 8.0,
            'lon_max': 12.0
        },
        'training_bbox': {
            'lat_min': 43.0,
            'lat_max': 47.0,
            'lon_min': 7.0,
            'lon_max': 13.0
        }
    }

    # Generation config
    config = {
        'frequency_mhz': 145.0,
        'tx_power_dbm': 37.0,
        'inside_ratio': 0.7,
        'min_snr_db': 3.0
    }

    # Generate a single sample
    sample_idx = 0
    seed = 42
    args = (sample_idx, receivers_list, training_config_dict, config, seed)

    result = _generate_single_sample(args)

    # Unpack result
    sample_idx_ret, receiver_features, extraction_metadata, quality_metrics, iq_samples, tx_position = result

    # Verify basic structure
    assert sample_idx_ret == sample_idx
    assert len(receiver_features) == 3  # 3 receivers
    assert 'extraction_method' in extraction_metadata
    assert extraction_metadata['extraction_method'] == 'synthetic'
    assert 'overall_confidence' in quality_metrics
    assert 'mean_snr_db' in quality_metrics
    assert 'num_receivers_detected' in quality_metrics
    assert 'tx_lat' in tx_position
    assert 'tx_lon' in tx_position

    # Verify feature structure for each receiver
    for rf in receiver_features:
        assert 'rx_id' in rf
        assert 'rx_lat' in rf
        assert 'rx_lon' in rf
        assert 'signal_present' in rf
        
        # Check aggregated features exist
        assert 'snr_db' in rf
        assert 'mean' in rf['snr_db']
        assert 'std' in rf['snr_db']
        assert 'min' in rf['snr_db']
        assert 'max' in rf['snr_db']

        # Check RSSI feature
        assert 'rssi_dbm' in rf
        assert 'mean' in rf['rssi_dbm']

        # Check frequency features
        assert 'frequency_offset_hz' in rf
        assert 'bandwidth_hz' in rf

    # Verify IQ samples (should be saved for sample_idx < 100)
    assert len(iq_samples) == 3  # All receivers should have IQ saved
    for rx_id, iq_sample in iq_samples.items():
        assert iq_sample.samples.dtype == np.complex64
        assert len(iq_sample.samples) == 200_000  # 200 kHz × 1 second
        assert iq_sample.sample_rate_hz == 200_000
        assert iq_sample.duration_ms == 1000.0

    print(f"✓ Generated sample {sample_idx} successfully")
    print(f"  - Receivers detected: {quality_metrics['num_receivers_detected']}")
    print(f"  - Mean SNR: {quality_metrics['mean_snr_db']:.1f} dB")
    print(f"  - Overall confidence: {quality_metrics['overall_confidence']:.2f}")
    print(f"  - GDOP: {quality_metrics['gdop']:.2f}")
    print(f"  - TX position: ({tx_position['tx_lat']:.4f}, {tx_position['tx_lon']:.4f})")


def test_generate_multiple_samples_reproducibility():
    """Test that same seed produces identical results."""
    receivers_list = [
        {
            'name': 'I4YOU',
            'latitude': 45.4642,
            'longitude': 9.1900,
            'altitude': 120.0
        }
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.0,
            'lat_max': 46.0,
            'lon_min': 8.0,
            'lon_max': 12.0
        },
        'training_bbox': {
            'lat_min': 43.0,
            'lat_max': 47.0,
            'lon_min': 7.0,
            'lon_max': 13.0
        }
    }

    config = {
        'frequency_mhz': 145.0,
        'tx_power_dbm': 37.0,
        'inside_ratio': 0.7,
        'min_snr_db': 3.0
    }

    # Generate sample twice with same seed
    seed = 12345
    sample_idx = 0
    args = (sample_idx, receivers_list, training_config_dict, config, seed)

    result1 = _generate_single_sample(args)
    result2 = _generate_single_sample(args)

    # Unpack results
    _, _, _, _, iq_samples1, tx_position1 = result1
    _, _, _, _, iq_samples2, tx_position2 = result2

    # Verify reproducibility
    assert tx_position1['tx_lat'] == tx_position2['tx_lat']
    assert tx_position1['tx_lon'] == tx_position2['tx_lon']

    # Verify IQ samples are identical
    for rx_id in iq_samples1.keys():
        assert np.allclose(iq_samples1[rx_id].samples, iq_samples2[rx_id].samples)

    print("✓ Reproducibility test passed")


def test_quality_validation():
    """Test that quality validation filters work correctly."""
    receivers_list = [
        {
            'name': 'I4YOU',
            'latitude': 45.4642,
            'longitude': 9.1900,
            'altitude': 120.0
        },
        {
            'name': 'IU1AZZ',
            'latitude': 44.4056,
            'longitude': 8.9463,
            'altitude': 4.0
        },
        {
            'name': 'IK4VET',
            'latitude': 44.5075,
            'longitude': 11.3514,
            'altitude': 55.0
        }
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.0,
            'lat_max': 46.0,
            'lon_min': 8.0,
            'lon_max': 12.0
        },
        'training_bbox': {
            'lat_min': 43.0,
            'lat_max': 47.0,
            'lon_min': 7.0,
            'lon_max': 13.0
        }
    }

    config = {
        'frequency_mhz': 145.0,
        'tx_power_dbm': 37.0,
        'inside_ratio': 0.7,
        'min_snr_db': 3.0
    }

    # Generate multiple samples and check quality metrics
    valid_samples = 0
    total_attempts = 10

    for i in range(total_attempts):
        args = (i, receivers_list, training_config_dict, config, 42 + i)
        _, _, _, quality_metrics, _, _ = _generate_single_sample(args)

        # Check if sample would pass validation
        min_snr_db = 3.0
        min_receivers = 2
        max_gdop = 50.0

        if (quality_metrics['num_receivers_detected'] >= min_receivers and
            quality_metrics['mean_snr_db'] >= min_snr_db and
            quality_metrics['gdop'] <= max_gdop):
            valid_samples += 1

    # Should have some valid samples
    success_rate = valid_samples / total_attempts
    print(f"✓ Quality validation test: {valid_samples}/{total_attempts} valid samples ({success_rate*100:.0f}%)")
    assert success_rate > 0.3, "Success rate too low - check quality thresholds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
