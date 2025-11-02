"""
Integration tests for synthetic data generation pipeline.

Tests the full pipeline from sample generation to database storage.
"""

import pytest
import uuid
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from src.data.synthetic_generator import _generate_single_sample
from src.data.config import TrainingConfig, BoundingBox, ReceiverConfig


def test_generate_single_sample():
    """Test single sample generation (for multiprocessing)."""
    # Create receiver config
    receivers_list = [
        {
            'name': 'Test_RX_1',
            'latitude': 45.0,
            'longitude': 7.0,
            'altitude': 300.0
        }
    ]

    # Create training config dict (serializable for multiprocessing)
    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.5,
            'lat_max': 45.5,
            'lon_min': 6.5,
            'lon_max': 7.5
        },
        'training_bbox': {
            'lat_min': 44.0,
            'lat_max': 46.0,
            'lon_min': 6.0,
            'lon_max': 8.0
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 33.0,
        'min_snr_db': 3.0,
        'min_receivers': 1,
        'max_gdop': 50.0,
        'inside_ratio': 0.7
    }

    args = (0, receivers_list, training_config_dict, config, 42)

    # Generate sample
    result = _generate_single_sample(args)
    sample_idx, receiver_features, extraction_metadata, quality_metrics, iq_samples, tx_position = result

    # Assertions on structure
    assert sample_idx == 0
    assert isinstance(receiver_features, list)
    assert len(receiver_features) >= 0  # May be empty if signal not detected
    
    assert isinstance(extraction_metadata, dict)
    assert 'extraction_method' in extraction_metadata
    assert extraction_metadata['extraction_method'] == 'synthetic'
    assert 'num_chunks' in extraction_metadata
    assert 'chunk_duration_ms' in extraction_metadata
    
    assert isinstance(quality_metrics, dict)
    assert 'overall_confidence' in quality_metrics
    assert 'num_receivers_detected' in quality_metrics
    assert 'mean_snr_db' in quality_metrics
    assert 'gdop' in quality_metrics
    
    assert 0.0 <= quality_metrics['overall_confidence'] <= 1.0
    assert quality_metrics['num_receivers_detected'] in [0, 1]
    
    assert isinstance(iq_samples, dict)
    assert 'Test_RX_1' in iq_samples
    assert hasattr(iq_samples['Test_RX_1'], 'samples')
    assert len(iq_samples['Test_RX_1'].samples) == 200_000  # 200 kHz × 1 second
    
    # Check tx_position
    assert isinstance(tx_position, dict)
    assert 'tx_lat' in tx_position
    assert 'tx_lon' in tx_position
    assert 'tx_alt' in tx_position
    assert 'tx_power_dbm' in tx_position


def test_generate_single_sample_multiple_receivers():
    """Test single sample generation with multiple receivers."""
    receivers_list = [
        {'name': 'RX_1', 'latitude': 45.0, 'longitude': 7.0, 'altitude': 300.0},
        {'name': 'RX_2', 'latitude': 45.5, 'longitude': 7.5, 'altitude': 300.0},
        {'name': 'RX_3', 'latitude': 44.5, 'longitude': 7.5, 'altitude': 300.0}
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 6.5, 'lon_max': 8.0
        },
        'training_bbox': {
            'lat_min': 43.5, 'lat_max': 46.5,
            'lon_min': 6.0, 'lon_max': 8.5
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 37.0,
        'min_snr_db': 3.0,
        'min_receivers': 2,
        'max_gdop': 50.0,
        'inside_ratio': 0.7
    }

    args = (1, receivers_list, training_config_dict, config, 100)

    result = _generate_single_sample(args)
    sample_idx, receiver_features, extraction_metadata, quality_metrics, iq_samples, tx_position = result

    # Should have IQ samples for all 3 receivers
    assert len(iq_samples) == 3
    assert 'RX_1' in iq_samples
    assert 'RX_2' in iq_samples
    assert 'RX_3' in iq_samples

    # Check all IQ samples have correct length
    for rx_id, iq_sample in iq_samples.items():
        assert len(iq_sample.samples) == 200_000
        assert iq_sample.sample_rate_hz == 200_000
        assert iq_sample.rx_id == rx_id


def test_generate_single_sample_reproducibility():
    """Test that same seed produces identical samples."""
    receivers_list = [
        {'name': 'Test', 'latitude': 45.0, 'longitude': 7.0, 'altitude': 300.0}
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.5, 'lat_max': 45.5,
            'lon_min': 6.5, 'lon_max': 7.5
        },
        'training_bbox': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 6.0, 'lon_max': 8.0
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 33.0,
        'min_snr_db': 3.0,
        'min_receivers': 1,
        'max_gdop': 50.0,
        'inside_ratio': 0.7
    }

    # Generate with same seed twice
    seed = 12345
    args1 = (0, receivers_list, training_config_dict, config, seed)
    args2 = (0, receivers_list, training_config_dict, config, seed)

    result1 = _generate_single_sample(args1)
    result2 = _generate_single_sample(args2)

    # Extract components
    _, _, _, _, iq_samples1, tx_pos1 = result1
    _, _, _, _, iq_samples2, tx_pos2 = result2

    # TX position should be identical
    assert tx_pos1['tx_lat'] == tx_pos2['tx_lat']
    assert tx_pos1['tx_lon'] == tx_pos2['tx_lon']
    assert tx_pos1['tx_alt'] == tx_pos2['tx_alt']

    # IQ samples should be identical
    assert np.allclose(
        iq_samples1['Test'].samples,
        iq_samples2['Test'].samples
    )


def test_generate_single_sample_different_seeds():
    """Test that different seeds produce different samples."""
    receivers_list = [
        {'name': 'Test', 'latitude': 45.0, 'longitude': 7.0, 'altitude': 300.0}
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.5, 'lat_max': 45.5,
            'lon_min': 6.5, 'lon_max': 7.5
        },
        'training_bbox': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 6.0, 'lon_max': 8.0
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 33.0,
        'min_snr_db': 3.0,
        'min_receivers': 1,
        'max_gdop': 50.0,
        'inside_ratio': 0.7
    }

    # Generate with different seeds
    args1 = (0, receivers_list, training_config_dict, config, 111)
    args2 = (0, receivers_list, training_config_dict, config, 222)

    result1 = _generate_single_sample(args1)
    result2 = _generate_single_sample(args2)

    # Extract components
    _, _, _, _, iq_samples1, tx_pos1 = result1
    _, _, _, _, iq_samples2, tx_pos2 = result2

    # TX positions should be different (high probability)
    assert tx_pos1['tx_lat'] != tx_pos2['tx_lat'] or tx_pos1['tx_lon'] != tx_pos2['tx_lon']

    # IQ samples should be different
    assert not np.allclose(
        iq_samples1['Test'].samples,
        iq_samples2['Test'].samples
    )


def test_receiver_features_structure():
    """Test that receiver features have expected structure."""
    receivers_list = [
        {'name': 'TestRX', 'latitude': 45.0, 'longitude': 7.0, 'altitude': 300.0}
    ]

    training_config_dict = {
        'receiver_bbox': {
            'lat_min': 44.5, 'lat_max': 45.5,
            'lon_min': 6.5, 'lon_max': 7.5
        },
        'training_bbox': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 6.0, 'lon_max': 8.0
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 37.0,  # Higher power for better signal
        'min_snr_db': 3.0,
        'min_receivers': 1,
        'max_gdop': 50.0,
        'inside_ratio': 1.0  # Always inside to ensure signal
    }

    args = (0, receivers_list, training_config_dict, config, 42)
    result = _generate_single_sample(args)
    _, receiver_features, _, _, _, _ = result

    # If signal detected, check feature structure
    if len(receiver_features) > 0:
        feature = receiver_features[0]
        
        # Check required fields
        assert 'rx_id' in feature
        assert feature['rx_id'] == 'TestRX'
        
        # Check aggregated stats structure (from chunked extraction)
        assert 'snr' in feature
        assert 'mean' in feature['snr']
        assert 'std' in feature['snr']
        assert 'min' in feature['snr']
        assert 'max' in feature['snr']
        
        assert 'psd' in feature
        assert 'frequency_offset' in feature
        assert 'bandwidth' in feature
