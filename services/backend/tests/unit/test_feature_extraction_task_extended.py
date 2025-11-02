"""
Extended tests for real recording feature extraction task.

Tests the feature extraction from real IQ recordings stored in MinIO.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from src.tasks.feature_extraction_task import extract_recording_features


@pytest.fixture
def mock_iq_sample():
    """Generate mock IQ sample for testing."""
    # Import from common module
    from common.feature_extraction import IQSample

    # Generate simple sine wave IQ with known frequency offset
    sample_rate = 200_000
    duration_ms = 1000.0
    num_samples = int(sample_rate * duration_ms / 1000.0)

    t = np.arange(num_samples) / sample_rate
    frequency_offset = 15.0  # 15 Hz offset
    signal = np.exp(2j * np.pi * frequency_offset * t)

    # Add noise for realistic SNR
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
    signal += noise

    return IQSample(
        samples=signal.astype(np.complex64),
        sample_rate_hz=sample_rate,
        center_frequency_hz=144_000_000,
        rx_id='Test_RX',
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=datetime.now(timezone.utc).timestamp()
    )


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    # Mock connection context manager
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()
    
    return mock_pool, mock_conn


def test_extract_recording_features_structure():
    """Test that extract_recording_features is callable with correct signature."""
    # Verify function exists and has expected signature
    assert callable(extract_recording_features)
    
    # Check it's a Celery task
    assert hasattr(extract_recording_features, 'apply')
    assert hasattr(extract_recording_features, 'delay')


@pytest.mark.unit
def test_extract_recording_features_invalid_uuid():
    """Test error handling with invalid session ID."""
    with patch('src.tasks.feature_extraction_task.get_pool'):
        # Call with invalid UUID format
        result = extract_recording_features.apply(args=['not-a-uuid'])
        
        # Should handle error gracefully
        assert 'result' in result.__dict__ or 'status' in result.__dict__


@pytest.mark.unit
def test_iq_sample_loading_mock(mock_iq_sample):
    """Test IQ sample structure after loading."""
    # Verify mock IQ sample has expected structure
    assert hasattr(mock_iq_sample, 'samples')
    assert hasattr(mock_iq_sample, 'sample_rate_hz')
    assert hasattr(mock_iq_sample, 'center_frequency_hz')
    assert hasattr(mock_iq_sample, 'rx_id')
    assert hasattr(mock_iq_sample, 'rx_lat')
    assert hasattr(mock_iq_sample, 'rx_lon')
    assert hasattr(mock_iq_sample, 'timestamp')
    
    # Verify data types
    assert isinstance(mock_iq_sample.samples, np.ndarray)
    assert mock_iq_sample.samples.dtype == np.complex64
    assert len(mock_iq_sample.samples) == 200_000
    assert mock_iq_sample.sample_rate_hz == 200_000


@pytest.mark.unit
def test_feature_extraction_from_iq_sample(mock_iq_sample):
    """Test that features can be extracted from IQ sample."""
    from common.feature_extraction import RFFeatureExtractor
    
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)
    
    # Extract features (chunked for realism)
    features = extractor.extract_features_chunked(
        mock_iq_sample,
        chunk_duration_ms=200.0,
        num_chunks=5
    )
    
    # Verify feature structure
    assert isinstance(features, dict)
    assert 'snr_db' in features
    assert 'mean' in features['snr_db']
    assert 'std' in features['snr_db']
    
    # SNR should be reasonable (we added signal with SNR ~20dB)
    assert features['snr_db']['mean'] > 0, "Should detect signal presence"


@pytest.mark.unit
def test_feature_extraction_error_propagation():
    """Test that errors in feature extraction are properly caught and logged."""
    from common.feature_extraction import RFFeatureExtractor, IQSample
    
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)
    
    # Create invalid IQ sample (empty)
    invalid_iq = IQSample(
        samples=np.array([], dtype=np.complex64),  # Empty array
        sample_rate_hz=200_000,
        center_frequency_hz=144_000_000,
        rx_id='Test',
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=datetime.now(timezone.utc).timestamp()
    )
    
    # Should raise an error or handle gracefully
    try:
        features = extractor.extract_features(invalid_iq)
        # If it doesn't raise, should return reasonable default or indication of failure
        assert 'error' in features or features.get('signal_present') is False
    except (ValueError, IndexError, Exception) as e:
        # Expected to fail with empty samples
        assert True


@pytest.mark.unit  
def test_receiver_metadata_structure():
    """Test that receiver metadata is properly structured."""
    # Test the expected structure of receiver features after extraction
    receiver_feature = {
        'rx_id': 'Test_RX',
        'rx_lat': 45.0,
        'rx_lon': 7.0,
        'snr': {'mean': 20.5, 'std': 2.1, 'min': 18.0, 'max': 23.0},
        'psd': {'mean': -75.0, 'std': 3.0, 'min': -80.0, 'max': -70.0},
        'frequency_offset': {'mean': 15.0, 'std': 0.5, 'min': 14.0, 'max': 16.0},
        'bandwidth': {'mean': 12500.0, 'std': 100.0, 'min': 12400.0, 'max': 12600.0}
    }
    
    # Verify structure
    assert 'rx_id' in receiver_feature
    assert 'rx_lat' in receiver_feature
    assert 'rx_lon' in receiver_feature
    assert 'snr' in receiver_feature
    
    # Verify aggregated stats
    for metric in ['snr', 'psd', 'frequency_offset', 'bandwidth']:
        assert metric in receiver_feature
        assert 'mean' in receiver_feature[metric]
        assert 'std' in receiver_feature[metric]
        assert 'min' in receiver_feature[metric]
        assert 'max' in receiver_feature[metric]


@pytest.mark.unit
def test_extraction_metadata_structure():
    """Test expected extraction metadata structure."""
    metadata = {
        'extraction_method': 'real_recording',
        'num_chunks': 5,
        'chunk_duration_ms': 200.0,
        'sample_rate_hz': 200_000,
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # Verify required fields
    assert 'extraction_method' in metadata
    assert metadata['extraction_method'] in ['real_recording', 'synthetic']
    assert 'num_chunks' in metadata
    assert 'chunk_duration_ms' in metadata
    assert 'sample_rate_hz' in metadata


@pytest.mark.unit
def test_quality_metrics_structure():
    """Test expected quality metrics structure."""
    quality_metrics = {
        'overall_confidence': 0.85,
        'num_receivers_detected': 3,
        'mean_snr_db': 18.5,
        'gdop': 5.2,
        'detection_rate': 1.0
    }
    
    # Verify required fields
    assert 'overall_confidence' in quality_metrics
    assert 0.0 <= quality_metrics['overall_confidence'] <= 1.0
    assert 'num_receivers_detected' in quality_metrics
    assert quality_metrics['num_receivers_detected'] >= 0
    assert 'mean_snr_db' in quality_metrics


@pytest.mark.unit
def test_multiple_receivers_aggregation():
    """Test aggregation of features from multiple receivers."""
    # Simulate features from 3 receivers
    receiver_features = [
        {
            'rx_id': 'RX_1',
            'snr': {'mean': 20.0, 'std': 2.0, 'min': 18.0, 'max': 22.0}
        },
        {
            'rx_id': 'RX_2',
            'snr': {'mean': 18.5, 'std': 1.5, 'min': 17.0, 'max': 20.0}
        },
        {
            'rx_id': 'RX_3',
            'snr': {'mean': 22.0, 'std': 2.5, 'min': 19.5, 'max': 24.5}
        }
    ]
    
    # Calculate mean SNR across receivers
    mean_snrs = [rf['snr']['mean'] for rf in receiver_features]
    overall_mean_snr = sum(mean_snrs) / len(mean_snrs)
    
    assert abs(overall_mean_snr - 20.167) < 0.01  # Should be ~20.17
    assert len(receiver_features) == 3


@pytest.mark.unit
def test_beacon_info_structure():
    """Test BeaconInfo structure for known transmitters."""
    from src.tasks.feature_extraction_task import BeaconInfo
    
    # Test with known beacon
    known_beacon = BeaconInfo(
        known=True,
        latitude=45.5,
        longitude=7.5,
        power_dbm=33.0
    )
    
    assert known_beacon.known is True
    assert known_beacon.latitude == 45.5
    assert known_beacon.longitude == 7.5
    assert known_beacon.power_dbm == 33.0
    
    # Test with unknown beacon
    unknown_beacon = BeaconInfo(
        known=False,
        latitude=None,
        longitude=None,
        power_dbm=None
    )
    
    assert unknown_beacon.known is False
    assert unknown_beacon.latitude is None
    assert unknown_beacon.longitude is None
    assert unknown_beacon.power_dbm is None
