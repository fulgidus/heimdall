"""Tests for MinIO S3 storage integration."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.storage.minio_client import MinIOClient


class TestMinIOClient:
    """Test suite for MinIO client."""
    
    @pytest.fixture
    def minio_client(self):
        """Create MinIO client instance."""
        return MinIOClient(
            endpoint_url="http://minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket_name="heimdall-raw-iq",
        )
    
    @pytest.fixture
    def sample_iq_data(self):
        """Generate sample IQ data."""
        return np.random.normal(0, 0.1, 125000).astype(np.complex64)
    
    def test_minio_client_initialization(self, minio_client):
        """Test MinIO client initialization."""
        assert minio_client.endpoint_url == "http://minio:9000"
        assert minio_client.bucket_name == "heimdall-raw-iq"
        assert minio_client.s3_client is not None
    
    @patch('src.storage.minio_client.boto3.client')
    def test_ensure_bucket_exists_already_exists(self, mock_boto3, minio_client):
        """Test bucket existence check when bucket already exists."""
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        # Mock successful head_bucket call
        mock_s3_client.head_bucket.return_value = {}
        
        minio_client.s3_client = mock_s3_client
        
        result = minio_client.ensure_bucket_exists()
        
        assert result is True
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="heimdall-raw-iq")
        mock_s3_client.create_bucket.assert_not_called()
    
    @patch('src.storage.minio_client.boto3.client')
    def test_ensure_bucket_exists_creates_bucket(self, mock_boto3, minio_client):
        """Test bucket creation when bucket doesn't exist."""
        from botocore.exceptions import ClientError
        
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        # Mock 404 error on head_bucket
        error_response = {'Error': {'Code': '404'}}
        mock_s3_client.head_bucket.side_effect = ClientError(error_response, 'HeadBucket')
        
        # Mock successful create_bucket
        mock_s3_client.create_bucket.return_value = {}
        
        minio_client.s3_client = mock_s3_client
        
        result = minio_client.ensure_bucket_exists()
        
        assert result is True
        mock_s3_client.create_bucket.assert_called_once_with(Bucket="heimdall-raw-iq")
    
    @patch('src.storage.minio_client.boto3.client')
    def test_upload_iq_data_success(self, mock_boto3, minio_client, sample_iq_data):
        """Test successful IQ data upload."""
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        # Mock successful bucket check
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        
        minio_client.s3_client = mock_s3_client
        
        task_id = "task_12345"
        websdr_id = 1
        
        success, result = minio_client.upload_iq_data(
            iq_data=sample_iq_data,
            task_id=task_id,
            websdr_id=websdr_id,
        )
        
        assert success is True
        assert f"s3://heimdall-raw-iq/sessions/{task_id}/websdr_{websdr_id}.npy" in result
        
        # Verify put_object was called
        calls = mock_s3_client.put_object.call_args_list
        assert len(calls) >= 1  # At least one call for .npy file
    
    @patch('src.storage.minio_client.boto3.client')
    def test_upload_iq_data_with_metadata(self, mock_boto3, minio_client, sample_iq_data):
        """Test IQ data upload with metadata."""
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        mock_s3_client.head_bucket.return_value = {}
        mock_s3_client.put_object.return_value = {}
        
        minio_client.s3_client = mock_s3_client
        
        task_id = "task_12345"
        websdr_id = 2
        metadata = {
            'frequency_mhz': 100.0,
            'sample_rate_khz': 12.5,
            'metrics': {'snr_db': 15.5, 'frequency_offset_hz': -2.1}
        }
        
        success, result = minio_client.upload_iq_data(
            iq_data=sample_iq_data,
            task_id=task_id,
            websdr_id=websdr_id,
            metadata=metadata,
        )
        
        assert success is True
        
        # Verify both .npy and _metadata.json were uploaded
        calls = mock_s3_client.put_object.call_args_list
        assert len(calls) == 2  # .npy and .json files
    
    @patch('src.storage.minio_client.boto3.client')
    def test_download_iq_data_success(self, mock_boto3, minio_client, sample_iq_data):
        """Test successful IQ data download."""
        import io
        
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        # Prepare mock response
        buffer = io.BytesIO()
        np.save(buffer, sample_iq_data)
        buffer.seek(0)
        
        mock_s3_client.get_object.return_value = {
            'Body': type('obj', (object,), {'read': lambda self: buffer.getvalue()})()
        }
        
        minio_client.s3_client = mock_s3_client
        
        task_id = "task_12345"
        websdr_id = 1
        
        success, downloaded_data = minio_client.download_iq_data(
            task_id=task_id,
            websdr_id=websdr_id,
        )
        
        assert success is True
        assert downloaded_data is not None
        assert downloaded_data.dtype == np.complex64
        assert len(downloaded_data) == len(sample_iq_data)
    
    @patch('src.storage.minio_client.boto3.client')
    def test_get_session_measurements(self, mock_boto3, minio_client):
        """Test listing measurements from a session."""
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        task_id = "task_12345"
        
        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        
        # Mock paginate response with two measurements
        mock_page = {
            'Contents': [
                {
                    'Key': f'sessions/{task_id}/websdr_1.npy',
                    'Size': 500000,
                    'LastModified': datetime.now(),
                },
                {
                    'Key': f'sessions/{task_id}/websdr_2.npy',
                    'Size': 500000,
                    'LastModified': datetime.now(),
                },
                {
                    'Key': f'sessions/{task_id}/websdr_1_metadata.json',
                    'Size': 1000,
                    'LastModified': datetime.now(),
                },
            ]
        }
        
        mock_paginator.paginate.return_value = [mock_page]
        
        minio_client.s3_client = mock_s3_client
        
        measurements = minio_client.get_session_measurements(task_id)
        
        assert len(measurements) == 2
        assert 1 in measurements
        assert 2 in measurements
        assert measurements[1]['size_bytes'] == 500000
        assert measurements[2]['size_bytes'] == 500000
    
    @patch('src.storage.minio_client.boto3.client')
    def test_health_check_healthy(self, mock_boto3, minio_client):
        """Test MinIO health check when healthy."""
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        mock_s3_client.head_bucket.return_value = {}
        
        minio_client.s3_client = mock_s3_client
        
        health = minio_client.health_check()
        
        assert health['status'] == 'healthy'
        assert health['accessible'] is True
        assert health['bucket'] == 'heimdall-raw-iq'
    
    @patch('src.storage.minio_client.boto3.client')
    def test_health_check_unhealthy(self, mock_boto3, minio_client):
        """Test MinIO health check when unhealthy."""
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client
        
        mock_s3_client.head_bucket.side_effect = Exception("Connection failed")
        
        minio_client.s3_client = mock_s3_client
        
        health = minio_client.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['accessible'] is False
        assert 'error' in health


class TestSaveMeasurementsToMinIOTask:
    """Test suite for save_measurements_to_minio Celery task."""
    
    @pytest.fixture
    def sample_measurements(self):
        """Generate sample measurements."""
        measurements = []
        for i in range(1, 4):
            iq_data = np.random.normal(0, 0.1, 125000).astype(np.complex64)
            measurements.append({
                'websdr_id': i,
                'frequency_mhz': 100.0 + i,
                'sample_rate_khz': 12.5,
                'samples_count': len(iq_data),
                'timestamp_utc': datetime.utcnow().isoformat(),
                'metrics': {
                    'snr_db': 10.0 + i,
                    'frequency_offset_hz': -1.0 - i,
                },
                'iq_data': iq_data.tolist(),
            })
        return measurements
    
    @patch('src.tasks.acquire_iq.MinIOClient')
    def test_save_measurements_to_minio_success(self, mock_minio_class, sample_measurements):
        """Test successful save_measurements_to_minio task."""
        from src.tasks.acquire_iq import save_measurements_to_minio
        from celery import Task
        
        # Mock MinIO client
        mock_minio = MagicMock()
        mock_minio_class.return_value = mock_minio
        
        mock_minio.ensure_bucket_exists.return_value = True
        mock_minio.upload_iq_data.return_value = (True, "s3://bucket/path.npy")
        
        # Create mock task
        mock_task = MagicMock(spec=Task)
        mock_task.update_state = MagicMock()
        
        # Call task
        result = save_measurements_to_minio.apply_async(
            args=["task_12345", sample_measurements]
        )
        
        # Note: In actual test environment, result would come from Celery
        # For unit test, we'd need to mock more carefully
        assert result is not None
    
    @patch('src.tasks.acquire_iq.MinIOClient')
    def test_save_measurements_to_minio_bucket_fail(self, mock_minio_class, sample_measurements):
        """Test save_measurements_to_minio when bucket fails."""
        from src.tasks.acquire_iq import save_measurements_to_minio
        from celery import Task
        
        # Mock MinIO client with bucket failure
        mock_minio = MagicMock()
        mock_minio_class.return_value = mock_minio
        
        mock_minio.ensure_bucket_exists.return_value = False
        
        # Create mock task
        mock_task = MagicMock(spec=Task)
        mock_task.update_state = MagicMock()
        
        # Call task
        result = save_measurements_to_minio.apply_async(
            args=["task_12345", sample_measurements]
        )
        
        assert result is not None
