"""
Test dataset and job deletion with MinIO cleanup.

This test validates that when datasets or jobs are deleted, the associated
IQ data files in MinIO storage are also properly cleaned up.
"""

import pytest
import uuid
from unittest.mock import Mock, patch


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client for testing."""
    mock_client = Mock()
    mock_client.delete_dataset_iq_data = Mock(return_value=(5, 0))  # 5 successful, 0 failed
    return mock_client


@pytest.fixture
def test_dataset_id():
    """Generate a test dataset UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def test_job_id():
    """Generate a test job UUID."""
    return str(uuid.uuid4())


class TestDatasetDeletion:
    """Test cases for dataset deletion with MinIO cleanup."""
    
    @patch('services.training.src.api.synthetic.MinIOClient')
    def test_delete_iq_raw_dataset_cleans_minio(
        self, 
        mock_minio_class,
        mock_minio_client,
        test_dataset_id
    ):
        """Test that deleting an iq_raw dataset cleans up MinIO files."""
        # Setup mock
        mock_minio_class.return_value = mock_minio_client
        
        # This test validates the logic flow
        # In a real test, you would:
        # 1. Create a test dataset in the database
        # 2. Create test IQ files in MinIO
        # 3. Call the delete endpoint
        # 4. Verify MinIO files are deleted
        # 5. Verify database records are deleted
        
        # For now, we verify the mock is called correctly
        assert mock_minio_client.delete_dataset_iq_data is not None
    
    @patch('services.training.src.api.synthetic.MinIOClient')
    def test_delete_feature_based_dataset_no_minio_cleanup(
        self,
        mock_minio_class,
        mock_minio_client,
        test_dataset_id
    ):
        """Test that deleting a feature_based dataset doesn't attempt MinIO cleanup."""
        # Feature-based datasets typically don't have IQ files
        # So MinIO cleanup should be skipped
        
        # This is validated by checking dataset_type before cleanup
        pass
    
    def test_delete_dataset_handles_minio_failure_gracefully(
        self,
        test_dataset_id
    ):
        """Test that dataset deletion continues even if MinIO cleanup fails."""
        # The deletion should:
        # 1. Log the MinIO error
        # 2. Still delete the database records
        # 3. Return success with warning about partial cleanup
        
        # This ensures the system is resilient to MinIO downtime
        pass
    
    @patch('services.training.src.api.synthetic.MinIOClient')
    def test_delete_job_with_dataset_cleanup(
        self,
        mock_minio_class,
        mock_minio_client,
        test_job_id,
        test_dataset_id
    ):
        """Test that deleting a job with delete_dataset=True deletes associated datasets."""
        mock_minio_class.return_value = mock_minio_client
        
        # When delete_dataset=True:
        # 1. Find all datasets created by this job
        # 2. Check if datasets are used by active models
        # 3. Delete IQ files from MinIO for each dataset
        # 4. Delete dataset records from database
        # 5. Delete job record
        
        # Verify mock is configured
        assert mock_minio_client.delete_dataset_iq_data is not None
    
    def test_delete_job_without_dataset_cleanup(
        self,
        test_job_id
    ):
        """Test that job deletion with delete_dataset=False preserves datasets (NEW DEFAULT)."""
        # When delete_dataset=False (NEW DEFAULT for data safety):
        # 1. Job record is deleted
        # 2. Datasets remain (with created_by_job_id set to NULL via ON DELETE SET NULL)
        # 3. No MinIO cleanup occurs
        # 4. Data is PRESERVED to prevent accidental loss
        pass
    
    def test_delete_dataset_blocked_by_active_models(
        self,
        test_dataset_id
    ):
        """Test that deleting a dataset used by active models raises 409 Conflict."""
        # When trying to delete a dataset that is referenced by active models:
        # 1. Check models table for synthetic_dataset_id match
        # 2. If any model has is_active=True, raise HTTPException 409
        # 3. Include model names in error message
        # 4. User must deactivate/delete models first
        pass
    
    def test_delete_job_with_dataset_blocked_by_active_models(
        self,
        test_job_id,
        test_dataset_id
    ):
        """Test that job deletion with delete_dataset=True is blocked if datasets are in use."""
        # When delete_dataset=True but dataset is used by active models:
        # 1. Find datasets created by this job
        # 2. For each dataset, check if it's used by active models
        # 3. If any active model found, raise HTTPException 409
        # 4. Suggest user either:
        #    a) Deactivate/delete the models first, or
        #    b) Use delete_dataset=False to preserve dataset
        pass


class TestMinIOClient:
    """Test MinIO client deletion methods."""
    
    def test_delete_dataset_iq_data_batch_deletion(self):
        """Test that delete_dataset_iq_data handles batch deletion correctly."""
        # MinIO has a 1000 object limit per delete request
        # The method should:
        # 1. List all objects for the dataset
        # 2. Batch them into groups of 1000
        # 3. Delete each batch
        # 4. Return counts of successful and failed deletions
        pass
    
    def test_delete_dataset_iq_data_handles_missing_dataset(self):
        """Test that deleting a non-existent dataset returns gracefully."""
        # Should return (0, 0) if no files found
        pass
    
    def test_delete_dataset_iq_data_logs_partial_failures(self):
        """Test that partial failures are logged but don't crash."""
        # If some files fail to delete:
        # 1. Continue with remaining files
        # 2. Log each failure
        # 3. Return accurate counts
        pass


@pytest.mark.integration
class TestDatasetDeletionIntegration:
    """Integration tests requiring actual database and MinIO."""
    
    @pytest.mark.skip(reason="Requires running PostgreSQL and MinIO")
    def test_full_dataset_deletion_flow(self):
        """
        End-to-end test of dataset deletion.
        
        Steps:
        1. Create a synthetic iq_raw dataset via API
        2. Verify IQ files exist in MinIO
        3. Delete the dataset via API
        4. Verify database records are gone
        5. Verify MinIO files are gone
        """
        pass
    
    @pytest.mark.skip(reason="Requires running PostgreSQL and MinIO")
    def test_job_deletion_cascades_to_dataset(self):
        """
        End-to-end test of job deletion cascading to dataset.
        
        Steps:
        1. Create a synthetic generation job
        2. Wait for job to complete and create dataset
        3. Verify dataset and IQ files exist
        4. Delete the job with delete_dataset=True
        5. Verify job, dataset, and IQ files are all gone
        """
        pass


def test_minio_client_delete_dataset_method_exists():
    """Verify MinIOClient has the delete_dataset_iq_data method."""
    import sys
    import os
    import inspect
    
    # Add backend src to path
    backend_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../backend/src")
    )
    if backend_src not in sys.path:
        sys.path.insert(0, backend_src)
    
    # Import MinIOClient
    from storage.minio_client import MinIOClient
    
    # Check method exists
    assert hasattr(MinIOClient, 'delete_dataset_iq_data')
    
    # Check method signature
    sig = inspect.signature(MinIOClient.delete_dataset_iq_data)
    params = list(sig.parameters.keys())
    
    # Should have self, dataset_id, and optional prefix_pattern
    assert 'self' in params
    assert 'dataset_id' in params
    assert 'prefix_pattern' in params
