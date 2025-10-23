"""
Tests for MLflow integration module.

Tests:
- MLflowTracker initialization
- Experiment creation and management
- Run tracking (start, end, params, metrics)
- Artifact logging
- Model registration
- Run queries
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from datetime import datetime

from src.mlflow_setup import MLflowTracker, initialize_mlflow


class TestMLflowTracker:
    """Test suite for MLflowTracker class."""
    
    @pytest.fixture
    def mlflow_tracker(self):
        """Create MLflowTracker instance for testing."""
        
        # Use patch as decorators/context managers, but keep patches active
        # by storing them and only stopping them after test completion
        patcher1 = patch('src.mlflow_setup.mlflow.set_tracking_uri')
        patcher2 = patch('src.mlflow_setup.mlflow.set_experiment')
        patcher3 = patch('src.mlflow_setup.MlflowClient')
        
        mock_set_uri = patcher1.start()
        mock_set_exp = patcher2.start()
        MockClient = patcher3.start()
        
        # Mock experiment ID
        mock_set_exp.return_value = "test-experiment-id"
        
        # Create a mock client instance that will persist
        mock_client_instance = Mock()
        MockClient.return_value = mock_client_instance
        
        tracker = MLflowTracker(
            tracking_uri="postgresql://test:test@localhost:5432/mlflow",
            artifact_uri="s3://test-mlflow",
            backend_store_uri="postgresql://test:test@localhost:5432/mlflow",
            registry_uri="postgresql://test:test@localhost:5432/mlflow",
            s3_endpoint_url="http://minio:9000",
            s3_access_key_id="testkey",
            s3_secret_access_key="testsecret",
            experiment_name="test-experiment",
        )
        
        # Ensure the client attribute is set and persists
        tracker.client = mock_client_instance
        
        yield tracker
        
        # Cleanup patches after test
        patcher1.stop()
        patcher2.stop()
        patcher3.stop()
    
    def test_initialization(self, mlflow_tracker):
        """Test MLflowTracker initialization."""
        
        assert mlflow_tracker.tracking_uri == "postgresql://test:test@localhost:5432/mlflow"
        assert mlflow_tracker.artifact_uri == "s3://test-mlflow"
        assert mlflow_tracker.experiment_name == "test-experiment"
        assert mlflow_tracker.experiment_id is not None
    
    @patch('mlflow.start_run')
    def test_start_run(self, mock_start_run, mlflow_tracker):
        """Test starting an MLflow run."""
        
        # Mock run
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id-123"
        mock_start_run.return_value = mock_run
        
        run_id = mlflow_tracker.start_run(
            run_name="test-run",
            tags={'model': 'LocalizationNet'},
        )
        
        assert run_id == "test-run-id-123"
        mock_start_run.assert_called_once()
    
    @patch('mlflow.end_run')
    def test_end_run(self, mock_end_run, mlflow_tracker):
        """Test ending an MLflow run."""
        
        mlflow_tracker.end_run(status="FINISHED")
        
        mock_end_run.assert_called_once()
    
    @patch('mlflow.log_param')
    def test_log_params(self, mock_log_param, mlflow_tracker):
        """Test logging parameters."""
        
        params = {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 100,
            'backbone': 'ConvNeXt-Large',
        }
        
        mlflow_tracker.log_params(params)
        
        # Verify each parameter was logged
        assert mock_log_param.call_count == len(params)
    
    @patch('mlflow.log_metric')
    def test_log_metrics(self, mock_log_metric, mlflow_tracker):
        """Test logging metrics."""
        
        metrics = {
            'train_loss': 0.523,
            'val_loss': 0.487,
            'train_mae': 12.3,
        }
        
        mlflow_tracker.log_metrics(metrics, step=1)
        
        # Verify each metric was logged
        assert mock_log_metric.call_count == len(metrics)
    
    @patch('mlflow.log_artifact')
    def test_log_artifact(self, mock_log_artifact, mlflow_tracker):
        """Test logging a single artifact."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test artifact content")
            temp_file = f.name
        
        try:
            mlflow_tracker.log_artifact(temp_file, artifact_path="artifacts")
            
            mock_log_artifact.assert_called_once()
        finally:
            Path(temp_file).unlink()
    
    @patch('mlflow.log_artifacts')
    def test_log_artifacts_dir(self, mock_log_artifacts, mlflow_tracker):
        """Test logging an entire directory."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            Path(temp_dir, "file1.txt").write_text("content1")
            Path(temp_dir, "file2.txt").write_text("content2")
            
            mlflow_tracker.log_artifacts_dir(temp_dir, artifact_path="artifacts")
            
            mock_log_artifacts.assert_called_once()
    
    @patch('mlflow.register_model')
    def test_register_model(self, mock_register_model, mlflow_tracker):
        """Test registering a model."""
        
        mock_version = Mock()
        mock_version.version = "1"
        mock_register_model.return_value = mock_version
        
        version = mlflow_tracker.register_model(
            model_name="heimdall-localization-v1",
            model_uri="runs://abc123/models/model",
            description="Test model",
            tags={'stage': 'production'},
        )
        
        assert version == "1"
        mock_register_model.assert_called_once()
    
    def test_transition_model_stage(self, mlflow_tracker):
        """Test transitioning model stage."""
        
        # The fixture already provides a mock client
        mlflow_tracker.transition_model_stage(
            model_name="heimdall-localization-v1",
            version="1",
            stage="Production",
        )
        
        mlflow_tracker.client.transition_model_version_stage.assert_called_once()
    
    def test_get_run_info(self, mlflow_tracker):
        """Test getting run information."""
        
        # Mock run
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id"
        mock_run.info.experiment_id = "test-exp-id"
        mock_run.info.status = "FINISHED"
        mock_run.data.params = {'lr': '1e-3'}
        mock_run.data.metrics = {'loss': 0.5}
        mock_run.data.tags = {'model': 'test'}
        
        mlflow_tracker.client.get_run.return_value = mock_run
        
        run_info = mlflow_tracker.get_run_info("test-run-id")
        
        assert run_info['run_id'] == "test-run-id"
        assert run_info['status'] == "FINISHED"
    
    @patch('mlflow.search_runs')
    def test_get_best_run(self, mock_search_runs, mlflow_tracker):
        """Test getting the best run."""
        
        # Mock dataframe with run data
        mock_df = Mock()
        mock_df.empty = False
        mock_df.iloc.__getitem__.return_value = {
            'run_id': 'best-run-id',
            'metrics.val/loss': 0.42,
        }
        
        mock_search_runs.return_value = mock_df
        
        best_run = mlflow_tracker.get_best_run(metric="val/loss", compare_fn=min)
        
        assert best_run is not None
        assert best_run['run_id'] == 'best-run-id'
    
    @patch('mlflow.search_runs')
    def test_get_best_run_no_runs(self, mock_search_runs, mlflow_tracker):
        """Test getting best run when no runs exist."""
        
        # Mock empty dataframe
        mock_df = Mock()
        mock_df.empty = True
        
        mock_search_runs.return_value = mock_df
        
        best_run = mlflow_tracker.get_best_run()
        
        assert best_run is None


class TestMLflowIntegration:
    """Integration tests for MLflow setup."""
    
    def test_log_params_with_complex_types(self):
        """Test logging complex types (lists, dicts)."""
        
        params = {
            'learning_rates': [1e-3, 1e-4, 1e-5],
            'model_config': {'backbone': 'ConvNeXt', 'size': 'large'},
        }
        
        # Should not raise an error
        with patch('mlflow.log_param'):
            tracker = MLflowTracker(
                tracking_uri="postgresql://test:test@localhost/mlflow",
                artifact_uri="s3://test",
                backend_store_uri="postgresql://test:test@localhost/mlflow",
                registry_uri="postgresql://test:test@localhost/mlflow",
                s3_endpoint_url="http://minio:9000",
                s3_access_key_id="key",
                s3_secret_access_key="secret",
            )
            
            # This should convert lists/dicts to JSON strings
            tracker.log_params(params)
    
    def test_error_handling_in_logging(self):
        """Test error handling when logging fails."""
        
        with patch('mlflow.log_metric') as mock_log:
            mock_log.side_effect = Exception("MLflow error")
            
            tracker = MLflowTracker(
                tracking_uri="postgresql://test:test@localhost/mlflow",
                artifact_uri="s3://test",
                backend_store_uri="postgresql://test:test@localhost/mlflow",
                registry_uri="postgresql://test:test@localhost/mlflow",
                s3_endpoint_url="http://minio:9000",
                s3_access_key_id="key",
                s3_secret_access_key="secret",
            )
            
            # Should not raise, should log warning instead
            tracker.log_metrics({'test': 1.0})


class TestInitializeMLflow:
    """Test initialize_mlflow helper function."""
    
    def test_initialize_from_settings(self):
        """Test initialization from settings object."""
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.mlflow_tracking_uri = "postgresql://test:test@localhost/mlflow"
        mock_settings.mlflow_artifact_uri = "s3://test-mlflow"
        mock_settings.mlflow_backend_store_uri = "postgresql://test:test@localhost/mlflow"
        mock_settings.mlflow_registry_uri = "postgresql://test:test@localhost/mlflow"
        mock_settings.mlflow_s3_endpoint_url = "http://minio:9000"
        mock_settings.mlflow_s3_access_key_id = "key"
        mock_settings.mlflow_s3_secret_access_key = "secret"
        mock_settings.mlflow_experiment_name = "test-exp"
        
        with patch('src.mlflow_setup.MLflowTracker'):
            tracker = initialize_mlflow(mock_settings)
            
            assert tracker is not None


class TestMLflowTrackingWorkflow:
    """End-to-end workflow tests."""
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.end_run')
    def test_complete_training_workflow(self, mock_end, mock_metric, mock_param, mock_start):
        """Test complete training workflow with MLflow."""
        
        # Setup mocks
        mock_run = Mock()
        mock_run.info.run_id = "training-run-123"
        mock_start.return_value = mock_run
        
        with patch('src.mlflow_setup.MLflowTracker.client'):
            tracker = MLflowTracker(
                tracking_uri="postgresql://test:test@localhost/mlflow",
                artifact_uri="s3://test-mlflow",
                backend_store_uri="postgresql://test:test@localhost/mlflow",
                registry_uri="postgresql://test:test@localhost/mlflow",
                s3_endpoint_url="http://minio:9000",
                s3_access_key_id="key",
                s3_secret_access_key="secret",
                experiment_name="training-exp",
            )
            
            # Simulate training
            run_id = tracker.start_run("training-run")
            
            # Log parameters
            tracker.log_params({
                'lr': 1e-3,
                'batch_size': 32,
                'epochs': 100,
            })
            
            # Log metrics over epochs
            for epoch in range(3):
                tracker.log_metrics({
                    'train_loss': 0.5 - epoch * 0.1,
                    'val_loss': 0.48 - epoch * 0.09,
                }, step=epoch)
            
            # End run
            tracker.end_run("FINISHED")
            
            # Verify calls
            assert mock_start.called
            assert mock_param.called
            assert mock_metric.called
            assert mock_end.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
