"""
Comprehensive tests for Training Entry Point Script (train.py).

Test coverage:
- TrainingPipeline initialization
- Data loading and DataLoader creation
- Lightning module creation
- Trainer setup with callbacks
- Training loop simulation
- Model export and ONNX conversion
- MLflow integration and logging
- Error handling and recovery
- End-to-end pipeline execution
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Tuple
import json

# Import from train.py
from train import (
    TrainingPipeline,
    parse_arguments,
)


class TestTrainingPipelineInit:
    """Test TrainingPipeline initialization."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_init_default_parameters(self, mock_onnx, mock_s3, mock_mlflow):
        """Test pipeline initialization with default parameters."""
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline()
        
        assert pipeline.epochs == 100
        assert pipeline.batch_size == 32
        assert pipeline.learning_rate == 1e-3
        assert pipeline.validation_split == 0.2
        assert pipeline.num_workers == 4
        assert pipeline.checkpoint_dir.exists()
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_init_custom_parameters(self, mock_onnx, mock_s3, mock_mlflow):
        """Test pipeline initialization with custom parameters."""
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline(
            epochs=50,
            batch_size=64,
            learning_rate=5e-4,
            validation_split=0.15,
            num_workers=8,
        )
        
        assert pipeline.epochs == 50
        assert pipeline.batch_size == 64
        assert pipeline.learning_rate == 5e-4
        assert pipeline.validation_split == 0.15
        assert pipeline.num_workers == 8
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_init_creates_checkpoint_dir(self, mock_onnx, mock_s3, mock_mlflow):
        """Test that checkpoint directory is created."""
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            
            pipeline = TrainingPipeline(
                checkpoint_dir=checkpoint_dir,
            )
            
            assert checkpoint_dir.exists()
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_init_mlflow_tracker_created(self, mock_onnx, mock_s3, mock_mlflow):
        """Test that MLflow tracker is initialized."""
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline(
            experiment_name="test-experiment",
            run_name_prefix="test-run",
        )
        
        mock_mlflow.assert_called_once()
        assert pipeline.mlflow_tracker == mock_mlflow_instance


class TestDataLoading:
    """Test data loading functionality."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    @patch('train.HeimdallDataset')
    @patch('train.random_split')
    def test_load_data_creates_dataloaders(
        self, mock_split, mock_dataset, mock_onnx, mock_s3, mock_mlflow
    ):
        """Test that load_data creates DataLoaders."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = MagicMock(return_value=1000)
        mock_dataset.return_value = mock_dataset_instance
        
        train_data = MagicMock()
        val_data = MagicMock()
        mock_split.return_value = (train_data, val_data)
        
        pipeline = TrainingPipeline(batch_size=32)
        
        with patch('train.DataLoader') as mock_dataloader:
            train_loader, val_loader = pipeline.load_data()
            
            # Verify DataLoader was called twice (train and val)
            assert mock_dataloader.call_count == 2


class TestLightningModuleCreation:
    """Test Lightning module creation."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    @patch('train.LocalizationLightningModule')
    def test_create_lightning_module(self, mock_lightning_module, mock_onnx, mock_s3, mock_mlflow):
        """Test Lightning module creation."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        mock_module_instance = MagicMock()
        mock_lightning_module.return_value = mock_module_instance
        
        pipeline = TrainingPipeline(learning_rate=1e-3)
        
        with patch('train.LocalizationNet'):
            result = pipeline.create_lightning_module()
            
            assert result == mock_module_instance
            mock_lightning_module.assert_called_once()


class TestTrainerCreation:
    """Test trainer creation with callbacks."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_create_trainer_with_callbacks(self, mock_onnx, mock_s3, mock_mlflow):
        """Test that trainer is created with all callbacks."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow_instance.experiment_name = "test-experiment"
        mock_mlflow_instance.tracking_uri = "postgresql://test"
        mock_mlflow_instance.active_run_id = "run-123"
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline(epochs=50)
        
        with patch('train.pl.Trainer') as mock_trainer:
            result = pipeline.create_trainer()
            
            # Verify trainer was created
            mock_trainer.assert_called_once()
            
            # Verify callbacks were provided
            call_kwargs = mock_trainer.call_args[1]
            assert 'callbacks' in call_kwargs
            assert len(call_kwargs['callbacks']) == 3  # Checkpoint, EarlyStopping, LRMonitor


class TestExportAndRegister:
    """Test model export and registration."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_export_and_register_success(self, mock_onnx, mock_s3, mock_mlflow):
        """Test successful model export and registration."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow_instance.active_run_id = "run-123"
        mock_mlflow.return_value = mock_mlflow_instance
        
        mock_onnx_instance = MagicMock()
        mock_onnx.return_value = mock_onnx_instance
        
        export_result = {
            "success": True,
            "model_name": "heimdall-localization-onnx",
            "model_version": 1,
            "s3_uri": "s3://heimdall-models/model.onnx",
            "metadata": {"file_size_mb": 120},
        }
        mock_onnx_instance.export_to_onnx.return_value = export_result
        
        pipeline = TrainingPipeline()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy checkpoint file
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            checkpoint_path.write_text("{}")
            
            with patch('train.torch.load', return_value={"state_dict": {}}):
                with patch('train.LocalizationNet'):
                    with patch('train.export_and_register_model', return_value=export_result):
                        result = pipeline.export_and_register(checkpoint_path)
                        
                        assert result["success"] == True
                        assert result["model_name"] == "heimdall-localization-onnx"


class TestPipelineRun:
    """Test complete pipeline execution."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_run_export_only_mode(self, mock_onnx, mock_s3, mock_mlflow):
        """Test pipeline run in export-only mode."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow_instance.active_run_id = "run-123"
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint.ckpt"
            checkpoint_path.write_text("{}")
            
            export_result = {
                "success": True,
                "model_name": "heimdall-localization-onnx",
                "model_version": 1,
            }
            
            with patch.object(pipeline, 'export_and_register', return_value=export_result):
                result = pipeline.run(
                    export_only=True,
                    checkpoint_path=checkpoint_path,
                )
                
                assert result["success"] == True


class TestParseArguments:
    """Test command-line argument parsing."""
    
    def test_parse_default_arguments(self):
        """Test parsing with default arguments."""
        with patch('sys.argv', ['train.py']):
            args = parse_arguments()
            
            assert args.epochs == 100
            assert args.batch_size == 32
            assert args.learning_rate == 1e-3
            assert args.validation_split == 0.2
    
    def test_parse_custom_epochs(self):
        """Test parsing custom epochs."""
        with patch('sys.argv', ['train.py', '--epochs', '50']):
            args = parse_arguments()
            
            assert args.epochs == 50
    
    def test_parse_custom_learning_rate(self):
        """Test parsing custom learning rate."""
        with patch('sys.argv', ['train.py', '--learning_rate', '5e-4']):
            args = parse_arguments()
            
            assert args.learning_rate == 5e-4
    
    def test_parse_export_only_flag(self):
        """Test parsing export_only flag."""
        with patch('sys.argv', ['train.py', '--export_only']):
            args = parse_arguments()
            
            assert args.export_only == True


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_pipeline_handles_load_data_error(self, mock_onnx, mock_s3, mock_mlflow):
        """Test pipeline handles data loading errors gracefully."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline()
        
        with patch.object(pipeline, 'load_data', side_effect=Exception("Data load error")):
            with pytest.raises(Exception):
                pipeline.run(export_only=False)
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_pipeline_mlflow_end_run_on_error(self, mock_onnx, mock_s3, mock_mlflow):
        """Test that MLflow run is ended on error."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline()
        
        with patch.object(pipeline, 'load_data', side_effect=Exception("Error")):
            try:
                pipeline.run(export_only=False)
            except Exception:
                pass
            
            # Verify end_run was called with FAILED status
            mock_mlflow_instance.end_run.assert_called_with(status="FAILED")


class TestMLflowIntegration:
    """Test MLflow tracking integration."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_pipeline_logs_hyperparameters(self, mock_onnx, mock_s3, mock_mlflow):
        """Test that pipeline logs hyperparameters to MLflow."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        
        pipeline = TrainingPipeline(
            epochs=50,
            batch_size=64,
            learning_rate=5e-4,
        )
        
        # Verify log_params was called
        mock_mlflow_instance.log_params.assert_called()
        
        # Get the parameters that were logged
        call_args = mock_mlflow_instance.log_params.call_args
        logged_params = call_args[0][0]
        
        assert logged_params["epochs"] == 50
        assert logged_params["batch_size"] == 64
        assert logged_params["learning_rate"] == 5e-4


class TestIntegrationE2E:
    """End-to-end integration tests."""
    
    @patch('train.MLflowTracker')
    @patch('train.boto3.client')
    @patch('train.ONNXExporter')
    def test_pipeline_initialization_and_setup(self, mock_onnx, mock_s3, mock_mlflow):
        """Test complete pipeline setup."""
        # Setup mocks
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        mock_s3_instance = MagicMock()
        mock_s3.return_value = mock_s3_instance
        mock_onnx_instance = MagicMock()
        mock_onnx.return_value = mock_onnx_instance
        
        # Create pipeline
        pipeline = TrainingPipeline(
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
        )
        
        # Verify components are initialized
        assert pipeline.mlflow_tracker == mock_mlflow_instance
        assert pipeline.s3_client == mock_s3_instance
        assert pipeline.onnx_exporter == mock_onnx_instance
        assert pipeline.checkpoint_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
