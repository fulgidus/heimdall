"""
Tests for ONNX export and model registration.

Coverage:
- ONNX export from PyTorch model
- ONNX model validation
- Inference accuracy testing
- MinIO upload
- MLflow registration
- End-to-end workflow
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np

from src.onnx_export import ONNXExporter, export_and_register_model


class TestONNXExporter:
    """Tests for ONNXExporter class."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock boto3 S3 client."""
        client = Mock()
        client.put_object = Mock(return_value={'ResponseMetadata': {'HTTPStatusCode': 200}})
        return client
    
    @pytest.fixture
    def mock_mlflow_tracker(self):
        """Mock MLflow tracker."""
        tracker = Mock()
        tracker.register_model = Mock(return_value=1)
        tracker.transition_model_stage = Mock()
        tracker.log_artifact = Mock()
        return tracker
    
    @pytest.fixture
    def exporter(self, mock_s3_client, mock_mlflow_tracker):
        """Create ONNXExporter instance."""
        return ONNXExporter(mock_s3_client, mock_mlflow_tracker)
    
    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model for testing."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3*128*32, 128)
                self.fc_pos = nn.Linear(128, 2)
                self.fc_unc = nn.Linear(128, 2)
            
            def forward(self, x):
                # Flatten input
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                positions = self.fc_pos(x)
                uncertainties = torch.clamp(self.fc_unc(x), min=0.01, max=1.0)
                return positions, uncertainties
        
        return DummyModel()
    
    def test_exporter_initialization(self, exporter):
        """Test ONNXExporter initialization."""
        assert exporter.s3_client is not None
        assert exporter.mlflow_tracker is not None
        assert exporter.device is not None
    
    def test_export_to_onnx(self, exporter, dummy_model):
        """Test ONNX export from PyTorch model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_model.onnx'
            
            # Export
            result = exporter.export_to_onnx(dummy_model, output_path)
            
            # Verify file exists and has content
            assert result == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_export_to_onnx_with_invalid_path(self, exporter, dummy_model):
        """Test ONNX export with invalid path."""
        invalid_path = Path('/invalid/nonexistent/path/model.onnx')
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="ONNX export failed"):
            exporter.export_to_onnx(dummy_model, invalid_path)
    
    def test_validate_onnx_model(self, exporter, dummy_model):
        """Test ONNX model validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_model.onnx'
            exporter.export_to_onnx(dummy_model, output_path)
            
            # Validate
            model_info = exporter.validate_onnx_model(output_path)
            
            # Check structure
            assert 'inputs' in model_info
            assert 'outputs' in model_info
            assert len(model_info['inputs']) > 0
            assert len(model_info['outputs']) > 0
            assert model_info['opset_version'] > 0
    
    def test_validate_invalid_onnx(self, exporter):
        """Test validation of invalid ONNX file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / 'invalid.onnx'
            # Create invalid file
            invalid_path.write_text('this is not a valid ONNX file')
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="ONNX model validation failed"):
                exporter.validate_onnx_model(invalid_path)
    
    def test_test_onnx_inference(self, exporter, dummy_model):
        """Test ONNX inference accuracy testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_model.onnx'
            exporter.export_to_onnx(dummy_model, output_path)
            
            # Test inference
            metrics = exporter.test_onnx_inference(
                output_path,
                dummy_model,
                num_batches=2,
                batch_size=4,
                tolerance=1e-4,
            )
            
            # Check metrics
            assert 'positions_mae' in metrics
            assert 'uncertainties_mae' in metrics
            assert 'inference_time_onnx_ms' in metrics
            assert 'inference_time_pytorch_ms' in metrics
            assert 'speedup' in metrics
            assert metrics['passed'] is True
            assert metrics['speedup'] > 0
    
    def test_upload_to_minio(self, exporter, dummy_model, mock_s3_client):
        """Test upload to MinIO."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_model.onnx'
            exporter.export_to_onnx(dummy_model, output_path)
            
            # Upload
            s3_uri = exporter.upload_to_minio(output_path)
            
            # Verify
            assert s3_uri.startswith('s3://')
            assert 'heimdall-models' in s3_uri
            mock_s3_client.put_object.assert_called_once()
    
    def test_upload_to_minio_with_custom_name(self, exporter, dummy_model, mock_s3_client):
        """Test upload with custom object name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_model.onnx'
            exporter.export_to_onnx(dummy_model, output_path)
            
            custom_name = 'custom/path/model.onnx'
            s3_uri = exporter.upload_to_minio(output_path, object_name=custom_name)
            
            # Verify custom name is used
            assert custom_name in s3_uri
    
    def test_get_model_metadata(self, exporter, dummy_model):
        """Test metadata generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_model.onnx'
            exporter.export_to_onnx(dummy_model, output_path)
            
            metadata = exporter.get_model_metadata(
                output_path,
                dummy_model,
                run_id='test_run_123',
                inference_metrics={'speedup': 2.5},
            )
            
            # Check metadata structure
            assert metadata['model_type'] == 'LocalizationNet'
            assert metadata['onnx_file_size_bytes'] > 0
            assert 'onnx_file_sha256' in metadata
            assert metadata['mlflow_run_id'] == 'test_run_123'
            assert 'pytorch_params' in metadata
    
    def test_register_with_mlflow(self, exporter, dummy_model, mock_mlflow_tracker):
        """Test MLflow model registration."""
        metadata = {'model_type': 'LocalizationNet'}
        s3_uri = 's3://heimdall-models/models/localization/v20240101_120000.onnx'
        
        result = exporter.register_with_mlflow(
            model_name='heimdall-localization',
            s3_uri=s3_uri,
            metadata=metadata,
            stage='Staging',
        )
        
        # Verify
        assert result['model_name'] == 'heimdall-localization'
        assert result['stage'] == 'Staging'
        mock_mlflow_tracker.register_model.assert_called_once()
        mock_mlflow_tracker.transition_model_stage.assert_called_once()


class TestONNXExportWorkflow:
    """Test end-to-end ONNX export workflow."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock S3 client and MLflow tracker."""
        s3_client = Mock()
        s3_client.put_object = Mock(return_value={'ResponseMetadata': {'HTTPStatusCode': 200}})
        
        mlflow_tracker = Mock()
        mlflow_tracker.register_model = Mock(return_value=1)
        mlflow_tracker.transition_model_stage = Mock()
        mlflow_tracker.log_artifact = Mock()
        
        return s3_client, mlflow_tracker
    
    @pytest.fixture
    def dummy_model(self):
        """Create simple model for testing."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3*128*32, 128)
                self.fc_pos = nn.Linear(128, 2)
                self.fc_unc = nn.Linear(128, 2)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                positions = self.fc_pos(x)
                uncertainties = torch.clamp(self.fc_unc(x), min=0.01, max=1.0)
                return positions, uncertainties
            
            def get_params_count(self):
                return {
                    'total': sum(p.numel() for p in self.parameters()),
                    'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                }
        
        return DummyModel()
    
    def test_complete_export_workflow(self, dummy_model, mock_components):
        """Test complete export and registration workflow."""
        s3_client, mlflow_tracker = mock_components
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_and_register_model(
                pytorch_model=dummy_model,
                run_id='test_run_123',
                s3_client=s3_client,
                mlflow_tracker=mlflow_tracker,
                output_dir=Path(tmpdir),
                model_name='test-localization',
            )
            
            # Verify success
            assert result['success'] is True
            assert result['model_name'] == 'test-localization'
            assert result['run_id'] == 'test_run_123'
            assert 's3_uri' in result
            assert 'metadata' in result
            assert 'inference_metrics' in result
            assert 'registration' in result
    
    def test_export_workflow_metrics(self, dummy_model, mock_components):
        """Test workflow includes valid metrics."""
        s3_client, mlflow_tracker = mock_components
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_and_register_model(
                pytorch_model=dummy_model,
                run_id='test_run_456',
                s3_client=s3_client,
                mlflow_tracker=mlflow_tracker,
                output_dir=Path(tmpdir),
            )
            
            # Check inference metrics
            metrics = result['inference_metrics']
            assert 'positions_mae' in metrics
            assert 'uncertainties_mae' in metrics
            assert 'speedup' in metrics
            assert metrics['speedup'] > 0  # ONNX should be faster
    
    def test_export_workflow_handles_errors(self, mock_components):
        """Test workflow error handling."""
        s3_client, mlflow_tracker = mock_components
        
        # Create invalid model (will fail at export)
        invalid_model = Mock(spec=nn.Module)
        invalid_model.side_effect = Exception("Model error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_and_register_model(
                pytorch_model=invalid_model,
                run_id='test_run_error',
                s3_client=s3_client,
                mlflow_tracker=mlflow_tracker,
                output_dir=Path(tmpdir),
            )
            
            # Verify error handling
            assert result['success'] is False
            assert 'error' in result


class TestONNXIntegration:
    """Integration tests for ONNX module."""
    
    def test_onnx_model_info_structure(self):
        """Test ONNX model info has correct structure."""
        s3_client = Mock()
        mlflow_tracker = Mock()
        exporter = ONNXExporter(s3_client, mlflow_tracker)
        
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(12288, 128)
                self.fc_pos = nn.Linear(128, 2)
                self.fc_unc = nn.Linear(128, 2)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                pos = self.fc_pos(x)
                unc = torch.clamp(self.fc_unc(x), min=0.01)
                return pos, unc
        
        model = SimpleModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'model.onnx'
            exporter.export_to_onnx(model, output_path)
            
            model_info = exporter.validate_onnx_model(output_path)
            
            # Verify structure
            assert isinstance(model_info, dict)
            assert all(k in model_info for k in ['inputs', 'outputs', 'opset_version', 'producer_name', 'ir_version'])
            assert isinstance(model_info['inputs'], list)
            assert isinstance(model_info['outputs'], list)
            assert len(model_info['inputs']) > 0
            assert len(model_info['outputs']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
