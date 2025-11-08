"""Tests for T6.1: ONNX Model Loader."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.onnx_loader import ONNXModelLoader


class TestONNXModelLoaderInit:
    """Test ONNX Model Loader initialization."""

    @patch("src.models.onnx_loader.mlflow")
    @patch("src.models.onnx_loader.MlflowClient")
    @patch.object(ONNXModelLoader, "_load_model")
    def test_init_success(self, mock_load, mock_client, mock_mlflow):
        """Test successful initialization."""
        loader = ONNXModelLoader(
            mlflow_uri="http://mlflow:5000", model_name="localization_model", stage="Production"
        )

        assert loader.mlflow_uri == "http://mlflow:5000"
        assert loader.model_name == "localization_model"
        assert loader.stage == "Production"
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_load.assert_called_once()

    @patch("src.models.onnx_loader.mlflow")
    @patch("src.models.onnx_loader.MlflowClient")
    def test_init_with_custom_stage(self, mock_client, mock_mlflow):
        """Test initialization with custom stage."""
        with patch.object(ONNXModelLoader, "_load_model"):
            loader = ONNXModelLoader(
                mlflow_uri="http://localhost:5000", model_name="my_model", stage="Staging"
            )

            assert loader.stage == "Staging"


class TestONNXModelLoaderLoadModel:
    """Test model loading from MLflow."""

    @patch("src.models.onnx_loader.mlflow")
    @patch("src.models.onnx_loader.ort.InferenceSession")
    @patch("src.models.onnx_loader.MlflowClient")
    def test_load_model_success(self, mock_client_class, mock_session, mock_mlflow):
        """Test successful model loading."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.run_id = "run123"
        mock_version.creation_timestamp = "2025-10-22T10:00:00"
        mock_version.status = "READY"

        mock_client.search_model_versions.return_value = [mock_version]
        mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/model"

        mock_onnx_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_mel_spectrogram"
        mock_input.shape = [1, 128, 100]
        mock_output = MagicMock()
        mock_output.name = "position"
        mock_output.shape = [1, 2]

        mock_onnx_session.get_inputs.return_value = [mock_input]
        mock_onnx_session.get_outputs.return_value = [mock_output]
        mock_session.return_value = mock_onnx_session

        # Create loader
        loader = ONNXModelLoader(
            mlflow_uri="http://mlflow:5000", model_name="localization_model", stage="Production"
        )

        # Verify model loaded
        assert loader.session is not None
        assert loader.model_metadata is not None
        assert loader.model_metadata["model_name"] == "localization_model"
        assert loader.model_metadata["version"] == "1"

    @patch("src.models.onnx_loader.mlflow")
    @patch("src.models.onnx_loader.MlflowClient")
    def test_load_model_not_found(self, mock_client_class, mock_mlflow):
        """Test error when model not found in registry."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.search_model_versions.return_value = []

        with pytest.raises(ValueError, match="not found in MLflow registry"):
            ONNXModelLoader(
                mlflow_uri="http://mlflow:5000", model_name="nonexistent_model", stage="Production"
            )

    @patch("src.models.onnx_loader.mlflow")
    @patch("src.models.onnx_loader.MlflowClient")
    def test_load_model_wrong_stage(self, mock_client_class, mock_mlflow):
        """Test error when model in wrong stage."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_version = MagicMock()
        mock_version.current_stage = "Staging"
        mock_client.search_model_versions.return_value = [mock_version]

        with pytest.raises(ValueError, match="not found in stage"):
            ONNXModelLoader(mlflow_uri="http://mlflow:5000", model_name="model", stage="Production")


class TestONNXModelLoaderPredict:
    """Test inference predictions."""

    @pytest.fixture
    def loader(self):
        """Create mock loader for testing."""
        with (
            patch("src.models.onnx_loader.mlflow"),
            patch("src.models.onnx_loader.MlflowClient"),
            patch.object(ONNXModelLoader, "_load_model"),
        ):
            loader = ONNXModelLoader("http://mlflow:5000")

            # Mock session
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            loader.session = mock_session
            loader.model_metadata = {"model_name": "test"}

            return loader

    def test_predict_1d_input(self, loader):
        """Test prediction with 1D input (adds batch dimension)."""
        # Mock ONNX output
        position = np.array([45.123, 7.456])
        uncertainty = np.array([50.0, 40.0, 25.0])
        confidence = np.array([0.95])

        loader.session.run.return_value = [
            np.array([position]),
            np.array([uncertainty]),
            np.array([confidence]),
        ]

        # Run prediction
        features = np.array([1.0, 2.0, 3.0])
        result = loader.predict(features)

        assert result["position"]["latitude"] == pytest.approx(45.123)
        assert result["position"]["longitude"] == pytest.approx(7.456)
        assert result["uncertainty"]["sigma_x"] == pytest.approx(50.0)
        assert result["uncertainty"]["sigma_y"] == pytest.approx(40.0)
        assert result["uncertainty"]["theta"] == pytest.approx(25.0)
        assert result["confidence"] == pytest.approx(0.95)

    def test_predict_2d_input(self, loader):
        """Test prediction with 2D input (batch already included)."""
        position = np.array([45.5, 7.5])
        uncertainty = np.array([30.0, 25.0, 15.0])
        confidence = np.array([0.88])

        loader.session.run.return_value = [
            np.array([position]),
            np.array([uncertainty]),
            np.array([confidence]),
        ]

        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = loader.predict(features)

        assert "position" in result
        assert "uncertainty" in result
        assert "confidence" in result

    def test_predict_no_session(self, loader):
        """Test error when model not loaded."""
        loader.session = None

        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.predict(np.array([1.0, 2.0]))

    def test_predict_converts_to_float32(self, loader):
        """Test that input is converted to float32."""
        loader.session.run.return_value = [
            np.array([[45.0, 7.0]]),
            np.array([[50.0, 40.0, 0.0]]),
            np.array([[0.9]]),
        ]

        # Pass integer array
        features = np.array([1, 2, 3], dtype=np.int32)
        loader.predict(features)

        # Check that run was called
        assert loader.session.run.called
        # Verify input was converted
        call_args = loader.session.run.call_args
        input_data = call_args[0][1]  # Second positional arg (dict values)
        assert next(iter(input_data.values())).dtype == np.float32


class TestONNXModelLoaderMetadata:
    """Test metadata management."""

    @pytest.fixture
    def loader(self):
        """Create mock loader."""
        with (
            patch("src.models.onnx_loader.mlflow"),
            patch("src.models.onnx_loader.MlflowClient"),
            patch.object(ONNXModelLoader, "_load_model"),
        ):
            loader = ONNXModelLoader("http://mlflow:5000")
            loader.model_metadata = {
                "model_name": "test_model",
                "version": "1.0",
                "stage": "Production",
                "run_id": "abc123",
            }
            return loader

    def test_get_metadata(self, loader):
        """Test getting model metadata."""
        metadata = loader.get_metadata()

        assert metadata["model_name"] == "test_model"
        assert metadata["version"] == "1.0"
        assert metadata["stage"] == "Production"

    def test_get_metadata_empty(self, loader):
        """Test getting metadata when none loaded."""
        loader.model_metadata = None
        metadata = loader.get_metadata()

        assert metadata == {}


class TestONNXModelLoaderReload:
    """Test model reloading."""

    @pytest.fixture
    def loader(self):
        """Create mock loader."""
        with (
            patch("src.models.onnx_loader.mlflow"),
            patch("src.models.onnx_loader.MlflowClient"),
            patch.object(ONNXModelLoader, "_load_model"),
        ):
            loader = ONNXModelLoader("http://mlflow:5000")
            return loader

    @patch.object(ONNXModelLoader, "_load_model")
    def test_reload_calls_load_model(self, mock_load, loader):
        """Test that reload calls _load_model."""
        loader.reload()

        # Should be called once on init and once on reload
        assert mock_load.call_count >= 1


class TestONNXModelLoaderStatus:
    """Test readiness status."""

    @pytest.fixture
    def loader(self):
        """Create mock loader."""
        with (
            patch("src.models.onnx_loader.mlflow"),
            patch("src.models.onnx_loader.MlflowClient"),
            patch.object(ONNXModelLoader, "_load_model"),
        ):
            loader = ONNXModelLoader("http://mlflow:5000")
            return loader

    def test_is_ready_true(self, loader):
        """Test is_ready returns True when loaded."""
        loader.session = MagicMock()
        loader.model_metadata = {"model_name": "test"}

        assert loader.is_ready() is True

    def test_is_ready_false_no_session(self, loader):
        """Test is_ready returns False without session."""
        loader.session = None
        loader.model_metadata = {"model_name": "test"}

        assert loader.is_ready() is False

    def test_is_ready_false_no_metadata(self, loader):
        """Test is_ready returns False without metadata."""
        loader.session = MagicMock()
        loader.model_metadata = None

        assert loader.is_ready() is False
