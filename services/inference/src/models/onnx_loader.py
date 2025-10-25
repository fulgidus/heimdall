"""ONNX Model Loader for Phase 6 Inference Service."""
import logging
from typing import Dict, Optional
import numpy as np
import onnxruntime as ort
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Load and manage ONNX model from MLflow registry."""
    
    def __init__(
        self,
        mlflow_uri: str,
        model_name: str = "localization_model",
        stage: str = "Production",
    ):
        """
        Initialize ONNX Model Loader.
        
        Args:
            mlflow_uri: MLflow tracking URI (e.g., "http://mlflow:5000")
            model_name: Registered model name in MLflow
            stage: Model stage ("Production", "Staging", "None")
        
        Raises:
            ValueError: If model not found in registry or stage
            RuntimeError: If ONNX session initialization fails
        """
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.stage = stage
        self.session = None
        self.model_metadata = None
        self.reload_count = 0
        
        # Initialize MLflow client
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load ONNX model from MLflow registry.
        
        Raises:
            ValueError: If model not found or stage invalid
            RuntimeError: If ONNX session init fails
        """
        try:
            logger.info(
                f"Loading model '{self.model_name}' from stage '{self.stage}' "
                f"(MLflow URI: {self.mlflow_uri})"
            )
            
            # Get all model versions
            model_versions = self.client.search_model_versions(
                f"name='{self.model_name}'"
            )
            
            if not model_versions:
                raise ValueError(
                    f"Model '{self.model_name}' not found in MLflow registry"
                )
            
            # Find version in requested stage
            version = None
            for mv in model_versions:
                if mv.current_stage == self.stage:
                    version = mv
                    break
            
            if version is None:
                available_stages = {mv.current_stage for mv in model_versions}
                raise ValueError(
                    f"Model '{self.model_name}' not found in stage '{self.stage}'. "
                    f"Available stages: {available_stages}"
                )
            
            # Download model artifact from MLflow
            model_uri = f"models:/{self.model_name}/{self.stage}"
            logger.info(f"Downloading model from {model_uri}")
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Initialize ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.log_severity_level = 3  # Suppress ONNX warnings
            
            model_path = f"{local_path}/model.onnx"
            logger.info(f"Creating ONNX Runtime session from {model_path}")
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            
            # Get model input/output info
            input_info = self.session.get_inputs()
            output_info = self.session.get_outputs()
            
            logger.info(
                f"ONNX Model loaded successfully. "
                f"Inputs: {len(input_info)}, Outputs: {len(output_info)}"
            )
            
            # Store metadata
            self.model_metadata = {
                "model_name": self.model_name,
                "version": version.version,
                "stage": self.stage,
                "run_id": version.run_id,
                "created_at": str(version.creation_timestamp),
                "status": version.status,
                "input_name": input_info[0].name,
                "input_shape": input_info[0].shape,
                "output_names": [out.name for out in output_info],
                "output_shapes": [out.shape for out in output_info],
                "reload_count": self.reload_count,
            }
            
            self.reload_count += 1
            logger.info(f"Model metadata: {self.model_metadata}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Run ONNX inference.
        
        Args:
            features: Input features (numpy array).
                     Shape depends on model (typically [batch, feature_dim] or [feature_dim])
        
        Returns:
            Dict with keys:
                - position: {latitude: float, longitude: float}
                - uncertainty: {sigma_x: float, sigma_y: float, theta: float}
                - confidence: float (0-1)
        
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input validation fails
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Validate and reshape input
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            if features.ndim == 1:
                features = features[np.newaxis, ...]  # Add batch dimension
            
            features = features.astype(np.float32)
            
            # Get model input/output names
            input_name = self.session.get_inputs()[0].name
            output_names = [out.name for out in self.session.get_outputs()]
            
            logger.debug(f"Running inference with input shape {features.shape}")
            
            # Run inference
            outputs = self.session.run(
                output_names,
                {input_name: features},
            )
            
            logger.debug(f"Inference complete. Outputs: {len(outputs)}")
            
            # Parse outputs (assuming Phase 5 model outputs):
            # Output 0: position [batch, 2] -> (lat, lon)
            # Output 1: uncertainty [batch, 3] -> (sigma_x, sigma_y, theta)
            # Output 2: confidence [batch, 1] -> probability
            
            position = outputs[0][0]  # First batch, position
            uncertainty = outputs[1][0] if len(outputs) > 1 else np.array([0.0, 0.0, 0.0])
            confidence = outputs[2][0] if len(outputs) > 2 else np.array([1.0])
            
            # Ensure arrays have enough elements
            if len(uncertainty) < 3:
                uncertainty = np.pad(uncertainty, (0, 3 - len(uncertainty)), 'constant')
            
            result = {
                "position": {
                    "latitude": float(position[0]),
                    "longitude": float(position[1]),
                },
                "uncertainty": {
                    "sigma_x": float(uncertainty[0]),
                    "sigma_y": float(uncertainty[1]),
                    "theta": float(uncertainty[2]),
                },
                "confidence": float(confidence[0]),
            }
            
            logger.debug(f"Prediction result: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise
    
    def get_metadata(self) -> Dict:
        """
        Return model metadata.
        
        Returns:
            Dict with model information from MLflow registry
        """
        if self.model_metadata is None:
            logger.warning("Model metadata not available")
            return {}
        
        return self.model_metadata.copy()
    
    def reload(self) -> None:
        """
        Reload model from MLflow registry (for graceful updates).
        
        Useful for updating model without restarting service.
        """
        logger.info(f"Reloading model '{self.model_name}' from MLflow...")
        try:
            self._load_model()
            logger.info("Model reloaded successfully")
        except Exception as e:
            logger.error(f"Model reload failed: {e}", exc_info=True)
            raise
    
    def is_ready(self) -> bool:
        """
        Check if model is ready for inference.
        
        Returns:
            True if model loaded and session active, False otherwise
        """
        return self.session is not None and self.model_metadata is not None
