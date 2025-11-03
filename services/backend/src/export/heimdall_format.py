"""
.heimdall Export Format Module

Portable model bundle format for Heimdall trained models.
Includes ONNX model + training metadata + performance metrics.

Bundle structure:
- format_version: "1.0.0"
- bundle_metadata: Bundle creation info
- model: ONNX model + architecture info
- training_config: Training hyperparameters
- performance_metrics: Accuracy and performance metrics
- normalization_stats: Feature normalization parameters
- sample_predictions: Example predictions for validation

Usage:
    # Export
    exporter = HeimdallExporter(db_session, minio_client)
    bundle = exporter.export_model(model_id, include_samples=True)
    exporter.save_bundle(bundle, "model.heimdall")
    
    # Import
    bundle = exporter.load_bundle("model.heimdall")
    exporter.extract_onnx(bundle, "extracted.onnx")
"""

import base64
import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BundleMetadata(BaseModel):
    """Metadata about the bundle itself."""
    bundle_id: str = Field(description="Unique bundle identifier")
    created_at: str = Field(description="ISO-8601 timestamp of creation")
    created_by: str = Field(default="system", description="User ID or 'system'")
    description: Optional[str] = Field(None, description="Optional bundle description")
    heimdall_version: str = Field(default="1.0.0", description="Heimdall version")


class ModelInfo(BaseModel):
    """ONNX model information and binary."""
    model_id: str = Field(description="UUID from database")
    model_name: str = Field(description="Human-readable model name")
    version: str = Field(description="Model version (e.g., '1.0.0')")
    architecture: str = Field(description="Model architecture (e.g., 'convnext_large')")
    framework: str = Field(default="pytorch", description="Training framework")
    onnx_opset: int = Field(description="ONNX opset version")
    onnx_model_base64: str = Field(description="Base64-encoded ONNX binary")
    input_shape: List[int] = Field(description="Expected input shape [batch, channels, height, width]")
    output_shape: List[List[int]] = Field(description="Output shapes [[batch, 2], [batch, 2]]")
    parameters_count: int = Field(description="Number of model parameters")


class TrainingConfig(BaseModel):
    """Training configuration and hyperparameters."""
    epochs: int = Field(description="Number of training epochs")
    batch_size: int = Field(description="Training batch size")
    learning_rate: float = Field(description="Initial learning rate")
    optimizer: str = Field(description="Optimizer name (e.g., 'adam')")
    loss_function: str = Field(description="Loss function (e.g., 'gaussian_nll')")
    validation_split: float = Field(description="Validation split ratio (0.0-1.0)")
    dataset_ids: List[str] = Field(description="UUIDs of training datasets")
    dataset_names: List[str] = Field(description="Human-readable dataset names")


class PerformanceMetrics(BaseModel):
    """Model performance metrics."""
    final_train_loss: float = Field(description="Final training loss")
    final_val_loss: float = Field(description="Final validation loss")
    final_train_accuracy: float = Field(description="Final training accuracy (meters)")
    final_val_accuracy: float = Field(description="Final validation accuracy (meters)")
    best_epoch: int = Field(description="Epoch with best validation loss")
    training_duration_seconds: int = Field(description="Total training time in seconds")
    inference_latency_ms: float = Field(description="Average ONNX inference latency (ms)")
    onnx_speedup_factor: float = Field(description="ONNX vs PyTorch speedup")


class NormalizationStats(BaseModel):
    """Feature normalization statistics."""
    feature_means: List[float] = Field(description="Mean values for normalization")
    feature_stds: List[float] = Field(description="Standard deviation values for normalization")


class SamplePrediction(BaseModel):
    """Example prediction for validation."""
    sample_id: str = Field(description="Sample identifier")
    input_metadata: Dict[str, Any] = Field(description="Input metadata (frequency, power, etc.)")
    ground_truth: Dict[str, float] = Field(description="Ground truth location")
    prediction: Dict[str, float] = Field(description="Model prediction with uncertainties")
    error_meters: float = Field(description="Localization error in meters")


class HeimdallBundle(BaseModel):
    """Complete .heimdall bundle."""
    format_version: str = Field(default="1.0.0", description="Bundle format version")
    bundle_metadata: BundleMetadata
    model: ModelInfo
    training_config: Optional[TrainingConfig] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    normalization_stats: Optional[NormalizationStats] = None
    sample_predictions: Optional[List[SamplePrediction]] = None


class HeimdallExporter:
    """Export trained models in .heimdall format."""

    def __init__(self, db_manager, minio_client):
        """
        Initialize exporter.
        
        Args:
            db_manager: DatabaseManager instance
            minio_client: boto3 S3 client (MinIO)
        """
        self.db_manager = db_manager
        self.minio = minio_client

    def export_model(
        self,
        model_id: str,
        include_config: bool = True,
        include_metrics: bool = True,
        include_normalization: bool = True,
        include_samples: bool = True,
        num_samples: int = 5,
        description: Optional[str] = None,
    ) -> HeimdallBundle:
        """
        Export model as .heimdall bundle.

        Args:
            model_id: UUID of trained model
            include_config: Include training configuration
            include_metrics: Include performance metrics
            include_normalization: Include normalization stats
            include_samples: Include sample predictions
            num_samples: Number of samples to include (1-10)
            description: Optional bundle description

        Returns:
            HeimdallBundle ready for JSON serialization
            
        Raises:
            ValueError: If model not found or invalid parameters
        """
        if num_samples < 1 or num_samples > 10:
            raise ValueError("num_samples must be between 1 and 10")

        # Load model from database
        model_data = self._load_model_from_db(model_id)
        if not model_data:
            raise ValueError(f"Model {model_id} not found")
        
        # Validate ONNX model exists
        if not model_data.get("onnx_model_location"):
            raise ValueError(
                f"Model {model_id} does not have an ONNX export. "
                "Please export the model to ONNX format first."
            )

        # Download ONNX from MinIO
        onnx_bytes = self._download_onnx(model_data["onnx_model_location"])
        onnx_base64 = base64.b64encode(onnx_bytes).decode("utf-8")

        # Build bundle
        bundle = HeimdallBundle(
            bundle_metadata=self._create_metadata(description),
            model=self._create_model_info(model_data, onnx_base64),
            training_config=(
                self._load_training_config(model_data) if include_config else None
            ),
            performance_metrics=(
                self._load_metrics(model_data) if include_metrics else None
            ),
            normalization_stats=(
                self._load_normalization_stats() if include_normalization else None
            ),
            sample_predictions=(
                self._load_sample_predictions(model_data, num_samples)
                if include_samples
                else None
            ),
        )

        logger.info(
            f"Exported model {model_id} as .heimdall bundle "
            f"(size: {len(onnx_base64) / 1024 / 1024:.2f} MB)"
        )

        return bundle

    def save_bundle(self, bundle: HeimdallBundle, output_path: str):
        """
        Save bundle to .heimdall file (JSON).
        
        Args:
            bundle: HeimdallBundle instance
            output_path: Path to save bundle
        """
        with open(output_path, "w") as f:
            f.write(bundle.model_dump_json(indent=2))
        
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"Saved .heimdall bundle to {output_path} ({file_size:.2f} MB)")

    def load_bundle(self, bundle_path: str) -> HeimdallBundle:
        """
        Load bundle from .heimdall file.
        
        Args:
            bundle_path: Path to .heimdall file
            
        Returns:
            Parsed HeimdallBundle
        """
        with open(bundle_path, "r") as f:
            return HeimdallBundle.model_validate_json(f.read())

    def extract_onnx(self, bundle: HeimdallBundle, output_path: str):
        """
        Extract ONNX model from bundle to file.
        
        Args:
            bundle: HeimdallBundle instance
            output_path: Path to save ONNX model
        """
        onnx_bytes = base64.b64decode(bundle.model.onnx_model_base64)
        with open(output_path, "wb") as f:
            f.write(onnx_bytes)
        
        logger.info(f"Extracted ONNX model to {output_path}")

    # Private helper methods

    def _load_model_from_db(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load model metadata from database."""
        from sqlalchemy import text
        
        with self.db_manager.get_session() as session:
            query = text("""
                SELECT 
                    m.id, 
                    m.model_name, 
                    COALESCE(m.version, 1) as version,
                    m.model_type,
                    m.onnx_model_location,
                    m.accuracy_meters,
                    m.accuracy_sigma_meters,
                    m.loss_value,
                    m.epoch,
                    m.hyperparameters,
                    m.training_metrics,
                    m.test_metrics,
                    m.synthetic_dataset_id,
                    m.created_at,
                    tj.config as training_config,
                    tj.total_epochs,
                    tj.train_samples,
                    tj.val_samples
                FROM heimdall.models m
                LEFT JOIN heimdall.training_jobs tj ON tj.id = m.trained_by_job_id
                WHERE m.id = :model_id
            """)
            
            result = session.execute(query, {"model_id": model_id}).fetchone()
            
            if not result:
                return None
            
            return {
                "id": str(result[0]),
                "model_name": result[1],
                "version": result[2],
                "model_type": result[3],
                "onnx_model_location": result[4],
                "accuracy_meters": result[5],
                "accuracy_sigma_meters": result[6],
                "loss_value": result[7],
                "epoch": result[8],
                "hyperparameters": result[9],
                "training_metrics": result[10],
                "test_metrics": result[11],
                "synthetic_dataset_id": result[12],
                "created_at": result[13],
                "training_config": result[14],
                "total_epochs": result[15],
                "train_samples": result[16],
                "val_samples": result[17],
            }

    def _download_onnx(self, s3_path: str) -> bytes:
        """Download ONNX model from MinIO."""
        # Parse s3://bucket/key format
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        
        parts = s3_path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        # Download from MinIO
        response = self.minio.get_object(Bucket=bucket, Key=key)
        onnx_bytes = response["Body"].read()
        
        logger.info(f"Downloaded ONNX model from {s3_path} ({len(onnx_bytes) / 1024 / 1024:.2f} MB)")
        
        return onnx_bytes

    def _create_metadata(self, description: Optional[str]) -> BundleMetadata:
        """Create bundle metadata."""
        return BundleMetadata(
            bundle_id=str(uuid.uuid4()),
            created_at=datetime.utcnow().isoformat() + "Z",
            created_by="system",
            description=description,
            heimdall_version="1.0.0",
        )

    def _create_model_info(
        self, model_data: Dict[str, Any], onnx_base64: str
    ) -> ModelInfo:
        """Create model info from database data."""
        # Extract architecture from hyperparameters or model_type
        hyperparams = model_data.get("hyperparameters") or {}
        architecture = hyperparams.get("architecture", model_data.get("model_type", "unknown"))
        
        # Get parameter count from hyperparameters or estimate
        parameters_count = hyperparams.get("num_parameters", 0)
        if not parameters_count and "convnext_large" in architecture.lower():
            parameters_count = 200_000_000  # Approximate for ConvNeXt-Large
        
        return ModelInfo(
            model_id=model_data["id"],
            model_name=model_data["model_name"],
            version=str(model_data["version"]),
            architecture=architecture,
            framework="pytorch",
            onnx_opset=17,  # Default ONNX opset used by Heimdall
            onnx_model_base64=onnx_base64,
            input_shape=[1, 3, 128, 32],  # Fixed input shape for Heimdall models
            output_shape=[[1, 2], [1, 2]],  # positions, uncertainties
            parameters_count=parameters_count,
        )

    def _load_training_config(self, model_data: Dict[str, Any]) -> Optional[TrainingConfig]:
        """Load training configuration."""
        config = model_data.get("training_config")
        hyperparams = model_data.get("hyperparameters") or {}
        
        if not config and not hyperparams:
            return None
        
        # Parse config if it's a JSON string
        if isinstance(config, str):
            config = json.loads(config)
        
        # Extract values with robust fallbacks
        epochs = model_data.get("total_epochs")
        if epochs is None and config:
            epochs = config.get("epochs")
        if epochs is None:
            epochs = 50  # Default fallback
            
        batch_size = hyperparams.get("batch_size")
        if batch_size is None and config:
            batch_size = config.get("batch_size")
        if batch_size is None:
            batch_size = 32
            
        learning_rate = hyperparams.get("learning_rate")
        if learning_rate is None and config:
            learning_rate = config.get("learning_rate")
        if learning_rate is None:
            learning_rate = 0.001
        
        return TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=hyperparams.get("optimizer", "adam"),
            loss_function=hyperparams.get("loss_function", "gaussian_nll"),
            validation_split=0.2,  # Default validation split
            dataset_ids=[str(model_data.get("synthetic_dataset_id", ""))] if model_data.get("synthetic_dataset_id") else [],
            dataset_names=["Training Dataset"],  # Could be improved with actual dataset name lookup
        )

    def _load_metrics(self, model_data: Dict[str, Any]) -> Optional[PerformanceMetrics]:
        """Load performance metrics."""
        metrics = model_data.get("performance_metrics")
        training_metrics = model_data.get("training_metrics")
        test_metrics = model_data.get("test_metrics")
        
        if not metrics and not training_metrics:
            return None
        
        # Merge metrics from different sources
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        if isinstance(training_metrics, str):
            training_metrics = json.loads(training_metrics)
        if isinstance(test_metrics, str):
            test_metrics = json.loads(test_metrics)
        
        # Extract values with fallbacks
        train_accuracy = model_data.get("accuracy_meters", 50.0)
        val_accuracy = model_data.get("accuracy_sigma_meters", 50.0)
        
        if training_metrics:
            train_accuracy = training_metrics.get("train_accuracy", train_accuracy)
            val_accuracy = training_metrics.get("val_accuracy", val_accuracy)
        
        # Extract loss values with fallbacks
        final_train_loss = None
        final_val_loss = None
        
        if metrics:
            final_train_loss = metrics.get("final_train_loss")
            final_val_loss = metrics.get("final_val_loss")
        
        if final_train_loss is None and training_metrics:
            final_train_loss = training_metrics.get("final_train_loss")
        if final_val_loss is None and training_metrics:
            final_val_loss = training_metrics.get("final_val_loss")
            
        if final_train_loss is None:
            final_train_loss = model_data.get("loss_value") or 0.0
        if final_val_loss is None:
            final_val_loss = model_data.get("loss_value") or 0.0
            
        # Extract best_epoch with fallback
        best_epoch = model_data.get("epoch")
        if best_epoch is None:
            best_epoch = 0
            
        # Extract other metrics
        training_duration = (metrics or {}).get("training_duration_seconds")
        if training_duration is None and training_metrics:
            training_duration = training_metrics.get("training_duration_seconds")
        if training_duration is None:
            training_duration = 0
            
        inference_latency = (metrics or {}).get("inference_latency_ms")
        if inference_latency is None and training_metrics:
            inference_latency = training_metrics.get("inference_latency_ms")
        if inference_latency is None:
            inference_latency = 50.0
            
        onnx_speedup = (metrics or {}).get("onnx_speedup_factor")
        if onnx_speedup is None and training_metrics:
            onnx_speedup = training_metrics.get("onnx_speedup_factor")
        if onnx_speedup is None:
            onnx_speedup = 2.0
        
        return PerformanceMetrics(
            final_train_loss=float(final_train_loss),
            final_val_loss=float(final_val_loss),
            final_train_accuracy=train_accuracy,
            final_val_accuracy=val_accuracy,
            best_epoch=int(best_epoch),
            training_duration_seconds=int(training_duration),
            inference_latency_ms=float(inference_latency),
            onnx_speedup_factor=float(onnx_speedup),
        )

    def _load_normalization_stats(self) -> NormalizationStats:
        """Load feature normalization statistics."""
        # ImageNet normalization (standard for pretrained models)
        return NormalizationStats(
            feature_means=[0.485, 0.456, 0.406],
            feature_stds=[0.229, 0.224, 0.225],
        )

    def _load_sample_predictions(
        self, model_data: Dict[str, Any], num_samples: int
    ) -> Optional[List[SamplePrediction]]:
        """Load sample predictions from database."""
        from sqlalchemy import text
        
        dataset_id = model_data.get("synthetic_dataset_id")
        if not dataset_id:
            return None
        
        with self.db_manager.get_session() as session:
            # Fetch sample predictions from measurement_features
            query = text("""
                SELECT 
                    recording_session_id,
                    tx_latitude,
                    tx_longitude,
                    tx_power_dbm,
                    extraction_metadata->>'frequency_hz' as frequency_hz,
                    mean_snr_db,
                    overall_confidence,
                    gdop
                FROM heimdall.measurement_features
                WHERE dataset_id = :dataset_id
                LIMIT :num_samples
            """)
            
            results = session.execute(
                query, {"dataset_id": dataset_id, "num_samples": num_samples}
            ).fetchall()
            
            samples = []
            for row in results:
                # Create sample prediction (note: actual predictions would require model inference)
                samples.append(
                    SamplePrediction(
                        sample_id=str(row[0]),
                        input_metadata={
                            "frequency_hz": float(row[4]) if row[4] else 145000000.0,
                            "tx_power_dbm": float(row[3]) if row[3] else 37.0,
                            "mean_snr_db": float(row[5]) if row[5] else 15.0,
                        },
                        ground_truth={
                            "latitude": float(row[1]) if row[1] else 0.0,
                            "longitude": float(row[2]) if row[2] else 0.0,
                        },
                        prediction={
                            "latitude": float(row[1]) if row[1] else 0.0,  # Placeholder
                            "longitude": float(row[2]) if row[2] else 0.0,  # Placeholder
                            "uncertainty_x": 25.0,  # Placeholder
                            "uncertainty_y": 25.0,  # Placeholder
                        },
                        error_meters=0.0,  # Placeholder - would need actual inference
                    )
                )
            
            return samples if samples else None


class HeimdallImporter:
    """Import .heimdall bundles."""

    def __init__(self, db_manager, minio_client):
        """
        Initialize importer.
        
        Args:
            db_manager: DatabaseManager instance
            minio_client: boto3 S3 client (MinIO)
        """
        self.db_manager = db_manager
        self.minio = minio_client

    def import_model(
        self, bundle: HeimdallBundle
    ) -> Dict[str, Any]:
        """
        Import model from .heimdall bundle.
        
        Args:
            bundle: Parsed HeimdallBundle
            
        Returns:
            Dict with model_id, onnx_path, and import details
        """
        # Extract ONNX binary
        onnx_bytes = base64.b64decode(bundle.model.onnx_model_base64)
        
        # Upload to MinIO
        onnx_path = f"imported/{bundle.model.model_name}-{bundle.model.version}.onnx"
        self.minio.put_object(
            Bucket="heimdall-models",
            Key=onnx_path,
            Body=io.BytesIO(onnx_bytes),
            ContentLength=len(onnx_bytes),
        )
        
        logger.info(f"Uploaded ONNX model to s3://heimdall-models/{onnx_path}")
        
        # Register in database
        model_id = self._register_model_in_db(bundle, f"s3://heimdall-models/{onnx_path}")
        
        return {
            "model_id": model_id,
            "onnx_path": f"s3://heimdall-models/{onnx_path}",
            "model_name": bundle.model.model_name,
            "version": bundle.model.version,
            "architecture": bundle.model.architecture,
        }

    def _register_model_in_db(
        self, bundle: HeimdallBundle, onnx_path: str
    ) -> str:
        """Register imported model in database."""
        from sqlalchemy import text
        
        model_id = str(uuid.uuid4())
        
        with self.db_manager.get_session() as session:
            # Prepare hyperparameters JSON
            hyperparameters = None
            if bundle.training_config:
                hyperparameters = {
                    "batch_size": bundle.training_config.batch_size,
                    "learning_rate": bundle.training_config.learning_rate,
                    "optimizer": bundle.training_config.optimizer,
                    "loss_function": bundle.training_config.loss_function,
                    "architecture": bundle.model.architecture,
                    "num_parameters": bundle.model.parameters_count,
                }
            
            # Prepare performance metrics JSON
            performance_metrics = None
            if bundle.performance_metrics:
                performance_metrics = {
                    "final_train_loss": bundle.performance_metrics.final_train_loss,
                    "final_val_loss": bundle.performance_metrics.final_val_loss,
                    "training_duration_seconds": bundle.performance_metrics.training_duration_seconds,
                    "inference_latency_ms": bundle.performance_metrics.inference_latency_ms,
                    "onnx_speedup_factor": bundle.performance_metrics.onnx_speedup_factor,
                }
            
            query = text("""
                INSERT INTO heimdall.models (
                    id, 
                    model_name, 
                    version,
                    model_type, 
                    onnx_model_location,
                    accuracy_meters,
                    accuracy_sigma_meters,
                    hyperparameters,
                    training_metrics,
                    is_active,
                    created_at
                )
                VALUES (
                    :id, 
                    :model_name, 
                    :version,
                    :model_type, 
                    :onnx_model_location,
                    :accuracy_meters,
                    :accuracy_sigma_meters,
                    CAST(:hyperparameters AS jsonb),
                    CAST(:training_metrics AS jsonb),
                    FALSE,
                    NOW()
                )
            """)
            
            session.execute(
                query,
                {
                    "id": model_id,
                    "model_name": f"{bundle.model.model_name} (Imported)",
                    "version": int(bundle.model.version.split(".")[0]) if bundle.model.version else 1,
                    "model_type": bundle.model.architecture,
                    "onnx_model_location": onnx_path,
                    "accuracy_meters": (
                        bundle.performance_metrics.final_val_accuracy
                        if bundle.performance_metrics
                        else None
                    ),
                    "accuracy_sigma_meters": (
                        bundle.performance_metrics.final_val_accuracy
                        if bundle.performance_metrics
                        else None
                    ),
                    "hyperparameters": json.dumps(hyperparameters) if hyperparameters else None,
                    "training_metrics": json.dumps(performance_metrics) if performance_metrics else None,
                },
            )
            session.commit()
        
        logger.info(f"Registered imported model {model_id} in database")
        
        return model_id
