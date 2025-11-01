"""
MLflow tracking integration for Heimdall training pipeline.

Provides:
- MLflow client initialization and configuration
- Experiment and run management
- Parameter and metric logging
- Artifact management and registration
- Model registry operations
"""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import structlog
from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)


class MLflowTracker:
    """
    Centralized MLflow tracking manager for training pipeline.

    Responsibilities:
    - Initialize MLflow tracking server
    - Create/manage experiments
    - Log training runs with parameters and metrics
    - Handle artifact storage (MinIO/S3)
    - Register models to MLflow Registry
    """

    def __init__(
        self,
        tracking_uri: str,
        artifact_uri: str,
        backend_store_uri: str,
        registry_uri: str,
        s3_endpoint_url: str,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        experiment_name: str = "heimdall-localization",
    ):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri (str): PostgreSQL URI for MLflow tracking server
                Format: postgresql://user:pass@host:port/dbname
            artifact_uri (str): S3 bucket URI for artifacts
                Format: s3://bucket-name
            backend_store_uri (str): PostgreSQL URI for backend store
            registry_uri (str): URI for model registry
            s3_endpoint_url (str): MinIO endpoint URL (e.g., http://minio:9000)
            s3_access_key_id (str): MinIO access key
            s3_secret_access_key (str): MinIO secret key
            experiment_name (str): MLflow experiment name
        """

        # Store configuration
        self.tracking_uri = tracking_uri
        self.artifact_uri = artifact_uri
        self.backend_store_uri = backend_store_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name

        # Set environment variables for S3/MinIO
        os.environ["AWS_ACCESS_KEY_ID"] = s3_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = s3_secret_access_key

        # Configure MLflow
        self._configure_mlflow(
            tracking_uri,
            artifact_uri,
            backend_store_uri,
            registry_uri,
            s3_endpoint_url,
        )

        # Create client
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Initialize experiment
        self.experiment_id = self._get_or_create_experiment(experiment_name)

        logger.info(
            "mlflow_tracker_initialized",
            tracking_uri=tracking_uri,
            artifact_uri=artifact_uri,
            experiment_name=experiment_name,
            experiment_id=self.experiment_id,
        )

    def _configure_mlflow(
        self,
        tracking_uri: str,
        artifact_uri: str,
        backend_store_uri: str,
        registry_uri: str,
        s3_endpoint_url: str,
    ):
        """
        Configure MLflow connection parameters.

        Args:
            tracking_uri (str): PostgreSQL URI for tracking
            artifact_uri (str): S3 artifact root
            backend_store_uri (str): Backend store URI
            registry_uri (str): Model registry URI
            s3_endpoint_url (str): MinIO endpoint
        """

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Configure S3/MinIO environment
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

        logger.debug(
            "mlflow_configured",
            tracking_uri=tracking_uri,
            artifact_uri=artifact_uri,
            s3_endpoint_url=s3_endpoint_url,
        )

    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment by name or create new one.

        Args:
            experiment_name (str): Name of experiment

        Returns:
            str: Experiment ID
        """

        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                logger.info(
                    "experiment_found",
                    experiment_name=experiment_name,
                    experiment_id=experiment.experiment_id,
                )
                return experiment.experiment_id
        except Exception as e:
            logger.warning(
                "experiment_lookup_failed",
                experiment_name=experiment_name,
                error=str(e),
            )

        # Create new experiment
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=self.artifact_uri,
            )
            logger.info(
                "experiment_created",
                experiment_name=experiment_name,
                experiment_id=experiment_id,
            )
            return experiment_id
        except Exception as e:
            logger.error(
                "experiment_creation_failed",
                experiment_name=experiment_name,
                error=str(e),
            )
            raise

    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name (str): Name for the run
            tags (dict, optional): Dictionary of tags to set

        Returns:
            str: Run ID
        """

        # Start run
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )

        run_id = run.info.run_id

        # Set default tags
        default_tags = {
            "phase": "training",
            "service": "training",
            "model": "LocalizationNet",
        }

        if tags:
            default_tags.update(tags)

        mlflow.set_tags(default_tags)

        logger.info(
            "mlflow_run_started",
            run_id=run_id,
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=default_tags,
        )

        return run_id

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status (str): Final run status (FINISHED, FAILED, KILLED)
        """

        mlflow.end_run(status=status)

        logger.info(
            "mlflow_run_ended",
            status=status,
        )

    def log_params(self, params: dict[str, Any]):
        """
        Log training parameters.

        Args:
            params (dict): Dictionary of parameters

        Example:
            tracker.log_params({
                'learning_rate': 1e-3,
                'batch_size': 32,
                'epochs': 100,
                'backbone': 'ConvNeXt-Large',
            })
        """

        for key, value in params.items():
            try:
                # Convert non-string types
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)

                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(
                    "parameter_logging_failed",
                    param_name=key,
                    param_value=str(value)[:100],
                    error=str(e),
                )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """
        Log training metrics.

        Args:
            metrics (dict): Dictionary of metric names and values
            step (int, optional): Step number (epoch)

        Example:
            tracker.log_metrics({
                'train_loss': 0.523,
                'val_loss': 0.487,
                'train_mae': 12.3,
            }, step=epoch)
        """

        for metric_name, metric_value in metrics.items():
            try:
                mlflow.log_metric(metric_name, metric_value, step=step)
            except Exception as e:
                logger.warning(
                    "metric_logging_failed",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    error=str(e),
                )

    def log_artifact(self, local_path: str, artifact_path: str = "artifacts"):
        """
        Log a local artifact file to MLflow.

        Args:
            local_path (str): Local file path
            artifact_path (str): Destination path in artifact store

        Example:
            tracker.log_artifact('model.onnx', 'models')
        """

        local_path = Path(local_path)

        if not local_path.exists():
            logger.warning(
                "artifact_not_found",
                local_path=str(local_path),
            )
            return

        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
            logger.info(
                "artifact_logged",
                local_path=str(local_path),
                artifact_path=artifact_path,
            )
        except Exception as e:
            logger.error(
                "artifact_logging_failed",
                local_path=str(local_path),
                artifact_path=artifact_path,
                error=str(e),
            )

    def log_artifacts_dir(self, local_dir: str, artifact_path: str = "artifacts"):
        """
        Log an entire directory of artifacts.

        Args:
            local_dir (str): Local directory path
            artifact_path (str): Destination path in artifact store
        """

        local_dir = Path(local_dir)

        if not local_dir.is_dir():
            logger.warning(
                "artifact_dir_not_found",
                local_dir=str(local_dir),
            )
            return

        try:
            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
            logger.info(
                "artifact_dir_logged",
                local_dir=str(local_dir),
                artifact_path=artifact_path,
                num_files=len(list(local_dir.rglob("*"))),
            )
        except Exception as e:
            logger.error(
                "artifact_dir_logging_failed",
                local_dir=str(local_dir),
                artifact_path=artifact_path,
                error=str(e),
            )

    def register_model(
        self,
        model_name: str,
        model_uri: str,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Register model to MLflow Model Registry.

        Args:
            model_name (str): Name for the model in registry
            model_uri (str): URI of model artifacts (runs://<run_id>/path/to/model)
            description (str): Model description
            tags (dict, optional): Model tags

        Returns:
            str: Model version

        Example:
            version = tracker.register_model(
                model_name="heimdall-localization-v1",
                model_uri="runs://abc123def/models/model",
                description="ConvNeXt-Large with uncertainty",
                tags={'stage': 'production'},
            )
        """

        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags or {},
                await_registration_for=300,
            )

            logger.info(
                "model_registered",
                model_name=model_name,
                model_version=model_version.version,
                model_uri=model_uri,
                description=description,
            )

            return model_version.version
        except Exception as e:
            logger.error(
                "model_registration_failed",
                model_name=model_name,
                model_uri=model_uri,
                error=str(e),
            )
            raise

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ):
        """
        Transition a registered model to a new stage.

        Args:
            model_name (str): Name of registered model
            version (str): Version number
            stage (str): Target stage (None, Staging, Production, Archived)

        Example:
            tracker.transition_model_stage(
                model_name="heimdall-localization-v1",
                version="1",
                stage="Production",
            )
        """

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )

            logger.info(
                "model_stage_transitioned",
                model_name=model_name,
                version=version,
                stage=stage,
            )
        except Exception as e:
            logger.error(
                "model_transition_failed",
                model_name=model_name,
                version=version,
                stage=stage,
                error=str(e),
            )

    def get_run_info(self, run_id: str) -> dict[str, Any]:
        """
        Get information about a specific run.

        Args:
            run_id (str): Run ID

        Returns:
            dict: Run information including metrics, parameters, artifacts
        """

        run = self.client.get_run(run_id)

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "parameters": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
        }

    def get_best_run(
        self,
        metric: str = "val/loss",
        compare_fn=min,
    ) -> dict[str, Any] | None:
        """
        Get the best run from current experiment based on a metric.

        Args:
            metric (str): Metric name to compare
            compare_fn: Comparison function (min or max)

        Returns:
            dict: Best run information or None if no runs

        Example:
            best = tracker.get_best_run(metric="val/loss", compare_fn=min)
        """

        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric} {'' if compare_fn == min else 'DESC'}"],
                max_results=1,
            )

            if runs.empty:
                logger.info("no_runs_found", experiment_id=self.experiment_id)
                return None

            best_run = runs.iloc[0]

            logger.info(
                "best_run_found",
                run_id=best_run["run_id"],
                metric=metric,
                value=best_run[f"metrics.{metric}"],
            )

            return {
                "run_id": best_run["run_id"],
                "metric_value": best_run[f"metrics.{metric}"],
                "params": best_run[[col for col in best_run.index if col.startswith("params.")]],
            }
        except Exception as e:
            logger.error(
                "best_run_lookup_failed",
                experiment_id=self.experiment_id,
                metric=metric,
                error=str(e),
            )
            return None


def initialize_mlflow(settings) -> MLflowTracker:
    """
    Initialize MLflow tracker from settings.

    Args:
        settings: Pydantic Settings object with MLflow configuration

    Returns:
        MLflowTracker: Initialized tracker instance
    """

    tracker = MLflowTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        artifact_uri=settings.mlflow_artifact_uri,
        backend_store_uri=settings.mlflow_backend_store_uri,
        registry_uri=settings.mlflow_registry_uri,
        s3_endpoint_url=settings.mlflow_s3_endpoint_url,
        s3_access_key_id=settings.mlflow_s3_access_key_id,
        s3_secret_access_key=settings.mlflow_s3_secret_access_key,
        experiment_name=settings.mlflow_experiment_name,
    )

    return tracker


if __name__ == "__main__":
    """Test MLflow setup"""
    from src.config import settings

    tracker = initialize_mlflow(settings)
    logger.info("✅ MLflow setup complete!")
