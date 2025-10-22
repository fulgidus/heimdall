"""Settings configuration for training service."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Training service configuration settings."""
    
    service_name: str = "training"
    service_port: int = 8002
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall:heimdall@postgres:5432/heimdall"
    redis_url: str = "redis://redis:6379/0"
    
    # MLflow configuration
    mlflow_tracking_uri: str = "postgresql://heimdall:heimdall@postgres:5432/mlflow_db"
    mlflow_artifact_uri: str = "s3://heimdall-mlflow"
    mlflow_s3_endpoint_url: str = "http://minio:9000"
    mlflow_s3_access_key_id: str = "minioadmin"
    mlflow_s3_secret_access_key: str = "minioadmin"
    mlflow_backend_store_uri: str = "postgresql://heimdall:heimdall@postgres:5432/mlflow_db"
    mlflow_default_artifact_root: str = "s3://heimdall-mlflow"
    mlflow_experiment_name: str = "heimdall-localization"
    mlflow_run_name_prefix: str = "rf-localization"
    mlflow_registry_uri: str = "postgresql://heimdall:heimdall@postgres:5432/mlflow_db"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
