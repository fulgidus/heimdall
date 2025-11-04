"""Settings configuration for training service."""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Training service configuration settings."""

    service_name: str = "training"
    service_port: int = 8002
    environment: str = "development"
    cors_origins: list[str] = ["*"]

    # Database configuration from environment variables
    postgres_user: str = os.getenv("POSTGRES_USER", "heimdall_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "changeme")
    postgres_host: str = os.getenv("POSTGRES_HOST", "postgres")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "heimdall")

    # Construct database URL from components
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    redis_url: str = "redis://redis:6379/0"

    # Celery configuration
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
    celery_result_backend_url: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

    # MLflow configuration - also built from environment variables
    mlflow_db_user: str = os.getenv("POSTGRES_USER", "heimdall_user")
    mlflow_db_password: str = os.getenv("POSTGRES_PASSWORD", "changeme")
    mlflow_db_host: str = os.getenv("POSTGRES_HOST", "postgres")
    mlflow_db_port: int = int(os.getenv("POSTGRES_PORT", "5432"))

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"postgresql://{self.mlflow_db_user}:{self.mlflow_db_password}@{self.mlflow_db_host}:{self.mlflow_db_port}/mlflow_db"

    @property
    def mlflow_backend_store_uri(self) -> str:
        return f"postgresql://{self.mlflow_db_user}:{self.mlflow_db_password}@{self.mlflow_db_host}:{self.mlflow_db_port}/mlflow_db"

    @property
    def mlflow_registry_uri(self) -> str:
        return f"postgresql://{self.mlflow_db_user}:{self.mlflow_db_password}@{self.mlflow_db_host}:{self.mlflow_db_port}/mlflow_db"

    mlflow_artifact_uri: str = "s3://heimdall-mlflow"
    mlflow_s3_endpoint_url: str = "http://minio:9000"
    mlflow_s3_access_key_id: str = "minioadmin"
    mlflow_s3_secret_access_key: str = "minioadmin"
    mlflow_default_artifact_root: str = "s3://heimdall-mlflow"
    mlflow_experiment_name: str = "heimdall-localization"
    mlflow_run_name_prefix: str = "rf-localization"

    # MinIO configuration
    minio_url: str = "http://minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    # Backend service URL for terrain downloads
    backend_url: str = os.getenv("BACKEND_URL", "http://backend:8000")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
