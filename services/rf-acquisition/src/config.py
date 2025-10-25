from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    service_name: str = "rf-acquisition"
    service_port: int = 8001
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_url: str = "redis://:changeme@redis:6379/0"
    
    # Celery configuration
    celery_broker_url: str = "amqp://guest:guest@rabbitmq:5672//"
    celery_result_backend_url: str = "redis://:changeme@redis:6379/1"
    celery_check_required: bool = False  # Set to True in production
    
    # MinIO configuration
    minio_url: str = "http://minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_raw_iq: str = "heimdall-raw-iq"
    
    # WebSDR configuration
    websdr_timeout_seconds: int = 30
    websdr_retry_count: int = 3
    websdr_concurrent_limit: int = 7
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = ""


settings = Settings()

