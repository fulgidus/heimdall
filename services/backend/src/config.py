from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    service_name: str = "backend"
    service_port: int = 8001
    environment: str = "development"
    
    # CORS configuration
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:8000"
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "GET,POST,PUT,DELETE,PATCH,OPTIONS"
    cors_allow_headers: str = "Authorization,Content-Type,Accept,Origin,X-Requested-With"
    cors_expose_headers: str = "*"
    cors_max_age: int = 3600
    
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_url: str = "redis://:changeme@redis:6379/0"
    
    # Celery configuration
    celery_broker_url: str = "amqp://guest:guest@rabbitmq:5672/"
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
    
    def get_cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    def get_cors_methods_list(self) -> List[str]:
        """Parse comma-separated CORS methods into a list."""
        return [method.strip() for method in self.cors_allow_methods.split(",") if method.strip()]
    
    def get_cors_headers_list(self) -> List[str]:
        """Parse comma-separated CORS headers into a list."""
        return [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]


settings = Settings()

