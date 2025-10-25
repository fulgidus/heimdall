from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    service_name: str = "inference"
    service_port: int = 8003
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_url: str = "redis://redis:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
