from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    service_name: str = "api-gateway"
    service_port: int = 8000
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_url: str = "redis://redis:6379/0"
    
    # Backend service URLs
    rf_acquisition_url: str = "http://rf-acquisition:8001"
    inference_url: str = "http://inference:8003"
    training_url: str = "http://training:8002"
    api_gateway_url: str = "http://localhost:8000"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
