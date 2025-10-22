from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    service_name: str = "training"
    service_port: int = 8002
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall:heimdall@postgres:5432/heimdall"
    redis_url: str = "redis://redis:6379/0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
