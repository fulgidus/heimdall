import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    service_name: str = "data-ingestion-web"
    service_port: int = 8004
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_password: str = os.getenv("REDIS_PASSWORD", "changeme")
    redis_url: str = f"redis://:{os.getenv('REDIS_PASSWORD', 'changeme')}@redis:6379/0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
