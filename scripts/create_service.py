#!/usr/bin/env python3
"""Service Scaffold Generator for Heimdall SDR"""

import sys
from pathlib import Path

class ServiceScaffoldGenerator:
    """Generate complete service scaffolding"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.service_dir = Path(__file__).parent.parent / "services" / service_name
        self.port_map = {
            "rf-acquisition": 8001,
            "training": 8002,
            "inference": 8003,
            "data-ingestion-web": 8004,
            "api-gateway": 8000,
        }
        self.port = self.port_map.get(service_name, 8005)
    
    def generate(self):
        """Generate all service files"""
        print(f"\nGenerating scaffold for: {self.service_name}")
        
        try:
            self._create_directories()
            self._generate_all_files()
            print(f"[SUCCESS] Service created at: {self.service_dir}")
            print(f"[INFO] Port: {self.port}")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _create_directories(self):
        """Create directory structure"""
        for path in [
            self.service_dir,
            self.service_dir / "src",
            self.service_dir / "src" / "models",
            self.service_dir / "src" / "routers",
            self.service_dir / "src" / "utils",
            self.service_dir / "tests",
            self.service_dir / "tests" / "unit",
            self.service_dir / "tests" / "integration",
            self.service_dir / "docs",
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _write_file(self, path, content):
        """Write file with UTF-8 encoding"""
        path.write_text(content, encoding='utf-8')
    
    def _generate_all_files(self):
        """Generate all service files"""
        # main.py
        self._write_file(
            self.service_dir / "src" / "main.py",
            self._template_main()
        )
        
        # config.py
        self._write_file(
            self.service_dir / "src" / "config.py",
            self._template_config()
        )
        
        # models/health.py
        self._write_file(
            self.service_dir / "src" / "models" / "__init__.py",
            ""
        )
        self._write_file(
            self.service_dir / "src" / "models" / "health.py",
            self._template_models()
        )
        
        # requirements.txt
        self._write_file(
            self.service_dir / "requirements.txt",
            self._template_requirements()
        )
        
        # Dockerfile
        self._write_file(
            self.service_dir / "Dockerfile",
            self._template_dockerfile()
        )
        
        # .gitignore
        self._write_file(
            self.service_dir / ".gitignore",
            self._template_gitignore()
        )
        
        # tests
        self._write_file(
            self.service_dir / "tests" / "__init__.py",
            ""
        )
        self._write_file(
            self.service_dir / "tests" / "unit" / "__init__.py",
            ""
        )
        self._write_file(
            self.service_dir / "tests" / "integration" / "__init__.py",
            ""
        )
        self._write_file(
            self.service_dir / "tests" / "conftest.py",
            self._template_conftest()
        )
        self._write_file(
            self.service_dir / "tests" / "test_main.py",
            self._template_test_main()
        )
        self._write_file(
            self.service_dir / "tests" / "unit" / ".gitkeep",
            ""
        )
        self._write_file(
            self.service_dir / "tests" / "integration" / ".gitkeep",
            ""
        )
        
        # README.md
        self._write_file(
            self.service_dir / "README.md",
            self._template_readme()
        )
      def _template_main(self):
        return f'"""Heimdall SDR - {self.service_name} Service"""\nfrom datetime import datetime\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\nimport uvicorn\n\nfrom config import settings\nfrom models.health import HealthResponse\n\nSERVICE_NAME = "{self.service_name}"\nSERVICE_VERSION = "0.1.0"\nSERVICE_PORT = {self.port}\n\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    print(f"Starting {{SERVICE_NAME}} service")\n    yield\n    print(f"Shutting down {{SERVICE_NAME}} service")\n\n\napp = FastAPI(\n    title=f"Heimdall SDR - {{SERVICE_NAME}}",\n    version=SERVICE_VERSION,\n    lifespan=lifespan,\n)\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=settings.cors_origins,\n    allow_credentials=True,\n    allow_methods=["*"],\n    allow_headers=["*"],\n)\n\n\n@app.get("/")\nasync def root():\n    return {{"service": SERVICE_NAME, "status": "running"}}\n\n\n@app.get("/health")\nasync def health_check():\n    return HealthResponse(\n        status="healthy",\n        service=SERVICE_NAME,\n        version=SERVICE_VERSION,\n        timestamp=datetime.utcnow(),\n    )\n\n\n@app.get("/ready")\nasync def readiness_check():\n    return {{"ready": True}}\n\n\nif __name__ == "__main__":\n    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)\n'
    
    def _template_config(self):
        return f"""'''Configuration for {self.service_name}'''
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "{self.service_name}"
    service_port: int = {self.port}
    environment: str = "development"
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_url: str = "redis://redis:6379/0"
    
    class Config:
        env_file = ".env"


settings = Settings()
"""
    
    def _template_models(self):
        return """'''Data models'''
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Version")
    timestamp: datetime = Field(..., description="Timestamp")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
"""
    
    def _template_requirements(self):
        return """# Heimdall SDR Service Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
structlog==24.1.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
aiohttp==3.9.1
celery==5.3.4
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.1
"""
    
    def _template_dockerfile(self):
        return f"""FROM python:3.11-slim as builder
WORKDIR /build
RUN apt-get update && apt-get install -y gcc postgresql-client
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appuser && useradd -r -g appuser appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser src/ ./src/
ENV PATH=/home/appuser/.local/bin:$PATH PYTHONUNBUFFERED=1
USER appuser
EXPOSE {self.port}
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "{self.port}"]
"""
    
    def _template_gitignore(self):
        return """__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
.env
.DS_Store
*.log
build/
dist/
.coverage
htmlcov/
.idea/
.vscode/
"""
    
    def _template_conftest(self):
        return """import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    return TestClient(app)
"""
    
    def _template_test_main(self):
        return """from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
"""
    
    def _template_readme(self):
        return f"""# {self.service_name.upper()} Service

Port: {self.port}

## Quick Start

pip install -r requirements.txt
python -m uvicorn src.main:app --reload

API: http://localhost:{self.port}/docs
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_service.py <service_name>")
        sys.exit(1)
    
    ServiceScaffoldGenerator(sys.argv[1]).generate()


if __name__ == "__main__":
    main()
