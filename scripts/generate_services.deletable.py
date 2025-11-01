#!/usr/bin/env python3
"""Generate all 5 Heimdall SDR services"""

from pathlib import Path

services = ["rf-acquisition", "training", "inference", "api-gateway"]
ports = {"rf-acquisition": 8001, "training": 8002, "inference": 8003, "api-gateway": 8000}

for service_name in services:
    service_dir = Path("services") / service_name
    port = ports[service_name]
    print(f"Generating {service_name}...", end=" ", flush=True)

    # Create directories
    for d in ["src/models", "src/routers", "src/utils", "tests/unit", "tests/integration", "docs"]:
        (service_dir / d).mkdir(parents=True, exist_ok=True)

    # main.py
    main_content = f'''"""Heimdall SDR - {service_name} Service"""
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from models.health import HealthResponse

SERVICE_NAME = "{service_name}"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = {port}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {{SERVICE_NAME}} service")
    yield
    print(f"Shutting down {{SERVICE_NAME}} service")


app = FastAPI(title=f"Heimdall SDR - {{SERVICE_NAME}}", version=SERVICE_VERSION, lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {{"service": SERVICE_NAME, "status": "running"}}


@app.get("/health")
async def health_check():
    return HealthResponse(status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow())


@app.get("/ready")
async def readiness_check():
    return {{"ready": True}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
'''
    (service_dir / "src" / "main.py").write_text(main_content, encoding="utf-8")

    # config.py
    config_content = f'''"""Configuration"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "{service_name}"
    service_port: int = {port}
    cors_origins: List[str] = ["*"]
    database_url: str = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"
    redis_url: str = "redis://redis:6379/0"


settings = Settings()
'''
    (service_dir / "src" / "config.py").write_text(config_content, encoding="utf-8")

    # models/health.py
    models_content = '''"""Models"""
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Version")
    timestamp: datetime = Field(..., description="Timestamp")
'''
    (service_dir / "src" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (service_dir / "src" / "models" / "health.py").write_text(models_content, encoding="utf-8")

    # requirements.txt
    req_content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
structlog==24.1.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
pytest==7.4.3
httpx==0.25.1
"""
    (service_dir / "requirements.txt").write_text(req_content, encoding="utf-8")

    # Dockerfile
    docker_content = f"""FROM python:3.11-slim as builder
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
EXPOSE {port}
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "{port}"]
"""
    (service_dir / "Dockerfile").write_text(docker_content, encoding="utf-8")

    # .gitignore
    gitignore_content = """__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
.env
.DS_Store
*.log
"""
    (service_dir / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    # tests
    (service_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (service_dir / "tests" / "unit" / "__init__.py").write_text("", encoding="utf-8")
    (service_dir / "tests" / "integration" / "__init__.py").write_text("", encoding="utf-8")
    (service_dir / "tests" / "unit" / ".gitkeep").write_text("", encoding="utf-8")
    (service_dir / "tests" / "integration" / ".gitkeep").write_text("", encoding="utf-8")

    conftest_content = """import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    return TestClient(app)
"""
    (service_dir / "tests" / "conftest.py").write_text(conftest_content, encoding="utf-8")

    test_content = """from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
"""
    (service_dir / "tests" / "test_main.py").write_text(test_content, encoding="utf-8")

    # README
    readme_content = f"""# {service_name.upper()} Service

Port: {port}

## Quick Start

pip install -r requirements.txt
python -m uvicorn src.main:app --reload

API: http://localhost:{port}/docs
"""
    (service_dir / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"[OK] {service_name} - {port}")

print("\n[SUCCESS] All services generated!")
