#!/usr/bin/env python3
"""Fast service generation"""
from pathlib import Path

services = [
    ("rf-acquisition", 8001),
    ("training", 8002),
    ("inference", 8003),
    ("data-ingestion-web", 8004),
    ("api-gateway", 8000),
]

for name, port in services:
    d = Path("services") / name
    (d / "src" / "models").mkdir(parents=True, exist_ok=True)
    (d / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    (d / "tests" / "integration").mkdir(parents=True, exist_ok=True)
    (d / "docs").mkdir(parents=True, exist_ok=True)
    
    # Write files
    (d / "src" / "main.py").write_text(f'from datetime import datetime\nfrom fastapi import FastAPI\nfrom config import settings\nfrom models.health import HealthResponse\napp = FastAPI()\n@app.get("/")\ndef root(): return {{"service": "{name}", "status": "running"}}\n@app.get("/health")\ndef health(): return HealthResponse(status="healthy", service="{name}", version="0.1.0", timestamp=datetime.utcnow())\n', encoding='utf-8')
    
    (d / "src" / "config.py").write_text(f'from pydantic_settings import BaseSettings\nclass Settings(BaseSettings):\n    service_name = "{name}"\n    service_port = {port}\n    cors_origins = ["*"]\n    database_url = "postgresql://heimdall_user:changeme@postgres:5432/heimdall"\n    redis_url = "redis://redis:6379/0"\nsettings = Settings()\n', encoding='utf-8')
    
    (d / "src" / "models" / "__init__.py").write_text("", encoding='utf-8')
    (d / "src" / "models" / "health.py").write_text('from datetime import datetime\nfrom pydantic import BaseModel, Field\nclass HealthResponse(BaseModel):\n    status: str = Field(..., description="Health status")\n    service: str = Field(..., description="Service name")\n    version: str = Field(..., description="Version")\n    timestamp: datetime = Field(..., description="Timestamp")\n', encoding='utf-8')
    
    (d / "requirements.txt").write_text("fastapi==0.104.1\nuvicorn[standard]==0.24.0\npydantic==2.5.0\npydantic-settings==2.1.0\npsycopg2-binary==2.9.9\nredis==5.0.1\npytest==7.4.3\nhttpx==0.25.1\n", encoding='utf-8')
    
    (d / "Dockerfile").write_text(f"FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY src/ ./src/\nEXPOSE {port}\nCMD [\"python\", \"-m\", \"uvicorn\", \"src.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"{port}\"]\n", encoding='utf-8')
    
    (d / ".gitignore").write_text("__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv/\n.env\n*.log\n", encoding='utf-8')
    
    (d / "tests" / "__init__.py").write_text("", encoding='utf-8')
    (d / "tests" / "unit" / "__init__.py").write_text("", encoding='utf-8')
    (d / "tests" / "integration" / "__init__.py").write_text("", encoding='utf-8')
    
    (d / "README.md").write_text(f"# {name.upper()}\nPort: {port}\n", encoding='utf-8')
    
    print(f"[+] {name}")

print("[OK] Done!")
