from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models.health import HealthResponse

SERVICE_NAME = "rf-acquisition"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8001

app = FastAPI(title=f"Heimdall SDR - {SERVICE_NAME}", version=SERVICE_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"service": SERVICE_NAME, "status": "running", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health")
async def health_check():
    return HealthResponse(status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow())


@app.get("/ready")
async def readiness_check():
    return {"ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
