from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models.health import HealthResponse
from .routers import predict, analytics

SERVICE_NAME = "inference"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8003

app = FastAPI(title=f"Heimdall SDR - {SERVICE_NAME}", version=SERVICE_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Include routers
# app.include_router(predict.router)  # Temporarily disabled
app.include_router(analytics.router)


@app.get("/api/v1/analytics/test")
async def test_analytics():
    return {"message": "Analytics router is working"}


@app.get("/")
async def root():
    return {"service": SERVICE_NAME, "status": "running", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health")
async def health_check():
    return HealthResponse(status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow())


@app.get("/api/v1/inference/health")
async def inference_health_check_public():
    """Public health check endpoint - used by API Gateway and dashboard monitoring."""
    return HealthResponse(status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow())


@app.get("/ready")
async def readiness_check():
    return {"ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
