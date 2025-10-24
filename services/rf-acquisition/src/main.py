import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from celery import Celery

from .config import settings
from .models.health import HealthResponse
from .routers.acquisition import router as acquisition_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "rf-acquisition"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8001

# Initialize FastAPI app
app = FastAPI(
    title=f"Heimdall SDR - {SERVICE_NAME}",
    version=SERVICE_VERSION,
    description="RF data acquisition service for Heimdall SDR platform"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize Celery
celery_app = Celery(
    SERVICE_NAME,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend_url
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

# Configure Celery Beat schedule for monitoring
celery_app.conf.beat_schedule = {
    'monitor-websdrs-uptime': {
        'task': 'monitor_websdrs_uptime',
        'schedule': 60.0,  # Every 60 seconds
    },
}

# Include routers
app.include_router(acquisition_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": SERVICE_NAME,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow()
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check Celery connectivity only if required
        if settings.celery_check_required:
            inspect = celery_app.control.inspect()
            if inspect is None or inspect.active() is None:
                return JSONResponse(
                    status_code=503,
                    content={"ready": False, "reason": "Celery broker not responding"}
                )
        
        return JSONResponse(
            status_code=200,
            content={"ready": True, "service": SERVICE_NAME}
        )
    except Exception as e:
        logger.warning("Readiness check warning: %s", str(e))
        # In development mode, still return ready=True even if Celery check fails
        if settings.environment == "development":
            return JSONResponse(
                status_code=200,
                content={"ready": True, "service": SERVICE_NAME, "warning": str(e)}
            )
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
