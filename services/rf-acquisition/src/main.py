import logging
import sys
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from celery import Celery

# Add parent directory to path for common module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .config import settings
from .models.health import HealthResponse
from .routers.acquisition import router as acquisition_router
from .routers.sessions import router as sessions_router

# Import common health utilities
from common.health import HealthChecker
from common.dependency_checkers import check_postgresql, check_redis, check_celery, check_minio

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
app.include_router(sessions_router)

# Initialize health checker
health_checker = HealthChecker(SERVICE_NAME, SERVICE_VERSION)

# Register dependency health checks
async def check_db():
    """Check database connectivity."""
    await check_postgresql(settings.database_url)

async def check_cache():
    """Check Redis cache connectivity."""
    await check_redis(settings.redis_url)

async def check_queue():
    """Check Celery broker and backend."""
    await check_celery(settings.celery_broker_url, settings.celery_result_backend_url)

async def check_storage():
    """Check MinIO object storage."""
    await check_minio(
        settings.minio_url,
        settings.minio_access_key,
        settings.minio_secret_key,
        secure=False
    )

health_checker.register_dependency("database", check_db)
health_checker.register_dependency("redis", check_cache)
health_checker.register_dependency("celery", check_queue)
health_checker.register_dependency("minio", check_storage)


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
    """
    Liveness probe - checks if service is alive.
    
    Returns basic health status without dependency checks.
    Used by Kubernetes liveness probe.
    """
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow()
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with dependency status.
    
    Returns comprehensive health information including all dependencies.
    """
    result = await health_checker.check_all()
    return JSONResponse(
        status_code=200 if result.ready else 503,
        content=result.to_dict()
    )


@app.get("/ready")
async def readiness_check():
    """
    Readiness probe - checks if service can handle requests.
    
    Validates all critical dependencies (database, cache, queue, storage).
    Used by Kubernetes readiness probe.
    """
    try:
        result = await health_checker.check_all()
        
        # In development mode, be more lenient
        if settings.environment == "development":
            # Allow service to be ready even if some non-critical deps are down
            critical_deps = ["database", "celery"]
            critical_status = [
                d.status for d in result.dependencies 
                if d.name in critical_deps
            ]
            
            from common.health import HealthStatus
            if HealthStatus.DOWN in critical_status:
                return JSONResponse(
                    status_code=503,
                    content={
                        "ready": False,
                        "service": SERVICE_NAME,
                        "dependencies": result.to_dict()["dependencies"]
                    }
                )
            
            return JSONResponse(
                status_code=200,
                content={
                    "ready": True,
                    "service": SERVICE_NAME,
                    "dependencies": result.to_dict()["dependencies"]
                }
            )
        
        # Production mode: all dependencies must be healthy
        return JSONResponse(
            status_code=200 if result.ready else 503,
            content={
                "ready": result.ready,
                "service": SERVICE_NAME,
                "dependencies": result.to_dict()["dependencies"]
            }
        )
    except Exception as e:
        logger.error("Readiness check failed: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"ready": False, "service": SERVICE_NAME, "error": str(e)}
        )


@app.get("/startup")
async def startup_check():
    """
    Startup probe - checks if service has finished starting up.
    
    Used by Kubernetes startup probe to know when to start sending traffic.
    """
    # For now, startup is the same as readiness
    # Can be customized for slower-starting services
    return await readiness_check()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
