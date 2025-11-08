import os
import sys
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for common module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from common.dependency_checkers import check_postgresql, check_redis

# Import common health utilities
from common.health import HealthChecker

from .config import settings
from .models.health import HealthResponse
from .routers import analytics

SERVICE_NAME = "inference"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8003

app = FastAPI(title=f"Heimdall SDR - {SERVICE_NAME}", version=SERVICE_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# app.include_router(predict.router)  # Temporarily disabled
app.include_router(analytics.router)

# Initialize health checker
health_checker = HealthChecker(SERVICE_NAME, SERVICE_VERSION)


# Register dependency health checks
async def check_db():
    """Check database connectivity."""
    await check_postgresql(settings.database_url)


async def check_cache():
    """Check Redis cache connectivity."""
    await check_redis(settings.redis_url)


health_checker.register_dependency("database", check_db)
health_checker.register_dependency("redis", check_cache)


@app.get("/api/v1/analytics/test")
async def test_analytics():
    return {"message": "Analytics router is working"}


@app.get("/")
async def root():
    return {
        "service": SERVICE_NAME,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check():
    """
    Liveness probe - checks if service is alive.

    Returns basic health status without dependency checks.
    """
    return HealthResponse(
        status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow()
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with dependency status.

    Returns comprehensive health information including all dependencies.
    """
    result = await health_checker.check_all()
    return JSONResponse(status_code=200 if result.ready else 503, content=result.to_dict())


@app.get("/api/v1/inference/health")
async def inference_health_check_public():
    """Public health check endpoint - used by API Gateway and dashboard monitoring."""
    return HealthResponse(
        status="healthy", service=SERVICE_NAME, version=SERVICE_VERSION, timestamp=datetime.utcnow()
    )


@app.get("/ready")
async def readiness_check():
    """
    Readiness probe - checks if service can handle requests.

    Validates all critical dependencies (database, cache).
    """
    try:
        result = await health_checker.check_all()

        return JSONResponse(
            status_code=200 if result.ready else 503,
            content={
                "ready": result.ready,
                "service": SERVICE_NAME,
                "dependencies": result.to_dict()["dependencies"],
            },
        )
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error("Readiness check failed: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=503, content={"ready": False, "service": SERVICE_NAME, "error": str(e)}
        )


@app.get("/startup")
async def startup_check():
    """
    Startup probe - checks if service has finished starting up.
    """
    return await readiness_check()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
