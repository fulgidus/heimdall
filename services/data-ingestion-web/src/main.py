import sys
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

# Add parent directory to path for common module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .config import settings
from .models.health import HealthResponse
from .db import get_pool, close_pool
from .routers import sessions

# Import common health utilities
from common.health import HealthChecker
from common.dependency_checkers import check_postgresql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_NAME = "data-ingestion-web"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8004


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle"""
    # Startup
    logger.info(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")
    try:
        await get_pool()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {SERVICE_NAME}")
    await close_pool()
    logger.info("Database connection pool closed")


app = FastAPI(
    title=f"Heimdall SDR - {SERVICE_NAME}",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions.router)

# Initialize health checker
health_checker = HealthChecker(SERVICE_NAME, SERVICE_VERSION)

# Register dependency health checks
async def check_db():
    """Check database connectivity."""
    await check_postgresql(settings.database_url)

health_checker.register_dependency("database", check_db)


@app.get("/")
async def root():
    return {
        "service": SERVICE_NAME,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": SERVICE_VERSION,
    }


@app.get("/health")
async def health_check():
    """
    Liveness probe - checks if service is alive.
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
    """
    try:
        result = await health_checker.check_all()
        
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
    """
    return await readiness_check()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
