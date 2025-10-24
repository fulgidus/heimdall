from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import settings
from .models.health import HealthResponse
from .db import get_pool, close_pool
from .routers import sessions

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

# Register routers
app.include_router(sessions.router)


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
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow()
    )


@app.get("/ready")
async def readiness_check():
    # Check database connection
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"ready": True, "database": "connected"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False, "database": "disconnected", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
