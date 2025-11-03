import logging
import os
import sys
import threading
from datetime import datetime

from celery import Celery
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for common module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from common.dependency_checkers import check_celery, check_minio, check_postgresql, check_redis

# Import common health utilities
from common.health import HealthChecker

from .config import settings
from .db import close_pool, init_pool
from .models.health import HealthResponse
from .routers.acquisition import router as acquisition_router
from .routers.admin import router as admin_router
from .routers.health import router as health_router
from .routers.import_export import router as import_export_router
from .routers.sessions import router as sessions_router
from .routers.settings import router as settings_router
from .routers.terrain import router as terrain_router
from .routers.training import router as training_router
from .routers.users import router as users_router
from .routers.websocket import router as websocket_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "backend"
SERVICE_VERSION = "0.1.0"
SERVICE_PORT = 8001

# Initialize FastAPI app
app = FastAPI(
    title=f"Heimdall SDR - {SERVICE_NAME}",
    version=SERVICE_VERSION,
    description="RF data acquisition service for Heimdall SDR platform",
)

# Configure CORS with environment-based settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.get_cors_methods_list(),
    allow_headers=settings.get_cors_headers_list(),
    expose_headers=settings.get_cors_expose_headers_list(),
    max_age=settings.cors_max_age,
)

# Initialize Celery
celery_app = Celery(
    SERVICE_NAME, broker=settings.celery_broker_url, backend=settings.celery_result_backend_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

# Configure Celery Beat schedule for monitoring
celery_app.conf.beat_schedule = {
    "monitor-websdrs-uptime": {
        "task": "monitor_websdrs_uptime",
        "schedule": 60.0,  # Every 60 seconds
    },
    "monitor-comprehensive-health": {
        "task": "monitor_comprehensive_health",
        "schedule": 30.0,  # Every 30 seconds
    },
    "batch-feature-extraction": {
        "task": "backend.tasks.batch_feature_extraction",
        "schedule": 300.0,  # Every 5 minutes
        "kwargs": {
            "batch_size": 50,
            "max_batches": 5  # Max 250 recordings per run
        }
    },
}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool and RabbitMQ listener on startup."""
    logger.info("Initializing database connection pool...")
    try:
        await init_pool()
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise

    # Start RabbitMQ listener for progress events in background thread
    logger.info("Starting RabbitMQ progress event listener...")
    listener_thread = threading.Thread(target=rabbitmq_progress_listener, daemon=True)
    listener_thread.start()
    logger.info("RabbitMQ listener thread started")

    # Start RabbitMQ event consumer for WebSocket broadcasting
    try:
        logger.info("Starting RabbitMQ events consumer for WebSocket broadcasting...")
        from .events.consumer import start_rabbitmq_consumer
        from .routers.websocket import manager as websocket_manager

        consumer_thread = threading.Thread(
            target=lambda: asyncio.run(start_rabbitmq_consumer(settings.celery_broker_url, websocket_manager)),
            daemon=True
        )
        consumer_thread.start()
        logger.info("RabbitMQ events consumer thread started")
    except Exception as e:
        logger.error(f"Failed to start RabbitMQ events consumer: {e}", exc_info=True)


def rabbitmq_progress_listener():
    """Listen for progress events from Celery tasks via RabbitMQ (runs in separate thread)."""
    import json

    from kombu import Connection, Exchange, Queue

    logger.info("RabbitMQ listener thread: started listening for progress events")

    try:
        connection = Connection(settings.celery_broker_url)
        exchange = Exchange("heimdall", type="topic", durable=True)
        queue = Queue(
            "progress_updates",
            exchange=exchange,
            routing_key="session.*.progress",
            durable=True,
            auto_delete=False,
        )

        def process_event(body, message):
            """Process incoming progress event and broadcast to WebSocket clients."""
            try:
                if isinstance(body, bytes):
                    body = body.decode("utf-8")

                event_data = json.loads(body) if isinstance(body, str) else body
                logger.debug(f"RabbitMQ listener: received event: {event_data.get('event', '?')}")

                # Schedule broadcast in FastAPI event loop
                import asyncio

                loop = asyncio.new_event_loop()

                async def broadcast():
                    from .routers.websocket import manager

                    await manager.broadcast(event_data)

                # Try to get existing event loop, else create new one
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.ensure_future(broadcast())
                except RuntimeError:
                    # No running loop, schedule it asynchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(broadcast())
                    loop.close()

                message.ack()

            except Exception as e:
                logger.error(f"RabbitMQ listener: error processing event: {e}")
                message.nack()

        with connection.Consumer(queue, callbacks=[process_event], auto_declare=True):
            logger.info("RabbitMQ listener: listening for messages...")
            while True:
                try:
                    connection.drain_events(timeout=1)
                except KeyboardInterrupt:
                    logger.info("RabbitMQ listener: interrupted by user")
                    break
                except Exception as e:
                    logger.warning(f"RabbitMQ listener: error during drain_events: {e}")
                    break

    except Exception as e:
        logger.error(f"RabbitMQ listener: fatal error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection pool on shutdown."""
    logger.info("Closing database connection pool...")
    try:
        await close_pool()
        logger.info("Database connection pool closed successfully")
    except Exception as e:
        logger.error(f"Failed to close database pool: {e}")


# Include routers
app.include_router(acquisition_router)
app.include_router(admin_router)
app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(settings_router)
app.include_router(terrain_router)
app.include_router(training_router)
app.include_router(users_router)
app.include_router(websocket_router)
app.include_router(import_export_router)

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
        settings.minio_url, settings.minio_access_key, settings.minio_secret_key, secure=False
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
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check():
    """
    Liveness probe - checks if service is alive.

    Returns basic health status without dependency checks.
    Used by Kubernetes liveness probe.
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
            critical_status = [d.status for d in result.dependencies if d.name in critical_deps]

            from common.health import HealthStatus

            if HealthStatus.DOWN in critical_status:
                return JSONResponse(
                    status_code=503,
                    content={
                        "ready": False,
                        "service": SERVICE_NAME,
                        "dependencies": result.to_dict()["dependencies"],
                    },
                )

            return JSONResponse(
                status_code=200,
                content={
                    "ready": True,
                    "service": SERVICE_NAME,
                    "dependencies": result.to_dict()["dependencies"],
                },
            )

        # Production mode: all dependencies must be healthy
        return JSONResponse(
            status_code=200 if result.ready else 503,
            content={
                "ready": result.ready,
                "service": SERVICE_NAME,
                "dependencies": result.to_dict()["dependencies"],
            },
        )
    except Exception as e:
        logger.error("Readiness check failed: %s", str(e), exc_info=True)
        return JSONResponse(
            status_code=503, content={"ready": False, "service": SERVICE_NAME, "error": str(e)}
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
