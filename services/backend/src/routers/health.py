"""
System health check endpoints
Provides detailed status of all backend components
"""

import logging
from typing import Any
from urllib.parse import urlparse

import redis
from celery import Celery
from fastapi import APIRouter, Response
from minio import Minio
from pika import BlockingConnection, URLParameters

from ..config import settings
from ..db import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("/detailed")
async def get_detailed_health():
    """
    Get detailed health status of all system components
    
    Returns:
        - PostgreSQL/TimescaleDB status
        - Redis cache status
        - RabbitMQ message queue status
        - MinIO object storage status
        - Celery worker status
        - WebSDR connectivity
        - Overall system ready status
    """
    components: dict[str, dict[str, Any]] = {}
    overall_ready = True
    
    # Check PostgreSQL/TimescaleDB
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                components["postgresql"] = {
                    "status": "healthy",
                    "message": "Database connection OK",
                    "type": "database"
                }
            else:
                components["postgresql"] = {
                    "status": "unhealthy",
                    "message": "Database query failed",
                    "type": "database"
                }
                overall_ready = False
    except Exception as e:
        components["postgresql"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "type": "database"
        }
        overall_ready = False
    
    # Check Redis
    try:
        # Parse Redis URL: redis://:password@host:port/db
        parsed = urlparse(settings.redis_url)
        r = redis.Redis(
            host=parsed.hostname or "redis",
            port=parsed.port or 6379,
            password=parsed.password,
            db=int(parsed.path.lstrip("/")) if parsed.path else 0,
            socket_connect_timeout=2
        )
        r.ping()
        components["redis"] = {
            "status": "healthy",
            "message": "Cache connection OK",
            "type": "cache"
        }
    except Exception as e:
        components["redis"] = {
            "status": "unhealthy",
            "message": f"Cache connection failed: {str(e)}",
            "type": "cache"
        }
        overall_ready = False
    
    # Check RabbitMQ
    try:
        connection = BlockingConnection(URLParameters(settings.celery_broker_url))
        channel = connection.channel()
        channel.close()
        connection.close()
        components["rabbitmq"] = {
            "status": "healthy",
            "message": "Message queue connection OK",
            "type": "queue"
        }
    except Exception as e:
        components["rabbitmq"] = {
            "status": "unhealthy",
            "message": f"Message queue connection failed: {str(e)}",
            "type": "queue"
        }
        overall_ready = False
    
    # Check MinIO
    try:
        # Parse MinIO URL to get host and port
        parsed = urlparse(settings.minio_url)
        endpoint = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname

        client = Minio(
            endpoint=endpoint or "minio:9000",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False
        )

        # Check if bucket exists
        if client.bucket_exists(settings.minio_bucket_raw_iq):
            components["minio"] = {
                "status": "healthy",
                "message": f"Object storage OK, bucket '{settings.minio_bucket_raw_iq}' exists",
                "type": "storage"
            }
        else:
            components["minio"] = {
                "status": "warning",
                "message": f"Object storage connected but bucket '{settings.minio_bucket_raw_iq}' not found",
                "type": "storage"
            }
    except Exception as e:
        components["minio"] = {
            "status": "unhealthy",
            "message": f"Object storage connection failed: {str(e)}",
            "type": "storage"
        }
        overall_ready = False
    
    # Check Celery Workers
    try:
        celery_app = Celery(broker=settings.celery_broker_url)
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if stats:
            worker_count = len(stats)
            components["celery_worker"] = {
                "status": "healthy",
                "message": f"{worker_count} worker(s) active",
                "type": "worker",
                "worker_count": worker_count
            }
        else:
            components["celery_worker"] = {
                "status": "unhealthy",
                "message": "No active workers found",
                "type": "worker",
                "worker_count": 0
            }
            overall_ready = False
    except Exception as e:
        components["celery_worker"] = {
            "status": "unhealthy",
            "message": f"Worker check failed: {str(e)}",
            "type": "worker"
        }
        overall_ready = False
    
    # Check WebSDR connectivity (check recent uptime records)
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            online_count = await conn.fetchval("""
                SELECT COUNT(DISTINCT websdr_id)
                FROM heimdall.websdrs_uptime_history
                WHERE timestamp > NOW() - INTERVAL '5 minutes'
                AND status = 'online'
            """)

            total_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM heimdall.websdr_stations
            """)

            if online_count and online_count > 0:
                components["websdrs"] = {
                    "status": "healthy",
                    "message": f"{online_count}/{total_count} WebSDRs online in last 5 minutes",
                    "type": "receiver",
                    "online_count": online_count,
                    "total_count": total_count
                }
            elif total_count == 0:
                components["websdrs"] = {
                    "status": "warning",
                    "message": "No WebSDRs configured",
                    "type": "receiver",
                    "online_count": 0,
                    "total_count": 0
                }
            else:
                components["websdrs"] = {
                    "status": "unknown",
                    "message": f"No recent data (0/{total_count} online)",
                    "type": "receiver",
                    "online_count": 0,
                    "total_count": total_count
                }
    except Exception as e:
        components["websdrs"] = {
            "status": "unknown",
            "message": f"WebSDR status check failed: {str(e)}",
            "type": "receiver"
        }
    
    return {
        "ready": overall_ready,
        "components": components,
        "summary": {
            "total": len(components),
            "healthy": sum(1 for c in components.values() if c["status"] == "healthy"),
            "unhealthy": sum(1 for c in components.values() if c["status"] == "unhealthy"),
            "warning": sum(1 for c in components.values() if c["status"] == "warning"),
            "unknown": sum(1 for c in components.values() if c["status"] == "unknown"),
        }
    }


@router.get("/ready")
async def get_ready(response: Response):
    """
    Kubernetes-style readiness probe
    Returns 200 if system is ready, 503 if not
    """
    health = await get_detailed_health()
    if not health["ready"]:
        response.status_code = 503
    return {"ready": health["ready"]}


@router.get("/liveness")
async def get_liveness():
    """
    Kubernetes-style liveness probe
    Always returns 200 if the service is running
    """
    return {"alive": True}
