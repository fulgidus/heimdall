"""
Comprehensive system health monitoring task.

Monitors ALL infrastructure components and broadcasts health status:
- PostgreSQL/TimescaleDB
- Redis
- RabbitMQ
- MinIO
- Celery Workers
- WebSDRs
- Backend service
- Training service
- Inference service
"""

import asyncio
import logging
from datetime import datetime
from urllib.parse import urlparse

import aiohttp
import redis
from celery import Celery, shared_task
from minio import Minio
from pika import BlockingConnection, URLParameters

from ..config import settings

logger = logging.getLogger(__name__)


async def check_microservice_health(service_name: str, base_url: str, timeout: int = 5) -> dict:
    """
    Check health of a microservice.
    
    Args:
        service_name: Service name
        base_url: Base URL of the service
        timeout: Request timeout in seconds
        
    Returns:
        dict with status, response_time_ms, version, and error info
    """
    health_url = f"{base_url}/health"
    start_time = datetime.utcnow()
    
    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(health_url) as resp:
                end_time = datetime.utcnow()
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                if resp.status == 200:
                    data = await resp.json()
                    result = {
                        "service": service_name,
                        "status": "healthy",
                        "response_time_ms": round(response_time_ms, 2),
                        "version": data.get("version", "unknown"),
                        "last_check": end_time.isoformat(),
                    }
                    
                    # For inference service, fetch additional model info
                    if service_name == "inference":
                        try:
                            model_info_url = f"{base_url}/api/v1/analytics/model/info"
                            async with session.get(model_info_url, timeout=timeout_obj) as model_resp:
                                if model_resp.status == 200:
                                    model_data = await model_resp.json()
                                    result["model_info"] = {
                                        "active_version": model_data.get("active_version"),
                                        "health_status": model_data.get("health_status"),
                                        "accuracy": model_data.get("accuracy"),
                                        "predictions_total": model_data.get("predictions_total"),
                                        "predictions_successful": model_data.get("predictions_successful"),
                                        "latency_p95_ms": model_data.get("latency_p95_ms"),
                                        "cache_hit_rate": model_data.get("cache_hit_rate"),
                                        "uptime_seconds": model_data.get("uptime_seconds"),
                                    }
                        except Exception as model_error:
                            logger.debug(f"Could not fetch model info for inference service: {model_error}")
                            # Non-critical, continue without model info
                    
                    return result
                else:
                    return {
                        "service": service_name,
                        "status": "unhealthy",
                        "response_time_ms": round(response_time_ms, 2),
                        "error": f"HTTP {resp.status}",
                        "last_check": end_time.isoformat(),
                    }
    except asyncio.TimeoutError:
        return {
            "service": service_name,
            "status": "unhealthy",
            "error": "Timeout",
            "last_check": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Health check failed for {service_name}: {e}")
        return {
            "service": service_name,
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }


async def check_postgresql_health() -> dict:
    """Check PostgreSQL/TimescaleDB health."""
    try:
        from ..db import get_pool
        try:
            pool = get_pool()
        except RuntimeError as re:
            return {
                "service": "postgresql",
                "status": "unhealthy",
                "message": "Database pool not initialized",
                "type": "database",
                "error": str(re),
                "last_check": datetime.utcnow().isoformat(),
            }
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                return {
                    "service": "postgresql",
                    "status": "healthy",
                    "message": "Database connection OK",
                    "type": "database",
                    "last_check": datetime.utcnow().isoformat(),
                }
        return {
            "service": "postgresql",
            "status": "unhealthy",
            "message": "Database query failed",
            "type": "database",
            "last_check": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "service": "postgresql",
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "type": "database",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }


def check_redis_health() -> dict:
    """Check Redis health."""
    try:
        parsed = urlparse(settings.redis_url)
        r = redis.Redis(
            host=parsed.hostname or "redis",
            port=parsed.port or 6379,
            password=parsed.password,
            db=int(parsed.path.lstrip("/")) if parsed.path else 0,
            socket_connect_timeout=2
        )
        r.ping()
        return {
            "service": "redis",
            "status": "healthy",
            "message": "Cache connection OK",
            "type": "cache",
            "last_check": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "service": "redis",
            "status": "unhealthy",
            "message": f"Cache connection failed: {str(e)}",
            "type": "cache",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }


def check_rabbitmq_health() -> dict:
    """Check RabbitMQ health."""
    try:
        connection = BlockingConnection(URLParameters(settings.celery_broker_url))
        channel = connection.channel()
        channel.close()
        connection.close()
        return {
            "service": "rabbitmq",
            "status": "healthy",
            "message": "Message queue connection OK",
            "type": "queue",
            "last_check": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "service": "rabbitmq",
            "status": "unhealthy",
            "message": f"Message queue connection failed: {str(e)}",
            "type": "queue",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }


def check_minio_health() -> dict:
    """Check MinIO health."""
    try:
        parsed = urlparse(settings.minio_url)
        endpoint = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname
        
        client = Minio(
            endpoint=endpoint or "minio:9000",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False
        )
        
        if client.bucket_exists(settings.minio_bucket_raw_iq):
            return {
                "service": "minio",
                "status": "healthy",
                "message": f"Object storage OK, bucket '{settings.minio_bucket_raw_iq}' exists",
                "type": "storage",
                "last_check": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "service": "minio",
                "status": "warning",
                "message": f"Object storage connected but bucket '{settings.minio_bucket_raw_iq}' not found",
                "type": "storage",
                "last_check": datetime.utcnow().isoformat(),
            }
    except Exception as e:
        return {
            "service": "minio",
            "status": "unhealthy",
            "message": f"Object storage connection failed: {str(e)}",
            "type": "storage",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }


def check_celery_health() -> dict:
    """Check Celery workers health."""
    try:
        celery_app = Celery(broker=settings.celery_broker_url)
        inspect = celery_app.control.inspect(timeout=5)
        stats = inspect.stats()
        
        if stats:
            worker_count = len(stats)
            return {
                "service": "celery",
                "status": "healthy",
                "message": f"{worker_count} worker(s) active",
                "type": "worker",
                "worker_count": worker_count,
                "last_check": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "service": "celery",
                "status": "unhealthy",
                "message": "No active workers found",
                "type": "worker",
                "worker_count": 0,
                "last_check": datetime.utcnow().isoformat(),
            }
    except Exception as e:
        return {
            "service": "celery",
            "status": "unhealthy",
            "message": f"Worker check failed: {str(e)}",
            "type": "worker",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }


@shared_task(bind=True, name="monitor_comprehensive_health")
def monitor_comprehensive_health(self):
    """
    Celery task: Check ALL infrastructure and microservices health.
    Broadcasts individual health updates to WebSocket clients via RabbitMQ.
    Runs periodically (every 30 seconds).
    """
    try:
        logger.info("Starting comprehensive health monitoring task")
        
        # Collect all health checks
        health_results = {}
        
        # Define all checks (both sync and async)
        async def run_all_checks():
            # Infrastructure components
            logger.debug("Checking infrastructure components...")
            infra_tasks = [
                check_postgresql_health(),
            ]
            
            # Synchronous checks (wrap in executor)
            loop = asyncio.get_running_loop()
            infra_tasks.extend([
                loop.run_in_executor(None, check_redis_health),
                loop.run_in_executor(None, check_rabbitmq_health),
                loop.run_in_executor(None, check_minio_health),
                loop.run_in_executor(None, check_celery_health),
            ])
            
            # Microservices
            logger.debug("Checking microservices...")
            services_config = [
                {"name": "backend", "url": "http://backend:8001"},
                {"name": "training", "url": "http://training:8002"},
                {"name": "inference", "url": "http://inference:8003"},
            ]
            
            microservice_tasks = [
                check_microservice_health(svc["name"], svc["url"])
                for svc in services_config
            ]
            
            # Run all checks concurrently
            all_results = await asyncio.gather(*infra_tasks, *microservice_tasks)
            return all_results
        
        all_results = asyncio.run(run_all_checks())
        
        for result in all_results:
            health_results[result["service"]] = result
        
        logger.info(f"Health check completed for {len(health_results)} components")
        
        # Publish comprehensive health update as single aggregated event + individual updates
        try:
            from ..events.publisher import get_event_publisher
            
            publisher = get_event_publisher()
            
            # Publish aggregated comprehensive health (all components at once)
            publisher.publish_comprehensive_health(health_results)
            
            # Also publish individual health updates for backwards compatibility
            for service_name, health_data in health_results.items():
                publisher.publish_service_health(
                    service_name=service_name,
                    status=health_data["status"],
                    response_time_ms=health_data.get("response_time_ms"),
                    error=health_data.get("error"),
                    version=health_data.get("version"),
                    last_check=health_data.get("last_check"),
                    message=health_data.get("message"),
                    type=health_data.get("type"),
                    model_info=health_data.get("model_info"),  # For inference service
                    worker_count=health_data.get("worker_count"),  # For celery
                    online_count=health_data.get("online_count"),  # For websdrs
                    total_count=health_data.get("total_count"),  # For websdrs
                )
            
            logger.debug(f"Published comprehensive + individual health updates for {len(health_results)} components")
        
        except Exception as publish_error:
            logger.warning(f"Failed to publish health updates to RabbitMQ: {publish_error}")
        
        return {
            "status": "success",
            "components_checked": len(health_results),
            "healthy": sum(1 for h in health_results.values() if h["status"] == "healthy"),
            "unhealthy": sum(1 for h in health_results.values() if h["status"] == "unhealthy"),
            "warning": sum(1 for h in health_results.values() if h["status"] == "warning"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.exception(f"Error in comprehensive health monitoring: {e}")
        raise
