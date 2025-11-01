"""Monitor microservices health and broadcast to WebSocket clients."""

import asyncio
import logging
from datetime import datetime

import aiohttp
from celery import shared_task

logger = logging.getLogger(__name__)


async def check_service_health(service_name: str, base_url: str, timeout: int = 5) -> dict:
    """
    Check health of a single microservice.
    
    Args:
        service_name: Name of the service
        base_url: Base URL of the service
        timeout: Request timeout in seconds
        
    Returns:
        dict with status, response_time_ms, and error info
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
                    return {
                        "service": service_name,
                        "status": "healthy",
                        "response_time_ms": round(response_time_ms, 2),
                        "version": data.get("version", "unknown"),
                        "last_check": end_time.isoformat(),
                    }
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


@shared_task(bind=True, name="monitor_services_health")
def monitor_services_health(self):
    """
    Celery task: Check microservices health and broadcast to WebSocket clients.
    Runs periodically (every 30 seconds) to provide real-time status updates.
    """
    try:
        logger.info("Starting microservices health monitoring task")
        
        # Define services to monitor
        services_config = [
            {"name": "backend", "url": "http://backend:8001"},
            {"name": "training", "url": "http://training:8002"},
            {"name": "inference", "url": "http://inference:8003"},
        ]
        
        async def check_all_services():
            """Check health of all services concurrently."""
            tasks = [
                check_service_health(svc["name"], svc["url"])
                for svc in services_config
            ]
            results = await asyncio.gather(*tasks)
            return {result["service"]: result for result in results}
        
        # Run async health checks
        loop = asyncio.get_event_loop()
        health_results = loop.run_until_complete(check_all_services())
        
        logger.info(f"Health check results: {health_results}")
        
        # Broadcast health status via WebSocket (if available)
        try:
            from ..routers.websocket import manager as ws_manager
            
            if ws_manager.active_connections:
                async def broadcast_health():
                    await ws_manager.broadcast({
                        "event": "services:health",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"health_status": health_results},
                    })
                
                loop = asyncio.get_event_loop()
                loop.run_until_complete(broadcast_health())
                logger.info(f"Broadcasted service health to {len(ws_manager.active_connections)} WebSocket clients")
        except Exception as ws_error:
            logger.warning(f"Failed to broadcast health update via WebSocket: {ws_error}")
        
        return {
            "status": "success",
            "services_checked": len(health_results),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.exception(f"Error in services health monitoring: {e}")
        raise
