"""Health check utilities and models."""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime
import asyncio
import structlog

logger = structlog.get_logger()


class HealthStatus(str, Enum):
    """Health status enumeration."""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class DependencyHealth:
    """Health status of a single dependency."""
    name: str
    status: HealthStatus
    response_time_ms: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time_ms": f"{self.response_time_ms:.2f}",
            "error_message": self.error_message,
        }


@dataclass
class HealthCheckResponse:
    """Complete health check response."""
    status: HealthStatus
    service_name: str
    version: str
    timestamp: datetime
    uptime_seconds: int
    dependencies: list[DependencyHealth]
    ready: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "service_name": self.service_name,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "ready": self.ready,
        }


class HealthChecker:
    """Manages health checks for a service."""
    
    def __init__(self, service_name: str, version: str):
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.utcnow()
        self.dependencies: Dict[str, Any] = {}
    
    def register_dependency(self, name: str, checker: callable):
        """Register a dependency health checker."""
        self.dependencies[name] = checker
    
    async def check_dependency(self, name: str) -> DependencyHealth:
        """Check health of a single dependency."""
        checker = self.dependencies.get(name)
        if not checker:
            return DependencyHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                error_message="No checker registered",
            )
        
        try:
            start = datetime.utcnow()
            await checker()
            response_time = (datetime.utcnow() - start).total_seconds() * 1000
            return DependencyHealth(
                name=name,
                status=HealthStatus.UP,
                response_time_ms=response_time,
            )
        except Exception as exc:
            logger.error("dependency_check_failed", dependency=name, error=str(exc))
            return DependencyHealth(
                name=name,
                status=HealthStatus.DOWN,
                response_time_ms=0,
                error_message=str(exc),
            )
    
    async def check_all(self) -> HealthCheckResponse:
        """Check health of all dependencies."""
        tasks = [self.check_dependency(name) for name in self.dependencies]
        dependency_results = await asyncio.gather(*tasks)
        
        # Determine overall status
        statuses = [d.status for d in dependency_results]
        if HealthStatus.DOWN in statuses:
            overall_status = HealthStatus.DOWN
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UP
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        ready = overall_status in [HealthStatus.UP]
        
        return HealthCheckResponse(
            status=overall_status,
            service_name=self.service_name,
            version=self.version,
            timestamp=datetime.utcnow(),
            uptime_seconds=int(uptime),
            dependencies=dependency_results,
            ready=ready,
        )
