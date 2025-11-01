"""Tests for health check utilities."""

import asyncio
from datetime import datetime

import pytest

from services.common.health import (
    DependencyHealth,
    HealthChecker,
    HealthCheckResponse,
    HealthStatus,
)


def test_health_status_enum():
    """Test HealthStatus enum values."""
    assert HealthStatus.UP == "up"
    assert HealthStatus.DOWN == "down"
    assert HealthStatus.DEGRADED == "degraded"
    assert HealthStatus.UNKNOWN == "unknown"


def test_dependency_health_to_dict():
    """Test DependencyHealth to_dict conversion."""
    dep = DependencyHealth(
        name="postgres",
        status=HealthStatus.UP,
        response_time_ms=15.5,
        error_message=None,
    )

    result = dep.to_dict()

    assert result["name"] == "postgres"
    assert result["status"] == "up"
    assert result["response_time_ms"] == "15.50"
    assert result["error_message"] is None


def test_dependency_health_with_error():
    """Test DependencyHealth with error message."""
    dep = DependencyHealth(
        name="redis",
        status=HealthStatus.DOWN,
        response_time_ms=0,
        error_message="Connection refused",
    )

    result = dep.to_dict()

    assert result["status"] == "down"
    assert result["error_message"] == "Connection refused"


def test_health_check_response_to_dict():
    """Test HealthCheckResponse to_dict conversion."""
    deps = [
        DependencyHealth("postgres", HealthStatus.UP, 10.0),
        DependencyHealth("redis", HealthStatus.UP, 5.0),
    ]

    response = HealthCheckResponse(
        status=HealthStatus.UP,
        service_name="test-service",
        version="1.0.0",
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        uptime_seconds=3600,
        dependencies=deps,
        ready=True,
    )

    result = response.to_dict()

    assert result["status"] == "up"
    assert result["service_name"] == "test-service"
    assert result["version"] == "1.0.0"
    assert result["uptime_seconds"] == 3600
    assert result["ready"] is True
    assert len(result["dependencies"]) == 2


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_init(self):
        """Test HealthChecker initialization."""
        checker = HealthChecker("test-service", "1.0.0")

        assert checker.service_name == "test-service"
        assert checker.version == "1.0.0"
        assert len(checker.dependencies) == 0
        assert isinstance(checker.start_time, datetime)

    def test_register_dependency(self):
        """Test registering a dependency checker."""
        checker = HealthChecker("test-service", "1.0.0")

        async def mock_checker():
            pass

        checker.register_dependency("postgres", mock_checker)

        assert "postgres" in checker.dependencies
        assert checker.dependencies["postgres"] == mock_checker

    @pytest.mark.asyncio
    async def test_check_dependency_success(self):
        """Test successful dependency check."""
        checker = HealthChecker("test-service", "1.0.0")

        async def mock_checker():
            await asyncio.sleep(0.01)  # Simulate some work

        checker.register_dependency("postgres", mock_checker)

        result = await checker.check_dependency("postgres")

        assert result.name == "postgres"
        assert result.status == HealthStatus.UP
        assert result.response_time_ms > 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_check_dependency_failure(self):
        """Test failed dependency check."""
        checker = HealthChecker("test-service", "1.0.0")

        async def failing_checker():
            raise ConnectionError("Database connection failed")

        checker.register_dependency("postgres", failing_checker)

        result = await checker.check_dependency("postgres")

        assert result.name == "postgres"
        assert result.status == HealthStatus.DOWN
        assert result.response_time_ms == 0
        assert "Database connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_check_dependency_unknown(self):
        """Test checking unregistered dependency."""
        checker = HealthChecker("test-service", "1.0.0")

        result = await checker.check_dependency("unknown")

        assert result.name == "unknown"
        assert result.status == HealthStatus.UNKNOWN
        assert "No checker registered" in result.error_message

    @pytest.mark.asyncio
    async def test_check_all_success(self):
        """Test checking all dependencies when all are healthy."""
        checker = HealthChecker("test-service", "1.0.0")

        async def postgres_checker():
            pass

        async def redis_checker():
            pass

        checker.register_dependency("postgres", postgres_checker)
        checker.register_dependency("redis", redis_checker)

        response = await checker.check_all()

        assert response.status == HealthStatus.UP
        assert response.service_name == "test-service"
        assert response.version == "1.0.0"
        assert response.ready is True
        assert len(response.dependencies) == 2
        assert all(d.status == HealthStatus.UP for d in response.dependencies)

    @pytest.mark.asyncio
    async def test_check_all_one_down(self):
        """Test checking all dependencies when one is down."""
        checker = HealthChecker("test-service", "1.0.0")

        async def postgres_checker():
            pass

        async def redis_checker():
            raise ConnectionError("Redis down")

        checker.register_dependency("postgres", postgres_checker)
        checker.register_dependency("redis", redis_checker)

        response = await checker.check_all()

        assert response.status == HealthStatus.DOWN
        assert response.ready is False
        assert len(response.dependencies) == 2

        # Find the failed dependency
        redis_dep = next(d for d in response.dependencies if d.name == "redis")
        assert redis_dep.status == HealthStatus.DOWN

    @pytest.mark.asyncio
    async def test_check_all_empty_dependencies(self):
        """Test checking all when no dependencies registered."""
        checker = HealthChecker("test-service", "1.0.0")

        response = await checker.check_all()

        assert response.status == HealthStatus.UP
        assert response.ready is True
        assert len(response.dependencies) == 0

    @pytest.mark.asyncio
    async def test_uptime_calculation(self):
        """Test uptime calculation."""
        checker = HealthChecker("test-service", "1.0.0")

        # Wait a bit to accumulate some uptime
        await asyncio.sleep(0.1)

        response = await checker.check_all()

        assert response.uptime_seconds >= 0
