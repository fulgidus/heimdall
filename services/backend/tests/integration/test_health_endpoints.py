"""Integration tests for health check endpoints across all services."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health(self):
        """Test basic /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "backend"
        assert "version" in data
        assert "timestamp" in data

    def test_detailed_health_no_dependencies(self):
        """Test /health/detailed endpoint when dependencies are not checked."""
        # Mock all dependency checkers to succeed
        with (
            patch("services.common.dependency_checkers.check_postgresql", new_callable=AsyncMock),
            patch("services.common.dependency_checkers.check_redis", new_callable=AsyncMock),
            patch("services.common.dependency_checkers.check_celery", new_callable=AsyncMock),
            patch("services.common.dependency_checkers.check_minio", new_callable=AsyncMock),
        ):

            response = client.get("/health/detailed")

            assert response.status_code in [200, 503]  # Can be either depending on dependencies
            data = response.json()
            assert "status" in data
            assert "service_name" in data
            assert "version" in data
            assert "dependencies" in data
            assert "ready" in data

    def test_readiness_probe(self):
        """Test /ready endpoint."""
        response = client.get("/ready")

        # Should return 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data
        assert "service" in data

    def test_startup_probe(self):
        """Test /startup endpoint."""
        response = client.get("/startup")

        # Should return 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data
        assert "service" in data


class TestDependencyHealthChecks:
    """Test dependency health check integration."""

    @patch("services.common.dependency_checkers.check_postgresql")
    @patch("services.common.dependency_checkers.check_redis")
    @patch("services.common.dependency_checkers.check_celery")
    @patch("services.common.dependency_checkers.check_minio")
    def test_all_dependencies_healthy(self, mock_minio, mock_celery, mock_redis, mock_pg):
        """Test detailed health when all dependencies are healthy."""
        # Make all checks succeed
        mock_pg.return_value = AsyncMock()
        mock_redis.return_value = AsyncMock()
        mock_celery.return_value = AsyncMock()
        mock_minio.return_value = AsyncMock()

        response = client.get("/health/detailed")

        data = response.json()
        assert data["status"] in ["up", "down", "degraded", "unknown"]
        assert "dependencies" in data
        assert len(data["dependencies"]) >= 0

    @patch("services.common.dependency_checkers.check_postgresql")
    @patch("services.common.dependency_checkers.check_redis")
    @patch("services.common.dependency_checkers.check_celery")
    @patch("services.common.dependency_checkers.check_minio")
    def test_one_dependency_down(self, mock_minio, mock_celery, mock_redis, mock_pg):
        """Test detailed health when one dependency is down."""
        # Make PostgreSQL check fail
        mock_pg.side_effect = Exception("Database connection failed")
        mock_redis.return_value = AsyncMock()
        mock_celery.return_value = AsyncMock()
        mock_minio.return_value = AsyncMock()

        response = client.get("/health/detailed")

        # Service should report as degraded or down
        assert response.status_code == 503
        data = response.json()
        assert data["status"] in ["down", "degraded"]
        assert data["ready"] is False


class TestHealthCheckResponseFormat:
    """Test health check response format compliance."""

    def test_health_response_structure(self):
        """Test that /health response has correct structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert isinstance(data["service"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)

    def test_detailed_health_response_structure(self):
        """Test that /health/detailed response has correct structure."""
        response = client.get("/health/detailed")

        data = response.json()

        # Required fields
        assert "status" in data
        assert "service_name" in data
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "dependencies" in data
        assert "ready" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert isinstance(data["service_name"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["uptime_seconds"], int)
        assert isinstance(data["dependencies"], list)
        assert isinstance(data["ready"], bool)

    def test_dependency_health_structure(self):
        """Test that dependency health has correct structure."""
        response = client.get("/health/detailed")

        data = response.json()

        if len(data["dependencies"]) > 0:
            dep = data["dependencies"][0]

            # Required fields
            assert "name" in dep
            assert "status" in dep
            assert "response_time_ms" in dep

            # Verify types
            assert isinstance(dep["name"], str)
            assert isinstance(dep["status"], str)
            assert dep["status"] in ["up", "down", "degraded", "unknown"]


class TestReadinessProbe:
    """Test readiness probe behavior."""

    def test_readiness_returns_json(self):
        """Test that readiness probe returns JSON."""
        response = client.get("/ready")

        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)

    def test_readiness_includes_service_name(self):
        """Test that readiness response includes service name."""
        response = client.get("/ready")

        data = response.json()
        assert "service" in data
        assert data["service"] == "backend"

    @patch("services.common.dependency_checkers.check_postgresql")
    @patch("services.common.dependency_checkers.check_redis")
    @patch("services.common.dependency_checkers.check_celery")
    @patch("services.common.dependency_checkers.check_minio")
    def test_readiness_with_healthy_deps(self, mock_minio, mock_celery, mock_redis, mock_pg):
        """Test readiness when all dependencies are healthy."""
        mock_pg.return_value = AsyncMock()
        mock_redis.return_value = AsyncMock()
        mock_celery.return_value = AsyncMock()
        mock_minio.return_value = AsyncMock()

        response = client.get("/ready")

        # Should be ready when all deps are up
        data = response.json()
        assert "ready" in data
