import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "api-gateway"


def test_health(client):
    """Test basic health endpoint (liveness probe)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "api-gateway"


def test_health_detailed(client):
    """Test detailed health endpoint with backend service status."""
    response = client.get("/health/detailed")
    # Can be 200 (all healthy) or 503 (some backend services down)
    # API Gateway can still be ready even if backends are down
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data or "message" in data


def test_ready(client):
    """Test readiness probe endpoint."""
    response = client.get("/ready")
    # API Gateway is always ready (it proxies requests even if backends are down)
    assert response.status_code == 200
    assert "ready" in response.json()


def test_startup(client):
    """Test startup probe endpoint."""
    response = client.get("/startup")
    # API Gateway is always ready on startup
    assert response.status_code == 200
    assert "ready" in response.json()
