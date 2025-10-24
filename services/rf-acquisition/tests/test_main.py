import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "rf-acquisition"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready(client):
    response = client.get("/ready")
    # Allow both 200 (ready) and 503 (service unavailable) for Celery
    assert response.status_code in [200, 503]
    assert "ready" in response.json()
