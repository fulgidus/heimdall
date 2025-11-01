"""Integration tests for acquisition API endpoints."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_acquisition_health():
    """Test health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "backend"
    assert "version" in data


def test_acquisition_config():
    """Test configuration endpoint."""
    response = client.get("/api/v1/acquisition/config")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "backend"
    assert "capabilities" in data
    assert "simultaneous-acquisition" in data["capabilities"]
    assert data["default_sample_rate_khz"] == 12.5


def test_list_websdrs():
    """Test WebSDR list endpoint."""
    response = client.get("/api/v1/acquisition/websdrs")

    assert response.status_code == 200
    websdrs = response.json()

    assert len(websdrs) == 7  # Default set of 7
    assert all("id" in ws for ws in websdrs)
    assert all("name" in ws for ws in websdrs)
    assert all("url" in ws for ws in websdrs)
    assert all("latitude" in ws for ws in websdrs)
    assert all("longitude" in ws for ws in websdrs)


def test_trigger_acquisition():
    """Test acquisition trigger endpoint."""
    request_data = {
        "frequency_mhz": 145.5,
        "duration_seconds": 10,
        "start_time": datetime.utcnow().isoformat(),
        "websdrs": None,
    }

    response = client.post("/api/v1/acquisition/acquire", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "task_id" in data
    assert data["status"] == "PENDING"
    assert data["frequency_mhz"] == 145.5
    assert data["websdrs_count"] == 7
    assert "task_id" in data


def test_trigger_acquisition_specific_websdrs():
    """Test acquisition with specific WebSDRs."""
    request_data = {
        "frequency_mhz": 145.5,
        "duration_seconds": 10,
        "start_time": datetime.utcnow().isoformat(),
        "websdrs": [1, 2, 3],
    }

    response = client.post("/api/v1/acquisition/acquire", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["websdrs_count"] == 3


def test_trigger_acquisition_invalid_frequency():
    """Test acquisition with invalid frequency."""
    request_data = {
        "frequency_mhz": -50,  # Invalid
        "duration_seconds": 10,
        "start_time": datetime.utcnow().isoformat(),
    }

    response = client.post("/api/v1/acquisition/acquire", json=request_data)

    assert response.status_code == 422  # Validation error


def test_trigger_acquisition_invalid_duration():
    """Test acquisition with too long duration."""
    request_data = {
        "frequency_mhz": 145.5,
        "duration_seconds": 600,  # > 300 seconds
        "start_time": datetime.utcnow().isoformat(),
    }

    response = client.post("/api/v1/acquisition/acquire", json=request_data)

    assert response.status_code == 422  # Validation error


def test_get_acquisition_status():
    """Test acquisition status endpoint."""
    # Mock Celery AsyncResult
    with patch("src.routers.acquisition.AsyncResult") as mock_result_class:
        # Create mock result object
        mock_result = MagicMock()
        mock_result.state = "PROGRESS"
        mock_result.info = {"progress": 50, "status": "Processing...", "successful": 3}
        mock_result.result = None
        mock_result_class.return_value = mock_result

        # Test getting status
        status_response = client.get("/api/v1/acquisition/status/test-task-id")

        assert status_response.status_code == 200
        data = status_response.json()

        assert data["task_id"] == "test-task-id"
        assert data["status"] in ["PROGRESS", "PENDING", "SUCCESS", "FAILURE"]
        assert "progress" in data
        assert "message" in data
        assert data["progress"] >= 0
        assert data["progress"] <= 100


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "backend"
    assert data["status"] == "running"
    assert "timestamp" in data


def test_readiness_endpoint():
    """Test readiness endpoint."""
    response = client.get("/ready")

    # Should return either 200 or 503 depending on Celery availability
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["ready"] is True
