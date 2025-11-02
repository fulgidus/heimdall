"""Integration tests for admin API endpoints."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_admin_batch_extract_endpoint():
    """Test manual batch extraction endpoint."""
    with patch("src.routers.admin.batch_feature_extraction_task") as mock_task:
        # Mock task result
        mock_result = MagicMock()
        mock_result.id = "test-task-123"
        mock_task.apply_async.return_value = mock_result

        response = client.post("/api/v1/admin/features/batch-extract?batch_size=10&max_batches=2")

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "test-task-123"
        assert "Batch extraction started" in data["message"]
        assert data["status"] == "queued"

        # Verify task was called with correct parameters
        mock_task.apply_async.assert_called_once_with(
            kwargs={"batch_size": 10, "max_batches": 2}, queue="celery"
        )


def test_admin_batch_extract_default_params():
    """Test batch extraction with default parameters."""
    with patch("src.routers.admin.batch_feature_extraction_task") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "test-task-456"
        mock_task.apply_async.return_value = mock_result

        response = client.post("/api/v1/admin/features/batch-extract")

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "test-task-456"
        assert data["status"] == "queued"

        # Verify default parameters
        mock_task.apply_async.assert_called_once_with(
            kwargs={"batch_size": 50, "max_batches": 5}, queue="celery"
        )


def test_admin_backfill_all_endpoint():
    """Test full backfill endpoint."""
    with patch("src.routers.admin.backfill_all_features") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "backfill-task-789"
        mock_task.apply_async.return_value = mock_result

        response = client.post("/api/v1/admin/features/backfill-all")

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "backfill-task-789"
        assert "Full backfill started" in data["message"]
        assert "may take hours" in data["message"]
        assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_admin_stats_endpoint_with_data():
    """Test stats endpoint with recording data."""
    # Mock database pool
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Mock database result
    mock_result = {
        "total_recordings": 100,
        "recordings_with_features": 85,
        "total_measurements": 700,
    }
    mock_conn.fetchrow = AsyncMock(return_value=mock_result)

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    with patch("src.routers.admin.get_pool", return_value=mock_pool):
        response = client.get("/api/v1/admin/features/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["total_recordings"] == 100
        assert data["recordings_with_features"] == 85
        assert data["recordings_without_features"] == 15
        assert data["total_measurements"] == 700
        assert data["coverage_percent"] == 85.0


@pytest.mark.asyncio
async def test_admin_stats_endpoint_empty_database():
    """Test stats endpoint with no data."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Mock empty result
    mock_conn.fetchrow = AsyncMock(return_value=None)

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    with patch("src.routers.admin.get_pool", return_value=mock_pool):
        response = client.get("/api/v1/admin/features/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["total_recordings"] == 0
        assert data["recordings_with_features"] == 0
        assert data["recordings_without_features"] == 0
        assert data["total_measurements"] == 0
        assert data["coverage_percent"] == 0.0


@pytest.mark.asyncio
async def test_admin_stats_partial_coverage():
    """Test stats endpoint with partial feature coverage."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    mock_result = {
        "total_recordings": 237,
        "recordings_with_features": 145,
        "total_measurements": 1659,
    }
    mock_conn.fetchrow = AsyncMock(return_value=mock_result)

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    with patch("src.routers.admin.get_pool", return_value=mock_pool):
        response = client.get("/api/v1/admin/features/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["total_recordings"] == 237
        assert data["recordings_with_features"] == 145
        assert data["recordings_without_features"] == 92
        assert data["total_measurements"] == 1659
        # 145/237 = 61.18%
        assert abs(data["coverage_percent"] - 61.18) < 0.01


def test_admin_batch_extract_error_handling():
    """Test error handling in batch extract endpoint."""
    with patch("src.routers.admin.batch_feature_extraction_task") as mock_task:
        # Mock task raising an error
        mock_task.apply_async.side_effect = Exception("Celery connection error")

        response = client.post("/api/v1/admin/features/batch-extract")

        assert response.status_code == 500
        assert "Celery connection error" in response.json()["detail"]


def test_admin_backfill_error_handling():
    """Test error handling in backfill endpoint."""
    with patch("src.routers.admin.backfill_all_features") as mock_task:
        mock_task.apply_async.side_effect = Exception("Task queue full")

        response = client.post("/api/v1/admin/features/backfill-all")

        assert response.status_code == 500
        assert "Task queue full" in response.json()["detail"]
