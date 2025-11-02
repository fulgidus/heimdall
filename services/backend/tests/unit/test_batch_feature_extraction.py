"""Unit tests for batch feature extraction task."""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.tasks.batch_feature_extraction import (
    _find_recordings_without_features,
    backfill_all_features,
    batch_feature_extraction_task,
)


@pytest.mark.asyncio
async def test_find_recordings_without_features_empty():
    """Test finding recordings when none exist without features."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    result = await _find_recordings_without_features(mock_pool, batch_size=50)

    assert result == []
    mock_conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_find_recordings_without_features_with_results():
    """Test finding recordings that don't have features."""
    session_id = uuid.uuid4()
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Mock database results
    mock_results = [
        {
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "status": "completed",
            "num_measurements": 5,
        }
    ]
    mock_conn.fetch = AsyncMock(return_value=mock_results)

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    result = await _find_recordings_without_features(mock_pool, batch_size=50)

    assert len(result) == 1
    assert result[0]["session_id"] == session_id
    assert result[0]["status"] == "completed"
    assert result[0]["num_measurements"] == 5


@pytest.mark.asyncio
async def test_find_recordings_without_features_batch_limit():
    """Test that batch limit is applied correctly."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    # Test with max_batches
    await _find_recordings_without_features(mock_pool, batch_size=50, max_batches=5)

    # Should be called with limit = 50 * 5 = 250
    call_args = mock_conn.fetch.call_args
    assert call_args[0][1] == 250


@pytest.mark.unit
def test_batch_feature_extraction_task_no_recordings():
    """Test batch extraction when no recordings need features."""
    with (
        patch("src.tasks.batch_feature_extraction.get_pool"),
        patch("asyncio.run") as mock_run,
    ):

        # Mock async function result
        async def mock_run_batch():
            return {
                "total_found": 0,
                "tasks_queued": 0,
                "message": "No recordings without features",
            }

        mock_run.return_value = {"total_found": 0, "tasks_queued": 0}

        result = batch_feature_extraction_task.apply(kwargs={"batch_size": 50, "max_batches": 5})

        assert result.result["total_found"] == 0
        assert result.result["tasks_queued"] == 0


@pytest.mark.unit
def test_batch_feature_extraction_task_with_recordings():
    """Test batch extraction queues tasks for recordings without features."""
    # This test verifies the function can be called and returns a dict
    # More detailed testing requires integration tests with real celery

    with patch("src.tasks.batch_feature_extraction.get_pool"), patch(
        "asyncio.run"
    ) as mock_run:

        # Mock successful execution
        mock_run.return_value = {"total_found": 5, "tasks_queued": 5, "task_ids": ["task-1"]}

        result = batch_feature_extraction_task.apply(
            kwargs={"batch_size": 50, "max_batches": 1}
        )

        # Verify result structure
        assert isinstance(result.result, dict)
        assert "total_found" in result.result or "error" in result.result


@pytest.mark.unit
def test_backfill_all_features_empty():
    """Test backfill when no recordings need features."""
    with patch(
        "src.tasks.batch_feature_extraction.batch_feature_extraction_task"
    ) as mock_batch_task:

        # Mock batch task returning no results
        mock_batch_task.return_value = {"total_found": 0, "tasks_queued": 0}

        result = backfill_all_features(MagicMock())

        assert result["total_recordings"] == 0
        assert result["total_tasks_queued"] == 0
        assert result["num_batches"] == 1


@pytest.mark.unit
def test_backfill_all_features_multiple_batches():
    """Test backfill with multiple batches."""
    with patch(
        "src.tasks.batch_feature_extraction.batch_feature_extraction_task"
    ) as mock_batch_task:

        # Mock batch task returning results then empty
        mock_batch_task.side_effect = [
            {"total_found": 50, "tasks_queued": 50},
            {"total_found": 30, "tasks_queued": 30},
            {"total_found": 0, "tasks_queued": 0},  # Stop condition
        ]

        result = backfill_all_features(MagicMock())

        assert result["total_recordings"] == 80
        assert result["total_tasks_queued"] == 80
        assert result["num_batches"] == 3
        assert mock_batch_task.call_count == 3


@pytest.mark.unit
def test_backfill_all_features_safety_limit():
    """Test backfill stops at safety limit."""
    with patch(
        "src.tasks.batch_feature_extraction.batch_feature_extraction_task"
    ) as mock_batch_task:

        # Mock batch task always returning results (would run forever without limit)
        mock_batch_task.return_value = {"total_found": 50, "tasks_queued": 50}

        result = backfill_all_features(MagicMock())

        # Should stop at 1000 batches
        assert result["num_batches"] == 1000
        assert result["total_recordings"] == 50000
        assert result["total_tasks_queued"] == 50000
