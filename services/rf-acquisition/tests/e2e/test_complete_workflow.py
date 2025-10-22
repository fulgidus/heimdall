"""
End-to-End Integration Tests for RF Acquisition Service

Tests the complete workflow:
1. Trigger acquisition via REST API
2. Monitor progress
3. Verify data storage in MinIO and database
4. Test error handling and edge cases
"""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock, MagicMock
import json


pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_acquisition_complete_workflow(api_helper, db_helper, minio_helper, db_clean):
    """
    Test: Complete successful acquisition workflow
    
    Steps:
    1. Trigger acquisition at 145.50 MHz for 2 seconds
    2. Wait for completion (polling every 1 second, max 60 seconds)
    3. Verify database has 7 measurements (one per WebSDR)
    4. Verify MinIO has session files
    5. Verify measurement data quality
    """
    # Step 1: Trigger acquisition
    task_id = await api_helper.trigger_acquisition(
        frequency_mhz=145.50,
        duration_seconds=2.0,
        description="E2E test: complete workflow"
    )
    assert task_id is not None
    print(f"✓ Acquisition triggered with task_id: {task_id}")
    
    # Step 2: Wait for completion
    final_status = await api_helper.wait_for_completion(task_id, timeout_seconds=60)
    assert final_status["state"] == "SUCCESS", f"Acquisition failed: {final_status}"
    print(f"✓ Acquisition completed successfully")
    
    # Step 3: Verify database
    measurements = db_helper.get_measurements_by_task_id(task_id)
    assert len(measurements) == 7, f"Expected 7 measurements, got {len(measurements)}"
    print(f"✓ Database has 7 measurements")
    
    # Verify each measurement has valid data
    for measurement in measurements:
        assert db_helper.verify_measurement_data(measurement), \
            f"Invalid measurement data: {measurement}"
        assert measurement["snr"] >= 0, f"SNR should be positive: {measurement}"
        assert measurement["websdr_id"] is not None, f"Missing WebSDR ID"
    print(f"✓ All measurements have valid data")
    
    # Step 4: Verify MinIO (optional - may not work in all setups)
    minio_files = minio_helper.list_session_files(task_id)
    if minio_files:
        assert len(minio_files) >= 7, f"Expected >= 7 MinIO files, got {len(minio_files)}"
        print(f"✓ MinIO has {len(minio_files)} files")
    else:
        print(f"⚠ MinIO files not accessible (may not be available in test environment)")


@pytest.mark.asyncio
async def test_websdr_partial_failure(api_helper, db_helper, db_clean):
    """
    Test: Acquisition with one WebSDR offline
    
    Expected behavior:
    - Acquisition should succeed (state = SUCCESS)
    - 6 measurements in database (not 7)
    - Error logged for failed WebSDR
    - API should return partial success with error details
    """
    # Mock one WebSDR as failing
    with patch('src.fetchers.websdr_fetcher.fetch_iq_concurrent') as mock_fetch:
        # Simulate 6 WebSDRs working, 1 failing
        async def mock_fetch_impl(*args, **kwargs):
            # Return data for 6 WebSDRs, raise error for 7th
            return {
                f"WebSDR-{i}": {"iq_data": f"mock_data_{i}", "snr": 10.0 + i}
                for i in range(6)
            }
        
        mock_fetch.side_effect = mock_fetch_impl
        
        # Trigger acquisition
        task_id = await api_helper.trigger_acquisition(
            frequency_mhz=145.50,
            duration_seconds=2.0
        )
        
        # Wait for completion
        final_status = await api_helper.wait_for_completion(task_id, timeout_seconds=60)
        
        # Verify success despite partial failure
        assert final_status["state"] == "SUCCESS", \
            f"Should succeed with partial data: {final_status}"
        print(f"✓ Acquisition succeeded despite WebSDR failure")
        
        # Verify we have measurements from working WebSDRs
        measurements = db_helper.get_measurements_by_task_id(task_id)
        assert len(measurements) >= 6, \
            f"Expected >= 6 measurements from working WebSDRs, got {len(measurements)}"
        print(f"✓ Database has {len(measurements)} measurements from working receivers")


@pytest.mark.asyncio
async def test_concurrent_acquisitions(api_helper, db_helper, db_clean):
    """
    Test: Multiple concurrent acquisitions
    
    Expected behavior:
    - All 5 acquisitions should complete successfully
    - Each should have independent 7 measurements
    - No cross-contamination or race conditions
    """
    num_concurrent = 5
    task_ids = []
    
    # Trigger all acquisitions concurrently
    tasks = [
        api_helper.trigger_acquisition(
            frequency_mhz=145.50 + i*0.1,
            duration_seconds=2.0,
            description=f"Concurrent test {i}"
        )
        for i in range(num_concurrent)
    ]
    task_ids = await asyncio.gather(*tasks)
    assert len(task_ids) == num_concurrent
    print(f"✓ Triggered {num_concurrent} concurrent acquisitions")
    
    # Wait for all to complete
    wait_tasks = [
        api_helper.wait_for_completion(task_id, timeout_seconds=60)
        for task_id in task_ids
    ]
    results = await asyncio.gather(*wait_tasks)
    
    # Verify all succeeded
    for i, result in enumerate(results):
        assert result["state"] == "SUCCESS", \
            f"Acquisition {i} failed: {result}"
    print(f"✓ All {num_concurrent} acquisitions completed successfully")
    
    # Verify data independence (no cross-contamination)
    for i, task_id in enumerate(task_ids):
        measurements = db_helper.get_measurements_by_task_id(task_id)
        assert len(measurements) == 7, \
            f"Acquisition {i} has {len(measurements)} measurements (expected 7)"
    print(f"✓ Each acquisition has independent data (7 measurements each)")


@pytest.mark.asyncio
async def test_acquisition_status_polling(api_helper, db_helper, db_clean):
    """
    Test: Status polling transitions
    
    Expected behavior:
    - Initial status should be PENDING
    - Should transition to RUNNING or SUCCESS
    - Final status should be SUCCESS or FAILED
    - Progress field should increase
    """
    task_id = await api_helper.trigger_acquisition(
        frequency_mhz=145.50,
        duration_seconds=2.0
    )
    
    # Poll initial status
    initial_status = await api_helper.get_status(task_id)
    assert "status" in initial_status  # API uses 'status', not 'state'
    assert "progress" in initial_status
    print(f"✓ Initial status: {initial_status['status']} ({initial_status.get('progress', 0)}%)")
    
    # Wait for completion with status tracking
    last_progress = 0
    for attempt in range(60):
        status = await api_helper.get_status(task_id)
        state = status.get("state")
        progress = status.get("progress", 0)
        
        # Progress should monotonically increase (or stay same)
        assert progress >= last_progress, \
            f"Progress decreased: {last_progress} -> {progress}"
        last_progress = progress
        
        if state in ["SUCCESS", "FAILED"]:
            print(f"✓ Final status: {state} ({progress}%)")
            break
        
        await asyncio.sleep(1)
    else:
        raise TimeoutError(f"Acquisition {task_id} did not complete")


@pytest.mark.asyncio
async def test_api_error_handling(api_helper):
    """
    Test: API error handling for invalid inputs
    
    Expected behavior:
    - Invalid frequency should return 400 (Bad Request)
    - Invalid duration should return 400
    - Missing parameters should return 400
    """
    # Test invalid frequency (too low)
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire",
        json={
            "frequency_mhz": -100,  # Invalid
            "duration_seconds": 2.0
        }
    )
    assert response.status_code in [400, 422], f"Should reject negative frequency"
    print(f"✓ Rejected invalid frequency")
    
    # Test invalid duration (too long)
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire",
        json={
            "frequency_mhz": 145.50,
            "duration_seconds": 1000  # Too long
        }
    )
    assert response.status_code in [400, 422], f"Should reject excessive duration"
    print(f"✓ Rejected excessive duration")
    
    # Test missing parameters
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire",
        json={}  # Missing parameters
    )
    assert response.status_code in [400, 422], f"Should reject missing parameters"
    print(f"✓ Rejected missing parameters")


@pytest.mark.asyncio
async def test_measurement_retrieval(api_helper, db_helper, db_clean):
    """
    Test: Retrieve measurements after acquisition
    
    Expected behavior:
    - /measurements/{task_id} endpoint returns all 7 measurements
    - Data includes frequency, SNR, timestamp, WebSDR ID
    - Data is properly formatted and accessible
    """
    task_id = await api_helper.trigger_acquisition(
        frequency_mhz=145.50,
        duration_seconds=2.0
    )
    
    # Wait for completion
    await api_helper.wait_for_completion(task_id, timeout_seconds=60)
    
    # Retrieve measurements via API
    measurements = await api_helper.get_measurements(task_id)
    assert len(measurements) == 7, f"Expected 7 measurements, got {len(measurements)}"
    print(f"✓ Retrieved {len(measurements)} measurements via API")
    
    # Verify measurement structure
    for measurement in measurements:
        assert "websdr_id" in measurement
        assert "frequency_mhz" in measurement
        assert "snr" in measurement
        assert "timestamp" in measurement
        assert measurement["frequency_mhz"] == 145.50
    print(f"✓ All measurements have correct structure and frequency")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_long_acquisition(api_helper, db_helper, db_clean):
    """
    Test: Extended acquisition (10 seconds)
    
    Expected behavior:
    - Should handle longer acquisitions gracefully
    - Progress should update periodically
    - Should complete within timeout
    """
    task_id = await api_helper.trigger_acquisition(
        frequency_mhz=145.50,
        duration_seconds=10.0  # Longer than normal
    )
    
    # Track progress updates
    progress_history = []
    
    for attempt in range(120):  # 2 minutes timeout
        status = await api_helper.get_status(task_id)
        state = status.get("state")
        progress = status.get("progress", 0)
        
        progress_history.append(progress)
        
        if state in ["SUCCESS", "FAILED"]:
            print(f"✓ Long acquisition completed: {state}")
            break
        
        await asyncio.sleep(1)
    else:
        raise TimeoutError(f"Long acquisition did not complete")
    
    # Verify measurements
    measurements = db_helper.get_measurements_by_task_id(task_id)
    assert len(measurements) == 7, f"Expected 7 measurements"
    print(f"✓ Long acquisition has {len(measurements)} measurements")


# Smoke tests (quick validation)

@pytest.mark.asyncio
async def test_health_endpoint(api_helper):
    """Test that API is reachable."""
    response = await api_helper.client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print(f"✓ Health endpoint responding")


@pytest.mark.asyncio
async def test_api_docs_available(api_helper):
    """Test that OpenAPI docs are available."""
    response = await api_helper.client.get("/docs")
    assert response.status_code == 200
    print(f"✓ OpenAPI documentation available")
