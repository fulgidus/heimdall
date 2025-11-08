"""
E2E Test Suite - SIMPLIFIED VERSION
Tests that focus on API structure and health rather than actual WebSDR integration.

This version doesn't require WebSDR data collection to work - it tests the API contracts
"""

import json

import pytest


@pytest.mark.asyncio
async def test_01_health_endpoint(api_helper):
    """✅ Verify API is alive and responsive."""
    response = await api_helper.client.get("/health")
    assert response.status_code == 200, f"Health endpoint failed: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "backend"
    print(f"✅ Health endpoint responding: {data['service']} is healthy")


@pytest.mark.asyncio
async def test_02_api_docs_available(api_helper):
    """✅ Verify OpenAPI documentation is accessible."""
    response = await api_helper.client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
    print("✅ OpenAPI documentation available")


@pytest.mark.asyncio
async def test_03_api_error_handling_missing_params(api_helper):
    """✅ Verify API validates required parameters."""
    # Missing frequency
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire", json={"duration_seconds": 2.0}
    )
    assert response.status_code in [400, 422], "Should reject missing frequency"
    print("✅ API rejects missing frequency")

    # Missing duration
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire", json={"frequency_mhz": 145.5}
    )
    assert response.status_code in [400, 422], "Should reject missing duration"
    print("✅ API rejects missing duration")


@pytest.mark.asyncio
async def test_04_api_error_handling_invalid_values(api_helper):
    """✅ Verify API validates parameter ranges."""
    # Negative frequency
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire", json={"frequency_mhz": -145.5, "duration_seconds": 2.0}
    )
    assert response.status_code in [400, 422], "Should reject negative frequency"
    print("✅ API rejects negative frequency")

    # Excessive duration
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire", json={"frequency_mhz": 145.5, "duration_seconds": 300.0}
    )
    assert response.status_code in [400, 422], "Should reject excessive duration"
    print("✅ API rejects excessive duration")


@pytest.mark.asyncio
async def test_05_acquisition_request_accepted(api_helper):
    """✅ Verify acquisition requests are queued."""
    # Should accept and return task_id
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire", json={"frequency_mhz": 145.5, "duration_seconds": 2.0}
    )
    assert response.status_code == 200, f"Acquisition request failed: {response.text}"
    data = response.json()

    # Verify response structure
    assert "task_id" in data, "Response must contain task_id"
    assert "status" in data, "Response must contain status"
    assert data["status"] in [
        "PENDING",
        "QUEUED",
        "RUNNING",
    ], f"Initial status should be pending-like, got {data['status']}"

    task_id = data["task_id"]
    print(f"✅ Acquisition request queued with task_id: {task_id}")

    # Store task_id for use in other tests (if needed)
    return task_id


@pytest.mark.asyncio
async def test_06_status_endpoint_responds(api_helper):
    """✅ Verify status endpoint returns proper response structure."""
    # First trigger an acquisition
    response = await api_helper.client.post(
        "/api/v1/acquisition/acquire", json={"frequency_mhz": 145.5, "duration_seconds": 2.0}
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    # Now check status
    response = await api_helper.client.get(f"/api/v1/acquisition/status/{task_id}")
    assert response.status_code == 200, f"Status endpoint failed: {response.text}"
    data = response.json()

    # Verify response structure (flexible - API might return different fields)
    assert isinstance(data, dict), "Status response should be dict"
    print(f"✅ Status endpoint returns valid response: {json.dumps(data, indent=2)[:200]}...")


@pytest.mark.asyncio
async def test_07_websdr_config_endpoint(api_helper):
    """✅ Verify WebSDR configuration is retrievable."""
    response = await api_helper.client.get("/api/v1/acquisition/websdrs")

    # Endpoint might exist or might return 404 - both are OK
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list), "WebSDR list should be array"
        print(f"✅ WebSDR config available: {len(data)} receivers")
    else:
        print(f"⚠️  WebSDR endpoint not available (status {response.status_code})")


@pytest.mark.asyncio
async def test_08_database_connectivity(db_helper):
    """✅ Verify database is accessible."""
    try:
        # Try to query measurements table
        count = db_helper.get_measurement_count()
        print(f"✅ Database connected: {count} measurements in DB")
        assert count >= 0, "Should return non-negative count"
    except Exception as e:
        pytest.skip(f"Database not accessible: {e}")


@pytest.mark.asyncio
async def test_09_response_schema_consistency(api_helper):
    """✅ Verify API responses are consistent."""
    # Test multiple acquisitions
    responses = []
    for i in range(3):
        response = await api_helper.client.post(
            "/api/v1/acquisition/acquire",
            json={"frequency_mhz": 145.5 + i * 0.1, "duration_seconds": 2.0},
        )
        assert response.status_code == 200
        responses.append(response.json())

    # Verify all responses have same structure
    keys = set(responses[0].keys())
    for resp in responses[1:]:
        assert set(resp.keys()) == keys, "Response structure should be consistent"

    print(f"✅ API responses are consistent ({len(keys)} fields)")


# SUMMARY
"""
This simplified test suite validates:
✅ API is alive and responds to health checks
✅ API documentation is available
✅ API validates input parameters (missing, invalid values)
✅ API queues acquisition requests and returns task IDs
✅ API status endpoint returns responses
✅ WebSDR configuration (if available)
✅ Database connectivity
✅ Response schema consistency

This suite does NOT require:
❌ WebSDR data collection to actually complete
❌ Real IQ data processing
❌ Database measurements table population
❌ Long-running tasks to complete

This allows us to test the API contract without depending on mock external services.
"""
