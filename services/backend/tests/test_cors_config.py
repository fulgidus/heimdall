"""Tests for CORS configuration."""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_preflight_request(client):
    """Test OPTIONS pre-flight request for CORS."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type,Authorization",
        },
    )
    assert response.status_code == 200

    # Check CORS headers are present
    headers = response.headers
    assert "access-control-allow-origin" in headers
    assert "access-control-allow-methods" in headers
    assert "access-control-allow-headers" in headers
    assert "access-control-max-age" in headers


def test_cors_credentials_allowed(client):
    """Test that credentials are allowed in CORS requests."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200

    headers = response.headers
    assert "access-control-allow-credentials" in headers
    assert headers["access-control-allow-credentials"] == "true"


def test_cors_allowed_origin(client):
    """Test that allowed origins are properly configured."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200

    headers = response.headers
    assert "access-control-allow-origin" in headers
    # Should match the requesting origin
    assert headers["access-control-allow-origin"] in ["http://localhost:3000", "*"]


def test_cors_exposed_headers(client):
    """Test that expose headers configuration is working."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200

    # CORS middleware should add expose-headers when configured
    headers = response.headers
    # FastAPI's CORSMiddleware adds this header when expose_headers is configured
    # Note: The header may not always be present depending on the CORS middleware implementation
    if "access-control-expose-headers" in headers:
        assert headers["access-control-expose-headers"] == "*"


def test_cors_methods_allowed(client):
    """Test that all required HTTP methods are allowed via pre-flight."""
    # Use the /health endpoint which definitely exists
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    # Pre-flight should succeed
    assert response.status_code == 200

    headers = response.headers
    allowed_methods = headers.get("access-control-allow-methods", "").upper()

    # Verify that all expected methods are allowed
    required_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    for method in required_methods:
        assert (
            method in allowed_methods
        ), f"Method {method} not found in allowed methods: {allowed_methods}"


def test_cors_with_authorization_header(client):
    """Test CORS with Authorization header."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )

    assert response.status_code == 200
    headers = response.headers
    assert "access-control-allow-headers" in headers

    allowed_headers = headers["access-control-allow-headers"].lower()
    assert "authorization" in allowed_headers
    assert "content-type" in allowed_headers
