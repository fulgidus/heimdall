"""
Test authentication endpoints in API Gateway.

Validates that:
1. POST /api/v1/auth/login endpoint exists
2. POST /api/v1/auth/refresh endpoint exists
3. GET /api/v1/auth/check endpoint exists
4. Endpoints return appropriate responses
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create test client for API Gateway."""
    return TestClient(app)


def test_auth_login_endpoint_exists(client):
    """Test that /api/v1/auth/login endpoint is accessible."""
    # POST without credentials should fail but endpoint should exist
    response = client.post("/api/v1/auth/login")
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404, "Login endpoint should exist at /api/v1/auth/login"
    # Will be 400 (missing credentials) or 500 (Keycloak not reachable), both are OK
    assert response.status_code in [400, 500], f"Expected 400 or 500, got {response.status_code}"


def test_auth_login_with_json_body(client):
    """Test login endpoint accepts JSON body."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "testpass"},
        headers={"Content-Type": "application/json"}
    )
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404, "Login endpoint should exist"
    # Will fail auth but endpoint is accessible
    assert response.status_code in [400, 401, 500], f"Expected 400/401/500, got {response.status_code}"


def test_auth_login_with_form_data(client):
    """Test login endpoint accepts form-urlencoded data."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "test@example.com", "password": "testpass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404, "Login endpoint should exist"
    # Will fail auth but endpoint is accessible
    assert response.status_code in [400, 401, 500], f"Expected 400/401/500, got {response.status_code}"


def test_auth_refresh_endpoint_exists(client):
    """Test that /api/v1/auth/refresh endpoint is accessible."""
    response = client.post("/api/v1/auth/refresh")
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404, "Refresh endpoint should exist at /api/v1/auth/refresh"
    # Will be 400 (missing refresh token) or 500 (Keycloak not reachable)
    assert response.status_code in [400, 500], f"Expected 400 or 500, got {response.status_code}"


def test_auth_check_endpoint_exists(client):
    """Test that /api/v1/auth/check endpoint is accessible."""
    response = client.get("/api/v1/auth/check")
    # Should not be 404 (endpoint exists)
    assert response.status_code != 404, "Auth check endpoint should exist at /api/v1/auth/check"
    # Should return 200 with auth status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "authenticated" in data, "Response should contain 'authenticated' field"
    assert "auth_enabled" in data, "Response should contain 'auth_enabled' field"


def test_all_auth_endpoints_use_correct_paths(client):
    """Verify all auth endpoints are registered with /api/v1/auth prefix."""
    # This test ensures our endpoints match what the frontend expects
    auth_endpoints = [
        ("/api/v1/auth/login", "POST"),
        ("/api/v1/auth/refresh", "POST"),
        ("/api/v1/auth/check", "GET"),
    ]
    
    for path, method in auth_endpoints:
        if method == "GET":
            response = client.get(path)
        else:
            response = client.post(path)
        
        # Most important: endpoint should exist (not 404)
        assert response.status_code != 404, f"{method} {path} should exist"
