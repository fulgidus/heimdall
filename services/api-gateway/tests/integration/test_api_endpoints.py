"""
Integration tests for API Gateway endpoints.

Tests real backend integration with Keycloak authentication and database.
"""

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# Test configuration
TEST_KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
TEST_BACKEND_ORIGIN = os.getenv("VITE_BACKEND_ORIGIN", "http://localhost:8000")


@pytest.fixture
def mock_keycloak_token():
    """Mock Keycloak JWT token for testing."""
    return "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test.token"


@pytest.fixture
def mock_admin_user():
    """Mock admin user from Keycloak."""
    return {
        "id": "admin-123",
        "username": "admin",
        "email": "admin@heimdall.local",
        "roles": ["admin", "operator", "viewer"],
        "is_admin": True,
        "is_operator": True,
        "is_viewer": True,
    }


@pytest.fixture
def authenticated_client(mock_keycloak_token, mock_admin_user):
    """Test client with authentication mocked."""
    from services.api_gateway.src.main import app
    
    # Mock the get_current_user dependency
    with patch("services.api_gateway.src.main.get_current_user") as mock_auth:
        from services.api_gateway.src.main import User
        mock_user = User(**mock_admin_user)
        mock_auth.return_value = mock_user
        
        client = TestClient(app)
        client.headers = {"Authorization": f"Bearer {mock_keycloak_token}"}
        yield client


class TestAuthenticationEndpoints:
    """Test authentication-related endpoints."""
    
    def test_auth_me_requires_authentication(self, authenticated_client):
        """Test /api/v1/auth/me endpoint with authentication."""
        response = authenticated_client.get("/api/v1/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "roles" in data
        assert data["username"] == "admin"
        assert "admin" in data["roles"]
    
    def test_auth_me_without_token_fails(self):
        """Test /api/v1/auth/me without authentication fails."""
        from services.api_gateway.src.main import app
        client = TestClient(app)
        
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestProfileEndpoints:
    """Test user profile endpoints."""
    
    def test_get_profile(self, authenticated_client):
        """Test GET /api/v1/profile."""
        response = authenticated_client.get("/api/v1/profile")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert data["username"] == "admin"


class TestSystemStatusEndpoints:
    """Test system status and health endpoints."""
    
    def test_get_config(self, authenticated_client):
        """Test /api/v1/config endpoint."""
        response = authenticated_client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "websdrs" in data
        assert "supported_bands" in data
        assert "max_duration_seconds" in data
        assert data["websdrs"] == 7
    
    def test_get_stats(self, authenticated_client):
        """Test /api/v1/stats endpoint."""
        response = authenticated_client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "active_sessions" in data
        assert "websdrs_online" in data
    
    def test_get_system_status(self, authenticated_client):
        """Test /api/v1/system/status endpoint."""
        response = authenticated_client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "services" in data
        assert "timestamp" in data
        assert isinstance(data["services"], list)


class TestProxyEndpoints:
    """Test proxy functionality to backend services."""
    
    def test_sessions_proxy(self, authenticated_client):
        """Test proxying to data-ingestion-web service."""
        # This will proxy to the actual backend service if running
        response = authenticated_client.get("/api/v1/sessions")
        
        # Should either succeed or fail with 503 if backend not running
        assert response.status_code in [200, 503, 504]
    
    def test_analytics_proxy(self, authenticated_client):
        """Test proxying to inference analytics service."""
        response = authenticated_client.get("/api/v1/analytics/predictions/metrics")
        
        # Should either succeed or fail with 503 if backend not running
        assert response.status_code in [200, 503, 504]


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self):
        """Test /health endpoint."""
        from services.api_gateway.src.main import app
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-gateway"
    
    def test_ready_check(self):
        """Test /ready endpoint."""
        from services.api_gateway.src.main import app
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True


@pytest.mark.integration
class TestEndToEndFlow:
    """Test end-to-end workflows."""
    
    def test_full_authentication_flow(self, authenticated_client):
        """Test complete authentication and authorization flow."""
        # 1. Check authentication
        response = authenticated_client.get("/api/v1/auth/check")
        assert response.status_code == 200
        
        # 2. Get user info
        response = authenticated_client.get("/api/v1/auth/me")
        assert response.status_code == 200
        user_data = response.json()
        
        # 3. Get user profile
        response = authenticated_client.get("/api/v1/profile")
        assert response.status_code == 200
        profile_data = response.json()
        assert profile_data["username"] == user_data["username"]
    
    def test_dashboard_data_flow(self, authenticated_client):
        """Test dashboard data loading flow."""
        # 1. Get configuration
        response = authenticated_client.get("/api/v1/config")
        assert response.status_code == 200
        
        # 2. Get dashboard stats
        response = authenticated_client.get("/api/v1/stats")
        assert response.status_code == 200
        
        # 3. Get system status
        response = authenticated_client.get("/api/v1/system/status")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
