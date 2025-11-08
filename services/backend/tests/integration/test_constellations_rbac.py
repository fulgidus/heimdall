"""
Integration tests for Constellations API with RBAC.

Tests permission checks for creating, viewing, editing, deleting, and sharing
constellations with different user roles (admin, owner, shared user, unauthorized user).

Requires:
- PostgreSQL database with RBAC schema (migrations 04 & 05)
- Mocked Keycloak JWT authentication
"""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main import app

client = TestClient(app)


# ============================================================================
# Mock User Data
# ============================================================================

ADMIN_USER = {
    "sub": "admin-user-123",
    "email": "admin@example.com",
    "preferred_username": "admin",
    "realm_access": {"roles": ["admin"]},
}

OWNER_USER = {
    "sub": "owner-user-456",
    "email": "owner@example.com",
    "preferred_username": "owner",
    "realm_access": {"roles": ["operator"]},
}

SHARED_READ_USER = {
    "sub": "shared-read-789",
    "email": "read@example.com",
    "preferred_username": "reader",
    "realm_access": {"roles": ["user"]},
}

SHARED_EDIT_USER = {
    "sub": "shared-edit-999",
    "email": "editor@example.com",
    "preferred_username": "editor",
    "realm_access": {"roles": ["operator"]},
}

UNAUTHORIZED_USER = {
    "sub": "unauthorized-111",
    "email": "unauthorized@example.com",
    "preferred_username": "unauthorized",
    "realm_access": {"roles": ["user"]},
}

REGULAR_USER = {
    "sub": "regular-user-222",
    "email": "regular@example.com",
    "preferred_username": "regular",
    "realm_access": {"roles": ["user"]},
}


# ============================================================================
# Mock Database Fixtures
# ============================================================================

@pytest.fixture
def mock_constellation_data():
    """Create mock constellation data for testing"""
    constellation_id = uuid4()
    websdr_id = uuid4()
    return {
        "constellation": {
            "id": constellation_id,
            "name": "Test Constellation",
            "description": "Test constellation for RBAC testing",
            "owner_id": OWNER_USER["sub"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
        "websdr": {
            "id": websdr_id,
            "name": "Test WebSDR",
            "url": "http://test-websdr.example.com",
        },
        "member": {
            "websdr_station_id": websdr_id,
            "station_name": "Test WebSDR",
            "added_at": datetime.utcnow(),
            "added_by": OWNER_USER["sub"],
        }
    }


@pytest.fixture
def mock_pool_with_constellations(mock_constellation_data):
    """Create mock asyncpg pool with constellation data"""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    
    constellation = mock_constellation_data["constellation"]
    member = mock_constellation_data["member"]

    # Mock fetchrow to return constellation data
    mock_conn.fetchrow = AsyncMock(return_value={**constellation})
    
    # Mock fetch to return list of constellations
    mock_conn.fetch = AsyncMock(return_value=[{**constellation, "member_count": 1}])
    
    # Mock fetchval for permission checks and counts
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])  # Default: owner
    
    # Mock execute for write operations
    mock_conn.execute = AsyncMock(return_value="UPDATE 1")
    
    # Mock transaction context
    mock_transaction = AsyncMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock()
    mock_conn.transaction = MagicMock(return_value=mock_transaction)

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    return mock_pool, mock_conn


# ============================================================================
# Test: GET /api/v1/constellations - List Constellations
# ============================================================================

@pytest.mark.asyncio
async def test_list_constellations_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test listing constellations as owner - should see owned constellations"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    
    # Mock fetch to return owned constellation
    mock_conn.fetch = AsyncMock(return_value=[{**constellation, "member_count": 1}])
    mock_conn.fetchval = AsyncMock(return_value=1)  # Total count
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get("/api/v1/constellations")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "constellations" in data
        assert data["total"] == 1
        assert len(data["constellations"]) == 1
        assert data["constellations"][0]["name"] == "Test Constellation"


@pytest.mark.asyncio
async def test_list_constellations_as_admin(mock_pool_with_constellations, mock_constellation_data):
    """Test listing constellations as admin - should see all constellations"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    
    # Mock fetch to return all constellations
    mock_conn.fetch = AsyncMock(return_value=[{**constellation, "member_count": 1}])
    mock_conn.fetchval = AsyncMock(return_value=10)  # Total count
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.get("/api/v1/constellations")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "constellations" in data
        assert data["total"] == 10


@pytest.mark.asyncio
async def test_list_constellations_as_shared_user(mock_pool_with_constellations, mock_constellation_data):
    """Test listing constellations as shared user - should see shared constellations"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    
    # Mock fetch to return shared constellation
    mock_conn.fetch = AsyncMock(return_value=[{**constellation, "member_count": 1}])
    mock_conn.fetchval = AsyncMock(return_value=1)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.get("/api/v1/constellations")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "constellations" in data


@pytest.mark.asyncio
async def test_list_constellations_pagination(mock_pool_with_constellations):
    """Test constellation listing with pagination"""
    mock_pool, mock_conn = mock_pool_with_constellations
    
    # Mock 25 constellations
    mock_conn.fetch = AsyncMock(return_value=[{"id": uuid4(), "name": f"Constellation {i}", 
                                                 "description": None, "owner_id": OWNER_USER["sub"],
                                                 "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
                                                 "member_count": i} for i in range(20)])
    mock_conn.fetchval = AsyncMock(return_value=25)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get("/api/v1/constellations?page=1&per_page=20")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["total"] == 25
        assert data["page"] == 1
        assert data["per_page"] == 20
        assert len(data["constellations"]) == 20


# ============================================================================
# Test: POST /api/v1/constellations - Create Constellation
# ============================================================================

@pytest.mark.asyncio
async def test_create_constellation_as_operator(mock_pool_with_constellations, mock_constellation_data):
    """Test creating constellation as operator - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    
    # Mock fetchrow to return created constellation
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.require_operator", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            "/api/v1/constellations",
            json={
                "name": "New Constellation",
                "description": "A new test constellation",
                "websdr_station_ids": []
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["name"] == constellation["name"]
        assert data["owner_id"] == OWNER_USER["sub"]


@pytest.mark.asyncio
async def test_create_constellation_as_user_fails(mock_pool_with_constellations):
    """Test creating constellation as regular user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    
    # Mock require_operator to raise HTTPException
    from fastapi import HTTPException
    
    def mock_require_operator():
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.require_operator", side_effect=mock_require_operator):
        
        response = client.post(
            "/api/v1/constellations",
            json={
                "name": "Should Fail",
                "description": "Regular user cannot create",
                "websdr_station_ids": []
            }
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_create_constellation_with_websdrs(mock_pool_with_constellations, mock_constellation_data):
    """Test creating constellation with initial WebSDR stations"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    websdr_id = mock_constellation_data["websdr"]["id"]
    
    # Mock fetchrow to return constellation
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    
    # Mock fetchval to confirm WebSDR exists
    mock_conn.fetchval = AsyncMock(return_value=True)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.require_operator", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            "/api/v1/constellations",
            json={
                "name": "Constellation with SDRs",
                "description": "Has initial WebSDRs",
                "websdr_station_ids": [str(websdr_id)]
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.asyncio
async def test_create_constellation_with_invalid_websdr(mock_pool_with_constellations):
    """Test creating constellation with non-existent WebSDR - should fail with 404"""
    mock_pool, mock_conn = mock_pool_with_constellations
    invalid_websdr_id = uuid4()
    
    # Mock fetchrow to return constellation
    mock_conn.fetchrow = AsyncMock(return_value={"id": uuid4(), "name": "Test", "description": None,
                                                   "owner_id": OWNER_USER["sub"], "created_at": datetime.utcnow(),
                                                   "updated_at": datetime.utcnow()})
    
    # Mock fetchval to return False (WebSDR doesn't exist)
    mock_conn.fetchval = AsyncMock(return_value=False)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.require_operator", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            "/api/v1/constellations",
            json={
                "name": "Invalid Constellation",
                "description": "Has invalid WebSDR",
                "websdr_station_ids": [str(invalid_websdr_id)]
            }
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Test: GET /api/v1/constellations/{id} - View Constellation Details
# ============================================================================

@pytest.mark.asyncio
async def test_view_constellation_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test viewing constellation details as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    member = mock_constellation_data["member"]
    constellation_id = constellation["id"]
    
    # Mock has_access check (not admin, so check is performed)
    mock_conn.fetchval = AsyncMock(return_value=True)  # Has access
    
    # Mock fetchrow to return constellation
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    
    # Mock fetch to return members
    mock_conn.fetch = AsyncMock(return_value=[member])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == "Test Constellation"
        assert data["owner_id"] == OWNER_USER["sub"]
        assert "members" in data
        assert len(data["members"]) == 1


@pytest.mark.asyncio
async def test_view_constellation_as_shared_read_user(mock_pool_with_constellations, mock_constellation_data):
    """Test viewing constellation as shared read user - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    constellation_id = constellation["id"]
    
    # Mock has_access check - user has read permission
    mock_conn.fetchval = AsyncMock(return_value=True)
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    mock_conn.fetch = AsyncMock(return_value=[])  # No members
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_view_constellation_as_unauthorized_user(mock_pool_with_constellations, mock_constellation_data):
    """Test viewing constellation as unauthorized user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock has_access check - no access
    mock_conn.fetchval = AsyncMock(return_value=False)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**UNAUTHORIZED_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_view_constellation_as_admin(mock_pool_with_constellations, mock_constellation_data):
    """Test viewing constellation as admin - should always succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    constellation_id = constellation["id"]
    
    # Admin bypasses access check
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    mock_conn.fetch = AsyncMock(return_value=[])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_view_nonexistent_constellation(mock_pool_with_constellations):
    """Test viewing constellation that doesn't exist - should return 404"""
    mock_pool, mock_conn = mock_pool_with_constellations
    nonexistent_id = uuid4()
    
    # Mock fetchval to allow access check to pass
    mock_conn.fetchval = AsyncMock(return_value=True)
    
    # Mock fetchrow to return None (constellation not found)
    mock_conn.fetchrow = AsyncMock(return_value=None)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get(f"/api/v1/constellations/{nonexistent_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Test: PUT /api/v1/constellations/{id} - Update Constellation
# ============================================================================

@pytest.mark.asyncio
async def test_update_constellation_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test updating constellation as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    constellation_id = constellation["id"]
    
    # Mock can_edit check
    mock_conn.fetchval = AsyncMock(return_value=True)
    
    # Mock fetchrow to return updated constellation
    updated_constellation = {**constellation, "name": "Updated Name"}
    mock_conn.fetchrow = AsyncMock(return_value=updated_constellation)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}",
            json={"name": "Updated Name", "description": "Updated description"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["name"] == "Updated Name"


@pytest.mark.asyncio
async def test_update_constellation_as_shared_edit_user(mock_pool_with_constellations, mock_constellation_data):
    """Test updating constellation as shared edit user - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    constellation_id = constellation["id"]
    
    # Mock can_edit check - has edit permission
    mock_conn.fetchval = AsyncMock(return_value=True)
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}",
            json={"name": "Updated by Editor"}
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_update_constellation_as_shared_read_user(mock_pool_with_constellations, mock_constellation_data):
    """Test updating constellation as shared read user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock can_edit check - only read permission
    mock_conn.fetchval = AsyncMock(return_value=False)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}",
            json={"name": "Should Not Update"}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_update_constellation_as_admin(mock_pool_with_constellations, mock_constellation_data):
    """Test updating constellation as admin - should always succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    constellation_id = constellation["id"]
    
    # Admin bypasses permission check
    mock_conn.fetchrow = AsyncMock(return_value={**constellation, "name": "Admin Updated"})
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}",
            json={"name": "Admin Updated"}
        )
        
        assert response.status_code == status.HTTP_200_OK


# ============================================================================
# Test: DELETE /api/v1/constellations/{id} - Delete Constellation
# ============================================================================

@pytest.mark.asyncio
async def test_delete_constellation_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test deleting constellation as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.asyncio
async def test_delete_constellation_as_shared_edit_user(mock_pool_with_constellations, mock_constellation_data):
    """Test deleting constellation as shared edit user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check - not owner
    mock_conn.fetchval = AsyncMock(return_value=None)  # Not owner
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}")
        
        # Shared edit users CANNOT delete
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_delete_constellation_as_admin(mock_pool_with_constellations, mock_constellation_data):
    """Test deleting constellation as admin - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Admin bypasses owner check
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT


# ============================================================================
# Test: POST /api/v1/constellations/{id}/members - Add WebSDR
# ============================================================================

@pytest.mark.asyncio
async def test_add_websdr_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test adding WebSDR to constellation as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    websdr_id = mock_constellation_data["websdr"]["id"]
    
    # Mock can_edit check
    mock_conn.fetchval = AsyncMock(side_effect=[True, True])  # Can edit, WebSDR exists
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/members",
            json={"websdr_station_id": str(websdr_id)}
        )
        
        assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.asyncio
async def test_add_websdr_as_shared_edit_user(mock_pool_with_constellations, mock_constellation_data):
    """Test adding WebSDR as shared edit user - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    websdr_id = mock_constellation_data["websdr"]["id"]
    
    # Mock can_edit check - has edit permission
    mock_conn.fetchval = AsyncMock(side_effect=[True, True])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/members",
            json={"websdr_station_id": str(websdr_id)}
        )
        
        assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.asyncio
async def test_add_websdr_as_shared_read_user(mock_pool_with_constellations, mock_constellation_data):
    """Test adding WebSDR as shared read user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    websdr_id = mock_constellation_data["websdr"]["id"]
    
    # Mock can_edit check - no edit permission
    mock_conn.fetchval = AsyncMock(return_value=False)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/members",
            json={"websdr_station_id": str(websdr_id)}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_add_nonexistent_websdr(mock_pool_with_constellations, mock_constellation_data):
    """Test adding non-existent WebSDR - should fail with 404"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    invalid_websdr_id = uuid4()
    
    # Mock can_edit check, then WebSDR not found
    mock_conn.fetchval = AsyncMock(side_effect=[True, False])  # Can edit, but WebSDR doesn't exist
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/members",
            json={"websdr_station_id": str(invalid_websdr_id)}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


# ============================================================================
# Test: DELETE /api/v1/constellations/{id}/members/{websdr_id} - Remove WebSDR
# ============================================================================

@pytest.mark.asyncio
async def test_remove_websdr_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test removing WebSDR from constellation as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    websdr_id = mock_constellation_data["websdr"]["id"]
    
    # Mock can_edit check
    mock_conn.fetchval = AsyncMock(return_value=True)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}/members/{websdr_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.asyncio
async def test_remove_websdr_as_shared_read_user(mock_pool_with_constellations, mock_constellation_data):
    """Test removing WebSDR as shared read user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    websdr_id = mock_constellation_data["websdr"]["id"]
    
    # Mock can_edit check - no edit permission
    mock_conn.fetchval = AsyncMock(return_value=False)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}/members/{websdr_id}")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: POST /api/v1/constellations/{id}/shares - Create Share
# ============================================================================

@pytest.mark.asyncio
async def test_create_share_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test creating share as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetchrow to return created share
    share_id = uuid4()
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": share_id,
        "constellation_id": constellation_id,
        "user_id": SHARED_READ_USER["sub"],
        "permission": "read",
        "shared_by": OWNER_USER["sub"],
        "shared_at": datetime.utcnow(),
    })
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "read"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["user_id"] == SHARED_READ_USER["sub"]
        assert data["permission"] == "read"


@pytest.mark.asyncio
async def test_create_share_as_shared_edit_user(mock_pool_with_constellations, mock_constellation_data):
    """Test creating share as shared edit user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check - not owner
    mock_conn.fetchval = AsyncMock(return_value=None)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "read"
            }
        )
        
        # Only owners can create shares
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_create_share_as_admin(mock_pool_with_constellations, mock_constellation_data):
    """Test creating share as admin - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Admin bypasses owner check
    share_id = uuid4()
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": share_id,
        "constellation_id": constellation_id,
        "user_id": REGULAR_USER["sub"],
        "permission": "edit",
        "shared_by": ADMIN_USER["sub"],
        "shared_at": datetime.utcnow(),
    })
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/shares",
            json={
                "user_id": REGULAR_USER["sub"],
                "permission": "edit"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED


# ============================================================================
# Test: GET /api/v1/constellations/{id}/shares - List Shares
# ============================================================================

@pytest.mark.asyncio
async def test_list_shares_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test listing shares as owner - should see all shares"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetch to return shares
    share_id = uuid4()
    mock_conn.fetch = AsyncMock(return_value=[{
        "id": share_id,
        "constellation_id": constellation_id,
        "user_id": SHARED_READ_USER["sub"],
        "permission": "read",
        "shared_by": OWNER_USER["sub"],
        "shared_at": datetime.utcnow(),
    }])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}/shares")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) >= 1


@pytest.mark.asyncio
async def test_list_shares_as_shared_user(mock_pool_with_constellations, mock_constellation_data):
    """Test listing shares as shared user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check - not owner
    mock_conn.fetchval = AsyncMock(return_value=None)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}/shares")
        
        # Only owners can list shares
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: PUT /api/v1/constellations/{id}/shares/{user_id} - Update Share
# ============================================================================

@pytest.mark.asyncio
async def test_update_share_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test updating share as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetchrow to return updated share
    share_id = uuid4()
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": share_id,
        "constellation_id": constellation_id,
        "user_id": SHARED_READ_USER["sub"],
        "permission": "edit",  # Updated from read
        "shared_by": OWNER_USER["sub"],
        "shared_at": datetime.utcnow(),
    })
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}/shares/{SHARED_READ_USER['sub']}",
            json={"permission": "edit"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["permission"] == "edit"


@pytest.mark.asyncio
async def test_update_share_as_non_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test updating share as non-owner - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check - not owner
    mock_conn.fetchval = AsyncMock(return_value=None)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}/shares/{SHARED_READ_USER['sub']}",
            json={"permission": "edit"}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: DELETE /api/v1/constellations/{id}/shares/{user_id} - Delete Share
# ============================================================================

@pytest.mark.asyncio
async def test_delete_share_as_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test deleting share as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}/shares/{SHARED_READ_USER['sub']}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT


@pytest.mark.asyncio
async def test_delete_share_as_non_owner(mock_pool_with_constellations, mock_constellation_data):
    """Test deleting share as non-owner - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check - not owner
    mock_conn.fetchval = AsyncMock(return_value=None)
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.delete(f"/api/v1/constellations/{constellation_id}/shares/{SHARED_READ_USER['sub']}")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: Edge Cases and Integration Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_create_duplicate_share(mock_pool_with_constellations, mock_constellation_data):
    """Test creating duplicate share - should return 409"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetchrow to raise UniqueViolationError
    from asyncpg.exceptions import UniqueViolationError
    mock_conn.fetchrow = AsyncMock(side_effect=UniqueViolationError("duplicate key"))
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "read"
            }
        )
        
        assert response.status_code == status.HTTP_409_CONFLICT


@pytest.mark.asyncio
async def test_invalid_permission_level(mock_pool_with_constellations, mock_constellation_data):
    """Test creating share with invalid permission level - should return 422"""
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation_id = mock_constellation_data["constellation"]["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/constellations/{constellation_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "invalid"  # Invalid permission
            }
        )
        
        # Pydantic validation should catch this
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_complete_workflow_create_share_view(mock_pool_with_constellations, mock_constellation_data):
    """
    Integration test: Complete workflow
    1. Operator creates constellation
    2. Operator shares with read permission
    3. User views constellation (succeeds)
    4. User tries to edit constellation (fails)
    """
    mock_pool, mock_conn = mock_pool_with_constellations
    constellation = mock_constellation_data["constellation"]
    constellation_id = constellation["id"]
    
    # Step 1: Create constellation (already covered above)
    # Step 2: Share with read permission (already covered above)
    
    # Step 3: User views constellation
    mock_conn.fetchval = AsyncMock(return_value=True)  # Has access
    mock_conn.fetchrow = AsyncMock(return_value=constellation)
    mock_conn.fetch = AsyncMock(return_value=[])
    
    with patch("src.routers.constellations.get_pool", return_value=mock_pool), \
         patch("src.routers.constellations.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.get(f"/api/v1/constellations/{constellation_id}")
        assert response.status_code == status.HTTP_200_OK
        
        # Step 4: User tries to edit (should fail)
        mock_conn.fetchval = AsyncMock(return_value=False)  # No edit permission
        
        response = client.put(
            f"/api/v1/constellations/{constellation_id}",
            json={"name": "Should Fail"}
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
