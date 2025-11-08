"""
Integration tests for Models API with RBAC.

Tests permission checks for viewing, editing, deleting, and sharing ML models
with different user roles (admin, owner, shared user, unauthorized user).

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


# ============================================================================
# Mock Database Fixtures
# ============================================================================

@pytest.fixture
def mock_model_data():
    """Create mock model data for testing"""
    model_id = uuid4()
    return {
        "id": model_id,
        "model_name": "TestModel",
        "version": 1,
        "model_type": "localization_cnn",
        "synthetic_dataset_id": None,
        "mlflow_run_id": "test-run-123",
        "mlflow_experiment_id": "exp-456",
        "onnx_model_location": "s3://models/test.onnx",
        "pytorch_model_location": None,
        "accuracy_meters": 25.5,
        "accuracy_sigma_meters": 10.2,
        "loss_value": 0.05,
        "epoch": 100,
        "is_active": True,
        "is_production": False,
        "hyperparameters": {"learning_rate": 0.001, "batch_size": 32},
        "training_metrics": {"train_loss": 0.05, "val_loss": 0.06},
        "test_metrics": {"test_accuracy": 0.95},
        "created_at": datetime.utcnow(),
        "trained_by_job_id": None,
        "parent_model_id": None,
        "owner_id": OWNER_USER["sub"],
    }


@pytest.fixture
def mock_pool_with_models(mock_model_data):
    """Create mock asyncpg pool with model data"""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Mock fetchrow to return model data
    mock_conn.fetchrow = AsyncMock(return_value=mock_model_data)
    
    # Mock fetch to return list of models
    mock_conn.fetch = AsyncMock(return_value=[mock_model_data])
    
    # Mock fetchval for permission checks
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])  # Default: owner
    
    # Mock execute for write operations
    mock_conn.execute = AsyncMock(return_value="UPDATE 1")

    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()

    return mock_pool, mock_conn


# ============================================================================
# Test: GET /api/v1/models - List Models
# ============================================================================

@pytest.mark.asyncio
async def test_list_models_as_owner(mock_pool_with_models, mock_model_data):
    """Test listing models as owner - should see owned models"""
    mock_pool, mock_conn = mock_pool_with_models
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get("/api/v1/models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "models" in data
        assert len(data["models"]) >= 1
        assert data["models"][0]["model_name"] == "TestModel"
        assert data["models"][0]["is_owner"] is True


@pytest.mark.asyncio
async def test_list_models_as_admin(mock_pool_with_models):
    """Test listing models as admin - should see all models"""
    mock_pool, mock_conn = mock_pool_with_models
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.get("/api/v1/models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "models" in data


@pytest.mark.asyncio
async def test_list_models_as_shared_user(mock_pool_with_models):
    """Test listing models as shared user - should see shared models"""
    mock_pool, mock_conn = mock_pool_with_models
    
    # Mock fetchval to return 'read' permission
    mock_conn.fetchval = AsyncMock(side_effect=['read', None])  # Has read permission, not owner
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.get("/api/v1/models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "models" in data


# ============================================================================
# Test: GET /api/v1/models/{model_id} - View Model Details
# ============================================================================

@pytest.mark.asyncio
async def test_view_model_as_owner(mock_pool_with_models, mock_model_data):
    """Test viewing model details as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get(f"/api/v1/models/{model_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["model_name"] == "TestModel"
        assert data["is_owner"] is True
        assert data["permission"] is None  # Owner doesn't have explicit permission


@pytest.mark.asyncio
async def test_view_model_as_shared_read_user(mock_pool_with_models, mock_model_data):
    """Test viewing model as shared read user - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, has read permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'read'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.get(f"/api/v1/models/{model_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["is_owner"] is False
        assert data["permission"] == "read"


@pytest.mark.asyncio
async def test_view_model_as_unauthorized_user(mock_pool_with_models, mock_model_data):
    """Test viewing model as unauthorized user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, no permission
    mock_conn.fetchval = AsyncMock(return_value=None)
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**UNAUTHORIZED_USER)):
        
        response = client.get(f"/api/v1/models/{model_id}")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_view_model_as_admin(mock_pool_with_models, mock_model_data):
    """Test viewing model as admin - should always succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.get(f"/api/v1/models/{model_id}")
        
        assert response.status_code == status.HTTP_200_OK


# ============================================================================
# Test: PATCH /api/v1/models/{model_id} - Update Model
# ============================================================================

@pytest.mark.asyncio
async def test_update_model_as_owner(mock_pool_with_models, mock_model_data):
    """Test updating model as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.patch(
            f"/api/v1/models/{model_id}",
            json={"model_name": "UpdatedModelName"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify execute was called (UPDATE query)
        assert mock_conn.execute.called


@pytest.mark.asyncio
async def test_update_model_as_shared_edit_user(mock_pool_with_models, mock_model_data):
    """Test updating model as shared edit user - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, has edit permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'edit'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.patch(
            f"/api/v1/models/{model_id}",
            json={"model_name": "UpdatedByEditor"}
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_update_model_as_shared_read_user(mock_pool_with_models, mock_model_data):
    """Test updating model as shared read user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, only read permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'read'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.patch(
            f"/api/v1/models/{model_id}",
            json={"model_name": "ShouldNotUpdate"}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: DELETE /api/v1/models/{model_id} - Delete Model
# ============================================================================

@pytest.mark.asyncio
async def test_delete_model_as_owner(mock_pool_with_models, mock_model_data):
    """Test deleting model as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.delete(f"/api/v1/models/{model_id}")
        
        assert response.status_code == status.HTTP_200_OK
        assert "deleted" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_delete_model_as_shared_edit_user(mock_pool_with_models, mock_model_data):
    """Test deleting model as shared edit user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, has edit permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'edit'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.delete(f"/api/v1/models/{model_id}")
        
        # Shared edit users CANNOT delete
        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.asyncio
async def test_delete_model_as_admin(mock_pool_with_models, mock_model_data):
    """Test deleting model as admin - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**ADMIN_USER)):
        
        response = client.delete(f"/api/v1/models/{model_id}")
        
        assert response.status_code == status.HTTP_200_OK


# ============================================================================
# Test: POST /api/v1/models/{model_id}/shares - Create Share
# ============================================================================

@pytest.mark.asyncio
async def test_create_share_as_owner(mock_pool_with_models, mock_model_data):
    """Test creating share as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetchrow to return created share
    share_id = uuid4()
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": share_id,
        "model_id": model_id,
        "user_id": SHARED_READ_USER["sub"],
        "permission": "read",
        "shared_by": OWNER_USER["sub"],
        "shared_at": datetime.utcnow(),
    })
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/shares",
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
async def test_create_share_as_shared_edit_user(mock_pool_with_models, mock_model_data):
    """Test creating share as shared edit user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, has edit permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'edit'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "read"
            }
        )
        
        # Only owners can create shares
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: GET /api/v1/models/{model_id}/shares - List Shares
# ============================================================================

@pytest.mark.asyncio
async def test_list_shares_as_owner(mock_pool_with_models, mock_model_data):
    """Test listing shares as owner - should see all shares"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetch to return shares
    share_id = uuid4()
    mock_conn.fetch = AsyncMock(return_value=[{
        "id": share_id,
        "model_id": model_id,
        "user_id": SHARED_READ_USER["sub"],
        "permission": "read",
        "shared_by": OWNER_USER["sub"],
        "shared_at": datetime.utcnow(),
    }])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get(f"/api/v1/models/{model_id}/shares")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "shares" in data
        assert len(data["shares"]) >= 1


@pytest.mark.asyncio
async def test_list_shares_as_shared_user(mock_pool_with_models, mock_model_data):
    """Test listing shares as shared user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'edit'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.get(f"/api/v1/models/{model_id}/shares")
        
        # Only owners can list shares
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: DELETE /api/v1/models/{model_id}/shares/{share_id} - Delete Share
# ============================================================================

@pytest.mark.asyncio
async def test_delete_share_as_owner(mock_pool_with_models, mock_model_data):
    """Test deleting share as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    share_id = uuid4()
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.delete(f"/api/v1/models/{model_id}/shares/{share_id}")
        
        assert response.status_code == status.HTTP_200_OK
        assert "removed" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_delete_share_as_shared_user(mock_pool_with_models, mock_model_data):
    """Test deleting share as shared user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    share_id = uuid4()
    
    # Mock permission check: not owner
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'edit'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.delete(f"/api/v1/models/{model_id}/shares/{share_id}")
        
        # Only owners can delete shares
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: POST /api/v1/models/{model_id}/deploy - Deploy Model
# ============================================================================

@pytest.mark.asyncio
async def test_deploy_model_as_owner(mock_pool_with_models, mock_model_data):
    """Test deploying model as owner - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/deploy",
            json={
                "is_active": True,
                "is_production": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert "deployed" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_deploy_model_as_shared_edit_user(mock_pool_with_models, mock_model_data):
    """Test deploying model as shared edit user - should succeed"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, has edit permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'edit'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_EDIT_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/deploy",
            json={
                "is_active": True,
                "is_production": False
            }
        )
        
        # Shared edit users CAN deploy models
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_deploy_model_as_shared_read_user(mock_pool_with_models, mock_model_data):
    """Test deploying model as shared read user - should fail with 403"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock permission check: not owner, only read permission
    mock_conn.fetchval = AsyncMock(side_effect=[None, 'read'])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**SHARED_READ_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/deploy",
            json={
                "is_active": True,
                "is_production": False
            }
        )
        
        # Read-only users cannot deploy
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test: Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_view_nonexistent_model(mock_pool_with_models):
    """Test viewing model that doesn't exist - should return 404"""
    mock_pool, mock_conn = mock_pool_with_models
    nonexistent_id = uuid4()
    
    # Mock fetchrow to return None (model not found)
    mock_conn.fetchrow = AsyncMock(return_value=None)
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.get(f"/api/v1/models/{nonexistent_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_create_duplicate_share(mock_pool_with_models, mock_model_data):
    """Test creating duplicate share - should return 409"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    # Mock fetchrow to raise UniqueViolationError
    from asyncpg.exceptions import UniqueViolationError
    mock_conn.fetchrow = AsyncMock(side_effect=UniqueViolationError("duplicate key"))
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "read"
            }
        )
        
        assert response.status_code == status.HTTP_409_CONFLICT


@pytest.mark.asyncio
async def test_invalid_permission_level(mock_pool_with_models, mock_model_data):
    """Test creating share with invalid permission level - should return 422"""
    mock_pool, mock_conn = mock_pool_with_models
    model_id = mock_model_data["id"]
    
    # Mock is_owner check
    mock_conn.fetchval = AsyncMock(return_value=OWNER_USER["sub"])
    
    with patch("src.routers.models.get_pool", return_value=mock_pool), \
         patch("src.routers.models.get_current_user", return_value=MagicMock(**OWNER_USER)):
        
        response = client.post(
            f"/api/v1/models/{model_id}/shares",
            json={
                "user_id": SHARED_READ_USER["sub"],
                "permission": "invalid"  # Invalid permission
            }
        )
        
        # Pydantic validation should catch this
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
