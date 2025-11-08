"""
Unit tests for RBAC (Role-Based Access Control) utilities.

Tests all permission checking functions with different user roles,
ownership scenarios, and sharing permissions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4, UUID

from ..rbac import (
    # Constellation permissions
    can_view_constellation,
    can_edit_constellation,
    can_delete_constellation,
    get_user_constellations,
    # Source permissions
    can_view_source,
    can_edit_source,
    can_delete_source,
    get_user_sources,
    # Model permissions
    can_view_model,
    can_edit_model,
    can_delete_model,
    get_user_models,
    # Helper functions
    is_constellation_owner,
    is_source_owner,
    is_model_owner,
    get_constellation_permission,
    get_source_permission,
    get_model_permission,
    is_source_public,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def admin_user_id():
    """Admin user ID."""
    return "admin-user-123"


@pytest.fixture
def operator_user_id():
    """Operator user ID."""
    return "operator-user-456"


@pytest.fixture
def regular_user_id():
    """Regular user ID."""
    return "regular-user-789"


@pytest.fixture
def other_user_id():
    """Another user ID for sharing tests."""
    return "other-user-999"


@pytest.fixture
def constellation_id():
    """Test constellation ID."""
    return uuid4()


@pytest.fixture
def source_id():
    """Test source ID."""
    return uuid4()


@pytest.fixture
def model_id():
    """Test model ID."""
    return uuid4()


@pytest.fixture
def mock_db():
    """Mock database connection."""
    db = AsyncMock()
    return db


# ============================================================================
# Constellation Permission Tests
# ============================================================================

class TestConstellationPermissions:
    """Test constellation permission functions."""
    
    @pytest.mark.asyncio
    async def test_admin_can_view_any_constellation(self, mock_db, admin_user_id, constellation_id):
        """Admin can view any constellation without ownership check."""
        result = await can_view_constellation(
            mock_db, admin_user_id, constellation_id, is_admin=True
        )
        assert result is True
        # Should not query database for admin
        mock_db.fetchval.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_owner_can_view_constellation(self, mock_db, operator_user_id, constellation_id):
        """Owner can view their constellation."""
        # Mock: user is owner
        mock_db.fetchval.return_value = True
        
        result = await can_view_constellation(
            mock_db, operator_user_id, constellation_id, is_admin=False
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_read_can_view(self, mock_db, regular_user_id, constellation_id):
        """User with 'read' permission can view constellation."""
        # Mock: not owner, but has 'read' permission
        mock_db.fetchval.side_effect = [False, 'read']  # is_owner, then permission
        
        result = await can_view_constellation(
            mock_db, regular_user_id, constellation_id, is_admin=False
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_edit_can_view(self, mock_db, regular_user_id, constellation_id):
        """User with 'edit' permission can view constellation."""
        # Mock: not owner, but has 'edit' permission
        mock_db.fetchval.side_effect = [False, 'edit']
        
        result = await can_view_constellation(
            mock_db, regular_user_id, constellation_id, is_admin=False
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_non_shared_user_cannot_view(self, mock_db, regular_user_id, constellation_id):
        """User without permissions cannot view constellation."""
        # Mock: not owner, no permission
        mock_db.fetchval.side_effect = [False, None]
        
        result = await can_view_constellation(
            mock_db, regular_user_id, constellation_id, is_admin=False
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_admin_can_edit_any_constellation(self, mock_db, admin_user_id, constellation_id):
        """Admin can edit any constellation."""
        result = await can_edit_constellation(
            mock_db, admin_user_id, constellation_id, is_admin=True
        )
        assert result is True
        mock_db.fetchval.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_owner_can_edit_constellation(self, mock_db, operator_user_id, constellation_id):
        """Owner can edit their constellation."""
        mock_db.fetchval.return_value = True
        
        result = await can_edit_constellation(
            mock_db, operator_user_id, constellation_id, is_admin=False
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_edit_can_edit(self, mock_db, regular_user_id, constellation_id):
        """User with 'edit' permission can edit constellation."""
        # Mock: not owner, but has 'edit' permission
        mock_db.fetchval.side_effect = [False, 'edit']
        
        result = await can_edit_constellation(
            mock_db, regular_user_id, constellation_id, is_admin=False
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_read_cannot_edit(self, mock_db, regular_user_id, constellation_id):
        """User with 'read' permission cannot edit constellation."""
        # Mock: not owner, only 'read' permission
        mock_db.fetchval.side_effect = [False, 'read']
        
        result = await can_edit_constellation(
            mock_db, regular_user_id, constellation_id, is_admin=False
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_admin_can_delete_any_constellation(self, mock_db, admin_user_id, constellation_id):
        """Admin can delete any constellation."""
        result = await can_delete_constellation(
            mock_db, admin_user_id, constellation_id, is_admin=True
        )
        assert result is True
        mock_db.fetchval.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_owner_can_delete_constellation(self, mock_db, operator_user_id, constellation_id):
        """Owner can delete their constellation."""
        mock_db.fetchval.return_value = True
        
        result = await can_delete_constellation(
            mock_db, operator_user_id, constellation_id, is_admin=False
        )
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_cannot_delete_even_with_edit(self, mock_db, regular_user_id, constellation_id):
        """User with 'edit' permission still cannot delete constellation."""
        # Mock: not owner (delete requires ownership)
        mock_db.fetchval.return_value = False
        
        result = await can_delete_constellation(
            mock_db, regular_user_id, constellation_id, is_admin=False
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_user_constellations_admin(self, mock_db, admin_user_id):
        """Admin gets all constellations."""
        constellation_ids = [uuid4(), uuid4(), uuid4()]
        mock_db.fetch.return_value = [{'id': cid} for cid in constellation_ids]
        
        result = await get_user_constellations(mock_db, admin_user_id, is_admin=True)
        
        assert len(result) == 3
        assert set(result) == set(constellation_ids)
    
    @pytest.mark.asyncio
    async def test_get_user_constellations_operator(self, mock_db, operator_user_id):
        """Operator gets owned + shared constellations."""
        owned_ids = [uuid4(), uuid4()]
        shared_ids = [uuid4()]
        
        # Mock: first call for owned, second call for shared
        mock_db.fetch.side_effect = [
            [{'id': cid} for cid in owned_ids],
            [{'constellation_id': cid} for cid in shared_ids]
        ]
        
        result = await get_user_constellations(mock_db, operator_user_id, is_admin=False)
        
        assert len(result) == 3
        assert set(result) == set(owned_ids + shared_ids)
    
    @pytest.mark.asyncio
    async def test_get_user_constellations_deduplicates(self, mock_db, operator_user_id):
        """get_user_constellations deduplicates if user owns and is shared the same constellation."""
        shared_id = uuid4()
        owned_ids = [uuid4(), shared_id]
        shared_ids = [shared_id]  # Same constellation
        
        mock_db.fetch.side_effect = [
            [{'id': cid} for cid in owned_ids],
            [{'constellation_id': cid} for cid in shared_ids]
        ]
        
        result = await get_user_constellations(mock_db, operator_user_id, is_admin=False)
        
        # Should have 2 unique constellations (owned_ids deduplicated)
        assert len(result) == 2


# ============================================================================
# Source Permission Tests
# ============================================================================

class TestSourcePermissions:
    """Test source permission functions."""
    
    @pytest.mark.asyncio
    async def test_admin_can_view_any_source(self, mock_db, admin_user_id, source_id):
        """Admin can view any source."""
        result = await can_view_source(mock_db, admin_user_id, source_id, is_admin=True)
        assert result is True
        mock_db.fetchval.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_anyone_can_view_public_source(self, mock_db, regular_user_id, source_id):
        """Any user can view public source."""
        # Mock: source is public
        mock_db.fetchval.return_value = True
        
        result = await can_view_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_owner_can_view_private_source(self, mock_db, operator_user_id, source_id):
        """Owner can view their private source."""
        # Mock: not public, but user is owner
        mock_db.fetchval.side_effect = [False, True]  # is_public, is_owner
        
        result = await can_view_source(mock_db, operator_user_id, source_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_read_can_view_source(self, mock_db, regular_user_id, source_id):
        """User with 'read' permission can view source."""
        # Mock: not public, not owner, but has 'read' permission
        mock_db.fetchval.side_effect = [False, False, 'read']
        
        result = await can_view_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_non_shared_user_cannot_view_private_source(self, mock_db, regular_user_id, source_id):
        """User without permissions cannot view private source."""
        # Mock: not public, not owner, no permission
        mock_db.fetchval.side_effect = [False, False, None]
        
        result = await can_view_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_admin_can_edit_any_source(self, mock_db, admin_user_id, source_id):
        """Admin can edit any source."""
        result = await can_edit_source(mock_db, admin_user_id, source_id, is_admin=True)
        assert result is True
        mock_db.fetchval.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_owner_can_edit_source(self, mock_db, operator_user_id, source_id):
        """Owner can edit their source."""
        mock_db.fetchval.return_value = True
        
        result = await can_edit_source(mock_db, operator_user_id, source_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_edit_can_edit_source(self, mock_db, regular_user_id, source_id):
        """User with 'edit' permission can edit source."""
        # Mock: not owner, but has 'edit' permission
        mock_db.fetchval.side_effect = [False, 'edit']
        
        result = await can_edit_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_read_cannot_edit_source(self, mock_db, regular_user_id, source_id):
        """User with 'read' permission cannot edit source."""
        # Mock: not owner, only 'read' permission
        mock_db.fetchval.side_effect = [False, 'read']
        
        result = await can_edit_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_public_source_cannot_be_edited_by_non_owner(self, mock_db, regular_user_id, source_id):
        """Public source can be viewed but not edited by non-owners."""
        # Mock: not owner, no edit permission (public doesn't grant edit)
        mock_db.fetchval.side_effect = [False, None]
        
        result = await can_edit_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_owner_can_delete_source(self, mock_db, operator_user_id, source_id):
        """Owner can delete their source."""
        mock_db.fetchval.return_value = True
        
        result = await can_delete_source(mock_db, operator_user_id, source_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_cannot_delete_source_even_with_edit(self, mock_db, regular_user_id, source_id):
        """User with 'edit' permission still cannot delete source."""
        mock_db.fetchval.return_value = False
        
        result = await can_delete_source(mock_db, regular_user_id, source_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_user_sources_admin(self, mock_db, admin_user_id):
        """Admin gets all sources."""
        source_ids = [uuid4(), uuid4(), uuid4()]
        mock_db.fetch.return_value = [{'id': sid} for sid in source_ids]
        
        result = await get_user_sources(mock_db, admin_user_id, is_admin=True)
        
        assert len(result) == 3
        assert set(result) == set(source_ids)
    
    @pytest.mark.asyncio
    async def test_get_user_sources_includes_public_owned_shared(self, mock_db, operator_user_id):
        """User gets public + owned + shared sources."""
        public_ids = [uuid4()]
        owned_ids = [uuid4(), uuid4()]
        shared_ids = [uuid4()]
        
        # Mock: public, owned, shared
        mock_db.fetch.side_effect = [
            [{'id': sid} for sid in public_ids],
            [{'id': sid} for sid in owned_ids],
            [{'source_id': sid} for sid in shared_ids]
        ]
        
        result = await get_user_sources(mock_db, operator_user_id, is_admin=False)
        
        assert len(result) == 4
        assert set(result) == set(public_ids + owned_ids + shared_ids)


# ============================================================================
# Model Permission Tests
# ============================================================================

class TestModelPermissions:
    """Test model permission functions."""
    
    @pytest.mark.asyncio
    async def test_admin_can_view_any_model(self, mock_db, admin_user_id, model_id):
        """Admin can view any model."""
        result = await can_view_model(mock_db, admin_user_id, model_id, is_admin=True)
        assert result is True
        mock_db.fetchval.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_owner_can_view_model(self, mock_db, operator_user_id, model_id):
        """Owner can view their model."""
        mock_db.fetchval.return_value = True
        
        result = await can_view_model(mock_db, operator_user_id, model_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_read_can_view_model(self, mock_db, regular_user_id, model_id):
        """User with 'read' permission can view model."""
        # Mock: not owner, but has 'read' permission
        mock_db.fetchval.side_effect = [False, 'read']
        
        result = await can_view_model(mock_db, regular_user_id, model_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_edit_can_view_model(self, mock_db, regular_user_id, model_id):
        """User with 'edit' permission can view model."""
        mock_db.fetchval.side_effect = [False, 'edit']
        
        result = await can_view_model(mock_db, regular_user_id, model_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_non_shared_user_cannot_view_model(self, mock_db, regular_user_id, model_id):
        """User without permissions cannot view model."""
        mock_db.fetchval.side_effect = [False, None]
        
        result = await can_view_model(mock_db, regular_user_id, model_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_owner_can_edit_model(self, mock_db, operator_user_id, model_id):
        """Owner can edit their model."""
        mock_db.fetchval.return_value = True
        
        result = await can_edit_model(mock_db, operator_user_id, model_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_edit_can_edit_model(self, mock_db, regular_user_id, model_id):
        """User with 'edit' permission can edit model."""
        mock_db.fetchval.side_effect = [False, 'edit']
        
        result = await can_edit_model(mock_db, regular_user_id, model_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_with_read_cannot_edit_model(self, mock_db, regular_user_id, model_id):
        """User with 'read' permission cannot edit model."""
        mock_db.fetchval.side_effect = [False, 'read']
        
        result = await can_edit_model(mock_db, regular_user_id, model_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_owner_can_delete_model(self, mock_db, operator_user_id, model_id):
        """Owner can delete their model."""
        mock_db.fetchval.return_value = True
        
        result = await can_delete_model(mock_db, operator_user_id, model_id, is_admin=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shared_user_cannot_delete_model_even_with_edit(self, mock_db, regular_user_id, model_id):
        """User with 'edit' permission still cannot delete model."""
        mock_db.fetchval.return_value = False
        
        result = await can_delete_model(mock_db, regular_user_id, model_id, is_admin=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_user_models_admin(self, mock_db, admin_user_id):
        """Admin gets all models."""
        model_ids = [uuid4(), uuid4(), uuid4()]
        mock_db.fetch.return_value = [{'id': mid} for mid in model_ids]
        
        result = await get_user_models(mock_db, admin_user_id, is_admin=True)
        
        assert len(result) == 3
        assert set(result) == set(model_ids)
    
    @pytest.mark.asyncio
    async def test_get_user_models_includes_owned_and_shared(self, mock_db, operator_user_id):
        """User gets owned + shared models."""
        owned_ids = [uuid4(), uuid4()]
        shared_ids = [uuid4()]
        
        mock_db.fetch.side_effect = [
            [{'id': mid} for mid in owned_ids],
            [{'model_id': mid} for mid in shared_ids]
        ]
        
        result = await get_user_models(mock_db, operator_user_id, is_admin=False)
        
        assert len(result) == 3
        assert set(result) == set(owned_ids + shared_ids)


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Test helper functions for ownership and permission checks."""
    
    @pytest.mark.asyncio
    async def test_is_constellation_owner_true(self, mock_db, operator_user_id, constellation_id):
        """is_constellation_owner returns True for owner."""
        mock_db.fetchval.return_value = True
        
        result = await is_constellation_owner(mock_db, constellation_id, operator_user_id)
        assert result is True
        
        # Verify correct SQL query
        call_args = mock_db.fetchval.call_args
        assert 'heimdall.constellations' in call_args[0][0]
        assert constellation_id in call_args[0][1:]
        assert operator_user_id in call_args[0][1:]
    
    @pytest.mark.asyncio
    async def test_is_constellation_owner_false(self, mock_db, regular_user_id, constellation_id):
        """is_constellation_owner returns False for non-owner."""
        mock_db.fetchval.return_value = False
        
        result = await is_constellation_owner(mock_db, constellation_id, regular_user_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_constellation_permission_returns_read(self, mock_db, regular_user_id, constellation_id):
        """get_constellation_permission returns 'read' for shared user."""
        mock_db.fetchval.return_value = 'read'
        
        result = await get_constellation_permission(mock_db, constellation_id, regular_user_id)
        assert result == 'read'
    
    @pytest.mark.asyncio
    async def test_get_constellation_permission_returns_edit(self, mock_db, regular_user_id, constellation_id):
        """get_constellation_permission returns 'edit' for shared user with edit access."""
        mock_db.fetchval.return_value = 'edit'
        
        result = await get_constellation_permission(mock_db, constellation_id, regular_user_id)
        assert result == 'edit'
    
    @pytest.mark.asyncio
    async def test_get_constellation_permission_returns_none(self, mock_db, regular_user_id, constellation_id):
        """get_constellation_permission returns None for non-shared user."""
        mock_db.fetchval.return_value = None
        
        result = await get_constellation_permission(mock_db, constellation_id, regular_user_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_is_source_public_true(self, mock_db, source_id):
        """is_source_public returns True for public source."""
        mock_db.fetchval.return_value = True
        
        result = await is_source_public(mock_db, source_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_source_public_false(self, mock_db, source_id):
        """is_source_public returns False for private source."""
        mock_db.fetchval.return_value = False
        
        result = await is_source_public(mock_db, source_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_source_public_handles_none(self, mock_db, source_id):
        """is_source_public returns False when source doesn't exist (None)."""
        mock_db.fetchval.return_value = None
        
        result = await is_source_public(mock_db, source_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_source_owner_true(self, mock_db, operator_user_id, source_id):
        """is_source_owner returns True for owner."""
        mock_db.fetchval.return_value = True
        
        result = await is_source_owner(mock_db, source_id, operator_user_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_model_owner_false(self, mock_db, regular_user_id, model_id):
        """is_model_owner returns False for non-owner."""
        mock_db.fetchval.return_value = False
        
        result = await is_model_owner(mock_db, model_id, regular_user_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_source_permission_workflow(self, mock_db, regular_user_id, source_id):
        """get_source_permission queries correct table."""
        mock_db.fetchval.return_value = 'edit'
        
        result = await get_source_permission(mock_db, source_id, regular_user_id)
        assert result == 'edit'
        
        # Verify correct SQL query
        call_args = mock_db.fetchval.call_args
        assert 'heimdall.source_shares' in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_get_model_permission_workflow(self, mock_db, regular_user_id, model_id):
        """get_model_permission queries correct table."""
        mock_db.fetchval.return_value = 'read'
        
        result = await get_model_permission(mock_db, model_id, regular_user_id)
        assert result == 'read'
        
        # Verify correct SQL query
        call_args = mock_db.fetchval.call_args
        assert 'heimdall.model_shares' in call_args[0][0]


# ============================================================================
# Integration Scenarios (still unit tests, but testing multiple functions)
# ============================================================================

class TestPermissionScenarios:
    """Test realistic permission scenarios combining multiple checks."""
    
    @pytest.mark.asyncio
    async def test_operator_creates_and_owns_constellation(self, mock_db, operator_user_id, constellation_id):
        """Operator creates constellation and has full permissions."""
        mock_db.fetchval.return_value = True
        
        # Can view, edit, and delete
        assert await can_view_constellation(mock_db, operator_user_id, constellation_id) is True
        assert await can_edit_constellation(mock_db, operator_user_id, constellation_id) is True
        assert await can_delete_constellation(mock_db, operator_user_id, constellation_id) is True
    
    @pytest.mark.asyncio
    async def test_operator_shares_with_read_permission(self, mock_db, operator_user_id, other_user_id, constellation_id):
        """Operator shares constellation with 'read', other user can view but not edit."""
        # Other user checks
        mock_db.fetchval.side_effect = [
            False, 'read',  # can_view: not owner, has 'read'
            False, 'read',  # can_edit: not owner, has 'read'
            False           # can_delete: not owner
        ]
        
        assert await can_view_constellation(mock_db, other_user_id, constellation_id) is True
        assert await can_edit_constellation(mock_db, other_user_id, constellation_id) is False
        assert await can_delete_constellation(mock_db, other_user_id, constellation_id) is False
    
    @pytest.mark.asyncio
    async def test_operator_shares_with_edit_permission(self, mock_db, operator_user_id, other_user_id, constellation_id):
        """Operator shares constellation with 'edit', other user can view and edit but not delete."""
        # Other user checks
        mock_db.fetchval.side_effect = [
            False, 'edit',  # can_view: not owner, has 'edit'
            False, 'edit',  # can_edit: not owner, has 'edit'
            False           # can_delete: not owner
        ]
        
        assert await can_view_constellation(mock_db, other_user_id, constellation_id) is True
        assert await can_edit_constellation(mock_db, other_user_id, constellation_id) is True
        assert await can_delete_constellation(mock_db, other_user_id, constellation_id) is False
    
    @pytest.mark.asyncio
    async def test_public_source_visible_to_all_editable_to_owner_only(self, mock_db, operator_user_id, regular_user_id, source_id):
        """Public source can be viewed by anyone, but only edited by owner."""
        # Regular user checks (not owner)
        mock_db.fetchval.side_effect = [
            True,          # is_public (view check)
            False, None    # not owner, no permission (edit check)
        ]
        
        assert await can_view_source(mock_db, regular_user_id, source_id) is True
        assert await can_edit_source(mock_db, regular_user_id, source_id) is False
        
        # Owner checks
        mock_db.fetchval.return_value = True  # is_owner
        assert await can_edit_source(mock_db, operator_user_id, source_id) is True
    
    @pytest.mark.asyncio
    async def test_user_with_no_permissions_cannot_access_anything(self, mock_db, regular_user_id, constellation_id, source_id, model_id):
        """User with no permissions cannot access any resources."""
        mock_db.fetchval.return_value = False
        
        # Constellations
        assert await can_view_constellation(mock_db, regular_user_id, constellation_id) is False
        assert await can_edit_constellation(mock_db, regular_user_id, constellation_id) is False
        assert await can_delete_constellation(mock_db, regular_user_id, constellation_id) is False
        
        # Sources (not public, not owner, no permission)
        mock_db.fetchval.side_effect = [False, False, None]  # not public, not owner, no permission
        assert await can_view_source(mock_db, regular_user_id, source_id) is False
        
        # Models
        mock_db.fetchval.side_effect = [False, None]  # not owner, no permission
        assert await can_view_model(mock_db, regular_user_id, model_id) is False
