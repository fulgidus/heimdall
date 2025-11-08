"""
RBAC (Role-Based Access Control) utilities for Heimdall SDR.

This module provides permission checking functions for resources like
Constellations, Sources, and Models. It implements ownership-based access
control with sharing capabilities.

Permission Hierarchy:
- Admins: Can do everything (always returns True)
- Owners: Have full control over their resources
- Shared Users: Have permission level defined in share tables ('read' or 'edit')
- Public Resources: Sources with is_public=True are visible to all

Usage:
    from common.auth.rbac import can_view_constellation, can_edit_constellation
    
    # In your FastAPI endpoint (with asyncpg):
    async with pool.acquire() as conn:
        if not await can_view_constellation(conn, user.id, constellation_id, user.is_admin):
            raise HTTPException(status_code=403, detail="Access denied")
"""

from typing import List, Optional, Union
from uuid import UUID
import asyncpg


# Type alias for asyncpg connection
DbConnection = Union[asyncpg.Connection, asyncpg.pool.PoolConnectionProxy]


# ============================================================================
# Constellation Permissions
# ============================================================================

async def can_view_constellation(
    db: DbConnection,
    user_id: str,
    constellation_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can view a constellation.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users with 'read' or 'edit': True
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        constellation_id: Constellation UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can view, False otherwise
    """
    if is_admin:
        return True
    
    # Check if user is owner
    if await is_constellation_owner(db, constellation_id, user_id):
        return True
    
    # Check if user has read or edit permission via sharing
    permission = await get_constellation_permission(db, constellation_id, user_id)
    return permission in ['read', 'edit']


async def can_edit_constellation(
    db: DbConnection,
    user_id: str,
    constellation_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can edit a constellation (modify name, description, add/remove SDRs).
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users with 'edit': True
    - Shared users with 'read': False
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        constellation_id: Constellation UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can edit, False otherwise
    """
    if is_admin:
        return True
    
    # Check if user is owner
    if await is_constellation_owner(db, constellation_id, user_id):
        return True
    
    # Check if user has edit permission via sharing
    permission = await get_constellation_permission(db, constellation_id, user_id)
    return permission == 'edit'


async def can_delete_constellation(
    db: DbConnection,
    user_id: str,
    constellation_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can delete a constellation.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users: False (cannot delete even with 'edit' permission)
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        constellation_id: Constellation UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can delete, False otherwise
    """
    if is_admin:
        return True
    
    # Only owners can delete (shared users cannot, even with 'edit')
    return await is_constellation_owner(db, constellation_id, user_id)


async def get_user_constellations(
    db: DbConnection,
    user_id: str,
    is_admin: bool = False
) -> List[UUID]:
    """
    Get list of constellation IDs accessible to the user.
    
    Permission Logic:
    - Admins: All constellations
    - Others: Owned constellations + shared constellations (read or edit)
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        is_admin: Whether user has admin role
    
    Returns:
        List of constellation UUIDs
    """
    if is_admin:
        # Admin sees all constellations
        rows = await db.fetch("SELECT id FROM heimdall.constellations")
        return [row['id'] for row in rows]
    
    # Get owned constellations
    owned_rows = await db.fetch(
        "SELECT id FROM heimdall.constellations WHERE owner_id = $1",
        user_id
    )
    owned_ids = [row['id'] for row in owned_rows]
    
    # Get shared constellations
    shared_rows = await db.fetch(
        "SELECT constellation_id FROM heimdall.constellation_shares WHERE user_id = $1",
        user_id
    )
    shared_ids = [row['constellation_id'] for row in shared_rows]
    
    # Combine and deduplicate
    return list(set(owned_ids + shared_ids))


# ============================================================================
# Source Permissions
# ============================================================================

async def can_view_source(
    db: DbConnection,
    user_id: str,
    source_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can view a known radio source.
    
    Permission Logic:
    - Admins: Always True
    - Public sources (is_public=True): Always True
    - Owners: Always True
    - Shared users with 'read' or 'edit': True
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        source_id: Source UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can view, False otherwise
    """
    if is_admin:
        return True
    
    # Check if source is public
    if await is_source_public(db, source_id):
        return True
    
    # Check if user is owner
    if await is_source_owner(db, source_id, user_id):
        return True
    
    # Check if user has read or edit permission via sharing
    permission = await get_source_permission(db, source_id, user_id)
    return permission in ['read', 'edit']


async def can_edit_source(
    db: DbConnection,
    user_id: str,
    source_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can edit a known radio source.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users with 'edit': True
    - Shared users with 'read': False
    - Public sources: False (unless owner/admin/shared with edit)
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        source_id: Source UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can edit, False otherwise
    """
    if is_admin:
        return True
    
    # Check if user is owner
    if await is_source_owner(db, source_id, user_id):
        return True
    
    # Check if user has edit permission via sharing
    permission = await get_source_permission(db, source_id, user_id)
    return permission == 'edit'


async def can_delete_source(
    db: DbConnection,
    user_id: str,
    source_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can delete a known radio source.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users: False (cannot delete even with 'edit' permission)
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        source_id: Source UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can delete, False otherwise
    """
    if is_admin:
        return True
    
    # Only owners can delete
    return await is_source_owner(db, source_id, user_id)


async def get_user_sources(
    db: DbConnection,
    user_id: str,
    is_admin: bool = False
) -> List[UUID]:
    """
    Get list of source IDs accessible to the user.
    
    Permission Logic:
    - Admins: All sources
    - Others: Public sources + owned sources + shared sources (read or edit)
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        is_admin: Whether user has admin role
    
    Returns:
        List of source UUIDs
    """
    if is_admin:
        # Admin sees all sources
        rows = await db.fetch("SELECT id FROM heimdall.known_sources")
        return [row['id'] for row in rows]
    
    # Get public sources
    public_rows = await db.fetch(
        "SELECT id FROM heimdall.known_sources WHERE is_public = true"
    )
    public_ids = [row['id'] for row in public_rows]
    
    # Get owned sources
    owned_rows = await db.fetch(
        "SELECT id FROM heimdall.known_sources WHERE owner_id = $1",
        user_id
    )
    owned_ids = [row['id'] for row in owned_rows]
    
    # Get shared sources
    shared_rows = await db.fetch(
        "SELECT source_id FROM heimdall.source_shares WHERE user_id = $1",
        user_id
    )
    shared_ids = [row['source_id'] for row in shared_rows]
    
    # Combine and deduplicate
    return list(set(public_ids + owned_ids + shared_ids))


# ============================================================================
# Model Permissions
# ============================================================================

async def can_view_model(
    db: DbConnection,
    user_id: str,
    model_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can view an ML model.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users with 'read' or 'edit': True
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        model_id: Model UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can view, False otherwise
    """
    if is_admin:
        return True
    
    # Check if user is owner
    if await is_model_owner(db, model_id, user_id):
        return True
    
    # Check if user has read or edit permission via sharing
    permission = await get_model_permission(db, model_id, user_id)
    return permission in ['read', 'edit']


async def can_edit_model(
    db: DbConnection,
    user_id: str,
    model_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can edit an ML model.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users with 'edit': True
    - Shared users with 'read': False
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        model_id: Model UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can edit, False otherwise
    """
    if is_admin:
        return True
    
    # Check if user is owner
    if await is_model_owner(db, model_id, user_id):
        return True
    
    # Check if user has edit permission via sharing
    permission = await get_model_permission(db, model_id, user_id)
    return permission == 'edit'


async def can_delete_model(
    db: DbConnection,
    user_id: str,
    model_id: UUID,
    is_admin: bool = False
) -> bool:
    """
    Check if user can delete an ML model.
    
    Permission Logic:
    - Admins: Always True
    - Owners: Always True
    - Shared users: False (cannot delete even with 'edit' permission)
    - Others: False
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        model_id: Model UUID
        is_admin: Whether user has admin role
    
    Returns:
        True if user can delete, False otherwise
    """
    if is_admin:
        return True
    
    # Only owners can delete
    return await is_model_owner(db, model_id, user_id)


async def get_user_models(
    db: DbConnection,
    user_id: str,
    is_admin: bool = False
) -> List[UUID]:
    """
    Get list of model IDs accessible to the user.
    
    Permission Logic:
    - Admins: All models
    - Others: Owned models + shared models (read or edit)
    
    Args:
        db: Database connection (asyncpg)
        user_id: User ID (from JWT 'sub' claim)
        is_admin: Whether user has admin role
    
    Returns:
        List of model UUIDs
    """
    if is_admin:
        # Admin sees all models
        rows = await db.fetch("SELECT id FROM heimdall.models")
        return [row['id'] for row in rows]
    
    # Get owned models
    owned_rows = await db.fetch(
        "SELECT id FROM heimdall.models WHERE owner_id = $1",
        user_id
    )
    owned_ids = [row['id'] for row in owned_rows]
    
    # Get shared models
    shared_rows = await db.fetch(
        "SELECT model_id FROM heimdall.model_shares WHERE user_id = $1",
        user_id
    )
    shared_ids = [row['model_id'] for row in shared_rows]
    
    # Combine and deduplicate
    return list(set(owned_ids + shared_ids))


# ============================================================================
# Helper Functions - Ownership Checks
# ============================================================================

async def is_constellation_owner(
    db: DbConnection,
    constellation_id: UUID,
    user_id: str
) -> bool:
    """Check if user owns a constellation."""
    result = await db.fetchval(
        """
        SELECT EXISTS(
            SELECT 1 FROM heimdall.constellations 
            WHERE id = $1 AND owner_id = $2
        )
        """,
        constellation_id, user_id
    )
    return bool(result)


async def is_source_owner(
    db: DbConnection,
    source_id: UUID,
    user_id: str
) -> bool:
    """Check if user owns a source."""
    result = await db.fetchval(
        """
        SELECT EXISTS(
            SELECT 1 FROM heimdall.known_sources 
            WHERE id = $1 AND owner_id = $2
        )
        """,
        source_id, user_id
    )
    return bool(result)


async def is_model_owner(
    db: DbConnection,
    model_id: UUID,
    user_id: str
) -> bool:
    """Check if user owns a model."""
    result = await db.fetchval(
        """
        SELECT EXISTS(
            SELECT 1 FROM heimdall.models 
            WHERE id = $1 AND owner_id = $2
        )
        """,
        model_id, user_id
    )
    return bool(result)


# ============================================================================
# Helper Functions - Permission Checks
# ============================================================================

async def get_constellation_permission(
    db: DbConnection,
    constellation_id: UUID,
    user_id: str
) -> Optional[str]:
    """
    Get user's permission level for a constellation.
    
    Returns:
        'read', 'edit', or None if no permission
    """
    result = await db.fetchval(
        """
        SELECT permission FROM heimdall.constellation_shares 
        WHERE constellation_id = $1 AND user_id = $2
        """,
        constellation_id, user_id
    )
    return result


async def get_source_permission(
    db: DbConnection,
    source_id: UUID,
    user_id: str
) -> Optional[str]:
    """
    Get user's permission level for a source.
    
    Returns:
        'read', 'edit', or None if no permission
    """
    result = await db.fetchval(
        """
        SELECT permission FROM heimdall.source_shares 
        WHERE source_id = $1 AND user_id = $2
        """,
        source_id, user_id
    )
    return result


async def get_model_permission(
    db: DbConnection,
    model_id: UUID,
    user_id: str
) -> Optional[str]:
    """
    Get user's permission level for a model.
    
    Returns:
        'read', 'edit', or None if no permission
    """
    result = await db.fetchval(
        """
        SELECT permission FROM heimdall.model_shares 
        WHERE model_id = $1 AND user_id = $2
        """,
        model_id, user_id
    )
    return result


async def is_source_public(
    db: DbConnection,
    source_id: UUID
) -> bool:
    """Check if a source is marked as public."""
    result = await db.fetchval(
        "SELECT is_public FROM heimdall.known_sources WHERE id = $1",
        source_id
    )
    return bool(result) if result is not None else False
