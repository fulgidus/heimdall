"""
Known Sources API endpoints - RBAC-enabled

This router provides CRUD operations and sharing management for Known Sources.
Known Sources are RF transmitter locations with known positions used for training
and validation of localization models.

Permission Logic:
- Admins: Full access to all sources
- Operators: Can create sources (become owner), edit owned/shared sources
- Users: Can view public sources and sources shared with them (read-only)
- Public sources (is_public=True): Viewable by everyone
- Owners: Full control (view, edit, delete, share)
- Shared users: Permission based on share level ('read' or 'edit')
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field
import asyncpg
import logging

from ..db import get_pool
from ..models.session import KnownSource, KnownSourceCreate, KnownSourceUpdate
from common.auth import (
    get_current_user,
    require_operator,
    User,
    can_view_source,
    can_edit_source,
    can_delete_source,
    get_user_sources,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sources", tags=["sources"])


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class SourceShareCreate(BaseModel):
    """Request model for sharing a source"""
    user_id: str = Field(..., min_length=1, description="Keycloak user ID to share with")
    permission: str = Field(..., pattern="^(read|edit)$", description="Permission level")


class SourceShareUpdate(BaseModel):
    """Request model for updating a share"""
    permission: str = Field(..., pattern="^(read|edit)$", description="Permission level")


class SourceShare(BaseModel):
    """Response model for a source share"""
    id: UUID
    source_id: UUID
    user_id: str
    permission: str
    shared_by: str
    shared_at: datetime


class KnownSourceWithOwnership(BaseModel):
    """Known source with ownership info"""
    id: UUID
    name: str
    description: Optional[str]
    frequency_hz: int
    latitude: float
    longitude: float
    power_dbm: Optional[float]
    source_type: Optional[str]
    is_validated: bool
    error_margin_meters: float
    owner_id: Optional[str]
    is_public: bool
    created_at: datetime
    updated_at: datetime
    is_owner: bool = False  # Set dynamically based on current user
    permission: Optional[str] = None  # 'read', 'edit', or None


class SourceListResponse(BaseModel):
    """Response for source list"""
    sources: List[KnownSourceWithOwnership]
    total: int
    page: int
    per_page: int


# ============================================================================
# CRUD Endpoints
# ============================================================================

@router.get("", response_model=SourceListResponse)
async def list_sources(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=200, description="Items per page"),
    is_public: Optional[bool] = Query(None, description="Filter by public status"),
    only_owned: bool = Query(False, description="Show only owned sources"),
    user: User = Depends(get_current_user),
):
    """
    List known sources accessible by the current user.
    
    - Admins see all sources
    - Operators see owned, shared, and public sources
    - Users see public sources and sources shared with them
    """
    pool = await get_pool()
    offset = (page - 1) * per_page
    
    async with pool.acquire() as conn:
        # Get accessible source IDs
        accessible_sources = await get_user_sources(conn, user.user_id, user.is_admin)
        
        # Build WHERE clause
        where_clauses = []
        params = []
        param_idx = 1
        
        if user.is_admin:
            # Admins see everything, no filtering needed
            pass
        elif only_owned:
            # Show only owned sources
            where_clauses.append(f"ks.owner_id = ${param_idx}")
            params.append(user.user_id)
            param_idx += 1
        else:
            # Show owned, shared, and public
            if accessible_sources:
                source_ids_str = "', '".join(str(s) for s in accessible_sources)
                where_clauses.append(f"(ks.id IN ('{source_ids_str}') OR ks.is_public = true)")
            else:
                # Only public sources
                where_clauses.append("ks.is_public = true")
        
        if is_public is not None:
            where_clauses.append(f"ks.is_public = ${param_idx}")
            params.append(is_public)
            param_idx += 1
        
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
        
        # Query sources
        query = f"""
            SELECT 
                ks.id, ks.name, ks.description, ks.frequency_hz, 
                ks.latitude, ks.longitude, ks.power_dbm, ks.source_type,
                ks.is_validated, ks.error_margin_meters, ks.owner_id, 
                ks.is_public, ks.created_at, ks.updated_at,
                ss.permission as share_permission
            FROM heimdall.known_sources ks
            LEFT JOIN heimdall.source_shares ss 
                ON ks.id = ss.source_id AND ss.user_id = ${param_idx}
            {where_sql}
            ORDER BY ks.name
            LIMIT ${param_idx + 1} OFFSET ${param_idx + 2}
        """
        
        params.extend([user.user_id, per_page, offset])
        
        rows = await conn.fetch(query, *params)
        
        # Count total
        count_query = f"""
            SELECT COUNT(DISTINCT ks.id)
            FROM heimdall.known_sources ks
            LEFT JOIN heimdall.source_shares ss 
                ON ks.id = ss.source_id AND ss.user_id = ${param_idx}
            {where_sql}
        """
        
        total = await conn.fetchval(count_query, *params[:param_idx])
        
        # Build response
        sources = []
        for row in rows:
            is_owner = row["owner_id"] == user.user_id if row["owner_id"] else False
            permission = row["share_permission"] if not is_owner else "edit"
            
            sources.append(KnownSourceWithOwnership(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                frequency_hz=row["frequency_hz"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                power_dbm=row["power_dbm"],
                source_type=row["source_type"],
                is_validated=row["is_validated"],
                error_margin_meters=row["error_margin_meters"],
                owner_id=row["owner_id"],
                is_public=row["is_public"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                is_owner=is_owner,
                permission=permission,
            ))
        
        return SourceListResponse(
            sources=sources,
            total=total,
            page=page,
            per_page=per_page,
        )


@router.post("", response_model=KnownSource, status_code=status.HTTP_201_CREATED)
async def create_source(
    source: KnownSourceCreate,
    user: User = Depends(require_operator),
):
    """
    Create a new known source (operator+ only).
    The creating user becomes the owner.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            query = """
                INSERT INTO heimdall.known_sources 
                (name, description, frequency_hz, latitude, longitude, power_dbm, 
                 source_type, is_validated, error_margin_meters, owner_id, is_public)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, false)
                RETURNING id, name, description, frequency_hz, latitude, longitude,
                          power_dbm, source_type, is_validated, error_margin_meters, 
                          owner_id, is_public, created_at, updated_at
            """
            
            row = await conn.fetchrow(
                query,
                source.name,
                source.description,
                source.frequency_hz,
                source.latitude,
                source.longitude,
                source.power_dbm,
                source.source_type,
                source.is_validated,
                source.error_margin_meters,
                user.user_id,
            )
            
            logger.info(f"User {user.user_id} created source {row['id']}: {source.name}")
            
            return KnownSource(**dict(row))
        
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A known source with this name already exists"
            )


@router.get("/{source_id}", response_model=KnownSourceWithOwnership)
async def get_source(
    source_id: UUID,
    user: User = Depends(get_current_user),
):
    """Get a specific known source by ID (checks permissions)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check view permission
        can_view = await can_view_source(conn, user.user_id, source_id, user.is_admin)
        
        if not can_view:
            logger.warning(f"User {user.user_id} attempted to view inaccessible source {source_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have permission to view this source"
            )
        
        # Get source details
        query = """
            SELECT 
                ks.id, ks.name, ks.description, ks.frequency_hz, 
                ks.latitude, ks.longitude, ks.power_dbm, ks.source_type,
                ks.is_validated, ks.error_margin_meters, ks.owner_id, 
                ks.is_public, ks.created_at, ks.updated_at,
                ss.permission as share_permission
            FROM heimdall.known_sources ks
            LEFT JOIN heimdall.source_shares ss 
                ON ks.id = ss.source_id AND ss.user_id = $2
            WHERE ks.id = $1
        """
        
        row = await conn.fetchrow(query, source_id, user.user_id)
        
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")
        
        is_owner = row["owner_id"] == user.user_id if row["owner_id"] else False
        permission = row["share_permission"] if not is_owner else "edit"
        
        return KnownSourceWithOwnership(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            frequency_hz=row["frequency_hz"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            power_dbm=row["power_dbm"],
            source_type=row["source_type"],
            is_validated=row["is_validated"],
            error_margin_meters=row["error_margin_meters"],
            owner_id=row["owner_id"],
            is_public=row["is_public"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_owner=is_owner,
            permission=permission,
        )


@router.put("/{source_id}", response_model=KnownSource)
async def update_source(
    source_id: UUID,
    source: KnownSourceUpdate,
    user: User = Depends(get_current_user),
):
    """Update a known source (requires edit permission)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check edit permission
        can_edit = await can_edit_source(conn, user.user_id, source_id, user.is_admin)
        
        if not can_edit:
            logger.warning(f"User {user.user_id} attempted to edit inaccessible source {source_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You don't have edit permission for this source"
            )
        
        # Build dynamic update query
        update_fields = []
        params = []
        param_idx = 1
        
        if source.name is not None:
            update_fields.append(f"name = ${param_idx}")
            params.append(source.name)
            param_idx += 1
        
        if source.description is not None:
            update_fields.append(f"description = ${param_idx}")
            params.append(source.description)
            param_idx += 1
        
        if source.frequency_hz is not None:
            update_fields.append(f"frequency_hz = ${param_idx}")
            params.append(source.frequency_hz)
            param_idx += 1
        
        if source.latitude is not None:
            update_fields.append(f"latitude = ${param_idx}")
            params.append(source.latitude)
            param_idx += 1
        
        if source.longitude is not None:
            update_fields.append(f"longitude = ${param_idx}")
            params.append(source.longitude)
            param_idx += 1
        
        if source.power_dbm is not None:
            update_fields.append(f"power_dbm = ${param_idx}")
            params.append(source.power_dbm)
            param_idx += 1
        
        if source.source_type is not None:
            update_fields.append(f"source_type = ${param_idx}")
            params.append(source.source_type)
            param_idx += 1
        
        if source.is_validated is not None:
            update_fields.append(f"is_validated = ${param_idx}")
            params.append(source.is_validated)
            param_idx += 1
        
        if source.error_margin_meters is not None:
            update_fields.append(f"error_margin_meters = ${param_idx}")
            params.append(source.error_margin_meters)
            param_idx += 1
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Always update updated_at
        update_fields.append(f"updated_at = ${param_idx}")
        params.append(datetime.utcnow())
        param_idx += 1
        
        # Add source_id as last parameter
        params.append(str(source_id))
        
        query = f"""
            UPDATE heimdall.known_sources
            SET {', '.join(update_fields)}
            WHERE id = ${param_idx}
            RETURNING id, name, description, frequency_hz, latitude, longitude,
                      power_dbm, source_type, is_validated, error_margin_meters,
                      owner_id, is_public, created_at, updated_at
        """
        
        try:
            updated_row = await conn.fetchrow(query, *params)
            
            if not updated_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Source not found"
                )
            
            logger.info(f"User {user.user_id} updated source {source_id}")
            
            return KnownSource(**dict(updated_row))
        
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A known source with this name already exists"
            )


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: UUID,
    user: User = Depends(get_current_user),
):
    """Delete a known source (owner or admin only)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check delete permission (only owner or admin)
        can_delete = await can_delete_source(conn, user.user_id, source_id, user.is_admin)
        
        if not can_delete:
            logger.warning(f"User {user.user_id} attempted to delete inaccessible source {source_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Only the owner or admin can delete this source"
            )
        
        # Check if source is in use
        usage_check = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.recording_sessions WHERE known_source_id = $1",
            source_id
        )
        
        if usage_check > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot delete source: it is referenced by {usage_check} recording session(s)"
            )
        
        # Delete source
        result = await conn.execute(
            "DELETE FROM heimdall.known_sources WHERE id = $1",
            source_id
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        logger.info(f"User {user.user_id} deleted source {source_id}")
        
        return None


# ============================================================================
# Sharing Endpoints
# ============================================================================

@router.get("/{source_id}/shares", response_model=List[SourceShare])
async def list_source_shares(
    source_id: UUID,
    user: User = Depends(get_current_user),
):
    """List all shares for a source (owner or admin only)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if user is owner or admin
        if not user.is_admin:
            owner_check = await conn.fetchval(
                "SELECT owner_id FROM heimdall.known_sources WHERE id = $1",
                source_id
            )
            
            if not owner_check:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Source not found"
                )
            
            if owner_check != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can view shares"
                )
        
        # Get shares
        query = """
            SELECT id, source_id, user_id, permission, shared_by, shared_at
            FROM heimdall.source_shares
            WHERE source_id = $1
            ORDER BY shared_at DESC
        """
        
        rows = await conn.fetch(query, source_id)
        
        return [SourceShare(**dict(row)) for row in rows]


@router.post("/{source_id}/shares", response_model=SourceShare, status_code=status.HTTP_201_CREATED)
async def create_source_share(
    source_id: UUID,
    share: SourceShareCreate,
    user: User = Depends(get_current_user),
):
    """Share a source with another user (owner or admin only)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if user is owner or admin
        if not user.is_admin:
            owner_check = await conn.fetchval(
                "SELECT owner_id FROM heimdall.known_sources WHERE id = $1",
                source_id
            )
            
            if not owner_check:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Source not found"
                )
            
            if owner_check != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can share this source"
                )
        
        # Create share
        try:
            query = """
                INSERT INTO heimdall.source_shares (source_id, user_id, permission, shared_by)
                VALUES ($1, $2, $3, $4)
                RETURNING id, source_id, user_id, permission, shared_by, shared_at
            """
            
            row = await conn.fetchrow(
                query,
                source_id,
                share.user_id,
                share.permission,
                user.user_id,
            )
            
            logger.info(
                f"User {user.user_id} shared source {source_id} with {share.user_id} "
                f"(permission: {share.permission})"
            )
            
            return SourceShare(**dict(row))
        
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This source is already shared with this user"
            )
        except asyncpg.ForeignKeyViolationError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )


@router.put("/{source_id}/shares/{shared_user_id}", response_model=SourceShare)
async def update_source_share(
    source_id: UUID,
    shared_user_id: str,
    share: SourceShareUpdate,
    user: User = Depends(get_current_user),
):
    """Update a source share's permission level (owner or admin only)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if user is owner or admin
        if not user.is_admin:
            owner_check = await conn.fetchval(
                "SELECT owner_id FROM heimdall.known_sources WHERE id = $1",
                source_id
            )
            
            if not owner_check:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Source not found"
                )
            
            if owner_check != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can modify shares"
                )
        
        # Update share
        query = """
            UPDATE heimdall.source_shares
            SET permission = $1
            WHERE source_id = $2 AND user_id = $3
            RETURNING id, source_id, user_id, permission, shared_by, shared_at
        """
        
        row = await conn.fetchrow(query, share.permission, source_id, shared_user_id)
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Share not found"
            )
        
        logger.info(
            f"User {user.user_id} updated share for source {source_id} with {shared_user_id} "
            f"to permission: {share.permission}"
        )
        
        return SourceShare(**dict(row))


@router.delete("/{source_id}/shares/{shared_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source_share(
    source_id: UUID,
    shared_user_id: str,
    user: User = Depends(get_current_user),
):
    """Remove a source share (owner or admin only)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if user is owner or admin
        if not user.is_admin:
            owner_check = await conn.fetchval(
                "SELECT owner_id FROM heimdall.known_sources WHERE id = $1",
                source_id
            )
            
            if not owner_check:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Source not found"
                )
            
            if owner_check != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can remove shares"
                )
        
        # Delete share
        result = await conn.execute(
            "DELETE FROM heimdall.source_shares WHERE source_id = $1 AND user_id = $2",
            source_id,
            shared_user_id,
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Share not found"
            )
        
        logger.info(f"User {user.user_id} removed share for source {source_id} with {shared_user_id}")
        
        return None


# ============================================================================
# Public Sources Management (Admin/Owner only)
# ============================================================================

@router.patch("/{source_id}/visibility", response_model=KnownSource)
async def update_source_visibility(
    source_id: UUID,
    is_public: bool,
    user: User = Depends(get_current_user),
):
    """Toggle source public visibility (owner or admin only)"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if user is owner or admin
        if not user.is_admin:
            owner_check = await conn.fetchval(
                "SELECT owner_id FROM heimdall.known_sources WHERE id = $1",
                source_id
            )
            
            if not owner_check:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Source not found"
                )
            
            if owner_check != user.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can change visibility"
                )
        
        # Update visibility
        query = """
            UPDATE heimdall.known_sources
            SET is_public = $1, updated_at = $2
            WHERE id = $3
            RETURNING id, name, description, frequency_hz, latitude, longitude,
                      power_dbm, source_type, is_validated, error_margin_meters,
                      owner_id, is_public, created_at, updated_at
        """
        
        row = await conn.fetchrow(query, is_public, datetime.utcnow(), source_id)
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        logger.info(
            f"User {user.user_id} set source {source_id} visibility to "
            f"{'public' if is_public else 'private'}"
        )
        
        return KnownSource(**dict(row))
