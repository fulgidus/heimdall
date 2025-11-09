"""
Constellations API endpoints - RBAC-enabled

This router provides CRUD operations and sharing management for Constellations.
Constellations are logical groupings of WebSDR stations that can be owned,
shared, and used for localization tasks.

Permission Logic:
- Admins: Full access to all constellations
- Operators: Can create constellations (become owner), edit owned/shared constellations
- Users: Can view constellations shared with them (read-only)
- Owners: Full control (view, edit, delete, share)
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field
import logging

from ..db import get_pool
from common.auth import get_current_user, require_operator, User
from common.auth.rbac import (
    can_view_constellation,
    can_edit_constellation,
    can_delete_constellation,
    get_user_constellations,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/constellations", tags=["constellations"])


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class ConstellationBase(BaseModel):
    """Base constellation fields"""
    name: str = Field(..., min_length=1, max_length=255, description="Constellation name")
    description: Optional[str] = Field(None, description="Optional description")


class ConstellationCreate(ConstellationBase):
    """Request model for creating a constellation"""
    websdr_station_ids: List[UUID] = Field(default_factory=list, description="Initial WebSDR stations")


class ConstellationUpdate(BaseModel):
    """Request model for updating a constellation"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class ConstellationMemberResponse(BaseModel):
    """Response model for a constellation member"""
    websdr_station_id: UUID
    station_name: str
    added_at: datetime
    added_by: Optional[str]


class ConstellationResponse(ConstellationBase):
    """Response model for a constellation"""
    id: UUID
    owner_id: str
    created_at: datetime
    updated_at: datetime
    member_count: int


class ConstellationDetailResponse(ConstellationResponse):
    """Detailed response model including members"""
    members: List[ConstellationMemberResponse]


class ConstellationListResponse(BaseModel):
    """Response model for constellation list"""
    constellations: List[ConstellationResponse]
    total: int
    page: int
    per_page: int


class ConstellationShareCreate(BaseModel):
    """Request model for creating a share"""
    user_id: str = Field(..., description="Keycloak user ID to share with")
    permission: str = Field(..., pattern="^(read|edit)$", description="Permission level: 'read' or 'edit'")


class ConstellationShareUpdate(BaseModel):
    """Request model for updating a share"""
    permission: str = Field(..., pattern="^(read|edit)$", description="Permission level: 'read' or 'edit'")


class ConstellationShareResponse(BaseModel):
    """Response model for a share"""
    id: UUID
    constellation_id: UUID
    user_id: str
    permission: str
    shared_by: str
    shared_at: datetime


class ConstellationMemberAdd(BaseModel):
    """Request model for adding WebSDR to constellation"""
    websdr_station_id: UUID = Field(..., description="WebSDR station ID to add")


# ============================================================================
# CRUD Endpoints
# ============================================================================

@router.get("", response_model=ConstellationListResponse)
async def list_constellations(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    user: User = Depends(get_current_user),
):
    """
    List all constellations accessible to the current user.
    
    - **Admins**: See all constellations
    - **Others**: See owned + shared constellations
    """
    pool = get_pool()
    offset = (page - 1) * per_page
    
    async with pool.acquire() as conn:
        if user.is_admin:
            # Admin sees everything
            query = """
                SELECT 
                    c.id,
                    c.name,
                    c.description,
                    c.owner_id,
                    c.created_at,
                    c.updated_at,
                    COUNT(cm.id) as member_count
                FROM heimdall.constellations c
                LEFT JOIN heimdall.constellation_members cm ON c.id = cm.constellation_id
                GROUP BY c.id
                ORDER BY c.created_at DESC
                LIMIT $1 OFFSET $2
            """
            count_query = "SELECT COUNT(*) FROM heimdall.constellations"
            
            rows = await conn.fetch(query, per_page, offset)
            total = await conn.fetchval(count_query)
        else:
            # Non-admins see owned + shared
            query = """
                SELECT DISTINCT
                    c.id,
                    c.name,
                    c.description,
                    c.owner_id,
                    c.created_at,
                    c.updated_at,
                    COUNT(cm.id) as member_count
                FROM heimdall.constellations c
                LEFT JOIN heimdall.constellation_members cm ON c.id = cm.constellation_id
                LEFT JOIN heimdall.constellation_shares cs ON c.id = cs.constellation_id
                WHERE c.owner_id = $1 OR cs.user_id = $1
                GROUP BY c.id
                ORDER BY c.created_at DESC
                LIMIT $2 OFFSET $3
            """
            count_query = """
                SELECT COUNT(DISTINCT c.id)
                FROM heimdall.constellations c
                LEFT JOIN heimdall.constellation_shares cs ON c.id = cs.constellation_id
                WHERE c.owner_id = $1 OR cs.user_id = $1
            """
            
            rows = await conn.fetch(query, user.id, per_page, offset)
            total = await conn.fetchval(count_query, user.id)
        
        constellations = [
            ConstellationResponse(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                owner_id=row["owner_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                member_count=row["member_count"],
            )
            for row in rows
        ]
        
        return ConstellationListResponse(
            constellations=constellations,
            total=total,
            page=page,
            per_page=per_page,
        )


@router.post("", response_model=ConstellationResponse, status_code=status.HTTP_201_CREATED)
async def create_constellation(
    data: ConstellationCreate,
    user: User = Depends(require_operator),  # Requires operator or admin role
):
    """
    Create a new constellation.
    
    - **Requires**: Operator or Admin role
    - **Ownership**: Creator becomes owner
    - **WebSDRs**: Optional list of initial WebSDR stations
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create constellation
            constellation_row = await conn.fetchrow(
                """
                INSERT INTO heimdall.constellations (name, description, owner_id)
                VALUES ($1, $2, $3)
                RETURNING id, name, description, owner_id, created_at, updated_at
                """,
                data.name,
                data.description,
                user.id,
            )
            
            constellation_id = constellation_row["id"]
            
            # Add initial WebSDR members if provided
            member_count = 0
            if data.websdr_station_ids:
                for websdr_id in data.websdr_station_ids:
                    # Verify WebSDR exists
                    websdr_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM heimdall.websdr_stations WHERE id = $1)",
                        websdr_id
                    )
                    
                    if not websdr_exists:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"WebSDR station {websdr_id} not found"
                        )
                    
                    # Add member
                    await conn.execute(
                        """
                        INSERT INTO heimdall.constellation_members (constellation_id, websdr_station_id, added_by)
                        VALUES ($1, $2, $3)
                        """,
                        constellation_id,
                        websdr_id,
                        user.id,
                    )
                    member_count += 1
            
            logger.info(f"Created constellation {constellation_id} with {member_count} members by user {user.id}")
            
            return ConstellationResponse(
                id=constellation_row["id"],
                name=constellation_row["name"],
                description=constellation_row["description"],
                owner_id=constellation_row["owner_id"],
                created_at=constellation_row["created_at"],
                updated_at=constellation_row["updated_at"],
                member_count=member_count,
            )


@router.get("/{constellation_id}", response_model=ConstellationDetailResponse)
async def get_constellation(
    constellation_id: UUID,
    user: User = Depends(get_current_user),
):
    """
    Get detailed information about a constellation including members.
    
    - **Permission**: View access required (owner, shared user, or admin)
    """
    pool = get_pool()
    
    # Check permission (using raw SQL approach since we're using asyncpg)
    # For async RBAC checks, we'd need to adapt them to asyncpg
    async with pool.acquire() as conn:
        # Check if user can view
        if not user.is_admin:
            # Check ownership or sharing
            has_access = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM heimdall.constellations c
                    LEFT JOIN heimdall.constellation_shares cs ON c.id = cs.constellation_id
                    WHERE c.id = $1 AND (c.owner_id = $2 OR cs.user_id = $2)
                )
                """,
                constellation_id,
                user.id,
            )
            
            if not has_access:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this constellation"
                )
        
        # Get constellation details
        constellation_row = await conn.fetchrow(
            """
            SELECT id, name, description, owner_id, created_at, updated_at
            FROM heimdall.constellations
            WHERE id = $1
            """,
            constellation_id,
        )
        
        if not constellation_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Constellation not found"
            )
        
        # Get members
        member_rows = await conn.fetch(
            """
            SELECT 
                cm.websdr_station_id,
                ws.name as station_name,
                cm.added_at,
                cm.added_by
            FROM heimdall.constellation_members cm
            JOIN heimdall.websdr_stations ws ON cm.websdr_station_id = ws.id
            WHERE cm.constellation_id = $1
            ORDER BY cm.added_at DESC
            """,
            constellation_id,
        )
        
        members = [
            ConstellationMemberResponse(
                websdr_station_id=row["websdr_station_id"],
                station_name=row["station_name"],
                added_at=row["added_at"],
                added_by=row["added_by"],
            )
            for row in member_rows
        ]
        
        return ConstellationDetailResponse(
            id=constellation_row["id"],
            name=constellation_row["name"],
            description=constellation_row["description"],
            owner_id=constellation_row["owner_id"],
            created_at=constellation_row["created_at"],
            updated_at=constellation_row["updated_at"],
            member_count=len(members),
            members=members,
        )


@router.put("/{constellation_id}", response_model=ConstellationResponse)
async def update_constellation(
    constellation_id: UUID,
    data: ConstellationUpdate,
    user: User = Depends(get_current_user),
):
    """
    Update constellation details (name, description).
    
    - **Permission**: Edit access required (owner, shared with 'edit', or admin)
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not user.is_admin:
            can_edit = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM heimdall.constellations c
                    LEFT JOIN heimdall.constellation_shares cs ON c.id = cs.constellation_id
                    WHERE c.id = $1 AND (c.owner_id = $2 OR (cs.user_id = $2 AND cs.permission = 'edit'))
                )
                """,
                constellation_id,
                user.id,
            )
            
            if not can_edit:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have edit permission for this constellation"
                )
        
        # Build update query dynamically
        update_fields = []
        params = []
        param_idx = 1
        
        if data.name is not None:
            update_fields.append(f"name = ${param_idx}")
            params.append(data.name)
            param_idx += 1
        
        if data.description is not None:
            update_fields.append(f"description = ${param_idx}")
            params.append(data.description)
            param_idx += 1
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        update_fields.append(f"updated_at = ${param_idx}")
        params.append(datetime.utcnow())
        param_idx += 1
        
        # Add constellation_id as last parameter
        update_fields_str = ', '.join(update_fields)
        params.append(str(constellation_id))
        
        query = f"""
            UPDATE heimdall.constellations
            SET {update_fields_str}
            WHERE id = ${param_idx}
            RETURNING id, name, description, owner_id, created_at, updated_at
        """
        
        updated_row = await conn.fetchrow(query, *params)
        
        if not updated_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Constellation not found"
            )
        
        # Get member count
        member_count = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.constellation_members WHERE constellation_id = $1",
            constellation_id,
        )
        
        logger.info(f"Updated constellation {constellation_id} by user {user.id}")
        
        return ConstellationResponse(
            id=updated_row["id"],
            name=updated_row["name"],
            description=updated_row["description"],
            owner_id=updated_row["owner_id"],
            created_at=updated_row["created_at"],
            updated_at=updated_row["updated_at"],
            member_count=member_count,
        )


@router.delete("/{constellation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_constellation(
    constellation_id: UUID,
    user: User = Depends(get_current_user),
):
    """
    Delete a constellation.
    
    - **Permission**: Owner or Admin only (shared users cannot delete)
    - **Cascade**: Removes all members and shares automatically
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission (only owner or admin)
        if not user.is_admin:
            is_owner = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM heimdall.constellations WHERE id = $1 AND owner_id = $2)",
                constellation_id,
                user.id,
            )
            
            if not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can delete a constellation"
                )
        
        # Delete constellation (cascades to members and shares)
        result = await conn.execute(
            "DELETE FROM heimdall.constellations WHERE id = $1",
            constellation_id,
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Constellation not found"
            )
        
        logger.info(f"Deleted constellation {constellation_id} by user {user.id}")


# ============================================================================
# Member Management Endpoints
# ============================================================================

@router.post("/{constellation_id}/members", status_code=status.HTTP_201_CREATED)
async def add_constellation_member(
    constellation_id: UUID,
    data: ConstellationMemberAdd,
    user: User = Depends(get_current_user),
):
    """
    Add a WebSDR station to a constellation.
    
    - **Permission**: Edit access required (owner, shared with 'edit', or admin)
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not user.is_admin:
            can_edit = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM heimdall.constellations c
                    LEFT JOIN heimdall.constellation_shares cs ON c.id = cs.constellation_id
                    WHERE c.id = $1 AND (c.owner_id = $2 OR (cs.user_id = $2 AND cs.permission = 'edit'))
                )
                """,
                constellation_id,
                user.id,
            )
            
            if not can_edit:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have edit permission for this constellation"
                )
        
        # Verify WebSDR exists
        websdr_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM heimdall.websdr_stations WHERE id = $1)",
            data.websdr_station_id,
        )
        
        if not websdr_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="WebSDR station not found"
            )
        
        # Check if already a member
        already_member = await conn.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM heimdall.constellation_members 
                WHERE constellation_id = $1 AND websdr_station_id = $2
            )
            """,
            constellation_id,
            data.websdr_station_id,
        )
        
        if already_member:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="WebSDR station is already a member of this constellation"
            )
        
        # Add member
        await conn.execute(
            """
            INSERT INTO heimdall.constellation_members (constellation_id, websdr_station_id, added_by)
            VALUES ($1, $2, $3)
            """,
            constellation_id,
            data.websdr_station_id,
            user.id,
        )
        
        logger.info(f"Added WebSDR {data.websdr_station_id} to constellation {constellation_id} by user {user.id}")
        
        return {"message": "WebSDR station added successfully"}


@router.delete("/{constellation_id}/members/{websdr_station_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_constellation_member(
    constellation_id: UUID,
    websdr_station_id: UUID,
    user: User = Depends(get_current_user),
):
    """
    Remove a WebSDR station from a constellation.
    
    - **Permission**: Edit access required (owner, shared with 'edit', or admin)
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not user.is_admin:
            can_edit = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM heimdall.constellations c
                    LEFT JOIN heimdall.constellation_shares cs ON c.id = cs.constellation_id
                    WHERE c.id = $1 AND (c.owner_id = $2 OR (cs.user_id = $2 AND cs.permission = 'edit'))
                )
                """,
                constellation_id,
                user.id,
            )
            
            if not can_edit:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have edit permission for this constellation"
                )
        
        # Remove member
        result = await conn.execute(
            """
            DELETE FROM heimdall.constellation_members
            WHERE constellation_id = $1 AND websdr_station_id = $2
            """,
            constellation_id,
            websdr_station_id,
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="WebSDR station is not a member of this constellation"
            )
        
        logger.info(f"Removed WebSDR {websdr_station_id} from constellation {constellation_id} by user {user.id}")


# ============================================================================
# Sharing Endpoints (Task 9)
# ============================================================================

@router.get("/{constellation_id}/shares", response_model=List[ConstellationShareResponse])
async def list_constellation_shares(
    constellation_id: UUID,
    user: User = Depends(get_current_user),
):
    """
    List all shares for a constellation.
    
    - **Permission**: Owner or Admin only
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission (only owner or admin can view shares)
        if not user.is_admin:
            is_owner = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM heimdall.constellations WHERE id = $1 AND owner_id = $2)",
                constellation_id,
                user.id,
            )
            
            if not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can view shares"
                )
        
        # Get shares
        rows = await conn.fetch(
            """
            SELECT id, constellation_id, user_id, permission, shared_by, shared_at
            FROM heimdall.constellation_shares
            WHERE constellation_id = $1
            ORDER BY shared_at DESC
            """,
            constellation_id,
        )
        
        return [
            ConstellationShareResponse(
                id=row["id"],
                constellation_id=row["constellation_id"],
                user_id=row["user_id"],
                permission=row["permission"],
                shared_by=row["shared_by"],
                shared_at=row["shared_at"],
            )
            for row in rows
        ]


@router.post("/{constellation_id}/shares", response_model=ConstellationShareResponse, status_code=status.HTTP_201_CREATED)
async def create_constellation_share(
    constellation_id: UUID,
    data: ConstellationShareCreate,
    user: User = Depends(get_current_user),
):
    """
    Share a constellation with another user.
    
    - **Permission**: Owner or Admin only
    - **Permission levels**: 'read' (view only) or 'edit' (modify)
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission (only owner or admin can share)
        if not user.is_admin:
            is_owner = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM heimdall.constellations WHERE id = $1 AND owner_id = $2)",
                constellation_id,
                user.id,
            )
            
            if not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can share a constellation"
                )
        
        # Verify constellation exists
        constellation_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM heimdall.constellations WHERE id = $1)",
            constellation_id,
        )
        
        if not constellation_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Constellation not found"
            )
        
        # Check if already shared with this user
        existing_share = await conn.fetchrow(
            """
            SELECT id FROM heimdall.constellation_shares
            WHERE constellation_id = $1 AND user_id = $2
            """,
            constellation_id,
            data.user_id,
        )
        
        if existing_share:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Constellation is already shared with this user. Use PUT to update permission."
            )
        
        # Create share
        share_row = await conn.fetchrow(
            """
            INSERT INTO heimdall.constellation_shares (constellation_id, user_id, permission, shared_by)
            VALUES ($1, $2, $3, $4)
            RETURNING id, constellation_id, user_id, permission, shared_by, shared_at
            """,
            constellation_id,
            data.user_id,
            data.permission,
            user.id,
        )
        
        logger.info(f"Shared constellation {constellation_id} with user {data.user_id} ({data.permission}) by {user.id}")
        
        return ConstellationShareResponse(
            id=share_row["id"],
            constellation_id=share_row["constellation_id"],
            user_id=share_row["user_id"],
            permission=share_row["permission"],
            shared_by=share_row["shared_by"],
            shared_at=share_row["shared_at"],
        )


@router.put("/{constellation_id}/shares/{share_user_id}", response_model=ConstellationShareResponse)
async def update_constellation_share(
    constellation_id: UUID,
    share_user_id: str,
    data: ConstellationShareUpdate,
    user: User = Depends(get_current_user),
):
    """
    Update permission level for an existing share.
    
    - **Permission**: Owner or Admin only
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission (only owner or admin can update shares)
        if not user.is_admin:
            is_owner = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM heimdall.constellations WHERE id = $1 AND owner_id = $2)",
                constellation_id,
                user.id,
            )
            
            if not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can update shares"
                )
        
        # Update share
        share_row = await conn.fetchrow(
            """
            UPDATE heimdall.constellation_shares
            SET permission = $1
            WHERE constellation_id = $2 AND user_id = $3
            RETURNING id, constellation_id, user_id, permission, shared_by, shared_at
            """,
            data.permission,
            constellation_id,
            share_user_id,
        )
        
        if not share_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Share not found"
            )
        
        logger.info(f"Updated share for constellation {constellation_id} user {share_user_id} to {data.permission} by {user.id}")
        
        return ConstellationShareResponse(
            id=share_row["id"],
            constellation_id=share_row["constellation_id"],
            user_id=share_row["user_id"],
            permission=share_row["permission"],
            shared_by=share_row["shared_by"],
            shared_at=share_row["shared_at"],
        )


@router.delete("/{constellation_id}/shares/{share_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_constellation_share(
    constellation_id: UUID,
    share_user_id: str,
    user: User = Depends(get_current_user),
):
    """
    Remove a share (revoke access).
    
    - **Permission**: Owner or Admin only
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Check permission (only owner or admin can delete shares)
        if not user.is_admin:
            is_owner = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM heimdall.constellations WHERE id = $1 AND owner_id = $2)",
                constellation_id,
                user.id,
            )
            
            if not is_owner:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Only the owner can remove shares"
                )
        
        # Delete share
        result = await conn.execute(
            """
            DELETE FROM heimdall.constellation_shares
            WHERE constellation_id = $1 AND user_id = $2
            """,
            constellation_id,
            share_user_id,
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Share not found"
            )
        
        logger.info(f"Removed share for constellation {constellation_id} user {share_user_id} by {user.id}")
