"""
ML Models API endpoints - RBAC-enabled

This router provides CRUD operations and sharing management for ML models.
Models are trained neural networks used for radio source localization.

Permission Logic:
- Admins: Full access to all models
- Operators: Can create models (via training jobs), edit owned/shared models
- Users: Can view models shared with them (read-only)
- Owners: Full control (view, edit, delete, share, deploy)
- Shared users: Permission based on share level ('read' or 'edit')
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field
import asyncpg
import logging
import json

from ..db import get_pool
from common.auth import (
    get_current_user,
    require_operator,
    User,
    can_view_model,
    can_edit_model,
    can_delete_model,
    get_user_models,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class ModelShareCreate(BaseModel):
    """Request model for sharing a model"""
    user_id: str = Field(..., min_length=1, description="Keycloak user ID to share with")
    permission: str = Field(..., pattern="^(read|edit)$", description="Permission level")


class ModelShareUpdate(BaseModel):
    """Request model for updating a share"""
    permission: str = Field(..., pattern="^(read|edit)$", description="Permission level")


class ModelShare(BaseModel):
    """Response model for a model share"""
    id: UUID
    model_id: UUID
    user_id: str
    permission: str
    shared_by: str
    shared_at: datetime


class ModelUpdate(BaseModel):
    """Request model for updating a model"""
    model_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Model name")


class ModelMetadataResponse(BaseModel):
    """Response model for model metadata"""
    id: UUID
    model_name: str
    version: int
    model_type: str
    synthetic_dataset_id: Optional[UUID]
    mlflow_run_id: Optional[str]
    mlflow_experiment_id: Optional[str]
    onnx_model_location: Optional[str]
    pytorch_model_location: Optional[str]
    accuracy_meters: Optional[float]
    accuracy_sigma_meters: Optional[float]
    loss_value: Optional[float]
    epoch: Optional[int]
    is_active: bool
    is_production: bool
    hyperparameters: Optional[dict]
    training_metrics: Optional[dict]
    test_metrics: Optional[dict]
    created_at: datetime
    trained_by_job_id: Optional[UUID]
    parent_model_id: Optional[UUID]
    owner_id: Optional[str]
    is_owner: bool = False
    permission: Optional[str] = None


class ModelListResponse(BaseModel):
    """Response for model list"""
    models: List[ModelMetadataResponse]
    total: int
    page: int
    per_page: int


# ============================================================================
# CRUD Endpoints
# ============================================================================

@router.get("", response_model=ModelListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=500, description="Items per page"),
    active_only: bool = Query(False, description="Only show active models"),
    current_user: User = Depends(get_current_user)
):
    """
    List all models accessible to the current user.
    
    Permission Logic:
    - Admins: See all models
    - Others: See only owned models + shared models
    
    Args:
        page: Page number (1-indexed)
        per_page: Items per page
        active_only: Filter for active models only
        current_user: Current authenticated user
    
    Returns:
        Paginated list of models with ownership info
    """
    pool = await get_pool()
    
    offset = (page - 1) * per_page
    
    async with pool.acquire() as conn:
        # Get accessible model IDs for non-admins
        accessible_ids = None
        if not current_user.is_admin:
            accessible_ids = await get_user_models(conn, current_user.id, current_user.is_admin)
            
            if not accessible_ids:
                # User has no accessible models
                return ModelListResponse(models=[], total=0, page=page, per_page=per_page)
        
        # Build query
        where_clauses = []
        params = {"limit": per_page, "offset": offset}
        
        if active_only:
            where_clauses.append("m.is_active = true")
        
        if accessible_ids is not None:
            # Filter by accessible IDs
            where_clauses.append("m.id = ANY($3::uuid[])")
            params["accessible_ids"] = accessible_ids
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # Count query
        count_query = f"""
            SELECT COUNT(*)
            FROM heimdall.models m
            {where_sql}
        """
        
        # Data query with ownership info
        data_query = f"""
            SELECT 
                m.id, m.model_name, m.version, m.model_type, m.synthetic_dataset_id,
                m.mlflow_run_id, m.mlflow_experiment_id, m.onnx_model_location,
                m.pytorch_model_location, m.accuracy_meters, m.accuracy_sigma_meters,
                m.loss_value, m.epoch, m.is_active, m.is_production, m.hyperparameters,
                m.training_metrics, m.test_metrics, m.created_at, m.trained_by_job_id,
                m.parent_model_id, m.owner_id,
                ms.permission as share_permission
            FROM heimdall.models m
            LEFT JOIN heimdall.model_shares ms ON ms.model_id = m.id AND ms.user_id = $4
            {where_sql}
            ORDER BY m.created_at DESC
            LIMIT $1 OFFSET $2
        """
        
        # Execute queries
        if accessible_ids is not None:
            total = await conn.fetchval(count_query, accessible_ids)
            rows = await conn.fetch(data_query, per_page, offset, accessible_ids, current_user.id)
        else:
            total = await conn.fetchval(count_query)
            rows = await conn.fetch(data_query, per_page, offset, current_user.id)
        
        models = []
        for row in rows:
            is_owner = row['owner_id'] == current_user.id if row['owner_id'] else False
            permission = row['share_permission'] if not is_owner else None
            
            models.append(ModelMetadataResponse(
                id=row['id'],
                model_name=row['model_name'],
                version=row['version'] or 1,
                model_type=row['model_type'],
                synthetic_dataset_id=row['synthetic_dataset_id'],
                mlflow_run_id=row['mlflow_run_id'],
                mlflow_experiment_id=row['mlflow_experiment_id'],
                onnx_model_location=row['onnx_model_location'],
                pytorch_model_location=row['pytorch_model_location'],
                accuracy_meters=row['accuracy_meters'],
                accuracy_sigma_meters=row['accuracy_sigma_meters'],
                loss_value=row['loss_value'],
                epoch=row['epoch'],
                is_active=row['is_active'],
                is_production=row['is_production'],
                hyperparameters=row['hyperparameters'],
                training_metrics=row['training_metrics'],
                test_metrics=row['test_metrics'],
                created_at=row['created_at'],
                trained_by_job_id=row['trained_by_job_id'],
                parent_model_id=row['parent_model_id'],
                owner_id=row['owner_id'],
                is_owner=is_owner,
                permission=permission
            ))
        
        return ModelListResponse(models=models, total=total, page=page, per_page=per_page)


@router.get("/{model_id}", response_model=ModelMetadataResponse)
async def get_model(
    model_id: UUID,
    current_user: User = Depends(get_current_user)
):
    """
    Get model details by ID.
    
    Permission Logic:
    - Admins: Can view any model
    - Owners: Can view their models
    - Shared users: Can view if shared with 'read' or 'edit'
    
    Args:
        model_id: Model UUID
        current_user: Current authenticated user
    
    Returns:
        Model metadata with ownership info
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not current_user.is_admin:
            has_permission = await can_view_model(conn, current_user.id, model_id, current_user.is_admin)
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to view this model"
                )
        
        # Fetch model
        query = """
            SELECT 
                m.id, m.model_name, m.version, m.model_type, m.synthetic_dataset_id,
                m.mlflow_run_id, m.mlflow_experiment_id, m.onnx_model_location,
                m.pytorch_model_location, m.accuracy_meters, m.accuracy_sigma_meters,
                m.loss_value, m.epoch, m.is_active, m.is_production, m.hyperparameters,
                m.training_metrics, m.test_metrics, m.created_at, m.trained_by_job_id,
                m.parent_model_id, m.owner_id,
                ms.permission as share_permission
            FROM heimdall.models m
            LEFT JOIN heimdall.model_shares ms ON ms.model_id = m.id AND ms.user_id = $2
            WHERE m.id = $1
        """
        
        row = await conn.fetchrow(query, model_id, current_user.id)
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        is_owner = row['owner_id'] == current_user.id if row['owner_id'] else False
        permission = row['share_permission'] if not is_owner else None
        
        return ModelMetadataResponse(
            id=row['id'],
            model_name=row['model_name'],
            version=row['version'] or 1,
            model_type=row['model_type'],
            synthetic_dataset_id=row['synthetic_dataset_id'],
            mlflow_run_id=row['mlflow_run_id'],
            mlflow_experiment_id=row['mlflow_experiment_id'],
            onnx_model_location=row['onnx_model_location'],
            pytorch_model_location=row['pytorch_model_location'],
            accuracy_meters=row['accuracy_meters'],
            accuracy_sigma_meters=row['accuracy_sigma_meters'],
            loss_value=row['loss_value'],
            epoch=row['epoch'],
            is_active=row['is_active'],
            is_production=row['is_production'],
            hyperparameters=row['hyperparameters'],
            training_metrics=row['training_metrics'],
            test_metrics=row['test_metrics'],
            created_at=row['created_at'],
            trained_by_job_id=row['trained_by_job_id'],
            parent_model_id=row['parent_model_id'],
            owner_id=row['owner_id'],
            is_owner=is_owner,
            permission=permission
        )


@router.patch("/{model_id}", response_model=ModelMetadataResponse)
async def update_model(
    model_id: UUID,
    update: ModelUpdate,
    current_user: User = Depends(require_operator)
):
    """
    Update model metadata (currently only name).
    
    Permission Logic:
    - Admins: Can update any model
    - Owners: Can update their models
    - Shared users with 'edit': Can update
    - Others: Forbidden
    
    Args:
        model_id: Model UUID
        update: Model update data
        current_user: Current authenticated user (must be operator or higher)
    
    Returns:
        Updated model metadata
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not current_user.is_admin:
            has_permission = await can_edit_model(conn, current_user.id, model_id, current_user.is_admin)
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to edit this model"
                )
        
        # Check if model exists
        exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM heimdall.models WHERE id = $1)", model_id)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Update model
        if update.model_name:
            await conn.execute(
                "UPDATE heimdall.models SET model_name = $1 WHERE id = $2",
                update.model_name, model_id
            )
            logger.info(f"User {current_user.id} updated model {model_id} name to '{update.model_name}'")
            
            # Broadcast WebSocket event
            from ..events.publisher import get_event_publisher
            publisher = get_event_publisher()
            event_data = {
                'event': 'model:name_updated',
                'timestamp': datetime.utcnow().isoformat(),
                'data': {
                    'model_id': str(model_id),
                    'model_name': update.model_name
                }
            }
            publisher._publish('model.name.updated', event_data)
        
        # Return updated model
        return await get_model(model_id, current_user)


@router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: UUID,
    set_production: bool = Query(False, description="Also set as production model"),
    current_user: User = Depends(require_operator)
):
    """
    Deploy model (set as active for inference).
    
    Permission Logic:
    - Admins: Can deploy any model
    - Owners: Can deploy their models
    - Shared users with 'edit': Can deploy
    - Others: Forbidden
    
    Args:
        model_id: Model UUID
        set_production: Also set as production model
        current_user: Current authenticated user (must be operator or higher)
    
    Returns:
        Deployment status
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not current_user.is_admin:
            has_permission = await can_edit_model(conn, current_user.id, model_id, current_user.is_admin)
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to deploy this model"
                )
        
        # Check if model exists
        exists = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM heimdall.models WHERE id = $1)", model_id)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Start transaction
        async with conn.transaction():
            # Deactivate other models
            await conn.execute("UPDATE heimdall.models SET is_active = FALSE")
            
            if set_production:
                await conn.execute("UPDATE heimdall.models SET is_production = FALSE")
            
            # Activate this model
            await conn.execute(
                """
                UPDATE heimdall.models
                SET is_active = TRUE, is_production = $2, updated_at = NOW()
                WHERE id = $1
                """,
                model_id, set_production
            )
        
        logger.info(f"User {current_user.id} deployed model {model_id} (production={set_production})")
        
        # Broadcast WebSocket event
        from .websocket import manager as ws_manager
        await ws_manager.broadcast({
            "event": "model_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "model_id": str(model_id),
                "action": "deployed",
                "is_production": set_production,
            }
        })
        
        return {"status": "deployed", "model_id": str(model_id), "is_production": set_production}


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: UUID,
    current_user: User = Depends(require_operator)
):
    """
    Delete model and associated artifacts.
    
    Permission Logic:
    - Admins: Can delete any model (even if active)
    - Owners: Can delete their models (only if not active)
    - Shared users: Cannot delete (even with 'edit' permission)
    - Others: Forbidden
    
    Args:
        model_id: Model UUID
        current_user: Current authenticated user (must be operator or higher)
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check permission
        if not current_user.is_admin:
            has_permission = await can_delete_model(conn, current_user.id, model_id, current_user.is_admin)
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to delete this model"
                )
        
        # Check if model exists and get status
        row = await conn.fetchrow(
            "SELECT is_active, owner_id FROM heimdall.models WHERE id = $1",
            model_id
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Prevent deletion of active models (unless admin)
        if row['is_active'] and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete active model. Deactivate first."
            )
        
        # Delete model (CASCADE will delete shares)
        await conn.execute("DELETE FROM heimdall.models WHERE id = $1", model_id)
        
        logger.info(f"User {current_user.id} deleted model {model_id}")
        
        # Broadcast WebSocket event
        from .websocket import manager as ws_manager
        await ws_manager.broadcast({
            "event": "model_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "model_id": str(model_id),
                "action": "deleted",
            }
        })


# ============================================================================
# Sharing Endpoints
# ============================================================================

@router.get("/{model_id}/shares", response_model=List[ModelShare])
async def list_model_shares(
    model_id: UUID,
    current_user: User = Depends(require_operator)
):
    """
    List all shares for a model.
    
    Only the owner or admins can view shares.
    
    Args:
        model_id: Model UUID
        current_user: Current authenticated user (must be operator or higher)
    
    Returns:
        List of model shares
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if model exists and get owner
        row = await conn.fetchrow(
            "SELECT owner_id FROM heimdall.models WHERE id = $1",
            model_id
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Check permission (owner or admin)
        is_owner = row['owner_id'] == current_user.id
        if not current_user.is_admin and not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the owner or admins can view shares"
            )
        
        # Fetch shares
        query = """
            SELECT id, model_id, user_id, permission, shared_by, shared_at
            FROM heimdall.model_shares
            WHERE model_id = $1
            ORDER BY shared_at DESC
        """
        
        rows = await conn.fetch(query, model_id)
        
        return [
            ModelShare(
                id=row['id'],
                model_id=row['model_id'],
                user_id=row['user_id'],
                permission=row['permission'],
                shared_by=row['shared_by'],
                shared_at=row['shared_at']
            )
            for row in rows
        ]


@router.post("/{model_id}/shares", response_model=ModelShare, status_code=status.HTTP_201_CREATED)
async def create_model_share(
    model_id: UUID,
    share: ModelShareCreate,
    current_user: User = Depends(require_operator)
):
    """
    Share a model with another user.
    
    Only the owner or admins can create shares.
    
    Args:
        model_id: Model UUID
        share: Share creation data
        current_user: Current authenticated user (must be operator or higher)
    
    Returns:
        Created share
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if model exists and get owner
        row = await conn.fetchrow(
            "SELECT owner_id FROM heimdall.models WHERE id = $1",
            model_id
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Check permission (owner or admin)
        is_owner = row['owner_id'] == current_user.id
        if not current_user.is_admin and not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the owner or admins can share models"
            )
        
        # Cannot share with self
        if share.user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot share model with yourself"
            )
        
        # Create share (ON CONFLICT UPDATE)
        query = """
            INSERT INTO heimdall.model_shares (model_id, user_id, permission, shared_by)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (model_id, user_id) DO UPDATE
            SET permission = EXCLUDED.permission, shared_by = EXCLUDED.shared_by
            RETURNING id, model_id, user_id, permission, shared_by, shared_at
        """
        
        result = await conn.fetchrow(
            query,
            model_id, share.user_id, share.permission, current_user.id
        )
        
        logger.info(
            f"User {current_user.id} shared model {model_id} with {share.user_id} "
            f"(permission={share.permission})"
        )
        
        return ModelShare(
            id=result['id'],
            model_id=result['model_id'],
            user_id=result['user_id'],
            permission=result['permission'],
            shared_by=result['shared_by'],
            shared_at=result['shared_at']
        )


@router.put("/{model_id}/shares/{user_id}", response_model=ModelShare)
async def update_model_share(
    model_id: UUID,
    user_id: str,
    share_update: ModelShareUpdate,
    current_user: User = Depends(require_operator)
):
    """
    Update a model share's permission level.
    
    Only the owner or admins can update shares.
    
    Args:
        model_id: Model UUID
        user_id: User ID to update share for
        share_update: Updated share data
        current_user: Current authenticated user (must be operator or higher)
    
    Returns:
        Updated share
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if model exists and get owner
        row = await conn.fetchrow(
            "SELECT owner_id FROM heimdall.models WHERE id = $1",
            model_id
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Check permission (owner or admin)
        is_owner = row['owner_id'] == current_user.id
        if not current_user.is_admin and not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the owner or admins can update shares"
            )
        
        # Update share
        query = """
            UPDATE heimdall.model_shares
            SET permission = $3
            WHERE model_id = $1 AND user_id = $2
            RETURNING id, model_id, user_id, permission, shared_by, shared_at
        """
        
        result = await conn.fetchrow(query, model_id, user_id, share_update.permission)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Share not found for user {user_id}"
            )
        
        logger.info(
            f"User {current_user.id} updated share for model {model_id} user {user_id} "
            f"(permission={share_update.permission})"
        )
        
        return ModelShare(
            id=result['id'],
            model_id=result['model_id'],
            user_id=result['user_id'],
            permission=result['permission'],
            shared_by=result['shared_by'],
            shared_at=result['shared_at']
        )


@router.delete("/{model_id}/shares/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_share(
    model_id: UUID,
    user_id: str,
    current_user: User = Depends(require_operator)
):
    """
    Remove a model share (revoke access).
    
    Only the owner or admins can delete shares.
    
    Args:
        model_id: Model UUID
        user_id: User ID to revoke access from
        current_user: Current authenticated user (must be operator or higher)
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if model exists and get owner
        row = await conn.fetchrow(
            "SELECT owner_id FROM heimdall.models WHERE id = $1",
            model_id
        )
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        # Check permission (owner or admin)
        is_owner = row['owner_id'] == current_user.id
        if not current_user.is_admin and not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the owner or admins can delete shares"
            )
        
        # Delete share
        result = await conn.execute(
            "DELETE FROM heimdall.model_shares WHERE model_id = $1 AND user_id = $2",
            model_id, user_id
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Share not found for user {user_id}"
            )
        
        logger.info(f"User {current_user.id} removed share for model {model_id} user {user_id}")
