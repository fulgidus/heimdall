"""
REST API endpoints for training job management.

Provides endpoints for:
- Creating new training jobs
- Listing training jobs with filters
- Getting job details and status
- Controlling job lifecycle (pause, resume, cancel)
- Retrieving training metrics
"""

import uuid
import json
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

# Create router with new RESTful path structure
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

# Database connection
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Database dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# REQUEST MODELS
# ============================================================================

class CreateTrainingJobRequest(BaseModel):
    """Request model for creating a new training job."""
    
    job_name: str = Field(..., description="Human-readable job name")
    model_architecture: str = Field(..., description="Model architecture ID (e.g., 'iq_resnet18')")
    
    # Training hyperparameters
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0, lt=1)
    total_epochs: int = Field(default=100, ge=1, le=1000)
    optimizer: str = Field(default="adam", description="Optimizer: adam, sgd, adamw")
    scheduler: Optional[str] = Field(default="reduce_on_plateau", description="LR scheduler")
    early_stopping_patience: int = Field(default=10, ge=0, description="Epochs to wait for improvement (0 = disabled)")
    
    # Dataset configuration
    dataset_id: Optional[str] = Field(default=None, description="Specific dataset UUID")
    train_split: float = Field(default=0.8, gt=0, lt=1)
    val_split: float = Field(default=0.2, gt=0, lt=1)
    
    # Hardware configuration
    use_gpu: bool = Field(default=True)
    num_workers: int = Field(default=4, ge=0, le=16)
    
    # Continuation training
    parent_model_id: Optional[str] = Field(default=None, description="Parent model for continuation")
    
    # Additional config
    config: Optional[dict] = Field(default=None, description="Additional configuration")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TrainingJobResponse(BaseModel):
    """Training job response model."""
    
    id: str
    job_name: str
    celery_task_id: Optional[str] = None
    status: str  # pending, running, completed, failed, cancelled, paused
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Configuration
    model_architecture: Optional[str] = None
    config: Optional[dict] = None
    
    # Progress
    current_epoch: int = 0
    total_epochs: int
    progress_percent: float = 0.0
    
    # Metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Best model info
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    
    # Artifacts
    checkpoint_path: Optional[str] = None
    onnx_model_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    
    # Dataset info
    dataset_size: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class TrainingJobListResponse(BaseModel):
    """List of training jobs with pagination."""
    
    jobs: List[TrainingJobResponse]
    total: int
    limit: int
    offset: int


class TrainingMetricPoint(BaseModel):
    """Single metric data point."""
    
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: datetime


class TrainingMetricsResponse(BaseModel):
    """Training metrics time series."""
    
    job_id: str
    metrics: List[TrainingMetricPoint]
    total_points: int


class JobActionResponse(BaseModel):
    """Response for job control actions."""
    
    job_id: str
    status: str
    message: str
    celery_task_id: Optional[str] = None


# ============================================================================
# API ENDPOINTS - TRAINING JOBS
# ============================================================================

@router.post(
    "/training",
    response_model=TrainingJobResponse,
    summary="Create new training job",
    description="Start a new model training job with specified configuration.",
)
async def create_training_job(
    request: CreateTrainingJobRequest,
    db: Session = Depends(get_db)
) -> TrainingJobResponse:
    """
    Create and start a new training job.
    
    Args:
        request: Training configuration
        db: Database session
    
    Returns:
        Created training job details
    """
    from ..tasks.training_task import start_training_job
    
    job_id = str(uuid.uuid4())
    
    # Build configuration
    config = {
        "model_architecture": request.model_architecture,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate,
        "optimizer": request.optimizer,
        "scheduler": request.scheduler,
        "early_stopping_patience": request.early_stopping_patience,
        "dataset_id": request.dataset_id,
        "train_split": request.train_split,
        "val_split": request.val_split,
        "use_gpu": request.use_gpu,
        "num_workers": request.num_workers,
        "parent_model_id": request.parent_model_id,
    }
    
    # Merge additional config
    if request.config:
        config.update(request.config)
    
    try:
        # Create job record
        insert_query = text("""
            INSERT INTO heimdall.training_jobs (
                id, job_name, status, config, total_epochs, model_architecture,
                created_at, updated_at
            )
            VALUES (
                :id, :job_name, 'pending', CAST(:config AS jsonb), :total_epochs,
                :model_architecture, NOW(), NOW()
            )
            RETURNING id, job_name, status, created_at, total_epochs, model_architecture
        """)
        
        result = db.execute(
            insert_query,
            {
                "id": job_id,
                "job_name": request.job_name,
                "config": json.dumps(config),
                "total_epochs": request.total_epochs,
                "model_architecture": request.model_architecture,
            }
        )
        db.commit()
        
        row = result.fetchone()
        
        # Submit Celery task
        task = start_training_job.apply_async(
            args=[job_id],
            task_id=job_id,
            queue='training'
        )
        
        # Update celery_task_id
        update_query = text("""
            UPDATE heimdall.training_jobs
            SET celery_task_id = :task_id
            WHERE id = :job_id
        """)
        db.execute(update_query, {"task_id": task.id, "job_id": job_id})
        db.commit()
        
        logger.info(
            "training_job_created",
            job_id=job_id,
            job_name=request.job_name,
            model_architecture=request.model_architecture,
            celery_task_id=task.id
        )
        
        return TrainingJobResponse(
            id=str(row[0]),
            job_name=row[1],
            status=row[2],
            created_at=row[3],
            total_epochs=row[4],
            model_architecture=row[5],
            celery_task_id=task.id,
            config=config,
            current_epoch=0,
            progress_percent=0.0
        )
        
    except Exception as e:
        db.rollback()
        logger.error("training_job_creation_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create training job: {str(e)}"
        )


@router.get(
    "/training",
    response_model=TrainingJobListResponse,
    summary="List training jobs",
    description="Get list of training jobs with optional filtering.",
)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
) -> TrainingJobListResponse:
    """
    List training jobs with optional filtering.
    
    Args:
        status: Filter by job status (pending, running, completed, etc.)
        limit: Maximum number of jobs to return
        offset: Pagination offset
        db: Database session
    
    Returns:
        List of training jobs
    """
    try:
        # Build query
        where_clause = ""
        params = {"limit": limit, "offset": offset}
        
        if status:
            where_clause = "WHERE status = :status"
            params["status"] = status
        
        # Query jobs
        query = text(f"""
            SELECT 
                id, job_name, celery_task_id, status, created_at, started_at,
                completed_at, updated_at, model_architecture, config, current_epoch,
                total_epochs, progress_percent, train_loss, val_loss, train_accuracy,
                val_accuracy, learning_rate, best_epoch, best_val_loss, checkpoint_path,
                onnx_model_path, mlflow_run_id, dataset_size, train_samples, val_samples,
                error_message
            FROM heimdall.training_jobs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        
        results = db.execute(query, params).fetchall()
        
        # Count total
        count_query = text(f"""
            SELECT COUNT(*) FROM heimdall.training_jobs {where_clause}
        """)
        total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
        
        jobs = []
        for row in results:
            jobs.append(TrainingJobResponse(
                id=str(row[0]),
                job_name=row[1],
                celery_task_id=row[2],
                status=row[3],
                created_at=row[4],
                started_at=row[5],
                completed_at=row[6],
                updated_at=row[7],
                model_architecture=row[8],
                config=row[9],
                current_epoch=row[10] or 0,
                total_epochs=row[11],
                progress_percent=row[12] or 0.0,
                train_loss=row[13],
                val_loss=row[14],
                train_accuracy=row[15],
                val_accuracy=row[16],
                learning_rate=row[17],
                best_epoch=row[18],
                best_val_loss=row[19],
                checkpoint_path=row[20],
                onnx_model_path=row[21],
                mlflow_run_id=row[22],
                dataset_size=row[23],
                train_samples=row[24],
                val_samples=row[25],
                error_message=row[26]
            ))
        
        logger.info(
            "training_jobs_listed",
            total=total,
            returned=len(jobs),
            status_filter=status
        )
        
        return TrainingJobListResponse(
            jobs=jobs,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error("training_jobs_list_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list training jobs: {str(e)}"
        )


@router.get(
    "/training/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get training job details",
    description="Get detailed information about a specific training job.",
)
async def get_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> TrainingJobResponse:
    """
    Get details for a specific training job.
    
    Args:
        job_id: Job UUID
        db: Database session
    
    Returns:
        Training job details
    
    Raises:
        404: Job not found
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        query = text("""
            SELECT 
                id, job_name, celery_task_id, status, created_at, started_at,
                completed_at, updated_at, model_architecture, config, current_epoch,
                total_epochs, progress_percent, train_loss, val_loss, train_accuracy,
                val_accuracy, learning_rate, best_epoch, best_val_loss, checkpoint_path,
                onnx_model_path, mlflow_run_id, dataset_size, train_samples, val_samples,
                error_message
            FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        return TrainingJobResponse(
            id=str(result[0]),
            job_name=result[1],
            celery_task_id=result[2],
            status=result[3],
            created_at=result[4],
            started_at=result[5],
            completed_at=result[6],
            updated_at=result[7],
            model_architecture=result[8],
            config=result[9],
            current_epoch=result[10] or 0,
            total_epochs=result[11],
            progress_percent=result[12] or 0.0,
            train_loss=result[13],
            val_loss=result[14],
            train_accuracy=result[15],
            val_accuracy=result[16],
            learning_rate=result[17],
            best_epoch=result[18],
            best_val_loss=result[19],
            checkpoint_path=result[20],
            onnx_model_path=result[21],
            mlflow_run_id=result[22],
            dataset_size=result[23],
            train_samples=result[24],
            val_samples=result[25],
            error_message=result[26]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("training_job_get_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training job: {str(e)}"
        )


@router.post(
    "/training/{job_id}/cancel",
    response_model=JobActionResponse,
    summary="Cancel training job",
    description="Cancel a running or pending training job.",
)
async def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> JobActionResponse:
    """Cancel a training job."""
    from celery import current_app
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Get celery task ID
        query = text("""
            SELECT celery_task_id, status FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        celery_task_id, current_status = result
        
        # Revoke Celery task
        if celery_task_id:
            current_app.control.revoke(celery_task_id, terminate=True, signal='SIGTERM')
        
        # Update status
        update_query = text("""
            UPDATE heimdall.training_jobs
            SET status = 'cancelled', completed_at = NOW(), updated_at = NOW()
            WHERE id = :job_id
        """)
        db.execute(update_query, {"job_id": str(job_uuid)})
        db.commit()
        
        logger.info("training_job_cancelled", job_id=job_id, celery_task_id=celery_task_id)
        
        return JobActionResponse(
            job_id=job_id,
            status="cancelled",
            message="Training job cancelled successfully",
            celery_task_id=celery_task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("training_job_cancel_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel training job: {str(e)}"
        )


@router.post(
    "/training/{job_id}/pause",
    response_model=JobActionResponse,
    summary="Pause training job",
    description="Pause a running training job (saves checkpoint).",
)
async def pause_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> JobActionResponse:
    """Pause a running training job."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Check if job exists and is running
        query = text("""
            SELECT status, celery_task_id FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        status, celery_task_id = result
        
        if status != 'running':
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pause job in status '{status}' (must be 'running')"
            )
        
        # Signal pause by updating status (training task monitors this)
        update_query = text("""
            UPDATE heimdall.training_jobs
            SET status = 'pausing', updated_at = NOW()
            WHERE id = :job_id
        """)
        db.execute(update_query, {"job_id": str(job_uuid)})
        db.commit()
        
        logger.info("training_job_pausing", job_id=job_id)
        
        return JobActionResponse(
            job_id=job_id,
            status="pausing",
            message="Training job pausing (will complete current epoch)",
            celery_task_id=celery_task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("training_job_pause_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause training job: {str(e)}"
        )


@router.post(
    "/training/{job_id}/resume",
    response_model=JobActionResponse,
    summary="Resume training job",
    description="Resume a paused training job from checkpoint.",
)
async def resume_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> JobActionResponse:
    """Resume a paused training job."""
    from ..tasks.training_task import start_training_job
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Check if job exists and is paused
        query = text("""
            SELECT status, pause_checkpoint_path FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        status, checkpoint_path = result
        
        if status != 'paused':
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume job in status '{status}' (must be 'paused')"
            )
        
        if not checkpoint_path:
            raise HTTPException(
                status_code=400,
                detail="No pause checkpoint found for this job"
            )
        
        # Submit new Celery task to resume
        task = start_training_job.apply_async(
            args=[job_id],
            task_id=f"{job_id}-resume-{uuid.uuid4().hex[:8]}",
            queue='training'
        )
        
        # Update job status
        update_query = text("""
            UPDATE heimdall.training_jobs
            SET status = 'pending', celery_task_id = :task_id, updated_at = NOW()
            WHERE id = :job_id
        """)
        db.execute(update_query, {"task_id": task.id, "job_id": str(job_uuid)})
        db.commit()
        
        logger.info("training_job_resuming", job_id=job_id, celery_task_id=task.id)
        
        return JobActionResponse(
            job_id=job_id,
            status="pending",
            message="Training job resuming from checkpoint",
            celery_task_id=task.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("training_job_resume_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume training job: {str(e)}"
        )


@router.post(
    "/training/{job_id}/continue",
    response_model=JobActionResponse,
    summary="Continue training",
    description="Continue training a completed model with more epochs.",
)
async def continue_training_job(
    job_id: str,
    additional_epochs: int = Query(default=10, ge=1, le=1000),
    db: Session = Depends(get_db)
) -> JobActionResponse:
    """Continue training a completed model with additional epochs."""
    from ..tasks.training_task import start_training_job
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Check if job exists and is completed
        query = text("""
            SELECT status, checkpoint_path, config, job_name, model_architecture, total_epochs
            FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        status, checkpoint_path, config, job_name, model_arch, completed_epochs = result
        
        if status != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Cannot continue job in status '{status}' (must be 'completed')"
            )
        
        if not checkpoint_path:
            raise HTTPException(
                status_code=400,
                detail="No checkpoint found for continuation"
            )
        
        # Create new job for continuation
        new_job_id = str(uuid.uuid4())
        new_config = config.copy() if isinstance(config, dict) else {}
        new_config["parent_model_id"] = job_id
        new_config["continuation"] = True
        new_config["checkpoint_path"] = checkpoint_path
        
        insert_query = text("""
            INSERT INTO heimdall.training_jobs (
                id, job_name, status, config, total_epochs, model_architecture,
                current_epoch, created_at, updated_at
            )
            VALUES (
                :id, :job_name, 'pending', CAST(:config AS jsonb), :total_epochs,
                :model_architecture, :current_epoch, NOW(), NOW()
            )
            RETURNING id
        """)
        
        db.execute(
            insert_query,
            {
                "id": new_job_id,
                "job_name": f"{job_name} (continued)",
                "config": json.dumps(new_config),
                "total_epochs": completed_epochs + additional_epochs,
                "model_architecture": model_arch,
                "current_epoch": completed_epochs
            }
        )
        
        # Submit Celery task
        task = start_training_job.apply_async(
            args=[new_job_id],
            task_id=new_job_id,
            queue='training'
        )
        
        # Update celery_task_id
        update_query = text("""
            UPDATE heimdall.training_jobs
            SET celery_task_id = :task_id
            WHERE id = :job_id
        """)
        db.execute(update_query, {"task_id": task.id, "job_id": new_job_id})
        db.commit()
        
        logger.info(
            "training_job_continued",
            original_job_id=job_id,
            new_job_id=new_job_id,
            additional_epochs=additional_epochs
        )
        
        return JobActionResponse(
            job_id=new_job_id,
            status="pending",
            message=f"Created continuation job with {additional_epochs} additional epochs",
            celery_task_id=task.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("training_job_continue_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to continue training job: {str(e)}"
        )


@router.delete(
    "/training/{job_id}",
    summary="Delete training job",
    description="Delete a training job and its associated data.",
)
async def delete_training_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """Delete a training job."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Check if job exists
        query = text("""
            SELECT status, celery_task_id FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        status, celery_task_id = result
        
        # Don't allow deletion of running jobs
        if status == 'running':
            raise HTTPException(
                status_code=400,
                detail="Cannot delete running job. Cancel it first."
            )
        
        # Delete associated metrics first
        delete_metrics = text("""
            DELETE FROM heimdall.training_metrics
            WHERE training_job_id = :job_id
        """)
        db.execute(delete_metrics, {"job_id": str(job_uuid)})
        
        # Delete job
        delete_query = text("""
            DELETE FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        db.execute(delete_query, {"job_id": str(job_uuid)})
        db.commit()
        
        logger.info("training_job_deleted", job_id=job_id)
        
        return {"message": "Training job deleted successfully", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("training_job_delete_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete training job: {str(e)}"
        )


@router.get(
    "/training/{job_id}/metrics",
    response_model=TrainingMetricsResponse,
    summary="Get training metrics",
    description="Get time-series training metrics for a job.",
)
async def get_training_metrics(
    job_id: str,
    limit: int = Query(default=1000, ge=1, le=10000),
    db: Session = Depends(get_db)
) -> TrainingMetricsResponse:
    """Get training metrics time series."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Verify job exists
        check_query = text("""
            SELECT id FROM heimdall.training_jobs WHERE id = :job_id
        """)
        if not db.execute(check_query, {"job_id": str(job_uuid)}).fetchone():
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        # Get metrics
        query = text("""
            SELECT 
                epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                learning_rate, timestamp
            FROM heimdall.training_metrics
            WHERE training_job_id = :job_id
            ORDER BY epoch ASC
            LIMIT :limit
        """)
        
        results = db.execute(query, {"job_id": str(job_uuid), "limit": limit}).fetchall()
        
        metrics = []
        for row in results:
            metrics.append(TrainingMetricPoint(
                epoch=row[0],
                train_loss=row[1],
                val_loss=row[2],
                train_accuracy=row[3],
                val_accuracy=row[4],
                learning_rate=row[5],
                timestamp=row[6]
            ))
        
        logger.info("training_metrics_retrieved", job_id=job_id, points=len(metrics))
        
        return TrainingMetricsResponse(
            job_id=job_id,
            metrics=metrics,
            total_points=len(metrics)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("training_metrics_get_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training metrics: {str(e)}"
        )


# ============================================================================
# API ENDPOINTS - SYNTHETIC GENERATION JOBS (for completeness)
# ============================================================================

@router.get(
    "/synthetic",
    summary="List synthetic generation jobs",
    description="Get list of synthetic data generation jobs.",
)
async def list_synthetic_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
) -> dict:
    """List synthetic data generation jobs."""
    try:
        where_clause = "WHERE job_type = 'synthetic_generation'"
        params = {"limit": limit, "offset": offset}
        
        if status:
            where_clause += " AND status = :status"
            params["status"] = status
        
        query = text(f"""
            SELECT 
                id, job_name, status, current_progress, total_progress,
                progress_percent, created_at, started_at, completed_at, error_message
            FROM heimdall.training_jobs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """)
        
        results = db.execute(query, params).fetchall()
        
        count_query = text(f"""
            SELECT COUNT(*) FROM heimdall.training_jobs {where_clause}
        """)
        total = db.execute(count_query, {k: v for k, v in params.items() if k not in ['limit', 'offset']}).scalar()
        
        jobs = []
        for row in results:
            jobs.append({
                "id": str(row[0]),
                "job_name": row[1],
                "status": row[2],
                "current_progress": row[3],
                "total_progress": row[4],
                "progress_percent": row[5],
                "created_at": row[6],
                "started_at": row[7],
                "completed_at": row[8],
                "error_message": row[9]
            })
        
        return {
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error("synthetic_jobs_list_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list synthetic jobs: {str(e)}"
        )


@router.get(
    "/synthetic/{job_id}",
    summary="Get synthetic job details",
    description="Get details for a specific synthetic generation job.",
)
async def get_synthetic_job(
    job_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """Get synthetic generation job details."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        query = text("""
            SELECT 
                id, job_name, job_type, status, current_progress, total_progress,
                progress_percent, progress_message, created_at, started_at,
                completed_at, error_message
            FROM heimdall.training_jobs
            WHERE id = :job_id AND job_type = 'synthetic_generation'
        """)
        
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Synthetic job not found: {job_id}")
        
        return {
            "job_id": str(result[0]),
            "job_name": result[1],
            "job_type": result[2],
            "status": result[3],
            "current_progress": result[4],
            "total_progress": result[5],
            "progress_percent": result[6],
            "progress_message": result[7],
            "created_at": result[8],
            "started_at": result[9],
            "completed_at": result[10],
            "error_message": result[11]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("synthetic_job_get_failed", job_id=job_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get synthetic job: {str(e)}"
        )
