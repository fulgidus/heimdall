"""
Training API router.

Endpoints for managing training jobs:
- POST /training/jobs - Create new training job
- GET /training/jobs - List all training jobs
- GET /training/jobs/{job_id} - Get job details with metrics
- DELETE /training/jobs/{job_id} - Cancel/delete training job
- GET /training/jobs/{job_id}/metrics - Get detailed metrics history
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import Response
from sqlalchemy import text

from ..models.training import (
    TrainingJobListResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingJobStatusResponse,
    TrainingMetrics,
    TrainingStatus,
)
from ..models.synthetic_data import SyntheticDataGenerationRequest
from ..storage.db_manager import get_db_manager
from ..storage.minio_client import MinIOClient
from ..export.heimdall_format import HeimdallExporter, HeimdallImporter, HeimdallBundle
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])


@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(request: TrainingJobRequest):
    """
    Create and start a new training job.

    The job will be queued and executed asynchronously via Celery.
    Use the WebSocket endpoint /ws/training/{job_id} to monitor progress.

    Args:
        request: Training job configuration

    Returns:
        Created training job with ID and WebSocket URL
    """
    db_manager = get_db_manager()

    try:
        # Create training job record in database
        with db_manager.get_session() as session:
            job_id = None
            query = text("""
                INSERT INTO heimdall.training_jobs (
                    job_name, status, config, total_epochs, model_architecture
                )
                VALUES (
                    :job_name, :status, CAST(:config AS jsonb), :total_epochs, :model_architecture
                )
                RETURNING id, created_at
            """)

            result = session.execute(
                query,
                {
                    "job_name": request.job_name,
                    "status": TrainingStatus.PENDING.value,
                    "config": request.config.model_dump_json(),
                    "total_epochs": request.config.epochs,
                    "model_architecture": request.config.model_architecture,
                },
            )
            row = result.fetchone()
            if row:
                job_id = row[0]
                created_at = row[1]

            session.commit()

            if not job_id:
                raise HTTPException(status_code=500, detail="Failed to create training job")

            logger.info(f"Created training job {job_id}: {request.job_name}")

        # Queue Celery task to training service
        from ..main import celery_app
        task = celery_app.send_task(
            'src.tasks.training_task.start_training_job',
            args=[str(job_id)],
            queue='training'
        )

        # Update job with Celery task ID
        with db_manager.get_session() as session:
            session.execute(
                text("""
                    UPDATE heimdall.training_jobs
                    SET celery_task_id = :task_id, status = :status
                    WHERE id = :job_id
                """),
                {
                    "task_id": task.id,
                    "status": TrainingStatus.QUEUED.value,
                    "job_id": str(job_id),
                },
            )
            session.commit()

        logger.info(f"Queued training job {job_id} with Celery task {task.id}")

        # Broadcast WebSocket update
        from .websocket import manager as ws_manager
        await ws_manager.broadcast({
            "event": "training_job_update",
            "data": {
                "job_id": str(job_id),
                "status": TrainingStatus.QUEUED.value,
                "action": "created",
            }
        })

        # Return job details
        return TrainingJobResponse(
            id=job_id,
            job_name=request.job_name,
            status=TrainingStatus.QUEUED,
            created_at=created_at,
            config=request.config.model_dump(),
            total_epochs=request.config.epochs,
            celery_task_id=task.id,
            model_architecture=request.config.model_architecture,
        )

    except Exception as e:
        logger.error(f"Error creating training job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {e!s}")


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    status: TrainingStatus | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    List all training jobs with optional filtering.

    Args:
        status: Filter by job status
        limit: Maximum number of jobs to return
        offset: Pagination offset

    Returns:
        List of training jobs
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Build query
            where_clause = ""
            params: dict[str, Any] = {"limit": limit, "offset": offset}

            if status:
                where_clause = "WHERE status = :status"
                params["status"] = status.value

            # Get jobs
            query = text(f"""
                SELECT id, job_name, status, created_at, started_at, completed_at,
                       config, current_epoch, total_epochs, progress_percent,
                       train_loss, val_loss, train_accuracy, val_accuracy, learning_rate,
                       best_epoch, best_val_loss, checkpoint_path, onnx_model_path,
                       mlflow_run_id, error_message, dataset_size, train_samples,
                       val_samples, model_architecture, celery_task_id,
                       current_progress, total_progress, progress_message
                FROM heimdall.training_jobs
                {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)

            results = session.execute(query, params).fetchall()

            # Get total count
            count_query = text(f"""
                SELECT COUNT(*) FROM heimdall.training_jobs
                {where_clause}
            """)
            count_params = {k: v for k, v in params.items() if k not in ["limit", "offset"]}
            total = session.execute(count_query, count_params).scalar() or 0

            # Convert to response models
            jobs = []
            for row in results:
                jobs.append(
                    TrainingJobResponse(
                        id=row[0],
                        job_name=row[1],
                        status=TrainingStatus(row[2]),
                        created_at=row[3],
                        started_at=row[4],
                        completed_at=row[5],
                        config=row[6] if row[6] else {},
                        current_epoch=row[7] or 0,
                        total_epochs=row[8],
                        progress_percent=row[9] or 0.0,
                        train_loss=row[10],
                        val_loss=row[11],
                        train_accuracy=row[12],
                        val_accuracy=row[13],
                        learning_rate=row[14],
                        best_epoch=row[15],
                        best_val_loss=row[16],
                        checkpoint_path=row[17],
                        onnx_model_path=row[18],
                        mlflow_run_id=row[19],
                        error_message=row[20],
                        dataset_size=row[21],
                        train_samples=row[22],
                        val_samples=row[23],
                        model_architecture=row[24],
                        celery_task_id=row[25],
                        current=row[26],
                        total=row[27],
                        message=row[28],
                    )
                )

            return TrainingJobListResponse(jobs=jobs, total=total)

    except Exception as e:
        logger.error(f"Error listing training jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list training jobs: {e!s}")


@router.get("/jobs/{job_id}", response_model=TrainingJobStatusResponse)
async def get_training_job(job_id: UUID):
    """
    Get detailed training job status including recent metrics.

    Args:
        job_id: Training job UUID

    Returns:
        Job details with recent metrics and WebSocket URL
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Get job details
            job_query = text("""
                SELECT id, job_name, status, created_at, started_at, completed_at,
                       config, current_epoch, total_epochs, progress_percent,
                       train_loss, val_loss, train_accuracy, val_accuracy, learning_rate,
                       best_epoch, best_val_loss, checkpoint_path, onnx_model_path,
                       mlflow_run_id, error_message, dataset_size, train_samples,
                       val_samples, model_architecture, celery_task_id,
                       current_progress, total_progress, progress_message
                FROM heimdall.training_jobs
                WHERE id = :job_id
            """)

            result = session.execute(job_query, {"job_id": str(job_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            job = TrainingJobResponse(
                id=result[0],
                job_name=result[1],
                status=TrainingStatus(result[2]),
                created_at=result[3],
                started_at=result[4],
                completed_at=result[5],
                config=result[6] if result[6] else {},
                current_epoch=result[7] or 0,
                total_epochs=result[8],
                progress_percent=result[9] or 0.0,
                train_loss=result[10],
                val_loss=result[11],
                train_accuracy=result[12],
                val_accuracy=result[13],
                learning_rate=result[14],
                best_epoch=result[15],
                best_val_loss=result[16],
                checkpoint_path=result[17],
                onnx_model_path=result[18],
                mlflow_run_id=result[19],
                error_message=result[20],
                dataset_size=result[21],
                train_samples=result[22],
                val_samples=result[23],
                model_architecture=result[24],
                celery_task_id=result[25],
                current=result[26],
                total=result[27],
                message=result[28],
            )

            # Get recent metrics (last 10 epochs)
            metrics_query = text("""
                SELECT DISTINCT ON (epoch)
                    epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                    learning_rate, gradient_norm
                FROM heimdall.training_metrics
                WHERE training_job_id = :job_id
                ORDER BY epoch DESC, timestamp DESC
                LIMIT 10
            """)

            metrics_results = session.execute(
                metrics_query, {"job_id": str(job_id)}
            ).fetchall()

            recent_metrics = [
                TrainingMetrics(
                    epoch=row[0],
                    train_loss=row[1],
                    val_loss=row[2],
                    train_accuracy=row[3],
                    val_accuracy=row[4],
                    learning_rate=row[5],
                    gradient_norm=row[6],
                )
                for row in metrics_results
            ]

            # Construct WebSocket URL
            websocket_url = f"ws://localhost:8001/ws/training/{job_id}"

            return TrainingJobStatusResponse(
                job=job, recent_metrics=recent_metrics, websocket_url=websocket_url
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get training job: {e!s}")


@router.post("/jobs/{job_id}/cancel", status_code=200)
async def cancel_training_job(job_id: UUID):
    """
    Cancel a running training job without deleting it.

    Sets the job status to 'cancelled' and stops the Celery task.

    Args:
        job_id: Training job UUID

    Returns:
        Updated job status
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Check if job exists and get status
            check_query = text("""
                SELECT status, celery_task_id FROM heimdall.training_jobs
                WHERE id = :job_id
            """)
            result = session.execute(check_query, {"job_id": str(job_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            status, celery_task_id = result

            # Can only cancel pending, queued, or running jobs
            if status not in ["pending", "queued", "running"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel job in status '{status}'. Only pending, queued, or running jobs can be cancelled."
                )

            # Cancel Celery task
            if celery_task_id:
                try:
                    from celery import current_app

                    current_app.control.revoke(celery_task_id, terminate=True, signal='SIGTERM')
                    logger.info(f"Cancelled Celery task {celery_task_id} for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel Celery task: {e}")

            # Update job status
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'cancelled', completed_at = NOW()
                WHERE id = :job_id
            """)
            session.execute(update_query, {"job_id": str(job_id)})
            session.commit()

            logger.info(f"Cancelled training job {job_id}")

            # Broadcast WebSocket update
            from .websocket import manager as ws_manager
            await ws_manager.broadcast({
                "event": "training_job_update",
                "data": {
                    "job_id": str(job_id),
                    "status": "cancelled",
                    "action": "cancelled",
                }
            })

            return {"status": "cancelled", "job_id": str(job_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel training job: {e!s}")


@router.post("/jobs/{job_id}/pause", status_code=200)
async def pause_training_job(job_id: UUID):
    """
    Pause a running training job.

    Sets the job status to 'paused' and saves a checkpoint at the end of the current epoch.
    The training task will detect this status change and gracefully pause after completing
    the current epoch.

    Args:
        job_id: Training job UUID

    Returns:
        Updated job status
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Check if job exists and get status
            check_query = text("""
                SELECT status, total_epochs, current_epoch FROM heimdall.training_jobs
                WHERE id = :job_id
            """)
            result = session.execute(check_query, {"job_id": str(job_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            status, total_epochs, current_epoch = result

            # Can only pause running jobs
            if status != "running":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot pause job in status '{status}'. Only running jobs can be paused."
                )

            # Don't allow pausing synthetic data generation jobs (total_epochs = 0)
            if total_epochs == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot pause synthetic data generation jobs."
                )

            # Update job status to paused
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'paused'
                WHERE id = :job_id
            """)
            session.execute(update_query, {"job_id": str(job_id)})
            session.commit()

            logger.info(f"Paused training job {job_id} at epoch {current_epoch}/{total_epochs}")

            # Broadcast WebSocket update
            from .websocket import manager as ws_manager
            await ws_manager.broadcast({
                "event": "training_job_update",
                "data": {
                    "job_id": str(job_id),
                    "status": "paused",
                    "action": "paused",
                    "current_epoch": current_epoch,
                }
            })

            return {
                "status": "paused",
                "job_id": str(job_id),
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "message": "Training will pause after completing the current epoch"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing training job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to pause training job: {e!s}")


@router.post("/jobs/{job_id}/resume", status_code=200)
async def resume_training_job(job_id: UUID):
    """
    Resume a paused training job.

    Restarts the training task from the pause checkpoint. The training will continue
    from the epoch where it was paused.

    Args:
        job_id: Training job UUID

    Returns:
        Updated job status with new Celery task ID
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Check if job exists and get status
            check_query = text("""
                SELECT status, pause_checkpoint_path, current_epoch, total_epochs
                FROM heimdall.training_jobs
                WHERE id = :job_id
            """)
            result = session.execute(check_query, {"job_id": str(job_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            status, pause_checkpoint_path, current_epoch, total_epochs = result

            # Can only resume paused jobs
            if status != "paused":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot resume job in status '{status}'. Only paused jobs can be resumed."
                )

            # Verify pause checkpoint exists
            if not pause_checkpoint_path:
                raise HTTPException(
                    status_code=400,
                    detail="No pause checkpoint found. Cannot resume training."
                )

            # Update job status to running and queue the training task
            from tasks.training_task import start_training_job

            task = start_training_job.apply_async(args=[str(job_id)])

            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'queued', celery_task_id = :task_id, started_at = NOW()
                WHERE id = :job_id
            """)
            session.execute(update_query, {"job_id": str(job_id), "task_id": task.id})
            session.commit()

            logger.info(f"Resumed training job {job_id} from epoch {current_epoch}/{total_epochs} (task: {task.id})")

            # Broadcast WebSocket update
            from .websocket import manager as ws_manager
            await ws_manager.broadcast({
                "event": "training_job_update",
                "data": {
                    "job_id": str(job_id),
                    "status": "queued",
                    "action": "resumed",
                    "current_epoch": current_epoch,
                    "celery_task_id": task.id,
                }
            })

            return {
                "status": "queued",
                "job_id": str(job_id),
                "celery_task_id": task.id,
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "message": f"Training resumed from epoch {current_epoch}"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming training job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume training job: {e!s}")


@router.post("/jobs/{job_id}/continue", status_code=202)
async def continue_synthetic_job(job_id: UUID):
    """
    Continue a cancelled synthetic data generation job from where it left off.
    
    Creates a new job that:
    - References the same dataset (reuses existing samples)
    - Generates only the remaining samples
    - Links to the original job via parent_job_id
    
    Only works for jobs that:
    - Are cancelled
    - Have job_type='synthetic_generation'
    - Have made some progress (current_progress > 0)
    
    Args:
        job_id: UUID of the cancelled job to continue
        
    Returns:
        New job details with continuation info
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Fetch original job details
            job_query = text("""
                SELECT 
                    tj.status, 
                    tj.job_type, 
                    tj.job_name,
                    tj.config, 
                    tj.current_progress, 
                    tj.total_progress,
                    sd.id as dataset_id,
                    sd.name as dataset_name
                FROM heimdall.training_jobs tj
                LEFT JOIN heimdall.synthetic_datasets sd ON sd.job_id = tj.id
                WHERE tj.id = :job_id
            """)
            result = session.execute(job_query, {"job_id": str(job_id)}).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            status, job_type, job_name, config, current_progress, total_progress, dataset_id, dataset_name = result
            
            # Validate job can be continued
            if job_type != 'synthetic_generation':
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only synthetic_generation jobs can be continued (found: {job_type})"
                )
            
            if status != 'cancelled':
                raise HTTPException(
                    status_code=400,
                    detail=f"Only cancelled jobs can be continued (current status: {status})"
                )
            
            if not current_progress or current_progress <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Job has no progress to continue from"
                )
            
            if not dataset_id:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot find associated dataset for this job"
                )
            
            # Count actual samples in database (may differ from current_progress)
            count_query = text("""
                SELECT COUNT(*) 
                FROM heimdall.measurement_features 
                WHERE dataset_id = :dataset_id
            """)
            actual_samples = session.execute(
                count_query, 
                {"dataset_id": str(dataset_id)}
            ).scalar()
            
            # Calculate remaining samples
            remaining_samples = total_progress - actual_samples
            
            if remaining_samples <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Job already complete ({actual_samples}/{total_progress} samples)"
                )
            
            # Parse original config and create continuation config
            import json
            original_config = json.loads(config) if isinstance(config, str) else config
            
            continuation_config = {
                **original_config,
                "is_continuation": True,
                "existing_dataset_id": str(dataset_id),
                "num_samples": remaining_samples,
                "samples_offset": actual_samples
            }
            
            # Create new job with parent reference
            new_job_query = text("""
                INSERT INTO heimdall.training_jobs (
                    job_name, 
                    job_type, 
                    status, 
                    config, 
                    total_epochs,
                    total_progress,
                    parent_job_id
                )
                VALUES (
                    :job_name, 
                    'synthetic_generation', 
                    'pending', 
                    CAST(:config AS jsonb), 
                    0,
                    :total_progress,
                    :parent_job_id
                )
                RETURNING id, created_at
            """)
            
            new_job_result = session.execute(
                new_job_query,
                {
                    "job_name": f"{job_name} (Continued)",
                    "config": json.dumps(continuation_config),
                    "total_progress": total_progress,
                    "parent_job_id": str(job_id)
                }
            )
            
            new_job_row = new_job_result.fetchone()
            new_job_id = new_job_row[0]
            created_at = new_job_row[1]
            
            session.commit()
        
        logger.info(
            f"Created continuation job {new_job_id} for cancelled job {job_id}. "
            f"Resuming from {actual_samples}/{total_progress} samples, "
            f"generating {remaining_samples} more."
        )
        
        # Queue Celery task
        from ..main import celery_app
        celery_app.send_task(
            'src.tasks.training_task.generate_synthetic_data_task',
            args=[str(new_job_id)],
            queue='training'
        )
        
        # Broadcast WebSocket update
        from .websocket import manager as ws_manager
        await ws_manager.broadcast({
            "event": "dataset_update",
            "data": {
                "job_id": str(new_job_id),
                "parent_job_id": str(job_id),
                "status": "pending",
                "action": "continued",
                "dataset_id": str(dataset_id),
                "samples_existing": actual_samples,
                "samples_remaining": remaining_samples
            }
        })
        
        return {
            "job_id": str(new_job_id),
            "parent_job_id": str(job_id),
            "dataset_id": str(dataset_id),
            "dataset_name": dataset_name,
            "status": "pending",
            "created_at": created_at,
            "samples_existing": actual_samples,
            "samples_remaining": remaining_samples,
            "total_samples": total_progress,
            "status_url": f"/api/v1/training/jobs/{new_job_id}",
            "message": f"Continuing from {actual_samples}/{total_progress} samples"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error continuing synthetic job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to continue job: {e!s}")


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_training_job(job_id: UUID):
    """
    Delete a training job.

    Can only delete jobs that are not running (completed, failed, or cancelled).
    For running jobs, cancel them first.

    Args:
        job_id: Training job UUID
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Check if job exists
            check_query = text("""
                SELECT status, celery_task_id FROM heimdall.training_jobs
                WHERE id = :job_id
            """)
            result = session.execute(check_query, {"job_id": str(job_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            status, celery_task_id = result

            # Cannot delete running jobs directly
            if status in ["pending", "queued", "running"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot delete job in status '{status}'. Cancel the job first."
                )

            # Delete job (cascade will delete metrics)
            delete_query = text("""
                DELETE FROM heimdall.training_jobs WHERE id = :job_id
            """)
            session.execute(delete_query, {"job_id": str(job_id)})
            session.commit()

            logger.info(f"Deleted training job {job_id}")

            # Broadcast WebSocket update
            from .websocket import manager as ws_manager
            await ws_manager.broadcast({
                "event": "training_job_update",
                "data": {
                    "job_id": str(job_id),
                    "action": "deleted",
                }
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting training job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete training job: {e!s}")


@router.get("/jobs/{job_id}/metrics", response_model=list[TrainingMetrics])
async def get_training_metrics(
    job_id: UUID,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get detailed training metrics history for a job.

    Returns per-epoch metrics for visualization.

    Args:
        job_id: Training job UUID
        limit: Maximum number of epochs to return

    Returns:
        List of epoch metrics
    """
    db_manager = get_db_manager()

    try:
        with db_manager.get_session() as session:
            # Verify job exists
            check_query = text("SELECT id FROM heimdall.training_jobs WHERE id = :job_id")
            if not session.execute(check_query, {"job_id": str(job_id)}).fetchone():
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            # Get metrics
            metrics_query = text("""
                SELECT DISTINCT ON (epoch)
                    epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                    learning_rate, gradient_norm
                FROM heimdall.training_metrics
                WHERE training_job_id = :job_id
                ORDER BY epoch ASC, timestamp DESC
                LIMIT :limit
            """)

            results = session.execute(
                metrics_query, {"job_id": str(job_id), "limit": limit}
            ).fetchall()

            return [
                TrainingMetrics(
                    epoch=row[0],
                    train_loss=row[1],
                    val_loss=row[2],
                    train_accuracy=row[3],
                    val_accuracy=row[4],
                    learning_rate=row[5],
                    gradient_norm=row[6],
                )
                for row in results
            ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get training metrics: {e!s}")


# ============================================================================
# SYNTHETIC DATA GENERATION ENDPOINTS
# ============================================================================

@router.post("/synthetic/generate", status_code=202)
async def generate_synthetic_data(request: SyntheticDataGenerationRequest):
    """
    Generate synthetic training dataset.

    Creates a background job to generate synthetic samples using RF propagation simulation.

    Args:
        request: Synthetic data generation configuration

    Returns:
        Job ID and status URL
    """
    # Convert to dict for storage
    request_dict = request.model_dump()
    
    db_manager = get_db_manager()
    
    try:
        # Create job record
        with db_manager.get_session() as session:
            job_query = text("""
                INSERT INTO heimdall.training_jobs (
                    job_name, job_type, status, config, total_epochs
                )
                VALUES (
                    :job_name, 'synthetic_generation', 'pending', CAST(:config AS jsonb), 0
                )
                RETURNING id, created_at
            """)
            
            import json
            result = session.execute(
                job_query,
                {
                    "job_name": f"Synthetic Data: {request_dict.get('name', 'Unnamed')}",
                    "config": json.dumps(request_dict)
                }
            )
            row = result.fetchone()
            job_id = row[0]
            created_at = row[1]
            
            session.commit()
        
        logger.info(f"Created synthetic data generation job {job_id}")
        
        # Queue Celery task to training service
        from ..main import celery_app
        celery_app.send_task(
            'src.tasks.training_task.generate_synthetic_data_task',
            args=[str(job_id)],
            queue='training'
        )

        # Broadcast WebSocket update
        from .websocket import manager as ws_manager
        await ws_manager.broadcast({
            "event": "dataset_update",
            "data": {
                "job_id": str(job_id),
                "status": "pending",
                "action": "created",
            }
        })
        
        return {
            "job_id": job_id,
            "status": "pending",
            "created_at": created_at,
            "status_url": f"/api/v1/training/jobs/{job_id}"
        }
    
    except Exception as e:
        logger.error(f"Error creating synthetic data job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create job: {e!s}")


@router.get("/synthetic/datasets")
async def list_synthetic_datasets(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
):
    """
    List all synthetic datasets.
    
    Args:
        limit: Maximum number of datasets to return
        offset: Pagination offset
    
    Returns:
        List of synthetic datasets
    """
    from ..models.synthetic_data import SyntheticDatasetResponse, SyntheticDatasetListResponse
    
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Get datasets with REAL-TIME sample counts from measurement_features
            query = text("""
                SELECT
                    sd.id,
                    sd.name,
                    sd.description,
                    COALESCE(COUNT(mf.recording_session_id), 0) as num_samples,
                    sd.config,
                    sd.quality_metrics,
                    sd.storage_table,
                    sd.created_at,
                    sd.created_by_job_id
                FROM heimdall.synthetic_datasets sd
                LEFT JOIN heimdall.measurement_features mf ON mf.dataset_id = sd.id
                GROUP BY sd.id, sd.name, sd.description, sd.config, sd.quality_metrics,
                         sd.storage_table, sd.created_at, sd.created_by_job_id
                ORDER BY sd.created_at DESC
                LIMIT :limit OFFSET :offset
            """)

            results = session.execute(query, {"limit": limit, "offset": offset}).fetchall()

            # Get total count
            count_query = text("SELECT COUNT(*) FROM heimdall.synthetic_datasets")
            total = session.execute(count_query).scalar() or 0
            
            # Convert to response models
            import json
            datasets = []
            for row in results:
                # JSONB columns are already dicts, not strings
                config = row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] else {})
                quality_metrics = row[5] if isinstance(row[5], dict) else (json.loads(row[5]) if row[5] else None)

                datasets.append(SyntheticDatasetResponse(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    num_samples=row[3],
                    config=config,
                    quality_metrics=quality_metrics,
                    storage_table=row[6],
                    created_at=row[7],
                    created_by_job_id=row[8]
                ))
            
            return SyntheticDatasetListResponse(datasets=datasets, total=total)
    
    except Exception as e:
        logger.error(f"Error listing synthetic datasets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e!s}")


@router.get("/synthetic/datasets/{dataset_id}")
async def get_synthetic_dataset(dataset_id: UUID):
    """
    Get synthetic dataset details.
    
    Args:
        dataset_id: Dataset UUID
    
    Returns:
        Dataset details
    """
    from ..models.synthetic_data import SyntheticDatasetResponse
    
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Get dataset with REAL-TIME sample count from measurement_features
            query = text("""
                SELECT
                    sd.id,
                    sd.name,
                    sd.description,
                    COALESCE(COUNT(mf.recording_session_id), 0) as num_samples,
                    sd.config,
                    sd.quality_metrics,
                    sd.storage_table,
                    sd.created_at,
                    sd.created_by_job_id
                FROM heimdall.synthetic_datasets sd
                LEFT JOIN heimdall.measurement_features mf ON mf.dataset_id = sd.id
                WHERE sd.id = :dataset_id
                GROUP BY sd.id, sd.name, sd.description, sd.config, sd.quality_metrics,
                         sd.storage_table, sd.created_at, sd.created_by_job_id
            """)

            result = session.execute(query, {"dataset_id": str(dataset_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            
            import json
            # JSONB columns are already dicts, not strings
            config = result[4] if isinstance(result[4], dict) else (json.loads(result[4]) if result[4] else {})
            quality_metrics = result[5] if isinstance(result[5], dict) else (json.loads(result[5]) if result[5] else None)

            return SyntheticDatasetResponse(
                id=result[0],
                name=result[1],
                description=result[2],
                num_samples=result[3],
                config=config,
                quality_metrics=quality_metrics,
                storage_table=result[6],
                created_at=result[7],
                created_by_job_id=result[8]
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e!s}")


@router.get("/synthetic/datasets/{dataset_id}/samples")
async def get_dataset_samples(
    dataset_id: UUID,
    limit: int = Query(default=10, ge=1, le=100, description="Number of samples to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    split: str = Query(default=None, description="Filter by split (train/val/test)")
):
    """
    Get samples from a synthetic dataset.
    
    Args:
        dataset_id: UUID of the dataset
        limit: Maximum number of samples to return (1-100)
        offset: Number of samples to skip
        split: Optional filter by split type
    
    Returns:
        List of samples with pagination info
    """
    from ..models.synthetic_data import SyntheticSampleResponse, SyntheticSamplesListResponse
    
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Build query - samples are in measurement_features table
            where_clause = "WHERE dataset_id = :dataset_id"
            params = {"dataset_id": str(dataset_id), "limit": limit, "offset": offset}
            
            # Note: split filter not applicable for measurement_features (no split column)
            # Splits are calculated at training time, not stored per-sample
            
            # Count total
            count_query = text(f"""
                SELECT COUNT(*) as total
                FROM heimdall.measurement_features
                {where_clause}
            """)
            
            count_result = session.execute(count_query, params).fetchone()
            total = count_result[0] if count_result else 0
            
            # Fetch samples from measurement_features
            samples_query = text(f"""
                SELECT recording_session_id, timestamp, tx_latitude, tx_longitude,
                       tx_power_dbm, extraction_metadata->>'frequency_hz' as frequency_hz,
                       receiver_features, gdop, num_receivers_detected, 
                       mean_snr_db, overall_confidence, created_at
                FROM heimdall.measurement_features
                {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """)
            
            rows = session.execute(samples_query, params).fetchall()
            
            samples = []
            for row in rows:
                # Convert receiver_features array to dict for response
                receivers_data = {
                    "num_receivers": row[8],
                    "mean_snr_db": row[9],
                    "overall_confidence": row[10],
                    "receiver_features": row[6]  # JSONB array
                }
                
                samples.append(SyntheticSampleResponse(
                    id=hash(str(row[0])),  # Use hash of UUID as int ID
                    timestamp=row[1],
                    tx_lat=row[2] or 0.0,
                    tx_lon=row[3] or 0.0,
                    tx_power_dbm=row[4] or 0.0,
                    frequency_hz=float(row[5]) if row[5] else 0.0,
                    receivers=receivers_data,
                    gdop=row[7] or 0.0,
                    num_receivers=row[8] or 0,
                    split="",  # Not stored at sample level
                    created_at=row[11]
                ))
            
            return SyntheticSamplesListResponse(
                samples=samples,
                total=total,
                limit=limit,
                offset=offset,
                dataset_id=str(dataset_id)
            )
    
    except Exception as e:
        logger.error(f"Error getting samples for dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get samples: {e!s}")


@router.delete("/synthetic/datasets/{dataset_id}", status_code=204)
async def delete_synthetic_dataset(dataset_id: UUID):
    """
    Delete synthetic dataset and all its samples.
    
    Args:
        dataset_id: Dataset UUID
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Check if dataset exists
            check_query = text("SELECT id FROM heimdall.synthetic_datasets WHERE id = :dataset_id")
            result = session.execute(check_query, {"dataset_id": str(dataset_id)}).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            
            # Delete dataset (cascade will delete samples)
            delete_query = text("DELETE FROM heimdall.synthetic_datasets WHERE id = :dataset_id")
            session.execute(delete_query, {"dataset_id": str(dataset_id)})
            session.commit()
            
            logger.info(f"Deleted synthetic dataset {dataset_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e!s}")


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/models")
async def list_models(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    active_only: bool = Query(default=False)
):
    """
    List all trained models.
    
    Args:
        limit: Maximum number of models to return
        offset: Pagination offset
        active_only: Only return active models
    
    Returns:
        List of trained models
    """
    from ..models.synthetic_data import ModelMetadataResponse, ModelListResponse
    
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Build query and parameters safely
            params = {"limit": limit, "offset": offset}
            if active_only:
                query = text("""
                    SELECT id, model_name, version, model_type, synthetic_dataset_id,
                           mlflow_run_id, mlflow_experiment_id, onnx_model_location,
                           pytorch_model_location, accuracy_meters, accuracy_sigma_meters,
                           loss_value, epoch, is_active, is_production, hyperparameters,
                           training_metrics, test_metrics, created_at, trained_by_job_id
                    FROM heimdall.models
                    WHERE is_active = :is_active
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                params["is_active"] = True
                count_query = text("SELECT COUNT(*) FROM heimdall.models WHERE is_active = :is_active")
                count_params = {"is_active": True}
            else:
                query = text("""
                    SELECT id, model_name, version, model_type, synthetic_dataset_id,
                           mlflow_run_id, mlflow_experiment_id, onnx_model_location,
                           pytorch_model_location, accuracy_meters, accuracy_sigma_meters,
                           loss_value, epoch, is_active, is_production, hyperparameters,
                           training_metrics, test_metrics, created_at, trained_by_job_id
                    FROM heimdall.models
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :offset
                """)
                count_query = text("SELECT COUNT(*) FROM heimdall.models")
                count_params = {}
            
            results = session.execute(query, params).fetchall()
            total = session.execute(count_query, count_params).scalar() or 0
            
            import json
            models = []
            for row in results:
                # JSONB columns are already dicts, not strings
                hyperparameters = row[15] if isinstance(row[15], dict) else (json.loads(row[15]) if row[15] else None)
                training_metrics = row[16] if isinstance(row[16], dict) else (json.loads(row[16]) if row[16] else None)
                test_metrics = row[17] if isinstance(row[17], dict) else (json.loads(row[17]) if row[17] else None)

                models.append(ModelMetadataResponse(
                    id=row[0],
                    model_name=row[1],
                    version=row[2] or 1,
                    model_type=row[3],
                    synthetic_dataset_id=row[4],
                    mlflow_run_id=row[5],
                    mlflow_experiment_id=row[6],
                    onnx_model_location=row[7],
                    pytorch_model_location=row[8],
                    accuracy_meters=row[9],
                    accuracy_sigma_meters=row[10],
                    loss_value=row[11],
                    epoch=row[12],
                    is_active=row[13],
                    is_production=row[14],
                    hyperparameters=hyperparameters,
                    training_metrics=training_metrics,
                    test_metrics=test_metrics,
                    created_at=row[18],
                    trained_by_job_id=row[19]
                ))
            
            return ModelListResponse(models=models, total=total)
    
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e!s}")


@router.get("/models/{model_id}")
async def get_model(model_id: UUID):
    """
    Get model details.
    
    Args:
        model_id: Model UUID
    
    Returns:
        Model details
    """
    from ..models.synthetic_data import ModelMetadataResponse
    
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            query = text("""
                SELECT id, model_name, version, model_type, synthetic_dataset_id,
                       mlflow_run_id, mlflow_experiment_id, onnx_model_location,
                       pytorch_model_location, accuracy_meters, accuracy_sigma_meters,
                       loss_value, epoch, is_active, is_production, hyperparameters,
                       training_metrics, test_metrics, created_at, trained_by_job_id
                FROM heimdall.models
                WHERE id = :model_id
            """)
            
            result = session.execute(query, {"model_id": str(model_id)}).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            import json
            # JSONB columns are already dicts, not strings
            hyperparameters = result[15] if isinstance(result[15], dict) else (json.loads(result[15]) if result[15] else None)
            training_metrics = result[16] if isinstance(result[16], dict) else (json.loads(result[16]) if result[16] else None)
            test_metrics = result[17] if isinstance(result[17], dict) else (json.loads(result[17]) if result[17] else None)

            return ModelMetadataResponse(
                id=result[0],
                model_name=result[1],
                version=result[2] or 1,
                model_type=result[3],
                synthetic_dataset_id=result[4],
                mlflow_run_id=result[5],
                mlflow_experiment_id=result[6],
                onnx_model_location=result[7],
                pytorch_model_location=result[8],
                accuracy_meters=result[9],
                accuracy_sigma_meters=result[10],
                loss_value=result[11],
                epoch=result[12],
                is_active=result[13],
                is_production=result[14],
                hyperparameters=hyperparameters,
                training_metrics=training_metrics,
                test_metrics=test_metrics,
                created_at=result[18],
                trained_by_job_id=result[19]
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model: {e!s}")


@router.post("/models/{model_id}/deploy", status_code=200)
async def deploy_model(model_id: UUID, set_production: bool = False):
    """
    Deploy model (set as active for inference).
    
    Args:
        model_id: Model UUID
        set_production: Also set as production model
    
    Returns:
        Updated model details
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Check if model exists
            check_query = text("SELECT id FROM heimdall.models WHERE id = :model_id")
            result = session.execute(check_query, {"model_id": str(model_id)}).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            # Deactivate other models
            deactivate_query = text("UPDATE heimdall.models SET is_active = FALSE")
            session.execute(deactivate_query)
            
            if set_production:
                unprod_query = text("UPDATE heimdall.models SET is_production = FALSE")
                session.execute(unprod_query)
            
            # Activate this model
            activate_query = text("""
                UPDATE heimdall.models
                SET is_active = TRUE, is_production = :set_production, updated_at = NOW()
                WHERE id = :model_id
            """)
            session.execute(activate_query, {"model_id": str(model_id), "set_production": set_production})
            session.commit()
            
            logger.info(f"Deployed model {model_id} (production={set_production})")

            # Broadcast WebSocket update
            from .websocket import manager as ws_manager
            await ws_manager.broadcast({
                "event": "model_update",
                "data": {
                    "model_id": str(model_id),
                    "action": "deployed",
                    "is_production": set_production,
                }
            })
            
            return {"status": "deployed", "model_id": model_id, "is_production": set_production}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to deploy model: {e!s}")


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(model_id: UUID):
    """
    Delete model and associated artifacts.
    
    Args:
        model_id: Model UUID
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_session() as session:
            # Check if model exists
            check_query = text("SELECT id, is_active FROM heimdall.models WHERE id = :model_id")
            result = session.execute(check_query, {"model_id": str(model_id)}).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            is_active = result[1]
            if is_active:
                raise HTTPException(status_code=400, detail="Cannot delete active model. Deactivate first.")
            
            # Delete model
            delete_query = text("DELETE FROM heimdall.models WHERE id = :model_id")
            session.execute(delete_query, {"model_id": str(model_id)})
            session.commit()
            
            logger.info(f"Deleted model {model_id}")

            # Broadcast WebSocket update
            from .websocket import manager as ws_manager
            await ws_manager.broadcast({
                "event": "model_update",
                "data": {
                    "model_id": str(model_id),
                    "action": "deleted",
                }
            })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e!s}")


@router.get("/models/{model_id}/export", response_class=Response)
async def export_model_heimdall(
    model_id: str,
    include_config: bool = Query(True, description="Include training configuration"),
    include_metrics: bool = Query(True, description="Include performance metrics"),
    include_normalization: bool = Query(True, description="Include normalization stats"),
    include_samples: bool = Query(True, description="Include sample predictions"),
    num_samples: int = Query(5, description="Number of sample predictions to include", ge=0, le=100),
    description: str = Query(None, description="Optional description for the bundle")
):
    """
    Export a trained model as a .heimdall bundle file.
    
    The bundle includes:
    - ONNX model (base64-encoded)
    - Model architecture details
    - Training configuration (optional)
    - Performance metrics (optional)
    - Normalization statistics (optional)
    - Sample predictions for validation (optional)
    
    Args:
        model_id: UUID of the model to export
        include_config: Include training hyperparameters
        include_metrics: Include accuracy and performance metrics
        include_normalization: Include feature normalization parameters
        include_samples: Include sample predictions for validation
        num_samples: Number of sample predictions (0-100)
        description: Optional description added to bundle metadata
        
    Returns:
        Downloadable .heimdall JSON file
    """
    try:
        db_manager = get_db_manager()
        minio_client = MinIOClient(
            endpoint_url=settings.minio_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket_name="heimdall-models"
        )
        exporter = HeimdallExporter(db_manager, minio_client.s3_client)
        
        logger.info(f"Exporting model {model_id} as .heimdall bundle")
        
        # Create bundle
        bundle = exporter.export_model(
            model_id=model_id,
            include_config=include_config,
            include_metrics=include_metrics,
            include_normalization=include_normalization,
            include_samples=include_samples,
            num_samples=num_samples,
            description=description
        )
        
        # Serialize to JSON
        bundle_json = bundle.model_dump_json(indent=2)
        
        # Get model name for filename
        model_name = bundle.model.model_name.replace(" ", "_")
        version = bundle.model.version
        filename = f"{model_name}-v{version}.heimdall"
        
        logger.info(f"Successfully exported model {model_id} to {filename} ({len(bundle_json)} bytes)")
        
        # Return as downloadable file
        return Response(
            content=bundle_json,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Bundle-ID": bundle.bundle_metadata.bundle_id,
                "X-Model-ID": model_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export model: {e!s}")


@router.post("/models/import", status_code=201)
async def import_model_heimdall(
    file: UploadFile = File(..., description=".heimdall bundle file to import")
):
    """
    Import a trained model from a .heimdall bundle file.
    
    The bundle is validated, the ONNX model is uploaded to MinIO,
    and the model is registered in the database.
    
    Args:
        file: Uploaded .heimdall bundle file
        
    Returns:
        Model ID and registration details
    """
    try:
        # Validate file extension
        if not file.filename.endswith(".heimdall"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Expected .heimdall file"
            )
        
        logger.info(f"Importing model from {file.filename}")
        
        # Read bundle contents
        bundle_json = await file.read()
        
        # Parse bundle
        bundle = HeimdallBundle.model_validate_json(bundle_json)
        
        # Import model
        db_manager = get_db_manager()
        minio_client = MinIOClient(
            endpoint_url=settings.minio_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket_name="heimdall-models"
        )
        importer = HeimdallImporter(db_manager, minio_client.s3_client)
        
        result = importer.import_model(bundle)
        model_id = result["model_id"]
        
        logger.info(f"Successfully imported model from {file.filename} as {model_id}")
        
        return {
            "status": "success",
            "model_id": str(model_id),
            "filename": file.filename,
            "message": f"Model imported successfully with ID {model_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing model from {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to import model: {e!s}")
