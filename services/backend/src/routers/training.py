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

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from ..models.training import (
    TrainingJobListResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingJobStatusResponse,
    TrainingMetrics,
    TrainingStatus,
)
from ..storage.db_manager import get_db_manager
from ..tasks.training_task import start_training_job

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
                    :job_name, :status, :config::jsonb, :total_epochs, :model_architecture
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

        # Queue Celery task
        task = start_training_job.delay(str(job_id))

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
                       val_samples, model_architecture, celery_task_id
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
                import json

                jobs.append(
                    TrainingJobResponse(
                        id=row[0],
                        job_name=row[1],
                        status=TrainingStatus(row[2]),
                        created_at=row[3],
                        started_at=row[4],
                        completed_at=row[5],
                        config=json.loads(row[6]) if row[6] else {},
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
                       val_samples, model_architecture, celery_task_id
                FROM heimdall.training_jobs
                WHERE id = :job_id
            """)

            result = session.execute(job_query, {"job_id": str(job_id)}).fetchone()

            if not result:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

            import json

            job = TrainingJobResponse(
                id=result[0],
                job_name=result[1],
                status=TrainingStatus(result[2]),
                created_at=result[3],
                started_at=result[4],
                completed_at=result[5],
                config=json.loads(result[6]) if result[6] else {},
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


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_training_job(job_id: UUID):
    """
    Cancel and delete a training job.

    If the job is running, it will be cancelled.
    All associated data (metrics, checkpoints) will be removed.

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

            # Cancel Celery task if running
            if status in ["pending", "queued", "running"] and celery_task_id:
                try:
                    from celery import current_app

                    current_app.control.revoke(celery_task_id, terminate=True)
                    logger.info(f"Cancelled Celery task {celery_task_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel Celery task: {e}")

            # Delete job (cascade will delete metrics)
            delete_query = text("""
                DELETE FROM heimdall.training_jobs WHERE id = :job_id
            """)
            session.execute(delete_query, {"job_id": str(job_id)})
            session.commit()

            logger.info(f"Deleted training job {job_id}")

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
