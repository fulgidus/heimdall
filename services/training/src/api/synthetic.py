"""
API endpoints for synthetic data management.
"""
import uuid
from typing import List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings

router = APIRouter(prefix="/api/v1/jobs/synthetic", tags=["synthetic-datasets"])

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
# RESPONSE MODELS
# ============================================================================

class SampleResponse(BaseModel):
    """Individual synthetic sample response."""
    id: Union[int, str]  # int for training samples, UUID string for IQ samples
    timestamp: datetime
    tx_lat: float
    tx_lon: float
    tx_power_dbm: float
    frequency_hz: float
    receivers: Union[dict, list]  # dict for feature_based, list for iq_raw
    gdop: float
    num_receivers: int
    split: Optional[str] = None  # Only for feature_based datasets
    created_at: datetime
    iq_available: bool = False  # Whether IQ data is available for this sample
    iq_metadata: Optional[dict] = None  # IQ metadata (sample_rate, duration, etc.)
    sample_idx: Optional[int] = None  # Sample index (for IQ data lookup)

    class Config:
        from_attributes = True


class SamplesListResponse(BaseModel):
    """List of samples with pagination."""
    samples: List[SampleResponse]
    total: int
    limit: int
    offset: int
    dataset_id: str


class DatasetResponse(BaseModel):
    """Individual dataset response."""
    id: str
    name: str
    description: Optional[str]
    num_samples: int
    config: dict
    quality_metrics: Optional[dict]
    storage_table: str
    created_at: datetime
    updated_at: datetime
    created_by_job_id: Optional[str]

    class Config:
        from_attributes = True


class DatasetsListResponse(BaseModel):
    """List of datasets with pagination."""
    datasets: List[DatasetResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/datasets", response_model=DatasetsListResponse)
async def list_datasets(
    limit: int = Query(default=100, ge=1, le=500, description="Number of datasets to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    List all synthetic datasets.
    
    Args:
        limit: Maximum number of datasets to return (1-500)
        offset: Number of datasets to skip
        db: Database session
    
    Returns:
        List of datasets with pagination info
    """
    # Count total
    count_query = text("""
        SELECT COUNT(*) as total
        FROM heimdall.synthetic_datasets
    """)
    
    count_result = db.execute(count_query).fetchone()
    total = count_result[0] if count_result else 0
    
    # Fetch datasets
    datasets_query = text("""
        SELECT id, name, description, num_samples, config, quality_metrics,
               storage_table, created_at, updated_at, created_by_job_id
        FROM heimdall.synthetic_datasets
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """)
    
    rows = db.execute(datasets_query, {"limit": limit, "offset": offset}).fetchall()
    
    datasets = []
    for row in rows:
        datasets.append(DatasetResponse(
            id=str(row[0]),
            name=row[1],
            description=row[2],
            num_samples=row[3],
            config=row[4],
            quality_metrics=row[5],
            storage_table=row[6],
            created_at=row[7],
            updated_at=row[8],
            created_by_job_id=str(row[9]) if row[9] else None
        ))
    
    return DatasetsListResponse(
        datasets=datasets,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/datasets/{dataset_id}/samples", response_model=SamplesListResponse)
async def get_dataset_samples(
    dataset_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Number of samples to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    split: Optional[str] = Query(default=None, description="Filter by split (train/val/test)"),
    db: Session = Depends(get_db)
):
    """
    Get samples from a synthetic dataset.
    
    Args:
        dataset_id: UUID of the dataset
        limit: Maximum number of samples to return (1-100)
        offset: Number of samples to skip
        split: Optional filter by split type
        db: Database session
    
    Returns:
        List of samples with pagination info
    """
    try:
        # Validate UUID
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")
    
    # First, determine the dataset type
    dataset_query = text("""
        SELECT dataset_type FROM heimdall.synthetic_datasets WHERE id = :dataset_id
    """)
    dataset_result = db.execute(dataset_query, {"dataset_id": str(dataset_uuid)}).fetchone()
    
    if not dataset_result:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    
    dataset_type = dataset_result[0]
    
    # Build query based on dataset type
    params = {"dataset_id": str(dataset_uuid), "limit": limit, "offset": offset}
    
    if dataset_type == "iq_raw":
        # For IQ datasets, query synthetic_iq_samples table
        where_clause = "WHERE dataset_id = :dataset_id"
        
        # Note: IQ datasets don't have split column, so we ignore that filter
        if split:
            raise HTTPException(
                status_code=400, 
                detail="Split filtering not supported for iq_raw datasets"
            )
        
        # Count total
        count_query = text(f"""
            SELECT COUNT(*) as total
            FROM heimdall.synthetic_iq_samples
            {where_clause}
        """)
        
        count_result = db.execute(count_query, params).fetchone()
        total = count_result[0] if count_result else 0
        
        # Fetch IQ samples
        samples_query = text(f"""
            SELECT 
                id::text as id, timestamp, tx_lat, tx_lon, tx_power_dbm, frequency_hz,
                receivers_metadata as receivers, gdop, num_receivers, 
                NULL as split, created_at,
                iq_metadata, sample_idx,
                TRUE as iq_available
            FROM heimdall.synthetic_iq_samples
            {where_clause}
            ORDER BY sample_idx
            LIMIT :limit OFFSET :offset
        """)
        
        rows = db.execute(samples_query, params).fetchall()
        
    else:  # feature_based dataset
        # For feature-based datasets, query synthetic_training_samples table
        where_clause = "WHERE dataset_id = :dataset_id"
        
        if split:
            where_clause += " AND split = :split"
            params["split"] = split
        
        # Count total
        count_query = text(f"""
            SELECT COUNT(*) as total
            FROM heimdall.synthetic_training_samples
            {where_clause}
        """)
        
        count_result = db.execute(count_query, params).fetchone()
        total = count_result[0] if count_result else 0
        
        # Fetch samples with IQ availability check
        # Use ROW_NUMBER() to calculate sample index within dataset
        samples_query = text(f"""
            WITH numbered_samples AS (
                SELECT 
                    s.id, s.timestamp, s.tx_lat, s.tx_lon, s.tx_power_dbm, s.frequency_hz,
                    s.receivers, s.gdop, s.num_receivers, s.split, s.created_at, s.dataset_id,
                    ROW_NUMBER() OVER (PARTITION BY s.dataset_id ORDER BY s.timestamp, s.id) - 1 as sample_idx
                FROM heimdall.synthetic_training_samples s
                {where_clause}
            )
            SELECT 
                ns.id, ns.timestamp, ns.tx_lat, ns.tx_lon, ns.tx_power_dbm, ns.frequency_hz,
                ns.receivers, ns.gdop, ns.num_receivers, ns.split, ns.created_at,
                iq.iq_metadata, ns.sample_idx,
                CASE WHEN iq.id IS NOT NULL THEN TRUE ELSE FALSE END as iq_available
            FROM numbered_samples ns
            LEFT JOIN heimdall.synthetic_iq_samples iq ON iq.dataset_id = ns.dataset_id 
                AND iq.sample_idx = ns.sample_idx
            ORDER BY ns.timestamp, ns.id
            LIMIT :limit OFFSET :offset
        """)
        
        rows = db.execute(samples_query, params).fetchall()
    
    # Parse results (same for both dataset types)
    samples = []
    for row in rows:
        samples.append(SampleResponse(
            id=row[0],
            timestamp=row[1],
            tx_lat=row[2],
            tx_lon=row[3],
            tx_power_dbm=row[4],
            frequency_hz=row[5],
            receivers=row[6],  # Already JSON from database
            gdop=row[7],
            num_receivers=row[8],
            split=row[9],
            created_at=row[10],
            iq_metadata=row[11],
            sample_idx=row[12],
            iq_available=row[13]
        ))
    
    return SamplesListResponse(
        samples=samples,
        total=total,
        limit=limit,
        offset=offset,
        dataset_id=dataset_id
    )


# ============================================================================
# DATASET GENERATION
# ============================================================================

class TxAntennaDistributionRequest(BaseModel):
    """TX antenna distribution configuration."""
    whip: float = 0.90
    rubber_duck: float = 0.08
    portable_directional: float = 0.02


class RxAntennaDistributionRequest(BaseModel):
    """RX antenna distribution configuration."""
    omni_vertical: float = 0.80
    yagi: float = 0.15
    collinear: float = 0.05


class GenerateDatasetRequest(BaseModel):
    """Request model for synthetic dataset generation and expansion."""
    name: str
    description: Optional[str] = None
    num_samples: int
    frequency_mhz: Optional[float] = Field(default=None)  # Optional for expansion (inherited from parent dataset)
    tx_power_dbm: Optional[float] = Field(default=None)  # Optional for expansion (inherited from parent dataset)
    min_snr_db: Optional[float] = Field(default=None)  # Optional for expansion (inherited from parent dataset)
    min_receivers: Optional[int] = Field(default=None)  # Optional for expansion (inherited from parent dataset)
    max_gdop: float = 150.0  # Default: 150 GDOP for clustered receivers (Italian WebSDRs) - relaxed for 50%+ success rate
    dataset_type: str = "feature_based"
    use_random_receivers: bool = False
    use_srtm_terrain: bool = True
    use_gpu: Optional[bool] = None  # None = auto-detect, True = force GPU, False = force CPU
    seed: Optional[int] = None
    tx_antenna_dist: Optional[TxAntennaDistributionRequest] = None
    rx_antenna_dist: Optional[RxAntennaDistributionRequest] = None
    expand_dataset_id: Optional[str] = None  # If provided, expand existing dataset instead of creating new one
    enable_meteorological: bool = True  # Tropospheric refraction and ducting effects
    enable_sporadic_e: bool = True  # Ionospheric skip propagation
    enable_knife_edge: bool = True  # Terrain diffraction effects
    enable_polarization: bool = True  # Polarization mismatch loss
    enable_antenna_patterns: bool = True  # Realistic antenna radiation patterns
    use_audio_library: bool = True  # Use real audio from library instead of formant synthesis
    audio_library_fallback: bool = True  # Fallback to formant synthesis if audio library fails


class GenerateDatasetResponse(BaseModel):
    """Response model for dataset generation."""
    job_id: str
    dataset_id: Optional[str] = None
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    job_name: str
    job_type: str
    status: str
    current_progress: Optional[int] = None
    total_progress: Optional[int] = None
    progress_percent: Optional[float] = None
    progress_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Optional[dict] = None

    class Config:
        from_attributes = True


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get status of a training or generation job.
    
    Args:
        job_id: UUID of the job
        db: Database session
    
    Returns:
        Job status and progress information
    """
    try:
        # Validate UUID
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    # Query job status
    query = text("""
        SELECT id, job_name, job_type, status, current_progress, total_progress,
               progress_percent, progress_message, created_at, started_at,
               completed_at, error_message
        FROM heimdall.training_jobs
        WHERE id = :job_id
    """)
    
    result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return JobStatusResponse(
        job_id=str(result[0]),
        job_name=result[1],
        job_type=result[2],
        status=result[3],
        current_progress=result[4],
        total_progress=result[5],
        progress_percent=result[6],
        progress_message=result[7],
        created_at=result[8],
        started_at=result[9],
        completed_at=result[10],
        error_message=result[11],
        result_data=None
    )


class JobActionResponse(BaseModel):
    """Response model for job action operations (cancel, pause, resume)."""
    job_id: str
    status: str
    message: str
    celery_task_id: Optional[str] = None

    class Config:
        from_attributes = True


@router.post("/{job_id}/cancel", response_model=JobActionResponse)
async def cancel_synthetic_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Cancel a synthetic data generation job.
    
    Args:
        job_id: UUID of the synthetic generation job
        db: Database session
    
    Returns:
        Cancellation status
    """
    from celery import current_app
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Get celery task ID and verify it's a synthetic generation job
        query = text("""
            SELECT celery_task_id, status, job_type FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        celery_task_id, current_status, job_type = result
        
        # Verify this is a synthetic generation job
        if job_type != "synthetic_generation":
            raise HTTPException(
                status_code=400, 
                detail=f"Job {job_id} is not a synthetic generation job (type: {job_type})"
            )
        
        # Check if job is already in a terminal state
        if current_status in ['completed', 'cancelled', 'failed']:
            return JobActionResponse(
                job_id=job_id,
                status=current_status,
                message=f"Job already in terminal state: {current_status}",
                celery_task_id=celery_task_id
            )
        
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
        
        logger.info("synthetic_job_cancelled", job_id=job_id, celery_task_id=celery_task_id)
        
        return JobActionResponse(
            job_id=job_id,
            status="cancelled",
            message="Synthetic generation job cancelled successfully",
            celery_task_id=celery_task_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("synthetic_job_cancel_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.delete("/{job_id}")
async def delete_synthetic_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a synthetic data generation job.
    
    This endpoint deletes the job record and associated data. If the job is
    currently running, it will be cancelled first.
    
    Args:
        job_id: UUID of the synthetic generation job
        db: Database session
    
    Returns:
        Deletion confirmation
    """
    from celery import current_app
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    
    try:
        # Check if job exists and get its details
        query = text("""
            SELECT status, celery_task_id, job_type FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        result = db.execute(query, {"job_id": str(job_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        status, celery_task_id, job_type = result
        
        # Verify this is a synthetic generation job
        if job_type != "synthetic_generation":
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not a synthetic generation job (type: {job_type})"
            )
        
        # If job is running, cancel it first
        if status == 'running':
            if celery_task_id:
                current_app.control.revoke(celery_task_id, terminate=True, signal='SIGTERM')
                logger.info("cancelled_running_job_before_delete", job_id=job_id, celery_task_id=celery_task_id)
        
        # Delete the job record
        delete_query = text("""
            DELETE FROM heimdall.training_jobs
            WHERE id = :job_id
        """)
        db.execute(delete_query, {"job_id": str(job_uuid)})
        db.commit()
        
        logger.info("synthetic_job_deleted", job_id=job_id)
        
        return {
            "message": f"Synthetic generation job {job_id} deleted successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("synthetic_job_delete_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_synthetic_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a synthetic dataset and all its samples.
    
    This endpoint deletes the dataset record and all associated samples.
    If the dataset was created by a job, the job record remains intact.
    
    Args:
        dataset_id: UUID of the dataset to delete
        db: Database session
    
    Returns:
        204 No Content on success
    """
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")
    
    try:
        # Check if dataset exists
        check_query = text("""
            SELECT id, name FROM heimdall.synthetic_datasets
            WHERE id = :dataset_id
        """)
        result = db.execute(check_query, {"dataset_id": str(dataset_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        
        dataset_name = result[1]
        
        # Delete dataset (CASCADE will delete associated samples)
        delete_query = text("""
            DELETE FROM heimdall.synthetic_datasets
            WHERE id = :dataset_id
        """)
        db.execute(delete_query, {"dataset_id": str(dataset_uuid)})
        db.commit()
        
        logger.info("synthetic_dataset_deleted", dataset_id=dataset_id, dataset_name=dataset_name)
        
        # Return 204 No Content (no body)
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("synthetic_dataset_delete_failed", dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.patch("/datasets/{dataset_id}", status_code=200)
async def rename_dataset(
    dataset_id: str,
    dataset_name: str = Query(..., min_length=1, max_length=200, description="New dataset name"),
    db: Session = Depends(get_db)
):
    """
    Update dataset name.
    
    Args:
        dataset_id: Dataset UUID
        dataset_name: New dataset name (1-200 characters)
        db: Database session
    
    Returns:
        Success message with updated dataset name
    """
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")
    
    try:
        # Check if dataset exists
        check_query = text("SELECT id, name FROM heimdall.synthetic_datasets WHERE id = :dataset_id")
        result = db.execute(check_query, {"dataset_id": str(dataset_uuid)}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        old_name = result[1]
        
        # Update dataset name
        update_query = text("""
            UPDATE heimdall.synthetic_datasets 
            SET name = :dataset_name, updated_at = NOW()
            WHERE id = :dataset_id
        """)
        db.execute(update_query, {"dataset_id": str(dataset_uuid), "dataset_name": dataset_name})
        db.commit()
        
        logger.info("dataset_renamed", dataset_id=dataset_id, old_name=old_name, new_name=dataset_name)
        
        return {"success": True, "dataset_id": str(dataset_uuid), "dataset_name": dataset_name}
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("dataset_rename_failed", dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to rename dataset: {str(e)}")




@router.get("/datasets/{dataset_id}/samples/{sample_idx}/iq/{rx_id}")
async def get_sample_iq_data(
    dataset_id: str,
    sample_idx: int,
    rx_id: str,
    db: Session = Depends(get_db)
):
    """
    Get IQ data for a specific receiver in a sample.
    
    Args:
        dataset_id: UUID of the dataset
        sample_idx: Sample index within dataset
        rx_id: Receiver ID (e.g., 'rx_0', 'rx_1')
        db: Database session
    
    Returns:
        IQ data as base64-encoded complex64 array + metadata
    """
    import base64
    import io
    import sys
    import os
    import structlog
    import numpy as np
    
    logger = structlog.get_logger(__name__)
    
    try:
        # Validate UUID
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")
    
    try:
        # Import MinIOClient from backend service
        sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
        from storage.minio_client import MinIOClient
        from config import settings as backend_settings
        
        # Query IQ sample metadata
        query = text("""
            SELECT iq_storage_paths, iq_metadata, receivers_metadata, 
                   tx_lat, tx_lon, tx_power_dbm, frequency_hz
            FROM heimdall.synthetic_iq_samples
            WHERE dataset_id = :dataset_id AND sample_idx = :sample_idx
        """)
        
        result = db.execute(query, {
            "dataset_id": str(dataset_uuid),
            "sample_idx": sample_idx
        }).fetchone()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Sample {sample_idx} not found in dataset {dataset_id}"
            )
        
        iq_storage_paths, iq_metadata, receivers_metadata, tx_lat, tx_lon, tx_power_dbm, frequency_hz = result
        
        # Check if rx_id exists in storage paths
        if rx_id not in iq_storage_paths:
            raise HTTPException(
                status_code=404,
                detail=f"Receiver {rx_id} not found in sample {sample_idx}"
            )
        
        # Get MinIO path for this receiver
        minio_path = iq_storage_paths[rx_id]
        
        # Initialize MinIO client (using synthetic-iq bucket)
        minio_client = MinIOClient(
            endpoint_url=backend_settings.minio_url,
            access_key=backend_settings.minio_access_key,
            secret_key=backend_settings.minio_secret_key,
            bucket_name="heimdall-synthetic-iq"
        )
        
        # Download IQ data from MinIO
        iq_bytes = minio_client.download_iq_data(s3_path=minio_path)
        
        # Load numpy array from bytes
        buffer = io.BytesIO(iq_bytes)
        iq_array = np.load(buffer)
        
        # Convert complex64 to separate real/imag arrays for JSON transport
        # Base64 encode the raw bytes (more efficient than JSON array)
        iq_real = iq_array.real.astype(np.float32)
        iq_imag = iq_array.imag.astype(np.float32)
        
        # Encode as base64
        real_b64 = base64.b64encode(iq_real.tobytes()).decode('utf-8')
        imag_b64 = base64.b64encode(iq_imag.tobytes()).decode('utf-8')
        
        # Find receiver metadata
        rx_metadata = None
        for rx in receivers_metadata:
            if rx.get('rx_id') == rx_id:
                rx_metadata = rx
                break
        
        logger.info(
            "iq_data_fetched",
            dataset_id=dataset_id,
            sample_idx=sample_idx,
            rx_id=rx_id,
            samples_count=len(iq_array),
            size_kb=len(iq_bytes) / 1024
        )
        
        return {
            "dataset_id": dataset_id,
            "sample_idx": sample_idx,
            "rx_id": rx_id,
            "iq_data": {
                "real_b64": real_b64,
                "imag_b64": imag_b64,
                "length": len(iq_array),
                "dtype": "complex64"
            },
            "iq_metadata": iq_metadata,
            "rx_metadata": rx_metadata,
            "tx_metadata": {
                "lat": tx_lat,
                "lon": tx_lon,
                "power_dbm": tx_power_dbm,
                "frequency_hz": frequency_hz
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("iq_data_fetch_failed", 
                    dataset_id=dataset_id, 
                    sample_idx=sample_idx, 
                    rx_id=rx_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch IQ data: {str(e)}")


@router.post("/generate", response_model=GenerateDatasetResponse)
async def generate_dataset(
    request: GenerateDatasetRequest,
    db: Session = Depends(get_db)
):
    """
    Start synthetic dataset generation job.
    
    Args:
        request: Generation configuration
        db: Database session
    
    Returns:
        Job ID and initial status
    """
    import json
    import structlog
    from ..tasks.training_task import generate_synthetic_data_task
    
    logger = structlog.get_logger()
    logger.info("generate_dataset_request", 
                expand_dataset_id=request.expand_dataset_id,
                frequency_mhz=request.frequency_mhz,
                tx_power_dbm=request.tx_power_dbm)
    
    # Create training job record
    job_id = str(uuid.uuid4())
    
    # If expanding an existing dataset, fetch parent config and inherit RF parameters
    if request.expand_dataset_id:
        logger.info("fetching_parent_dataset", parent_id=request.expand_dataset_id)
        parent_query = text("""
            SELECT config FROM heimdall.synthetic_datasets
            WHERE id = :dataset_id
        """)
        parent_result = db.execute(parent_query, {"dataset_id": request.expand_dataset_id}).fetchone()
        
        if not parent_result:
            logger.error("parent_dataset_not_found", parent_id=request.expand_dataset_id)
            raise HTTPException(
                status_code=404, 
                detail=f"Parent dataset {request.expand_dataset_id} not found"
            )
        
        parent_config = parent_result[0]  # config is jsonb, may be dict or string
        logger.info("parent_config_fetched", config_type=type(parent_config).__name__, config_sample=str(parent_config)[:100])
        
        # Parse config if it's a string
        if isinstance(parent_config, str):
            parent_config = json.loads(parent_config)
        
        logger.info("parent_config_parsed", 
                   freq=parent_config.get("frequency_mhz"),
                   power=parent_config.get("tx_power_dbm"),
                   snr=parent_config.get("min_snr_db"))
        
        # Inherit RF parameters from parent if not explicitly provided
        frequency_mhz = request.frequency_mhz if request.frequency_mhz is not None else parent_config.get("frequency_mhz")
        tx_power_dbm = request.tx_power_dbm if request.tx_power_dbm is not None else parent_config.get("tx_power_dbm")
        min_snr_db = request.min_snr_db if request.min_snr_db is not None else parent_config.get("min_snr_db")
        min_receivers = request.min_receivers if request.min_receivers is not None else parent_config.get("min_receivers")
        
        logger.info("inherited_parameters", freq=frequency_mhz, power=tx_power_dbm, snr=min_snr_db, min_rx=min_receivers)
    else:
        # For new datasets, use provided values (which are now required to be set by frontend or will be None)
        frequency_mhz = request.frequency_mhz
        tx_power_dbm = request.tx_power_dbm
        min_snr_db = request.min_snr_db
        min_receivers = request.min_receivers
    
    config = {
        "name": request.name,
        "description": request.description,
        "num_samples": request.num_samples,
        "frequency_mhz": frequency_mhz,
        "tx_power_dbm": tx_power_dbm,
        "min_snr_db": min_snr_db,
        "min_receivers": min_receivers,
        "max_gdop": request.max_gdop,
        "dataset_type": request.dataset_type,
        "use_random_receivers": request.use_random_receivers,
        "use_srtm_terrain": request.use_srtm_terrain,
        "use_gpu": request.use_gpu,
        "seed": request.seed,
        "expand_dataset_id": request.expand_dataset_id,
        "enable_meteorological": request.enable_meteorological,
        "enable_sporadic_e": request.enable_sporadic_e,
        "enable_knife_edge": request.enable_knife_edge,
        "enable_polarization": request.enable_polarization,
        "enable_antenna_patterns": request.enable_antenna_patterns,
        "use_audio_library": request.use_audio_library,
        "audio_library_fallback": request.audio_library_fallback
    }
    
    # Add antenna distributions if provided
    if request.tx_antenna_dist is not None:
        config["tx_antenna_dist"] = {
            "whip": request.tx_antenna_dist.whip,
            "rubber_duck": request.tx_antenna_dist.rubber_duck,
            "portable_directional": request.tx_antenna_dist.portable_directional
        }
    
    if request.rx_antenna_dist is not None:
        config["rx_antenna_dist"] = {
            "omni_vertical": request.rx_antenna_dist.omni_vertical,
            "yagi": request.rx_antenna_dist.yagi,
            "collinear": request.rx_antenna_dist.collinear
        }
    
    try:
        # Insert job record
        insert_query = text("""
            INSERT INTO heimdall.training_jobs (
                id, job_name, job_type, status, config, total_epochs, 
                total_progress, created_at
            )
            VALUES (
                :id, :name, 'synthetic_generation', 'pending', CAST(:config AS jsonb), 
                0, :total_progress, NOW()
            )
        """)
        
        db.execute(
            insert_query,
            {
                "id": job_id,
                "name": request.name,
                "config": json.dumps(config),
                "total_progress": request.num_samples
            }
        )
        db.commit()
        
        # Submit Celery task with explicit queue routing
        task = generate_synthetic_data_task.apply_async(
            args=[job_id],
            task_id=job_id,
            queue='training'  # Route to training queue where worker listens
        )
        
        # Update celery_task_id in database
        update_query = text("""
            UPDATE heimdall.training_jobs
            SET celery_task_id = :task_id
            WHERE id = :job_id
        """)
        db.execute(update_query, {"task_id": str(task.id), "job_id": job_id})
        db.commit()
        
        return GenerateDatasetResponse(
            job_id=job_id,
            status="pending",
            message=f"Dataset generation job created: {request.name}"
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create generation job: {str(e)}")


# ============================================================================
# TERRAIN MANAGEMENT ENDPOINTS
# ============================================================================

class TerrainTileStatus(BaseModel):
    """Terrain tile status information."""
    tile_name: str
    exists: bool
    lat_min: int
    lat_max: int
    lon_min: int
    lon_max: int


class TerrainCoverageRequest(BaseModel):
    """Request model for terrain coverage check."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class TerrainCoverageResponse(BaseModel):
    """Response model for terrain coverage status."""
    total_tiles: int
    available_tiles: int
    missing_tiles: int
    coverage_percent: float
    tiles: List[TerrainTileStatus]
    missing_tile_names: List[str]


class TerrainDownloadRequest(BaseModel):
    """Request model for SRTM tile download."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class TerrainDownloadResponse(BaseModel):
    """Response model for terrain download operation."""
    message: str
    tiles_to_download: List[str]
    backend_url: str


@router.post("/terrain/coverage", response_model=TerrainCoverageResponse)
async def check_terrain_coverage(
    request: TerrainCoverageRequest
):
    """
    Check terrain tile coverage for a geographic region.
    
    This endpoint checks which SRTM tiles are available in MinIO storage
    for the specified region and which ones are missing.
    
    Args:
        request: Geographic bounds to check
    
    Returns:
        Coverage status with available and missing tiles
    """
    import math
    import sys
    import os
    
    # Import MinIOClient from backend service
    sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
    from storage.minio_client import MinIOClient
    from config import settings as backend_settings
    
    # Import training service settings
    from ..config import settings
    
    # Calculate required tiles (1°×1° tiles)
    tile_coords = []
    lat_start = int(math.floor(request.lat_min))
    lat_end = int(math.floor(request.lat_max))
    lon_start = int(math.floor(request.lon_min))
    lon_end = int(math.floor(request.lon_max))
    
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tile_coords.append((lat, lon))
    
    # Initialize MinIO client using backend settings
    minio_client = MinIOClient(
        endpoint_url=backend_settings.minio_url,
        access_key=backend_settings.minio_access_key,
        secret_key=backend_settings.minio_secret_key,
        bucket_name="heimdall-terrain"
    )
    
    # Check which tiles exist in MinIO
    tiles_status = []
    available_count = 0
    missing_tile_names = []
    
    for lat, lon in tile_coords:
        # Format tile name
        lat_prefix = "N" if lat >= 0 else "S"
        lon_prefix = "E" if lon >= 0 else "W"
        tile_name = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"
        
        # Check if tile exists in MinIO
        object_name = f"tiles/{tile_name}.tif"
        try:
            minio_client.s3_client.head_object(
                Bucket="heimdall-terrain",
                Key=object_name
            )
            exists = True
            available_count += 1
        except Exception:
            exists = False
            missing_tile_names.append(tile_name)
        
        tiles_status.append(TerrainTileStatus(
            tile_name=tile_name,
            exists=exists,
            lat_min=lat,
            lat_max=lat + 1,
            lon_min=lon,
            lon_max=lon + 1
        ))
    
    total_tiles = len(tile_coords)
    missing_count = total_tiles - available_count
    coverage_percent = (available_count / total_tiles * 100) if total_tiles > 0 else 0.0
    
    return TerrainCoverageResponse(
        total_tiles=total_tiles,
        available_tiles=available_count,
        missing_tiles=missing_count,
        coverage_percent=coverage_percent,
        tiles=tiles_status,
        missing_tile_names=missing_tile_names
    )


@router.post("/terrain/download", response_model=TerrainDownloadResponse)
async def download_terrain_tiles(
    request: TerrainDownloadRequest
):
    """
    Trigger SRTM tile download for a region.
    
    This endpoint redirects the user to the backend service's terrain download
    endpoint, as the training service should not directly download tiles.
    
    The backend service has the proper event publishing infrastructure for
    real-time progress updates via WebSocket.
    
    Args:
        request: Geographic bounds for tiles to download
    
    Returns:
        Information about the download request with backend service URL
    """
    import math
    from ..config import settings
    
    # Calculate required tiles
    tile_coords = []
    lat_start = int(math.floor(request.lat_min))
    lat_end = int(math.floor(request.lat_max))
    lon_start = int(math.floor(request.lon_min))
    lon_end = int(math.floor(request.lon_max))
    
    tiles_to_download = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"
            tile_name = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"
            tiles_to_download.append(tile_name)
    
    # Backend service URL for terrain downloads
    backend_url = f"{settings.backend_url}/api/v1/terrain/download"
    
    return TerrainDownloadResponse(
        message=f"Please use the backend service to download {len(tiles_to_download)} tiles",
        tiles_to_download=tiles_to_download,
        backend_url=backend_url
    )


@router.get("/terrain/status")
async def get_terrain_status():
    """
    Get overall terrain system status.
    
    Returns information about SRTM support, MinIO connectivity,
    and general terrain system health.
    
    Returns:
        Terrain system status
    """
    import sys
    import os
    
    # Import MinIOClient from backend service
    sys.path.insert(0, os.environ.get('BACKEND_SRC_PATH', '/app/backend/src'))
    from storage.minio_client import MinIOClient
    from config import settings as backend_settings
    
    # Import training service settings
    from ..config import settings
    
    status = {
        "srtm_enabled": True,
        "minio_configured": bool(backend_settings.minio_url),
        "bucket_name": "heimdall-terrain",
        "backend_service_url": settings.backend_url
    }
    
    # Check MinIO connectivity
    try:
        minio_client = MinIOClient(
            endpoint_url=backend_settings.minio_url,
            access_key=backend_settings.minio_access_key,
            secret_key=backend_settings.minio_secret_key,
            bucket_name="heimdall-terrain"
        )
        
        # Try to check if bucket exists
        try:
            minio_client.s3_client.head_bucket(Bucket="heimdall-terrain")
            status["minio_connection"] = "healthy"
            status["bucket_exists"] = True
        except Exception as e:
            status["minio_connection"] = "degraded"
            status["bucket_exists"] = False
            status["bucket_error"] = str(e)
    except Exception as e:
        status["minio_connection"] = "failed"
        status["minio_error"] = str(e)
    
    return status
