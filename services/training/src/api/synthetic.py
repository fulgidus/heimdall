"""
API endpoints for synthetic data management.
"""
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings

router = APIRouter(prefix="/synthetic", tags=["synthetic"])

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
    id: int
    timestamp: datetime
    tx_lat: float
    tx_lon: float
    tx_power_dbm: float
    frequency_hz: float
    receivers: dict  # JSON data
    gdop: float
    num_receivers: int
    split: str
    created_at: datetime

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
    
    # Build query
    where_clause = "WHERE dataset_id = :dataset_id"
    params = {"dataset_id": str(dataset_uuid), "limit": limit, "offset": offset}
    
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
    
    # Fetch samples
    samples_query = text(f"""
        SELECT id, timestamp, tx_lat, tx_lon, tx_power_dbm, frequency_hz,
               receivers, gdop, num_receivers, split, created_at
        FROM heimdall.synthetic_training_samples
        {where_clause}
        ORDER BY id
        LIMIT :limit OFFSET :offset
    """)
    
    rows = db.execute(samples_query, params).fetchall()
    
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
            created_at=row[10]
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
    """Request model for synthetic dataset generation."""
    name: str
    description: Optional[str] = None
    num_samples: int
    frequency_mhz: float
    tx_power_dbm: float
    min_snr_db: float
    min_receivers: int
    max_gdop: float = 150.0  # Default: 150 GDOP for clustered receivers (Italian WebSDRs) - relaxed for 50%+ success rate
    dataset_type: str = "feature_based"
    use_random_receivers: bool = False
    use_srtm_terrain: bool = True
    seed: Optional[int] = None
    tx_antenna_dist: Optional[TxAntennaDistributionRequest] = None
    rx_antenna_dist: Optional[RxAntennaDistributionRequest] = None


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


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
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
    from ..tasks.training_task import generate_synthetic_data_task
    
    # Create training job record
    job_id = str(uuid.uuid4())
    config = {
        "name": request.name,
        "description": request.description,
        "num_samples": request.num_samples,
        "frequency_mhz": request.frequency_mhz,
        "tx_power_dbm": request.tx_power_dbm,
        "min_snr_db": request.min_snr_db,
        "min_receivers": request.min_receivers,
        "max_gdop": request.max_gdop,
        "dataset_type": request.dataset_type,
        "use_random_receivers": request.use_random_receivers,
        "use_srtm_terrain": request.use_srtm_terrain,
        "seed": request.seed
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
