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
