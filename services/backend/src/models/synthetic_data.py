"""
Pydantic models for synthetic data generation API.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic training data."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    num_samples: int = Field(..., ge=1000, le=5000000, description="Number of samples to generate")
    inside_ratio: float = Field(default=0.7, ge=0.0, le=1.0, description="Ratio of TX inside receiver network")
    train_ratio: float = Field(default=0.7, ge=0.0, le=1.0, description="Training set ratio")
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0, description="Validation set ratio")
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0, description="Test set ratio")
    frequency_mhz: float = Field(default=145.0, ge=144.0, le=148.0, description="Frequency in MHz")
    tx_power_dbm: float = Field(default=37.0, ge=0.0, le=50.0, description="TX power in dBm")
    min_snr_db: float = Field(default=3.0, ge=0.0, description="Minimum SNR threshold")
    min_receivers: int = Field(default=3, ge=2, le=7, description="Minimum receivers with signal")
    max_gdop: float = Field(default=10.0, ge=1.0, description="Maximum GDOP")


class SyntheticDatasetResponse(BaseModel):
    """Response with synthetic dataset details."""
    
    id: UUID
    name: str
    description: Optional[str]
    num_samples: int
    train_count: Optional[int]
    val_count: Optional[int]
    test_count: Optional[int]
    config: dict[str, Any]
    quality_metrics: Optional[dict[str, Any]]
    storage_table: str
    created_at: datetime
    created_by_job_id: Optional[UUID]
    
    class Config:
        from_attributes = True


class SyntheticDatasetListResponse(BaseModel):
    """List of synthetic datasets."""
    
    datasets: list[SyntheticDatasetResponse]
    total: int


class TerrainDownloadRequest(BaseModel):
    """Request to download terrain data."""
    
    force_redownload: bool = Field(default=False, description="Force redownload even if cached")


class TerrainStatus(BaseModel):
    """Terrain download status."""
    
    tiles_required: list[str]
    tiles_downloaded: list[str]
    tiles_pending: list[str]
    tiles_failed: list[str]
    progress_percent: float
    status: str  # 'ready', 'downloading', 'partial', 'failed'


class ModelMetadataResponse(BaseModel):
    """Response with trained model metadata."""
    
    id: UUID
    model_name: str
    version: int
    model_type: Optional[str]
    synthetic_dataset_id: Optional[UUID]
    mlflow_run_id: Optional[str]
    mlflow_experiment_id: Optional[int]
    onnx_model_location: Optional[str]
    pytorch_model_location: Optional[str]
    accuracy_meters: Optional[float]
    accuracy_sigma_meters: Optional[float]
    loss_value: Optional[float]
    epoch: Optional[int]
    is_active: bool
    is_production: bool
    hyperparameters: Optional[dict[str, Any]]
    training_metrics: Optional[dict[str, Any]]
    test_metrics: Optional[dict[str, Any]]
    created_at: datetime
    trained_by_job_id: Optional[UUID]
    
    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """List of trained models."""
    
    models: list[ModelMetadataResponse]
    total: int


class ModelEvaluationRequest(BaseModel):
    """Request to evaluate a model."""
    
    dataset_id: Optional[UUID] = Field(default=None, description="Dataset to evaluate on (uses test split)")


class ModelEvaluationResponse(BaseModel):
    """Response with model evaluation results."""
    
    id: UUID
    model_id: UUID
    dataset_id: Optional[UUID]
    metrics: dict[str, Any]
    visualization_paths: Optional[dict[str, Any]]
    evaluated_at: datetime
    
    class Config:
        from_attributes = True


class ModelExportRequest(BaseModel):
    """Request to export model to ONNX."""
    
    optimize: bool = Field(default=True, description="Apply ONNX optimizations")


class ModelExportResponse(BaseModel):
    """Response with ONNX export details."""
    
    onnx_path: str
    file_size_mb: float
    export_time_seconds: float


class ModelDeployRequest(BaseModel):
    """Request to deploy model (set as active)."""
    
    set_production: bool = Field(default=False, description="Also set as production model")
