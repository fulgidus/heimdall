"""
Pydantic models for synthetic data generation API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    """Dataset type enumeration."""
    
    FEATURE_BASED = "feature_based"  # Traditional feature extraction (mel-spec, MFCC)
    IQ_RAW = "iq_raw"  # Raw IQ samples for CNN training


class SyntheticDataGenerationRequest(BaseModel):
    """
    Request to generate synthetic training data.

    Two modes:
    - feature_based: Traditional feature extraction (mel-spectrograms, MFCCs) with fixed receivers
    - iq_raw: Raw IQ samples for CNN training with random receivers per sample (5-10 count)

    Note: Data splits (train/val/test) are NOT assigned during generation.
    They are calculated at training time for maximum flexibility.
    """

    name: str = Field(..., min_length=1, max_length=100, description="Dataset name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    dataset_type: DatasetType = Field(
        default=DatasetType.FEATURE_BASED, 
        description="Dataset type: feature_based (mel-spec/MFCC) or iq_raw (CNN training)"
    )
    num_samples: int = Field(..., ge=100, le=5000000, description="Number of samples to generate")
    expand_dataset_id: Optional[UUID] = Field(
        default=None, 
        description="If provided, adds samples to existing dataset instead of creating new one"
    )
    inside_ratio: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Ratio of TX inside receiver network"
    )
    frequency_mhz: float = Field(
        default=145.0, ge=0.5, le=3000.0, description="Frequency in MHz (HF/VHF/UHF/SHF)"
    )
    tx_power_dbm: float = Field(default=37.0, ge=0.0, le=50.0, description="TX power in dBm")
    min_snr_db: float = Field(default=3.0, ge=0.0, description="Minimum SNR threshold")
    min_receivers: int = Field(
        default=3, ge=2, le=10, 
        description="Minimum receivers with signal (applies to quality filtering)"
    )
    max_gdop: float = Field(default=150.0, ge=1.0, description="Maximum GDOP (150 recommended for clustered receivers)")

    # Random receiver generation parameters (for iq_raw datasets)
    # For iq_raw mode: use_random_receivers is forced to True
    use_random_receivers: bool = Field(
        default=False, 
        description="Use random receivers per sample (automatically True for iq_raw datasets)"
    )
    min_receivers_count: int = Field(
        default=5, ge=3, le=15, 
        description="Minimum number of receivers per sample (iq_raw mode)"
    )
    max_receivers_count: int = Field(
        default=10, ge=3, le=15, 
        description="Maximum number of receivers per sample (iq_raw mode)"
    )
    receiver_seed: Optional[int] = Field(
        default=None, description="Random seed for receiver generation (reproducibility)"
    )
    area_lat_min: float = Field(
        default=44.0, ge=-90.0, le=90.0, description="Minimum latitude for receiver area"
    )
    area_lat_max: float = Field(
        default=46.0, ge=-90.0, le=90.0, description="Maximum latitude for receiver area"
    )
    area_lon_min: float = Field(
        default=7.0, ge=-180.0, le=180.0, description="Minimum longitude for receiver area"
    )
    area_lon_max: float = Field(
        default=10.0, ge=-180.0, le=180.0, description="Maximum longitude for receiver area"
    )
    
    # Processing acceleration (GPU/CPU selection)
    use_gpu: Optional[bool] = Field(
        default=None,
        description="Processing unit: None=auto-detect, True=force GPU (CuPy), False=force CPU (NumPy)"
    )
    
    # Audio library configuration
    use_audio_library: bool = Field(
        default=True,
        description="Use real audio from library instead of synthetic formant tones"
    )
    audio_library_fallback: bool = Field(
        default=True,
        description="Fallback to formant synthesis if audio library fails or is empty"
    )


class SyntheticDatasetResponse(BaseModel):
    """
    Response with synthetic dataset details.

    Note: train_count, val_count, test_count are no longer stored.
    Data splits are calculated dynamically at training time.
    """

    id: UUID
    name: str
    description: Optional[str]
    dataset_type: Optional[DatasetType] = DatasetType.FEATURE_BASED  # Default for backward compatibility
    num_samples: int
    config: dict[str, Any]
    quality_metrics: Optional[dict[str, Any]]
    storage_table: str
    storage_size_bytes: Optional[int] = None  # Total storage (PostgreSQL + MinIO), NULL if not calculated
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
    parent_model_id: Optional[UUID]  # Track model evolution lineage

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


class SyntheticSampleResponse(BaseModel):
    """Individual synthetic sample response."""

    id: int
    timestamp: datetime
    tx_lat: float
    tx_lon: float
    tx_power_dbm: float
    frequency_hz: float
    receivers: dict[str, Any]  # JSON data with receiver details
    gdop: float
    num_receivers: int
    split: str  # train/val/test
    created_at: datetime

    class Config:
        from_attributes = True


class SyntheticSamplesListResponse(BaseModel):
    """List of samples with pagination."""

    samples: list[SyntheticSampleResponse]
    total: int
    limit: int
    offset: int
    dataset_id: str
