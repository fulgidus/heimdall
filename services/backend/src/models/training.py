"""
Pydantic models for training API.

Defines request/response models for training job management:
- Training job creation and configuration
- Job status and progress tracking
- Training metrics and results
- WebSocket update messages
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    """Training job status states."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingPhase(str, Enum):
    """Training phase indicator."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class TrainingConfig(BaseModel):
    """Training configuration parameters."""

    # Dataset
    dataset_ids: list[str] = Field(..., description="List of synthetic dataset UUIDs to train on (required)", min_length=1)

    # Model architecture
    model_architecture: str = Field(default="triangulation", description="Model architecture")
    pretrained: bool = Field(default=False, description="Use pretrained weights")
    freeze_backbone: bool = Field(default=False, description="Freeze backbone layers")

    # Data parameters
    batch_size: int = Field(default=128, ge=1, le=1024, description="Batch size (larger=better GPU utilization)")
    num_workers: int = Field(default=0, ge=0, le=32, description="DataLoader workers (0=auto-detect, recommended)")
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Validation split ratio")

    # Feature extraction
    n_mels: int = Field(default=128, description="Mel-spectrogram frequency bins")
    n_fft: int = Field(default=2048, description="FFT size")
    hop_length: int = Field(default=512, description="STFT hop length")

    # Training hyperparameters
    epochs: int = Field(default=100, ge=1, le=1000, description="Training epochs")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="L2 regularization")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout rate")

    # Learning rate scheduler
    lr_scheduler: str = Field(default="cosine", description="LR scheduler type")
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs")

    # Early stopping
    early_stop_patience: int = Field(default=20, ge=1, description="Early stopping patience")
    early_stop_delta: float = Field(default=0.001, ge=0.0, description="Minimum change for early stopping")

    # Gradient clipping
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Maximum gradient norm for clipping")

    # Hardware
    accelerator: str = Field(default="auto", description="Training device (auto/cpu/gpu)")
    devices: int = Field(default=1, ge=1, description="Number of devices")

    # Data filters
    min_snr_db: Optional[float] = Field(default=10.0, description="Minimum SNR for training samples")
    max_gdop: Optional[float] = Field(default=5.0, description="Maximum GDOP for validation metrics")
    only_approved: bool = Field(default=True, description="Use only approved recordings")


class TrainingJobRequest(BaseModel):
    """Request to create a new training job."""

    job_name: str = Field(..., min_length=1, max_length=255, description="Human-readable job name")
    config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    description: Optional[str] = Field(default=None, description="Job description")


class TrainingMetrics(BaseModel):
    """Training metrics for an epoch."""

    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None


class TrainingJobResponse(BaseModel):
    """Response with training job details."""

    id: UUID
    job_name: str
    job_type: str  # Job type: 'training', 'synthetic_generation', etc.
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Configuration
    config: dict[str, Any]

    # Progress
    current_epoch: int = 0
    total_epochs: int
    progress_percent: float = 0.0

    # Progress tracking (for synthetic data and training)
    current: Optional[int] = None  # current_progress from DB
    total: Optional[int] = None    # total_progress from DB
    message: Optional[str] = None  # progress_message from DB

    # Metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None

    # Best model
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None

    # Artifacts
    checkpoint_path: Optional[str] = None
    onnx_model_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None

    # Error handling
    error_message: Optional[str] = None

    # Metadata
    dataset_size: Optional[int] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None
    model_architecture: Optional[str] = None

    celery_task_id: Optional[str] = None

    class Config:
        from_attributes = True


class TrainingJobListResponse(BaseModel):
    """List of training jobs."""

    jobs: list[TrainingJobResponse]
    total: int


class TrainingProgressUpdate(BaseModel):
    """WebSocket update message for training progress."""

    event: str = "training_progress"
    job_id: UUID
    status: TrainingStatus
    current_epoch: int
    total_epochs: int
    progress_percent: float
    metrics: Optional[TrainingMetrics] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingEpochComplete(BaseModel):
    """WebSocket update when epoch completes."""

    event: str = "epoch_complete"
    job_id: UUID
    epoch: int
    metrics: TrainingMetrics
    is_best: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingJobComplete(BaseModel):
    """WebSocket update when training job completes."""

    event: str = "training_complete"
    job_id: UUID
    status: TrainingStatus
    best_epoch: Optional[int] = None
    best_val_loss: Optional[float] = None
    onnx_model_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingJobStatusResponse(BaseModel):
    """Detailed training job status."""

    job: TrainingJobResponse
    recent_metrics: list[TrainingMetrics] = Field(default_factory=list, description="Last 10 epochs")
    websocket_url: str = Field(..., description="WebSocket URL for live updates")
