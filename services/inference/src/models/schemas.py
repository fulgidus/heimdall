"""Pydantic schemas for inference service API."""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request for single prediction endpoint."""

    iq_data: list[list[float]] = Field(
        ...,
        description="IQ data as [[I1, Q1], [I2, Q2], ...] or complex array",
        examples=[[[1.0, 0.5], [0.8, 0.3]]],
    )
    session_id: str | None = Field(
        None, description="Optional session ID for tracking and audit trail"
    )
    cache_enabled: bool = Field(True, description="Enable Redis caching for this prediction")

    class Config:
        schema_extra = {
            "example": {
                "iq_data": [[1.0, 0.5], [0.8, 0.3]],
                "session_id": "session-123",
                "cache_enabled": True,
            }
        }


class UncertaintyResponse(BaseModel):
    """Uncertainty ellipse parameters for visualization."""

    sigma_x: float = Field(..., description="Standard deviation in X direction (meters)", ge=0.0)
    sigma_y: float = Field(..., description="Standard deviation in Y direction (meters)", ge=0.0)
    theta: float = Field(
        0.0, description="Rotation angle of ellipse in degrees (-180 to 180)", ge=-180.0, le=180.0
    )
    confidence_interval: float = Field(
        0.68, description="Confidence interval (1-sigma = 68%, 2-sigma = 95%, etc.)", ge=0.0, le=1.0
    )


class PositionResponse(BaseModel):
    """Predicted position in geographic coordinates."""

    latitude: float = Field(..., description="Latitude in decimal degrees", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="Longitude in decimal degrees", ge=-180.0, le=180.0)


class PredictionResponse(BaseModel):
    """Response for single prediction endpoint."""

    position: PositionResponse = Field(..., description="Predicted position")
    uncertainty: UncertaintyResponse = Field(
        ..., description="Uncertainty parameters for visualization"
    )
    confidence: float = Field(
        ..., description="Model confidence (0-1, higher is more confident)", ge=0.0, le=1.0
    )
    model_version: str = Field(..., description="Version of model used for this prediction")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds", ge=0.0)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp when prediction was made"
    )

    class Config:
        schema_extra = {
            "example": {
                "position": {"latitude": 45.123, "longitude": 7.456},
                "uncertainty": {
                    "sigma_x": 45.0,
                    "sigma_y": 38.0,
                    "theta": 25.0,
                    "confidence_interval": 0.68,
                },
                "confidence": 0.92,
                "model_version": "v1.2.0",
                "inference_time_ms": 125.5,
                "timestamp": "2025-10-22T10:30:00",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction endpoint."""

    iq_samples: list[list[list[float]]] = Field(
        ...,
        description="List of IQ data samples, each as [[I1, Q1], [I2, Q2], ...]",
        min_items=1,
        max_items=100,
    )
    session_id: str | None = Field(None, description="Optional session ID")
    cache_enabled: bool = Field(True, description="Enable Redis caching")

    class Config:
        schema_extra = {
            "example": {
                "iq_samples": [
                    [[1.0, 0.5], [0.8, 0.3]],
                    [[0.9, 0.4], [0.7, 0.2]],
                ],
                "session_id": "batch-session-456",
                "cache_enabled": True,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""

    predictions: list[PredictionResponse] = Field(..., description="List of prediction results")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds", ge=0.0)
    samples_per_second: float = Field(
        ..., description="Throughput: samples processed per second", ge=0.0
    )


class ModelInfoResponse(BaseModel):
    """Response for model information endpoint."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version (semantic versioning)")
    stage: str = Field(..., description="Current model stage (Production, Staging, None)")
    created_at: datetime = Field(..., description="Model creation timestamp")
    mlflow_run_id: str = Field(..., description="MLflow run ID for reproducibility")
    accuracy_meters: float | None = Field(
        None, description="Localization accuracy (sigma) in meters from training"
    )
    training_samples: int | None = Field(None, description="Number of samples in training set")
    last_reloaded: datetime = Field(
        ..., description="Timestamp when model was last reloaded in service"
    )
    inference_count: int = Field(..., description="Total number of inferences performed", ge=0)
    avg_latency_ms: float = Field(
        ..., description="Average inference latency in milliseconds", ge=0.0
    )
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)", ge=0.0, le=1.0)
    status: str = Field(..., description="Service status (ready, loading, error)")

    class Config:
        schema_extra = {
            "example": {
                "name": "localization_model",
                "version": "1.2.0",
                "stage": "Production",
                "created_at": "2025-10-20T15:30:00",
                "mlflow_run_id": "abc123def456",
                "accuracy_meters": 30.0,
                "training_samples": 5000,
                "last_reloaded": "2025-10-22T08:00:00",
                "inference_count": 1250,
                "avg_latency_ms": 145.5,
                "cache_hit_rate": 0.82,
                "status": "ready",
            }
        }


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    model_ready: bool = Field(..., description="Is model ready for inference")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")

    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "service": "inference",
                "version": "0.1.0",
                "model_ready": True,
                "timestamp": "2025-10-22T10:30:00",
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: str | None = Field(None, description="Request ID for tracing")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "IQ data validation failed",
                "timestamp": "2025-10-22T10:30:00",
                "request_id": "req-789",
            }
        }
