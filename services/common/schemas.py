"""Pydantic validation schemas for API requests/responses."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AcquisitionRequest(BaseModel):
    """Request to trigger RF acquisition."""

    frequency_mhz: float = Field(..., gt=2.0, lt=1000.0, description="Frequency in MHz")
    duration_seconds: int = Field(..., ge=1, le=600, description="Duration in seconds")
    receiver_ids: list[str] | None = Field(None, description="Specific receivers, or all if None")

    model_config = ConfigDict(strict=True)


class MeasurementResponse(BaseModel):
    """Single measurement response."""

    receiver_id: str
    frequency_mhz: float
    snr_db: float
    signal_strength_dbm: float
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class AcquisitionResponse(BaseModel):
    """Response from acquisition request."""

    task_id: str
    status: str = Field(..., pattern=r"^(pending|processing|success|failed|partial_success)$")
    measurements: list[MeasurementResponse] | None = None
    error_message: str | None = None

    model_config = ConfigDict(strict=True)


class StatusEnum(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., pattern=r"^(ok|degraded|error)$")
    timestamp: datetime
    service_name: str
    version: str
    uptime_seconds: int

    model_config = ConfigDict(strict=True)
