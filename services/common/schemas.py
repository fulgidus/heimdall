"""Pydantic validation schemas for API requests/responses."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from enum import Enum


class AcquisitionRequest(BaseModel):
    """Request to trigger RF acquisition."""

    frequency_mhz: float = Field(..., gt=2.0, lt=1000.0, description="Frequency in MHz")
    duration_seconds: int = Field(..., ge=1, le=600, description="Duration in seconds")
    receiver_ids: Optional[list[str]] = Field(
        None, description="Specific receivers, or all if None"
    )

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
    measurements: Optional[list[MeasurementResponse]] = None
    error_message: Optional[str] = None

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
