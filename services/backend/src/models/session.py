"""
Session management models
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class KnownSource(BaseModel):
    """Known RF source for training"""

    id: UUID
    name: str
    description: str | None = None
    frequency_hz: int
    latitude: float
    longitude: float
    power_dbm: float | None = None
    source_type: str | None = None
    is_validated: bool = False
    error_margin_meters: float | None = None  # Optional, can be set later
    created_at: datetime
    updated_at: datetime


class KnownSourceCreate(BaseModel):
    """Create a new known source"""

    name: str
    description: str | None = None
    frequency_hz: int = Field(..., gt=0, description="Frequency in Hz")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    power_dbm: float | None = None
    source_type: str | None = None
    is_validated: bool = False
    error_margin_meters: float | None = Field(
        None, gt=0, description="Error margin radius in meters"
    )


class KnownSourceUpdate(BaseModel):
    """Update a known source"""

    name: str | None = None
    description: str | None = None
    frequency_hz: int | None = Field(None, gt=0, description="Frequency in Hz")
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    power_dbm: float | None = None
    source_type: str | None = None
    is_validated: bool | None = None
    error_margin_meters: float | None = Field(
        None, gt=0, description="Error margin radius in meters"
    )


class RecordingSession(BaseModel):
    """Recording session model"""

    id: UUID
    known_source_id: UUID
    session_name: str
    session_start: datetime
    session_end: datetime | None = None
    duration_seconds: float | None = None
    celery_task_id: str | None = None
    status: str  # pending, in_progress, completed, failed
    approval_status: str = "pending"  # pending, approved, rejected
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


class RecordingSessionCreate(BaseModel):
    """Create a new recording session"""

    known_source_id: UUID
    session_name: str
    frequency_hz: int = Field(..., gt=0, description="Frequency in Hz")
    duration_seconds: float = Field(
        ..., gt=0, le=3600, description="Duration in seconds (max 1 hour)"
    )
    notes: str | None = None


class RecordingSessionUpdate(BaseModel):
    """Update recording session metadata"""

    session_name: str | None = Field(None, min_length=1, max_length=255)
    notes: str | None = None
    approval_status: str | None = Field(None, pattern="^(pending|approved|rejected)$")


class RecordingSessionWithDetails(RecordingSession):
    """Recording session with source details"""

    source_name: str
    source_frequency: int
    source_latitude: float
    source_longitude: float
    measurements_count: int = 0


class SessionListResponse(BaseModel):
    """Response for session list"""

    sessions: list[RecordingSessionWithDetails]
    total: int
    page: int
    per_page: int


class SessionAnalytics(BaseModel):
    """Analytics for sessions"""

    total_sessions: int
    completed_sessions: int
    failed_sessions: int
    pending_sessions: int
    success_rate: float
    total_measurements: int = 0
    average_duration_seconds: float | None = None
    average_accuracy_meters: float | None = None
