"""
Session management models
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class KnownSource(BaseModel):
    """Known RF source for training"""

    id: UUID
    name: str
    description: str | None = None
    frequency_hz: int | None = None  # Optional - may be unknown for amateur stations
    latitude: float | None = None  # Optional - may be unknown initially
    longitude: float | None = None  # Optional - may be unknown initially
    power_dbm: float | None = None
    source_type: str | None = None
    is_validated: bool = False
    error_margin_meters: float | None = None  # Optional, can be set later
    owner_id: str | None = None  # Keycloak user ID of the owner
    is_public: bool = False  # Public visibility flag
    created_at: datetime
    updated_at: datetime


class KnownSourceCreate(BaseModel):
    """Create a new known source"""

    name: str
    description: str | None = None
    frequency_hz: int | None = Field(None, gt=0, description="Frequency in Hz (optional)")
    latitude: float | None = Field(None, ge=-90, le=90, description="Latitude (optional)")
    longitude: float | None = Field(None, ge=-180, le=180, description="Longitude (optional)")
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
    known_source_id: UUID | None = None  # NULL when source is unknown
    constellation_id: UUID | None = None  # NULL when not in a constellation
    session_name: str
    session_start: datetime
    session_end: datetime | None = None
    duration_seconds: float | None = None
    celery_task_id: str | None = None
    status: str  # pending, in_progress, completed, failed
    approval_status: str = "pending"  # pending, approved, rejected
    uncertainty_meters: float | None = None
    notes: str = ""  # Default to empty string instead of None
    created_at: datetime
    updated_at: datetime

    @field_validator('notes', mode='before')
    @classmethod
    def notes_to_empty_string(cls, v):
        """Convert None to empty string for frontend compatibility"""
        return v if v is not None else ""


class RecordingSessionCreate(BaseModel):
    """Create a new recording session"""

    known_source_id: UUID | None = None  # None for unknown sources
    constellation_id: UUID | None = Field(None, description="Optional constellation for collaboration")
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
    uncertainty_meters: float | None = Field(None, gt=0, description="Uncertainty in meters")


class RecordingSessionWithDetails(RecordingSession):
    """Recording session with source details"""

    source_name: str | None = None  # None when source is unknown
    source_frequency: int | None = None
    source_latitude: float | None = None
    source_longitude: float | None = None
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
