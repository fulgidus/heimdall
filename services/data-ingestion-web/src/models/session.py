"""
Session management models
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID


class KnownSource(BaseModel):
    """Known RF source for training"""
    id: UUID
    name: str
    description: Optional[str] = None
    frequency_hz: int
    latitude: float
    longitude: float
    power_dbm: Optional[float] = None
    source_type: Optional[str] = None
    is_validated: bool = False
    created_at: datetime
    updated_at: datetime


class KnownSourceCreate(BaseModel):
    """Create a new known source"""
    name: str
    description: Optional[str] = None
    frequency_hz: int = Field(..., gt=0, description="Frequency in Hz")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    power_dbm: Optional[float] = None
    source_type: Optional[str] = None
    is_validated: bool = False


class RecordingSession(BaseModel):
    """Recording session model"""
    id: UUID
    known_source_id: UUID
    session_name: str
    session_start: datetime
    session_end: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    celery_task_id: Optional[str] = None
    status: str  # pending, in_progress, completed, failed
    approval_status: str = "pending"  # pending, approved, rejected
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class RecordingSessionCreate(BaseModel):
    """Create a new recording session"""
    known_source_id: UUID
    session_name: str
    frequency_hz: int = Field(..., gt=0, description="Frequency in Hz")
    duration_seconds: float = Field(..., gt=0, le=3600, description="Duration in seconds (max 1 hour)")
    notes: Optional[str] = None


class RecordingSessionWithDetails(RecordingSession):
    """Recording session with source details"""
    source_name: str
    source_frequency: int
    source_latitude: float
    source_longitude: float
    measurements_count: int = 0
    

class SessionListResponse(BaseModel):
    """Response for session list"""
    sessions: List[RecordingSessionWithDetails]
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
    total_measurements: int
    average_duration_seconds: Optional[float] = None
    average_accuracy_meters: Optional[float] = None
