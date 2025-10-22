"""Recording Session models (SQLAlchemy + Pydantic)"""
from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()


class SessionStatus(str, Enum):
    """Session lifecycle states"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== DATABASE MODELS ====================

class RecordingSessionORM(Base):
    """SQLAlchemy ORM model for recording sessions"""
    __tablename__ = "recording_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, nullable=False, index=True)
    frequency_mhz = Column(Float, nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    status = Column(String, default=SessionStatus.PENDING.value)
    
    # Celery tracking
    celery_task_id = Column(String, unique=True, index=True, nullable=True)
    
    # Results
    result_metadata = Column(JSON, nullable=True)  # Contains SNR, offset, etc.
    minio_path = Column(String, nullable=True)    # s3://heimdall-raw-iq/sessions/{id}/...
    error_message = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # WebSDR details
    websdrs_enabled = Column(Integer, default=7)  # How many WebSDR were enabled
    
    __table_args__ = {"extend_existing": True}


# ==================== PYDANTIC SCHEMAS ====================

class RecordingSessionCreate(BaseModel):
    """Request schema for creating a new recording session"""
    session_name: str = Field(..., min_length=1, max_length=255, description="Friendly name for this session")
    frequency_mhz: float = Field(default=145.500, ge=100, le=1000, description="Frequency in MHz")
    duration_seconds: int = Field(default=30, ge=5, le=300, description="Recording duration in seconds")


class RecordingSessionResponse(BaseModel):
    """Response schema for session queries"""
    id: int
    session_name: str
    frequency_mhz: float
    duration_seconds: int
    status: SessionStatus
    celery_task_id: Optional[str] = None
    result_metadata: Optional[dict] = None
    minio_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    websdrs_enabled: int
    
    class Config:
        from_attributes = True


class RecordingSessionList(BaseModel):
    """List of sessions with pagination"""
    total: int
    offset: int
    limit: int
    sessions: list[RecordingSessionResponse]


class SessionStatus_Update(BaseModel):
    """Internal model to update session status"""
    status: SessionStatus
    result_metadata: Optional[dict] = None
    error_message: Optional[str] = None
