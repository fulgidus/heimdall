"""Session endpoints for recording management"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db, init_db
from ..models.session import (
    RecordingSessionCreate,
    RecordingSessionResponse,
    RecordingSessionList,
    SessionStatus,
)
from ..repository import SessionRepository
from ..tasks import trigger_acquisition

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


# Initialize DB on startup
@router.on_event("startup")
async def startup_event():
    """Initialize database tables"""
    init_db()


@router.post("/create", response_model=RecordingSessionResponse, status_code=201)
async def create_session(
    request: RecordingSessionCreate,
    db: Session = Depends(get_db),
) -> RecordingSessionResponse:
    """
    Create a new recording session and queue RF acquisition.
    
    Returns the session details with status PENDING (will move to PROCESSING shortly).
    """
    # Create session in DB
    repo = SessionRepository()
    session = repo.create(
        db=db,
        session_name=request.session_name,
        frequency_mhz=request.frequency_mhz,
        duration_seconds=request.duration_seconds,
    )
    
    # Queue the acquisition task asynchronously
    trigger_acquisition.delay(
        session_id=session.id,
        frequency_mhz=request.frequency_mhz,
        duration_seconds=request.duration_seconds,
    )
    
    return RecordingSessionResponse.from_orm(session)


@router.get("/{session_id}", response_model=RecordingSessionResponse)
async def get_session(
    session_id: int,
    db: Session = Depends(get_db),
) -> RecordingSessionResponse:
    """Get details of a specific recording session"""
    repo = SessionRepository()
    session = repo.get_by_id(db, session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return RecordingSessionResponse.from_orm(session)


@router.get("", response_model=RecordingSessionList)
async def list_sessions(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> RecordingSessionList:
    """List all recording sessions with pagination"""
    repo = SessionRepository()
    total, sessions = repo.list_sessions(db, offset=offset, limit=limit)
    
    return RecordingSessionList(
        total=total,
        offset=offset,
        limit=limit,
        sessions=[RecordingSessionResponse.from_orm(s) for s in sessions],
    )


@router.get("/{session_id}/status")
async def get_session_status(
    session_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Get current status and progress of a session"""
    repo = SessionRepository()
    session = repo.get_by_id(db, session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate progress
    progress = 0
    if session.status == SessionStatus.PENDING.value:
        progress = 0
    elif session.status == SessionStatus.PROCESSING.value:
        progress = 50
    elif session.status == SessionStatus.COMPLETED.value:
        progress = 100
    elif session.status == SessionStatus.FAILED.value:
        progress = 0
    
    return {
        "session_id": session.id,
        "status": session.status,
        "progress": progress,
        "created_at": session.created_at,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "error_message": session.error_message,
    }
