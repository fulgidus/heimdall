"""Repository pattern for database operations"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc
from .models.session import RecordingSessionORM, RecordingSessionResponse, SessionStatus


class SessionRepository:
    """Data access layer for recording sessions"""

    @staticmethod
    def create(db: Session, session_name: str, frequency_mhz: float, duration_seconds: int) -> RecordingSessionORM:
        """Create a new recording session"""
        session = RecordingSessionORM(
            session_name=session_name,
            frequency_mhz=frequency_mhz,
            duration_seconds=duration_seconds,
            status=SessionStatus.PENDING.value,
            websdrs_enabled=7,
            created_at=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_by_id(db: Session, session_id: int) -> Optional[RecordingSessionORM]:
        """Fetch session by ID"""
        return db.query(RecordingSessionORM).filter(RecordingSessionORM.id == session_id).first()

    @staticmethod
    def get_by_task_id(db: Session, task_id: str) -> Optional[RecordingSessionORM]:
        """Fetch session by Celery task ID"""
        return db.query(RecordingSessionORM).filter(RecordingSessionORM.celery_task_id == task_id).first()

    @staticmethod
    def list_sessions(db: Session, offset: int = 0, limit: int = 20) -> tuple[int, List[RecordingSessionORM]]:
        """List all sessions with pagination (newest first)"""
        total = db.query(RecordingSessionORM).count()
        sessions = (
            db.query(RecordingSessionORM)
            .order_by(desc(RecordingSessionORM.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return total, sessions

    @staticmethod
    def update_status(
        db: Session,
        session_id: int,
        status: SessionStatus,
        celery_task_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
    ) -> Optional[RecordingSessionORM]:
        """Update session status"""
        session = db.query(RecordingSessionORM).filter(RecordingSessionORM.id == session_id).first()
        if session:
            session.status = status.value
            if celery_task_id:
                session.celery_task_id = celery_task_id
            if started_at:
                session.started_at = started_at
            db.commit()
            db.refresh(session)
        return session

    @staticmethod
    def update_completed(
        db: Session,
        session_id: int,
        result_metadata: dict,
        minio_path: str,
    ) -> Optional[RecordingSessionORM]:
        """Mark session as completed with results"""
        session = db.query(RecordingSessionORM).filter(RecordingSessionORM.id == session_id).first()
        if session:
            session.status = SessionStatus.COMPLETED.value
            session.result_metadata = result_metadata
            session.minio_path = minio_path
            session.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(session)
        return session

    @staticmethod
    def update_failed(db: Session, session_id: int, error_message: str) -> Optional[RecordingSessionORM]:
        """Mark session as failed with error"""
        session = db.query(RecordingSessionORM).filter(RecordingSessionORM.id == session_id).first()
        if session:
            session.status = SessionStatus.FAILED.value
            session.error_message = error_message
            session.completed_at = datetime.utcnow()
            db.commit()
            db.refresh(session)
        return session

    @staticmethod
    def delete(db: Session, session_id: int) -> bool:
        """Delete a session by ID. Returns True if deleted, False if not found"""
        session = db.query(RecordingSessionORM).filter(RecordingSessionORM.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
