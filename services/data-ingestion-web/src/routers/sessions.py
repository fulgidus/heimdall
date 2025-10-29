
"""
Recording sessions API endpoints
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
import asyncpg
import logging

from ..models.session import (
    RecordingSession,
    RecordingSessionCreate,
    RecordingSessionWithDetails,
    SessionListResponse,
    SessionAnalytics,
    KnownSource,
    KnownSourceCreate,
    KnownSourceUpdate,
)
from ..db import get_pool
from ..rf_client import RFAcquisitionClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

# Initialize RF acquisition client
rf_client = RFAcquisitionClient()


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    approval_status: Optional[str] = Query(None, description="Filter by approval status"),
):
    """List all recording sessions with pagination"""
    pool = await get_pool()
    
    offset = (page - 1) * per_page
    
    # Build query with filters
    where_clauses = []
    if status:
        where_clauses.append(f"rs.status = '{status}'")
    if approval_status:
        where_clauses.append(f"rs.approval_status = '{approval_status}'")
    
    where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
    
    query = f"""
        SELECT 
            rs.id,
            rs.known_source_id,
            rs.session_name,
            rs.session_start,
            rs.session_end,
            rs.duration_seconds,
            rs.celery_task_id,
            rs.status,
            rs.approval_status,
            rs.notes,
            rs.created_at,
            rs.updated_at,
            ks.name as source_name,
            ks.frequency_hz as source_frequency,
            ks.latitude as source_latitude,
            ks.longitude as source_longitude,
            COUNT(m.id) as measurements_count
        FROM heimdall.recording_sessions rs
        JOIN heimdall.known_sources ks ON rs.known_source_id = ks.id
        LEFT JOIN heimdall.measurements m ON m.created_at >= rs.session_start 
            AND (rs.session_end IS NULL OR m.created_at <= rs.session_end)
        {where_sql}
        GROUP BY rs.id, ks.name, ks.frequency_hz, ks.latitude, ks.longitude
        ORDER BY rs.session_start DESC
        LIMIT $1 OFFSET $2
    """
    
    count_query = f"""
        SELECT COUNT(*) 
        FROM heimdall.recording_sessions rs
        WHERE 1=1 {where_sql}
    """
    
    async with pool.acquire() as conn:
        # Get sessions
        rows = await conn.fetch(query, per_page, offset)
        
        # Get total count
        total = await conn.fetchval(count_query)
        
        sessions = [
            RecordingSessionWithDetails(
                id=row["id"],
                known_source_id=row["known_source_id"],
                session_name=row["session_name"],
                session_start=row["session_start"],
                session_end=row["session_end"],
                duration_seconds=row["duration_seconds"],
                celery_task_id=row["celery_task_id"],
                status=row["status"],
                approval_status=row["approval_status"],
                notes=row["notes"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                source_name=row["source_name"],
                source_frequency=row["source_frequency"],
                source_latitude=row["source_latitude"],
                source_longitude=row["source_longitude"],
                measurements_count=row["measurements_count"],
            )
            for row in rows
        ]
        
        return SessionListResponse(
            sessions=sessions,
            total=total,
            page=page,
            per_page=per_page,
        )


@router.get("/analytics", response_model=SessionAnalytics)
async def get_session_analytics():
    """Get analytics for all sessions"""
    pool = await get_pool()
    
    query = """
        SELECT 
            COUNT(*) as total_sessions,
            COUNT(*) FILTER (WHERE status = 'completed') as completed_sessions,
            COUNT(*) FILTER (WHERE status = 'failed') as failed_sessions,
            COUNT(*) FILTER (WHERE status = 'pending') as pending_sessions,
            AVG(duration_seconds) as average_duration_seconds,
            SUM((SELECT COUNT(*) FROM heimdall.measurements m 
                 WHERE m.created_at >= rs.session_start 
                 AND (rs.session_end IS NULL OR m.created_at <= rs.session_end))) as total_measurements
        FROM heimdall.recording_sessions rs
    """
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query)
        
        total = row["total_sessions"] or 0
        completed = row["completed_sessions"] or 0
        failed = row["failed_sessions"] or 0
        pending = row["pending_sessions"] or 0
        
        success_rate = (completed / total * 100) if total > 0 else 0.0
        
        # Calculate average accuracy from inference results if available
        accuracy_query = """
            SELECT AVG(uncertainty_meters) as avg_accuracy
            FROM heimdall.recording_sessions rs
            WHERE rs.status = 'completed'
        """
        
        accuracy_row = await conn.fetchrow(accuracy_query)
        average_accuracy = accuracy_row["avg_accuracy"] if accuracy_row else None
        
        return SessionAnalytics(
            total_sessions=total,
            completed_sessions=completed,
            failed_sessions=failed,
            pending_sessions=pending,
            success_rate=success_rate,
            total_measurements=row["total_measurements"] or 0,
            average_duration_seconds=row["average_duration_seconds"],
            average_accuracy_meters=average_accuracy,
        )


@router.get("/{session_id}", response_model=RecordingSessionWithDetails)
async def get_session(session_id: UUID):
    """Get a specific recording session by ID"""
    pool = await get_pool()
    
    query = """
        SELECT 
            rs.id,
            rs.known_source_id,
            rs.session_name,
            rs.session_start,
            rs.session_end,
            rs.duration_seconds,
            rs.celery_task_id,
            rs.status,
            rs.approval_status,
            rs.notes,
            rs.created_at,
            rs.updated_at,
            ks.name as source_name,
            ks.frequency_hz as source_frequency,
            ks.latitude as source_latitude,
            ks.longitude as source_longitude,
            COUNT(m.id) as measurements_count
        FROM heimdall.recording_sessions rs
        JOIN heimdall.known_sources ks ON rs.known_source_id = ks.id
        LEFT JOIN heimdall.measurements m ON m.created_at >= rs.session_start 
            AND (rs.session_end IS NULL OR m.created_at <= rs.session_end)
        WHERE rs.id = $1
        GROUP BY rs.id, ks.name, ks.frequency_hz, ks.latitude, ks.longitude
    """
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, session_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return RecordingSessionWithDetails(
            id=row["id"],
            known_source_id=row["known_source_id"],
            session_name=row["session_name"],
            session_start=row["session_start"],
            session_end=row["session_end"],
            duration_seconds=row["duration_seconds"],
            celery_task_id=row["celery_task_id"],
            status=row["status"],
            approval_status=row["approval_status"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            source_name=row["source_name"],
            source_frequency=row["source_frequency"],
            source_latitude=row["source_latitude"],
            source_longitude=row["source_longitude"],
            measurements_count=row["measurements_count"],
        )


async def trigger_rf_acquisition_task(
    session_id: UUID,
    frequency_hz: int,
    duration_seconds: float,
):
    """Background task to trigger RF acquisition and update session status"""
    pool = await get_pool()
    
    try:
        # Update status to in_progress
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE heimdall.recording_sessions
                SET status = 'in_progress', updated_at = NOW()
                WHERE id = $1
                """,
                session_id,
            )
        
        # Trigger RF acquisition
        result = await rf_client.trigger_acquisition(
            frequency_hz=frequency_hz,
            duration_seconds=duration_seconds,
        )
        
        task_id = result.get("task_id")
        
        # Update session with task ID
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE heimdall.recording_sessions
                SET celery_task_id = $1, updated_at = NOW()
                WHERE id = $2
                """,
                task_id,
                session_id,
            )
        
        logger.info(
            f"RF acquisition triggered for session {session_id}, task_id={task_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger RF acquisition for session {session_id}: {e}")
        
        # Update session status to failed
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE heimdall.recording_sessions
                SET status = 'failed', 
                    session_end = NOW(),
                    duration_seconds = EXTRACT(EPOCH FROM (NOW() - session_start)),
                    updated_at = NOW()
                WHERE id = $1
                """,
                session_id,
            )


@router.post("", response_model=RecordingSession, status_code=201)
async def create_session(
    session: RecordingSessionCreate,
    background_tasks: BackgroundTasks,
):
    """Create a new recording session and trigger RF acquisition"""
    pool = await get_pool()
    
    # Verify known source exists
    async with pool.acquire() as conn:
        source_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM heimdall.known_sources WHERE id = $1)",
            session.known_source_id
        )
        
        if not source_exists:
            raise HTTPException(status_code=404, detail="Known source not found")
        
        # Insert new session
        query = """
            INSERT INTO heimdall.recording_sessions 
            (known_source_id, session_name, session_start, status, approval_status, notes)
            VALUES ($1, $2, $3, 'pending', 'pending', $4)
            RETURNING 
                id, known_source_id, session_name, session_start, session_end,
                duration_seconds, celery_task_id, status, approval_status,
                notes, created_at, updated_at
        """
        
        row = await conn.fetchrow(
            query,
            session.known_source_id,
            session.session_name,
            datetime.utcnow(),
            session.notes,
        )
        
        session_result = RecordingSession(**dict(row))
        
        # Trigger RF acquisition in background
        background_tasks.add_task(
            trigger_rf_acquisition_task,
            session_result.id,
            session.frequency_hz,
            session.duration_seconds,
        )
        
        logger.info(f"Created session {session_result.id}, queuing RF acquisition")
        
        return session_result


@router.patch("/{session_id}/status")
async def update_session_status(
    session_id: UUID,
    status: str,
    celery_task_id: Optional[str] = None,
):
    """Update session status"""
    pool = await get_pool()
    
    # Validate status
    valid_statuses = ["pending", "in_progress", "completed", "failed"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    async with pool.acquire() as conn:
        # Update session
        query = """
            UPDATE heimdall.recording_sessions
            SET status = $1, 
                celery_task_id = COALESCE($2, celery_task_id),
                session_end = CASE WHEN $1 IN ('completed', 'failed') THEN NOW() ELSE session_end END,
                duration_seconds = CASE WHEN $1 IN ('completed', 'failed') 
                    THEN EXTRACT(EPOCH FROM (NOW() - session_start))
                    ELSE duration_seconds END,
                updated_at = NOW()
            WHERE id = $3
            RETURNING 
                id, known_source_id, session_name, session_start, session_end,
                duration_seconds, celery_task_id, status, approval_status,
                notes, created_at, updated_at
        """
        
        row = await conn.fetchrow(query, status, celery_task_id, session_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return RecordingSession(**dict(row))


@router.patch("/{session_id}/approval")
async def update_session_approval(session_id: UUID, approval_status: str):
    """Update session approval status"""
    pool = await get_pool()
    
    # Validate approval status
    valid_statuses = ["pending", "approved", "rejected"]
    if approval_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid approval status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    async with pool.acquire() as conn:
        query = """
            UPDATE heimdall.recording_sessions
            SET approval_status = $1, updated_at = NOW()
            WHERE id = $2
            RETURNING 
                id, known_source_id, session_name, session_start, session_end,
                duration_seconds, celery_task_id, status, approval_status,
                notes, created_at, updated_at
        """
        
        row = await conn.fetchrow(query, approval_status, session_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return RecordingSession(**dict(row))


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: UUID):
    """Delete a recording session"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM heimdall.recording_sessions WHERE id = $1",
            session_id
        )
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Session not found")
        
        return None


@router.get("/known-sources", response_model=List[KnownSource])
async def list_known_sources():
    """List all known RF sources"""
    pool = await get_pool()
    
    query = """
        SELECT id, name, description, frequency_hz, latitude, longitude,
               power_dbm, source_type, is_validated, error_margin_meters, created_at, updated_at
        FROM heimdall.known_sources
        ORDER BY name
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        return [KnownSource(**dict(row)) for row in rows]


@router.get("/known-sources/{source_id}", response_model=KnownSource)
async def get_known_source(source_id: UUID):
    """Get a specific known RF source"""
    pool = await get_pool()
    
    query = """
        SELECT id, name, description, frequency_hz, latitude, longitude,
               power_dbm, source_type, is_validated, error_margin_meters, created_at, updated_at
        FROM heimdall.known_sources
        WHERE id = $1
    """
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, source_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Known source not found")
        
        return KnownSource(**dict(row))


@router.post("/known-sources", response_model=KnownSource, status_code=201)
async def create_known_source(source: KnownSourceCreate):
    """Create a new known RF source"""
    pool = await get_pool()
    
    query = """
        INSERT INTO heimdall.known_sources 
        (name, description, frequency_hz, latitude, longitude, power_dbm, source_type, is_validated, error_margin_meters)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id, name, description, frequency_hz, latitude, longitude,
                  power_dbm, source_type, is_validated, error_margin_meters, created_at, updated_at
    """
    
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                query,
                source.name,
                source.description,
                source.frequency_hz,
                source.latitude,
                source.longitude,
                source.power_dbm,
                source.source_type,
                source.is_validated,
                source.error_margin_meters,
            )
            
            return KnownSource(**dict(row))
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=400,
                detail="A known source with this name already exists"
            )


@router.put("/known-sources/{source_id}", response_model=KnownSource)
async def update_known_source(source_id: UUID, source: KnownSourceUpdate):
    """Update a known RF source"""
    pool = await get_pool()
    
    # Build dynamic update query based on provided fields
    update_fields = []
    params = []
    param_count = 1
    
    if source.name is not None:
        update_fields.append(f"name = ${param_count}")
        params.append(source.name)
        param_count += 1
    
    if source.description is not None:
        update_fields.append(f"description = ${param_count}")
        params.append(source.description)
        param_count += 1
    
    if source.frequency_hz is not None:
        update_fields.append(f"frequency_hz = ${param_count}")
        params.append(source.frequency_hz)
        param_count += 1
    
    if source.latitude is not None:
        update_fields.append(f"latitude = ${param_count}")
        params.append(source.latitude)
        param_count += 1
    
    if source.longitude is not None:
        update_fields.append(f"longitude = ${param_count}")
        params.append(source.longitude)
        param_count += 1
    
    if source.power_dbm is not None:
        update_fields.append(f"power_dbm = ${param_count}")
        params.append(source.power_dbm)
        param_count += 1
    
    if source.source_type is not None:
        update_fields.append(f"source_type = ${param_count}")
        params.append(source.source_type)
        param_count += 1
    
    if source.is_validated is not None:
        update_fields.append(f"is_validated = ${param_count}")
        params.append(source.is_validated)
        param_count += 1
    
    if source.error_margin_meters is not None:
        update_fields.append(f"error_margin_meters = ${param_count}")
        params.append(source.error_margin_meters)
        param_count += 1
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    # Always update updated_at
    update_fields.append("updated_at = NOW()")
    
    params.append(source_id)
    
    query = f"""
        UPDATE heimdall.known_sources
        SET {', '.join(update_fields)}
        WHERE id = ${param_count}
        RETURNING id, name, description, frequency_hz, latitude, longitude,
                  power_dbm, source_type, is_validated, error_margin_meters, created_at, updated_at
    """
    
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(query, *params)
            
            if not row:
                raise HTTPException(status_code=404, detail="Known source not found")
            
            return KnownSource(**dict(row))
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=400,
                detail="A known source with this name already exists"
            )


@router.delete("/known-sources/{source_id}", status_code=204)
async def delete_known_source(source_id: UUID):
    """Delete a known RF source"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        # Check if source is in use by any recording sessions
        usage_check = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.recording_sessions WHERE known_source_id = $1",
            source_id
        )
        
        if usage_check > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete source: it is referenced by {usage_check} recording session(s)"
            )
        
        result = await conn.execute(
            "DELETE FROM heimdall.known_sources WHERE id = $1",
            source_id
        )
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Known source not found")
        
        return None
