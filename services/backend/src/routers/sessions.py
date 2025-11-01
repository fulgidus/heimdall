"""
Recording sessions API endpoints
"""

import logging
from datetime import datetime
from uuid import UUID

import asyncpg
from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query

from ..db import get_pool
from ..models.session import (
    KnownSource,
    KnownSourceCreate,
    KnownSourceUpdate,
    RecordingSession,
    RecordingSessionCreate,
    RecordingSessionUpdate,
    RecordingSessionWithDetails,
    SessionAnalytics,
    SessionListResponse,
)
from ..rf_client import RFAcquisitionClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

# Initialize RF acquisition client
rf_client = RFAcquisitionClient()


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: str | None = Query(None, description="Filter by status"),
    approval_status: str | None = Query(None, description="Filter by approval status"),
):
    """List all recording sessions with pagination"""
    pool = get_pool()

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
            rs.uncertainty_meters,
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
    pool = get_pool()

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

        # Calculate average accuracy from completed sessions
        accuracy_query = """
            SELECT AVG(uncertainty_meters) as avg_accuracy
            FROM heimdall.recording_sessions
            WHERE status = 'completed' AND uncertainty_meters IS NOT NULL
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


@router.patch("/{session_id}", response_model=RecordingSessionWithDetails)
async def update_session(
    session_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
    session_update: RecordingSessionUpdate = ...,
):
    """Update recording session metadata (session_name, notes, approval_status)"""
    UUID(session_id)
    pool = await get_pool()

    # Build dynamic update query based on provided fields
    update_fields = []
    params = []
    param_count = 1

    if session_update.session_name is not None:
        update_fields.append(f"session_name = ${param_count}")
        params.append(session_update.session_name)
        param_count += 1

    if session_update.notes is not None:
        update_fields.append(f"notes = ${param_count}")
        params.append(session_update.notes)
        param_count += 1

    if session_update.approval_status is not None:
        update_fields.append(f"approval_status = ${param_count}")
        params.append(session_update.approval_status)
        param_count += 1

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Always update updated_at
    update_fields.append("updated_at = NOW()")

    params.append(session_id)

    update_query = f"""
        UPDATE heimdall.recording_sessions
        SET {', '.join(update_fields)}
        WHERE id = ${param_count}
        RETURNING id
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(update_query, *params)

        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        # Fetch full session details to return
        detail_query = """
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
                rs.uncertainty_meters,
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

        detail_row = await conn.fetchrow(detail_query, session_id)

        return RecordingSessionWithDetails(
            id=detail_row["id"],
            known_source_id=detail_row["known_source_id"],
            session_name=detail_row["session_name"],
            session_start=detail_row["session_start"],
            session_end=detail_row["session_end"],
            duration_seconds=detail_row["duration_seconds"],
            celery_task_id=detail_row["celery_task_id"],
            status=detail_row["status"],
            approval_status=detail_row["approval_status"],
            notes=detail_row["notes"],
            created_at=detail_row["created_at"],
            updated_at=detail_row["updated_at"],
            source_name=detail_row["source_name"],
            source_frequency=detail_row["source_frequency"],
            source_latitude=detail_row["source_latitude"],
            source_longitude=detail_row["source_longitude"],
            measurements_count=detail_row["measurements_count"],
        )


async def update_session_to_failed(pool, session_id: UUID):
    """Update session status to failed with end time"""
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


async def trigger_rf_acquisition_task(
    session_id: UUID,
    frequency_hz: int,
    duration_seconds: float,
):
    """Background task to trigger RF acquisition and update session status"""
    pool = get_pool()

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

        # Get WebSDR configurations
        from .acquisition import get_websdrs_config
        from ..tasks.acquire_iq import acquire_iq_chunked

        websdrs_config = get_websdrs_config()

        if not websdrs_config:
            raise ValueError("No active WebSDRs available")

        # Trigger RF acquisition using chunked task (1-second samples)
        task = acquire_iq_chunked.delay(
            frequency_hz=frequency_hz,
            duration_seconds=int(duration_seconds),
            session_id=str(session_id),
            websdrs_config_list=websdrs_config,
        )

        task_id = task.id

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

        logger.info(f"RF acquisition triggered for session {session_id}, task_id={task_id}")

    except Exception as e:
        logger.error(f"Failed to trigger RF acquisition for session {session_id}: {e}")
        await update_session_to_failed(pool, session_id)


@router.post("", response_model=RecordingSession, status_code=201)
async def create_session(
    session: RecordingSessionCreate,
    background_tasks: BackgroundTasks,
):
    """Create a new recording session and trigger RF acquisition"""
    pool = get_pool()

    # Verify known source exists
    async with pool.acquire() as conn:
        source_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM heimdall.known_sources WHERE id = $1)",
            session.known_source_id,
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
    session_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
    status: str = ...,
    celery_task_id: str | None = None,
):
    """Update session status"""
    UUID(session_id)
    pool = get_pool()

    # Validate status
    valid_statuses = ["pending", "in_progress", "completed", "failed"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400, detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
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
async def update_session_approval(
    session_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
    approval_status: str = ...,
):
    """Update session approval status"""
    UUID(session_id)
    pool = get_pool()

    # Validate approval status
    valid_statuses = ["pending", "approved", "rejected"]
    if approval_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid approval status. Must be one of: {', '.join(valid_statuses)}",
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
async def delete_session(
    session_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
):
    """Delete a recording session"""
    session_uuid = UUID(session_id)
    pool = get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM heimdall.recording_sessions WHERE id = $1", session_uuid
        )

        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Session not found")

        return None


@router.get("/sources", response_model=list[KnownSource])
async def list_known_sources():
    """List all known RF sources"""
    pool = get_pool()

    query = """
        SELECT id, name, description, frequency_hz, latitude, longitude,
               power_dbm, source_type, is_validated, created_at, updated_at
        FROM heimdall.known_sources
        ORDER BY name
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        return [KnownSource(**dict(row)) for row in rows]


# Alias for backward compatibility - /known-sources also works
@router.get("/known-sources", response_model=list[KnownSource])
async def list_known_sources_compat():
    """List all known RF sources (alias for /sources for backward compatibility)"""
    return await list_known_sources()


@router.get("/known-sources/{source_id}", response_model=KnownSource)
async def get_known_source(
    source_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
):
    """Get a specific known RF source"""
    source_uuid = UUID(source_id)
    pool = get_pool()

    query = """
        SELECT id, name, description, frequency_hz, latitude, longitude,
               power_dbm, source_type, is_validated, created_at, updated_at
        FROM heimdall.known_sources
        WHERE id = $1
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, source_uuid)

        if not row:
            raise HTTPException(status_code=404, detail="Known source not found")

        return KnownSource(**dict(row))


@router.post("/known-sources", response_model=KnownSource, status_code=201)
async def create_known_source(source: KnownSourceCreate):
    """Create a new known RF source"""
    pool = get_pool()

    query = """
        INSERT INTO heimdall.known_sources
        (name, description, frequency_hz, latitude, longitude, power_dbm, source_type, is_validated)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id, name, description, frequency_hz, latitude, longitude,
                  power_dbm, source_type, is_validated, created_at, updated_at
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
            )

            return KnownSource(**dict(row))
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=400, detail="A known source with this name already exists"
            )


@router.put("/known-sources/{source_id}", response_model=KnownSource)
async def update_known_source(
    source_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
    source: KnownSourceUpdate = ...,
):
    """Update a known RF source"""
    source_uuid = UUID(source_id)
    pool = get_pool()

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

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Always update updated_at
    update_fields.append("updated_at = NOW()")

    params.append(source_uuid)

    query = f"""
        UPDATE heimdall.known_sources
        SET {', '.join(update_fields)}
        WHERE id = ${param_count}
        RETURNING id, name, description, frequency_hz, latitude, longitude,
                  power_dbm, source_type, is_validated, created_at, updated_at
    """

    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(query, *params)

            if not row:
                raise HTTPException(status_code=404, detail="Known source not found")

            return KnownSource(**dict(row))
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=400, detail="A known source with this name already exists"
            )


@router.delete("/known-sources/{source_id}", status_code=204)
async def delete_known_source(
    source_id: str = Path(
        ..., regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
):
    """Delete a known RF source"""
    source_uuid = UUID(source_id)
    pool = get_pool()

    async with pool.acquire() as conn:
        # Check if source is in use by any recording sessions
        usage_check = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.recording_sessions WHERE known_source_id = $1",
            source_uuid,
        )

        if usage_check > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete source: it is referenced by {usage_check} recording session(s)",
            )

        result = await conn.execute("DELETE FROM heimdall.known_sources WHERE id = $1", source_uuid)

        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Known source not found")

        return None


@router.get("/{session_id}", response_model=RecordingSessionWithDetails)
async def get_session(
    session_id: str = Path(
        ...,
        regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="Session ID as UUID",
    )
):
    """Get a specific recording session by ID"""
    # Convert to UUID
    session_uuid = UUID(session_id)
    pool = get_pool()

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
        row = await conn.fetchrow(query, session_uuid)

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


# WebSocket handlers for real-time session management
async def handle_session_start_ws(session_data: dict):
    """Handle session start command via WebSocket - now only validates configuration"""
    session_name = session_data.get("session_name")
    frequency_mhz = session_data.get("frequency_mhz")
    duration_seconds = session_data.get("duration_seconds")
    notes = session_data.get("notes", "")

    if not session_name or not frequency_mhz or not duration_seconds:
        raise ValueError("Missing required fields: session_name, frequency_mhz, duration_seconds")

    # Just validate and return the configuration - session creation happens in session:complete
    return {
        "status": "ready",
        "session_name": session_name,
        "frequency_mhz": frequency_mhz,
        "duration_seconds": duration_seconds,
        "notes": notes,
    }


async def handle_session_assign_source_ws(assignment_data: dict):
    """Handle source assignment via WebSocket"""
    pool = get_pool()

    session_id = assignment_data.get("session_id")
    source_id = assignment_data.get("source_id")

    if not session_id:
        raise ValueError("Missing required field: session_id")

    # If source_id is "unknown" or not provided, use the unknown source
    if not source_id or source_id == "unknown":
        async with pool.acquire() as conn:
            source_id = await conn.fetchval(
                "SELECT id FROM heimdall.known_sources WHERE name = 'Unknown'"
            )

            if not source_id:
                raise ValueError("Unknown source not found in database")

    async with pool.acquire() as conn:
        # Verify source exists
        source_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM heimdall.known_sources WHERE id = $1)",
            UUID(source_id) if isinstance(source_id, str) else source_id,
        )

        if not source_exists:
            raise ValueError("Source not found")

        # Update session with source
        query = """
            UPDATE heimdall.recording_sessions
            SET known_source_id = $1,
                status = 'source_assigned',
                updated_at = NOW()
            WHERE id = $2
            RETURNING
                id, known_source_id, session_name, session_start, session_end,
                duration_seconds, celery_task_id, status, approval_status,
                notes, created_at, updated_at
        """

        row = await conn.fetchrow(
            query,
            UUID(source_id) if isinstance(source_id, str) else source_id,
            UUID(session_id) if isinstance(session_id, str) else session_id,
        )

        if not row:
            raise ValueError("Session not found")

        session = RecordingSession(**dict(row))

        return {
            "session_id": str(session.id),
            "source_id": str(session.known_source_id),
            "status": session.status,
        }


async def handle_session_complete_ws(complete_data: dict):
    """Handle session completion and trigger acquisition via WebSocket"""
    pool = get_pool()

    # New workflow: create session when acquisition starts
    session_name = complete_data.get("session_name")
    frequency_hz = complete_data.get("frequency_hz")
    duration_seconds = complete_data.get("duration_seconds")
    source_id = complete_data.get("source_id")
    notes = complete_data.get("notes", "")

    if not session_name or not frequency_hz or not duration_seconds or not source_id:
        raise ValueError(
            "Missing required fields: session_name, frequency_hz, duration_seconds, source_id"
        )

    async with pool.acquire() as conn:
        # Handle "unknown" source - find or create it
        if source_id == "unknown":
            # Check if "Unknown" source exists
            unknown_id = await conn.fetchval(
                "SELECT id FROM heimdall.known_sources WHERE name = 'Unknown' LIMIT 1"
            )

            # If not, create it with the frequency from this session
            if unknown_id is None:
                unknown_id = await conn.fetchval(
                    """
                    INSERT INTO heimdall.known_sources
                    (name, description, frequency_hz, latitude, longitude, is_validated)
                    VALUES ('Unknown', 'Placeholder for unknown sources', $1, 0.0, 0.0, false)
                    RETURNING id
                    """,
                    frequency_hz,  # Already in Hz
                )

            source_id = str(unknown_id)

        # Create session with selected source
        query = """
            INSERT INTO heimdall.recording_sessions
            (known_source_id, session_name, session_start, status, approval_status, notes)
            VALUES ($1, $2, $3, 'in_progress', 'pending', $4)
            RETURNING
                id, known_source_id, session_name, session_start, session_end,
                duration_seconds, celery_task_id, status, approval_status,
                notes, created_at, updated_at
        """

        row = await conn.fetchrow(
            query,
            UUID(source_id) if isinstance(source_id, str) else source_id,
            session_name,
            datetime.utcnow(),
            notes,
        )

        if not row:
            raise ValueError("Failed to create session")

        session_id = row[0]
        session = RecordingSession(**dict(row))

    # Trigger RF acquisition with 1-second chunking
    # Each second will be acquired as a separate sample
    from ..routers.acquisition import get_websdrs_config
    from ..tasks.acquire_iq import acquire_iq_chunked

    try:
        websdrs_config = get_websdrs_config()

        if not websdrs_config:
            raise ValueError("No active WebSDRs available")

        # Queue chunked acquisition task (splits into 1-second samples)
        task = acquire_iq_chunked.delay(
            frequency_hz=frequency_hz,
            duration_seconds=duration_seconds,
            session_id=str(session_id),
            websdrs_config_list=websdrs_config,
        )

        # Update session with task ID
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE heimdall.recording_sessions
                SET celery_task_id = $1, updated_at = NOW()
                WHERE id = $2
                """,
                task.id,
                UUID(session_id) if isinstance(session_id, str) else session_id,
            )

        logger.info(f"Started chunked acquisition for session {session_id}, task_id={task.id}")

        return {
            "session_id": str(session.id),
            "task_id": task.id,
            "status": "in_progress",
            "chunks": duration_seconds,  # Number of 1-second chunks
        }

    except Exception as e:
        logger.error(f"Failed to start acquisition for session {session_id}: {e}")
        await update_session_to_failed(
            pool, UUID(session_id) if isinstance(session_id, str) else session_id
        )
        raise
