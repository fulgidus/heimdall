""""""

Recording sessions and known sources management API endpoints.Recording sessions and known sources management API endpoints.



Sessions management functionality for RF acquisition service.

""""""



import loggingimport logging

from typing import List, Optionalfrom typing import List, Optional

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from fastapi import APIRouter, HTTPException, Query

from sqlalchemy import select, func, textlogger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

from ..storage.db_manager import DatabaseManager



logger = logging.getLogger(__name__)@router.get("/known-sources")

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])async def list_known_sources(

    skip: int = Query(0, ge=0),

    limit: int = Query(100, ge=1, le=1000),

@router.get("/known-sources")    validated_only: bool = Query(False),

async def list_known_sources() -> List[dict]:

    skip: int = Query(0, ge=0),    """

    limit: int = Query(100, ge=1, le=1000),    List known RF sources with optional filtering.

    validated_only: bool = Query(False),    """

) -> List[dict]:    # Placeholder - return empty list for now

    """    return []

    List known RF sources with optional filtering.

    """

    try:@router.get("")

        db_manager = DatabaseManager()async def list_sessions(

        async with db_manager.get_session() as session:    page: int = Query(1, ge=1),

            query = "SELECT id, name, frequency_hz, latitude, longitude, is_validated FROM known_sources"    per_page: int = Query(20, ge=1, le=100),

            if validated_only:    status_filter: Optional[str] = Query(None),

                query += " WHERE is_validated = TRUE") -> dict:

            query += f" LIMIT {limit} OFFSET {skip}"    """

                List recording sessions with pagination.

            result = await session.execute(text(query))    """

            sources = result.fetchall()    # Placeholder - return empty response for now

                return {

            return [        "total": 0,

                {        "page": page,

                    "id": str(row[0]),        "per_page": per_page,

                    "name": row[1],        "items": []

                    "frequency_hz": row[2],    }

                    "latitude": row[3],

                    "longitude": row[4],

                    "is_validated": row[5],@router.get("/analytics")

                }async def get_session_analytics() -> dict:

                for row in sources    """

            ]    Get session analytics and statistics.

    except Exception as e:    """

        logger.error(f"Error fetching known sources: {e}")    # Placeholder - return empty analytics for now

        return []    return {

        "total_sessions": 0,

        "completed_sessions": 0,

@router.get("")        "failed_sessions": 0,

async def list_sessions(        "pending_sessions": 0,

    page: int = Query(1, ge=1),        "success_rate": 0.0,

    per_page: int = Query(20, ge=1, le=100),    }

    status_filter: Optional[str] = Query(None),
) -> dict:
    """
    List recording sessions with pagination.
    """
    try:
        db_manager = DatabaseManager()
        async with db_manager.get_session() as session:
            # Count total
            count_query = "SELECT COUNT(*) FROM recording_sessions"
            if status_filter:
                count_query += f" WHERE status = '{status_filter}'"
            
            count_result = await session.execute(text(count_query))
            total = count_result.scalar() or 0
            
            # Fetch sessions
            offset = (page - 1) * per_page
            query = f"SELECT id, status, created_at FROM recording_sessions"
            if status_filter:
                query += f" WHERE status = '{status_filter}'"
            query += f" ORDER BY created_at DESC LIMIT {per_page} OFFSET {offset}"
            
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            items = [
                {
                    "id": str(row[0]),
                    "status": row[1],
                    "created_at": row[2].isoformat() if row[2] else None,
                }
                for row in rows
            ]
            
            return {
                "total": total,
                "page": page,
                "per_page": per_page,
                "items": items,
            }
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return {
            "total": 0,
            "page": page,
            "per_page": per_page,
            "items": [],
        }


@router.get("/analytics")
async def get_session_analytics() -> dict:
    """
    Get session analytics and statistics.
    """
    try:
        db_manager = DatabaseManager()
        async with db_manager.get_session() as session:
            # Get totals
            total_result = await session.execute(text("SELECT COUNT(*) FROM recording_sessions"))
            total_sessions = total_result.scalar() or 0
            
            completed_result = await session.execute(text("SELECT COUNT(*) FROM recording_sessions WHERE status = 'completed'"))
            completed_sessions = completed_result.scalar() or 0
            
            failed_result = await session.execute(text("SELECT COUNT(*) FROM recording_sessions WHERE status = 'failed'"))
            failed_sessions = failed_result.scalar() or 0
            
            pending_result = await session.execute(text("SELECT COUNT(*) FROM recording_sessions WHERE status = 'pending'"))
            pending_sessions = pending_result.scalar() or 0
            
            success_rate = (
                (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0.0
            )
            
            return {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "failed_sessions": failed_sessions,
                "pending_sessions": pending_sessions,
                "success_rate": success_rate,
            }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return {
            "total_sessions": 0,
            "completed_sessions": 0,
            "failed_sessions": 0,
            "pending_sessions": 0,
            "success_rate": 0.0,
        }
