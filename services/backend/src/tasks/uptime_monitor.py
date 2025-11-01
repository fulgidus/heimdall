"""Monitor WebSDR uptime and record to database."""

import asyncio
import logging
from datetime import datetime

from celery import shared_task

try:
    from ..config import settings
    from ..fetchers.websdr_fetcher import WebSDRFetcher
    from ..models.db import Base
    from ..storage.db_manager import DatabaseManager
except ImportError:
    # Fallback for direct execution
    from storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# Define websdrs_uptime_history table model (raw SQL approach)
UPTIME_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS heimdall.websdrs_uptime_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    websdr_id INT NOT NULL,
    websdr_name VARCHAR(255),
    status VARCHAR(20) NOT NULL CHECK (status IN ('online', 'offline')),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

SELECT create_hypertable(
    'heimdall.websdrs_uptime_history',
    'timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_websdrs_uptime_history_websdr_time
ON heimdall.websdrs_uptime_history(websdr_id, timestamp DESC);
"""


def get_default_websdrs() -> list[dict]:
    """Get default WebSDR configuration."""
    return [
        {
            "id": 1,
            "name": "Aquila di Giaveno",
            "url": "http://sdr1.ik1jns.it:8076/",
            "location_name": "Giaveno, Italy",
            "latitude": 45.02,
            "longitude": 7.29,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
        {
            "id": 2,
            "name": "Montanaro",
            "url": "http://cbfenis.ddns.net:43510/",
            "location_name": "Montanaro, Italy",
            "latitude": 45.234,
            "longitude": 7.857,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
        {
            "id": 3,
            "name": "Torino",
            "url": "http://vst-aero.it:8073/",
            "location_name": "Torino, Italy",
            "latitude": 45.044,
            "longitude": 7.672,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
        {
            "id": 4,
            "name": "Coazze",
            "url": "http://94.247.189.130:8076/",
            "location_name": "Coazze, Italy",
            "latitude": 45.03,
            "longitude": 7.27,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
        {
            "id": 5,
            "name": "Passo del Giovi",
            "url": "http://iz1mlt.ddns.net:8074/",
            "location_name": "Passo del Giovi, Italy",
            "latitude": 44.561,
            "longitude": 8.956,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
        {
            "id": 6,
            "name": "Genova",
            "url": "http://iq1zw.ddns.net:42154/",
            "location_name": "Genova, Italy",
            "latitude": 44.395,
            "longitude": 8.956,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
        {
            "id": 7,
            "name": "Milano - Baggio",
            "url": "http://iu2mch.duckdns.org:8073/",
            "location_name": "Milano (Baggio), Italy",
            "latitude": 45.478,
            "longitude": 9.123,
            "is_active": True,
            "timeout_seconds": 3,
            "retry_count": 1,
        },
    ]


@shared_task(bind=True, name="monitor_websdrs_uptime")
def monitor_websdrs_uptime(self):
    """
    Celery task: Check WebSDR health and record status to database.
    Runs periodically (every minute) to build uptime history.
    """
    try:
        logger.info("Starting WebSDR uptime monitoring task")

        # Get WebSDR configs as dict
        websdrs = get_default_websdrs()

        # Direct health check without WebSDRFetcher (to avoid object attribute issues)
        logger.info(f"Checking health of {len(websdrs)} WebSDRs")
        import aiohttp

        async def direct_health_check():
            """Direct health check without using WebSDRFetcher."""
            results = {}
            timeout = aiohttp.ClientTimeout(total=3)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                for ws in websdrs:
                    ws_id = ws["id"]
                    ws_url = ws["url"]
                    ws_name = ws["name"]

                    try:
                        async with session.head(
                            ws_url, timeout=timeout, allow_redirects=False
                        ) as resp:
                            # 501 means HEAD not supported, try GET
                            if resp.status == 501:
                                async with session.get(
                                    ws_url, timeout=timeout, allow_redirects=False
                                ) as resp2:
                                    is_online = 200 <= resp2.status < 400
                                    logger.debug(
                                        f"  {ws_name} ({ws_id}): GET {resp2.status} → {'online' if is_online else 'offline'}"
                                    )
                                    results[ws_id] = is_online
                            else:
                                is_online = 200 <= resp.status < 400
                                logger.debug(
                                    f"  {ws_name} ({ws_id}): HEAD {resp.status} → {'online' if is_online else 'offline'}"
                                )
                                results[ws_id] = is_online
                    except Exception as e:
                        logger.warning(
                            f"  {ws_name} ({ws_id}): Exception {type(e).__name__} → offline"
                        )
                        results[ws_id] = False

            return results

        # Run async health check
        loop = asyncio.get_event_loop()
        health_results = loop.run_until_complete(direct_health_check())

        logger.info(f"Health check results: {health_results}")

        # Get database connection
        db_manager = DatabaseManager()

        # Record status for each WebSDR
        records_inserted = 0
        with db_manager.get_session() as session:
            for ws_config in websdrs:
                ws_id = ws_config["id"]
                ws_name = ws_config["name"]

                # Get online/offline status
                is_online = health_results.get(ws_id, False)
                status = "online" if is_online else "offline"
                timestamp = datetime.utcnow()

                # Insert into websdrs_uptime_history
                try:
                    from sqlalchemy import text

                    # Raw SQL insert for uptime history using text()
                    query = text(
                        """
                        INSERT INTO heimdall.websdrs_uptime_history
                        (websdr_id, websdr_name, status, timestamp)
                        VALUES (:websdr_id, :websdr_name, :status, :timestamp)
                    """
                    )
                    session.execute(
                        query,
                        {
                            "websdr_id": ws_id,
                            "websdr_name": ws_name,
                            "status": status,
                            "timestamp": timestamp,
                        },
                    )
                    records_inserted += 1
                    logger.debug(f"Recorded {ws_name} ({ws_id}): {status}")
                except Exception as e:
                    logger.error(f"Failed to record uptime for {ws_name}: {e}")

            session.commit()

        logger.info(f"Uptime monitoring complete: {records_inserted} records inserted")
        
        # Broadcast health status via WebSocket (if available)
        try:
            # Import here to avoid circular dependency
            from ..routers.websocket import manager as ws_manager
            
            if ws_manager.active_connections:
                # Build health status message
                health_data = {}
                for ws_config in websdrs:
                    ws_id = ws_config["id"]
                    ws_name = ws_config["name"]
                    is_online = health_results.get(ws_id, False)
                    
                    # Use UUID as key if available (for frontend compatibility)
                    # For now, use string ID as key
                    health_data[str(ws_id)] = {
                        "websdr_id": str(ws_id),
                        "name": ws_name,
                        "status": "online" if is_online else "offline",
                        "last_check": datetime.utcnow().isoformat(),
                    }
                
                # Broadcast asynchronously
                async def broadcast_health():
                    await ws_manager.broadcast({
                        "event": "websdrs:health",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"health_status": health_data},
                    })
                
                loop = asyncio.get_event_loop()
                loop.run_until_complete(broadcast_health())
                logger.debug(f"Broadcasted health update to {len(ws_manager.active_connections)} WebSocket clients")
        except Exception as ws_error:
            logger.warning(f"Failed to broadcast health update via WebSocket: {ws_error}")

        return {
            "status": "success",
            "records_inserted": records_inserted,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error in uptime monitoring: {e}")
        raise


def calculate_uptime_percentage(websdr_id: int, hours: int = 24) -> float:
    """
    Calculate uptime percentage for a WebSDR over the last N hours.

    Args:
        websdr_id: ID of the WebSDR
        hours: Time window in hours (default: 24)

    Returns:
        Uptime percentage (0-100)
    """
    try:
        from datetime import timedelta

        from sqlalchemy import text

        db_manager = DatabaseManager()

        with db_manager.get_session() as session:
            # Query uptime history for the time window
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Raw SQL query using SQLAlchemy text()
            query = text(
                """
                SELECT status, COUNT(*) as count
                FROM heimdall.websdrs_uptime_history
                WHERE websdr_id = :websdr_id
                  AND timestamp >= :cutoff_time
                GROUP BY status
            """
            )

            results = session.execute(query, {"websdr_id": websdr_id, "cutoff_time": cutoff_time})

            status_counts = {row[0]: row[1] for row in results}

            total_checks = sum(status_counts.values())
            if total_checks == 0:
                return 0.0

            online_checks = status_counts.get("online", 0)
            uptime_pct = (online_checks / total_checks) * 100

            logger.debug(
                f"SDR {websdr_id}: {online_checks}/{total_checks} online over {hours}h = {uptime_pct:.1f}%"
            )

            return uptime_pct

    except Exception as e:
        logger.error(f"Error calculating uptime for SDR {websdr_id}: {e}")
        return 0.0
