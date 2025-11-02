"""Database connection pool management using asyncpg."""

import logging
from urllib.parse import urlparse

import asyncpg

from .config import settings

logger = logging.getLogger(__name__)

# Global pool instance
_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    """Initialize asyncpg connection pool."""
    global _pool

    if _pool is not None:
        return _pool

    try:
        # Parse DATABASE_URL
        db_url = urlparse(settings.database_url)

        _pool = await asyncpg.create_pool(
            user=db_url.username,
            password=db_url.password,
            database=db_url.path.lstrip("/"),
            host=db_url.hostname,
            port=db_url.port or 5432,
            min_size=5,
            max_size=20,
            command_timeout=60,
        )
        logger.info("Database connection pool initialized")
        return _pool
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


def get_pool() -> asyncpg.Pool:
    """Get the active connection pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def get_db_session():
    """
    FastAPI dependency to get a database connection from the pool.

    Usage:
        @router.get("/endpoint")
        async def endpoint(db: asyncpg.Connection = Depends(get_db_session)):
            result = await db.fetch("SELECT * FROM table")
            return result
    """
    pool = get_pool()
    async with pool.acquire() as connection:
        yield connection
