"""
Database connection and utilities
"""
import asyncpg
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# Database configuration from environment
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "heimdall")
DB_USER = os.getenv("POSTGRES_USER", "heimdall_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme")

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create database connection pool"""
    global _pool
    
    if _pool is None:
        logger.info(f"Creating database connection pool to {DB_HOST}:{DB_PORT}/{DB_NAME}")
        _pool = await asyncpg.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database connection pool created successfully")
    
    return _pool


async def close_pool():
    """Close database connection pool"""
    global _pool
    
    if _pool is not None:
        logger.info("Closing database connection pool")
        await _pool.close()
        _pool = None


async def get_connection():
    """Get a database connection from the pool"""
    pool = await get_pool()
    return await pool.acquire()


async def release_connection(conn):
    """Release a database connection back to the pool"""
    pool = await get_pool()
    await pool.release(conn)
