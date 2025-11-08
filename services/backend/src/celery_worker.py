"""
Celery worker initialization signals.

This module handles worker process initialization, including database pool setup.
"""

import asyncio
import logging

from celery.signals import worker_process_init, worker_process_shutdown

from .db import init_pool, close_pool

logger = logging.getLogger(__name__)

# Global event loop for this worker process
_worker_loop = None


def get_worker_loop():
    """Get the worker's event loop (created during init)."""
    global _worker_loop
    if _worker_loop is None:
        raise RuntimeError("Worker event loop not initialized")
    return _worker_loop


@worker_process_init.connect
def init_worker_pool(**kwargs):
    """
    Initialize database pool when worker process starts.
    
    This signal is fired when a new worker process is spawned (prefork pool model).
    Each worker process needs its own database connection pool.
    """
    global _worker_loop
    logger.info("Worker process starting - initializing database pool")
    
    try:
        # Create a new event loop for this worker process
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
        
        # Initialize database pool
        _worker_loop.run_until_complete(init_pool())
        
        logger.info("Worker process database pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize worker database pool: {e}", exc_info=True)
        raise


@worker_process_shutdown.connect
def shutdown_worker_pool(**kwargs):
    """
    Close database pool when worker process shuts down.
    
    This signal is fired when a worker process is shutting down.
    Ensures clean shutdown of database connections.
    """
    logger.info("Worker process shutting down - closing database pool")
    
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(close_pool())
        logger.info("Worker process database pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing worker database pool: {e}", exc_info=True)
