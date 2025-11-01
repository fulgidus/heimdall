"""Health checkers for external dependencies."""

import asyncio

import structlog

logger = structlog.get_logger()


async def check_postgresql(connection_string: str) -> None:
    """
    Check PostgreSQL database connectivity.

    Args:
        connection_string: PostgreSQL connection string

    Raises:
        Exception: If database is not reachable or connection fails
    """
    try:
        import asyncpg

        # Parse connection string to extract components
        # Format: postgresql://user:pass@host:port/db
        conn = await asyncpg.connect(connection_string)
        try:
            # Simple query to verify connectivity
            await conn.execute("SELECT 1")
        finally:
            await conn.close()
    except ImportError:
        # Fallback to psycopg2 if asyncpg not available
        import psycopg2

        conn = psycopg2.connect(connection_string)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        finally:
            conn.close()
    except Exception as exc:
        logger.error("postgresql_check_failed", error=str(exc))
        raise Exception(f"PostgreSQL health check failed: {str(exc)}")


async def check_redis(redis_url: str) -> None:
    """
    Check Redis connectivity.

    Args:
        redis_url: Redis connection URL (redis://[:password]@host:port/db)

    Raises:
        Exception: If Redis is not reachable or connection fails
    """
    try:
        import redis.asyncio as redis

        client = redis.from_url(redis_url, decode_responses=True)
        try:
            await client.ping()
        finally:
            await client.close()
    except ImportError:
        # Fallback to sync redis
        import redis as redis_sync

        client = redis_sync.from_url(redis_url, decode_responses=True)
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:
        logger.error("redis_check_failed", error=str(exc))
        raise Exception(f"Redis health check failed: {str(exc)}")


async def check_rabbitmq(broker_url: str) -> None:
    """
    Check RabbitMQ connectivity.

    Args:
        broker_url: RabbitMQ broker URL (amqp://user:pass@host:port/vhost)

    Raises:
        Exception: If RabbitMQ is not reachable or connection fails
    """
    try:
        import aio_pika

        connection = await aio_pika.connect_robust(broker_url)
        try:
            # Create a channel to verify connection works
            channel = await connection.channel()
            await channel.close()
        finally:
            await connection.close()
    except ImportError:
        # Fallback to pika (synchronous)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_check_rabbitmq, broker_url)
    except Exception as exc:
        logger.error("rabbitmq_check_failed", error=str(exc))
        raise Exception(f"RabbitMQ health check failed: {str(exc)}")


def _sync_check_rabbitmq(broker_url: str) -> None:
    """Synchronous RabbitMQ check for fallback."""
    import pika

    connection = pika.BlockingConnection(pika.URLParameters(broker_url))
    try:
        channel = connection.channel()
        channel.close()
    finally:
        connection.close()


async def check_minio(
    endpoint: str, access_key: str, secret_key: str, secure: bool = False
) -> None:
    """
    Check MinIO/S3 connectivity.

    Args:
        endpoint: MinIO endpoint (host:port)
        access_key: MinIO access key
        secret_key: MinIO secret key
        secure: Whether to use HTTPS

    Raises:
        Exception: If MinIO is not reachable or connection fails
    """
    try:
        from minio import Minio

        # Remove http:// or https:// prefix if present
        if endpoint.startswith("http://"):
            endpoint = endpoint.replace("http://", "")
            secure = False
        elif endpoint.startswith("https://"):
            endpoint = endpoint.replace("https://", "")
            secure = True

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, client.list_buckets)
    except Exception as exc:
        logger.error("minio_check_failed", error=str(exc))
        raise Exception(f"MinIO health check failed: {str(exc)}")


async def check_celery(broker_url: str, backend_url: str | None = None) -> None:
    """
    Check Celery broker and backend connectivity.

    Args:
        broker_url: Celery broker URL (RabbitMQ)
        backend_url: Optional Celery result backend URL (Redis)

    Raises:
        Exception: If Celery broker/backend is not reachable
    """
    # Check broker (RabbitMQ)
    await check_rabbitmq(broker_url)

    # Check backend if provided (Redis)
    if backend_url:
        await check_redis(backend_url)
