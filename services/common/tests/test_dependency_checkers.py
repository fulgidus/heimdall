"""Tests for dependency checker utilities."""
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio

from services.common.dependency_checkers import (
    check_postgresql,
    check_redis,
    check_rabbitmq,
    check_minio,
    check_celery,
)


class TestPostgreSQLChecker:
    """Test PostgreSQL health checker."""
    
    @pytest.mark.asyncio
    async def test_check_postgresql_success_asyncpg(self):
        """Test successful PostgreSQL check with asyncpg."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.close = AsyncMock()
        
        with patch("services.common.dependency_checkers.asyncpg") as mock_asyncpg:
            mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
            
            await check_postgresql("postgresql://user:pass@localhost/db")
            
            mock_asyncpg.connect.assert_called_once()
            mock_conn.execute.assert_called_once_with("SELECT 1")
            mock_conn.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_postgresql_failure(self):
        """Test PostgreSQL check failure."""
        with patch("services.common.dependency_checkers.asyncpg") as mock_asyncpg:
            mock_asyncpg.connect = AsyncMock(side_effect=ConnectionError("DB down"))
            
            with pytest.raises(Exception, match="PostgreSQL health check failed"):
                await check_postgresql("postgresql://user:pass@localhost/db")


class TestRedisChecker:
    """Test Redis health checker."""
    
    @pytest.mark.asyncio
    async def test_check_redis_success(self):
        """Test successful Redis check."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        mock_client.close = AsyncMock()
        
        with patch("services.common.dependency_checkers.redis") as mock_redis:
            mock_redis.from_url = Mock(return_value=mock_client)
            
            await check_redis("redis://localhost:6379/0")
            
            mock_redis.from_url.assert_called_once()
            mock_client.ping.assert_called_once()
            mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_redis_failure(self):
        """Test Redis check failure."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_client.close = AsyncMock()
        
        with patch("services.common.dependency_checkers.redis") as mock_redis:
            mock_redis.from_url = Mock(return_value=mock_client)
            
            with pytest.raises(Exception, match="Redis health check failed"):
                await check_redis("redis://localhost:6379/0")


class TestRabbitMQChecker:
    """Test RabbitMQ health checker."""
    
    @pytest.mark.asyncio
    async def test_check_rabbitmq_success(self):
        """Test successful RabbitMQ check."""
        mock_channel = AsyncMock()
        mock_channel.close = AsyncMock()
        
        mock_conn = AsyncMock()
        mock_conn.channel = AsyncMock(return_value=mock_channel)
        mock_conn.close = AsyncMock()
        
        with patch("services.common.dependency_checkers.aio_pika") as mock_pika:
            mock_pika.connect_robust = AsyncMock(return_value=mock_conn)
            
            await check_rabbitmq("amqp://guest:guest@localhost:5672//")
            
            mock_pika.connect_robust.assert_called_once()
            mock_conn.channel.assert_called_once()
            mock_channel.close.assert_called_once()
            mock_conn.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_rabbitmq_failure(self):
        """Test RabbitMQ check failure."""
        with patch("services.common.dependency_checkers.aio_pika") as mock_pika:
            mock_pika.connect_robust = AsyncMock(side_effect=ConnectionError("RabbitMQ down"))
            
            with pytest.raises(Exception, match="RabbitMQ health check failed"):
                await check_rabbitmq("amqp://guest:guest@localhost:5672//")


class TestMinIOChecker:
    """Test MinIO health checker."""
    
    @pytest.mark.asyncio
    async def test_check_minio_success(self):
        """Test successful MinIO check."""
        mock_client = Mock()
        mock_client.list_buckets = Mock(return_value=[])
        
        with patch("services.common.dependency_checkers.Minio") as mock_minio:
            mock_minio.return_value = mock_client
            
            # Mock the executor to run synchronously
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=[])
                mock_loop.return_value.run_in_executor = mock_executor
                
                await check_minio("localhost:9000", "access", "secret", False)
                
                mock_minio.assert_called_once()
                mock_executor.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_minio_strip_http(self):
        """Test MinIO check strips http:// prefix."""
        mock_client = Mock()
        mock_client.list_buckets = Mock(return_value=[])
        
        with patch("services.common.dependency_checkers.Minio") as mock_minio:
            mock_minio.return_value = mock_client
            
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock(return_value=[])
                mock_loop.return_value.run_in_executor = mock_executor
                
                await check_minio("http://localhost:9000", "access", "secret", False)
                
                # Verify http:// was stripped
                call_args = mock_minio.call_args
                assert call_args[0][0] == "localhost:9000"
    
    @pytest.mark.asyncio
    async def test_check_minio_failure(self):
        """Test MinIO check failure."""
        with patch("services.common.dependency_checkers.Minio") as mock_minio:
            mock_minio.side_effect = Exception("MinIO down")
            
            with pytest.raises(Exception, match="MinIO health check failed"):
                await check_minio("localhost:9000", "access", "secret", False)


class TestCeleryChecker:
    """Test Celery health checker."""
    
    @pytest.mark.asyncio
    async def test_check_celery_broker_only(self):
        """Test Celery check with broker only."""
        mock_channel = AsyncMock()
        mock_channel.close = AsyncMock()
        
        mock_conn = AsyncMock()
        mock_conn.channel = AsyncMock(return_value=mock_channel)
        mock_conn.close = AsyncMock()
        
        with patch("services.common.dependency_checkers.aio_pika") as mock_pika:
            mock_pika.connect_robust = AsyncMock(return_value=mock_conn)
            
            await check_celery("amqp://guest:guest@localhost:5672//")
            
            mock_pika.connect_robust.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_celery_with_backend(self):
        """Test Celery check with broker and backend."""
        # Mock RabbitMQ
        mock_channel = AsyncMock()
        mock_channel.close = AsyncMock()
        
        mock_conn = AsyncMock()
        mock_conn.channel = AsyncMock(return_value=mock_channel)
        mock_conn.close = AsyncMock()
        
        # Mock Redis
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.close = AsyncMock()
        
        with patch("services.common.dependency_checkers.aio_pika") as mock_pika, \
             patch("services.common.dependency_checkers.redis") as mock_redis:
            
            mock_pika.connect_robust = AsyncMock(return_value=mock_conn)
            mock_redis.from_url = Mock(return_value=mock_redis_client)
            
            await check_celery(
                "amqp://guest:guest@localhost:5672//",
                "redis://localhost:6379/1"
            )
            
            mock_pika.connect_robust.assert_called_once()
            mock_redis.from_url.assert_called_once()
