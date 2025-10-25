"""
Shared Test Fixtures for Heimdall Services

This module provides reusable pytest fixtures for testing across all microservices.
Includes database, Redis, RabbitMQ, and MinIO fixtures.
"""

import pytest
import os
import asyncio
from typing import Generator

# Test database configuration
TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_engine():
    """Create in-memory test database."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        
        engine = create_engine(
            TEST_DB_URL,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        yield engine
        engine.dispose()
    except ImportError:
        pytest.skip("SQLAlchemy not installed")


@pytest.fixture
def test_db_session(test_db_engine):
    """Provide isolated database session per test."""
    try:
        from sqlalchemy.orm import sessionmaker, Session
        
        connection = test_db_engine.connect()
        transaction = connection.begin()
        session = sessionmaker(bind=connection)()
        
        yield session
        
        session.close()
        transaction.rollback()
        connection.close()
    except ImportError:
        pytest.skip("SQLAlchemy not installed")


@pytest.fixture
def mock_redis():
    """Mock Redis for testing."""
    try:
        import redis
        with redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=15,  # Use separate DB for tests
            decode_responses=True
        ) as r:
            r.flushdb()  # Clean test database
            yield r
            r.flushdb()
    except (ImportError, Exception):
        pytest.skip("Redis not available for testing")


@pytest.fixture
def mock_rabbitmq():
    """Mock RabbitMQ connection."""
    try:
        import pika
        credentials = pika.PlainCredentials(
            os.getenv('RABBITMQ_USER', 'guest'),
            os.getenv('RABBITMQ_PASS', 'guest')
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=os.getenv('RABBITMQ_HOST', 'localhost'),
                port=int(os.getenv('RABBITMQ_PORT', 5672)),
                credentials=credentials,
                virtual_host='/'
            )
        )
        channel = connection.channel()
        yield channel
        connection.close()
    except (ImportError, Exception):
        pytest.skip("RabbitMQ not available for testing")


@pytest.fixture
def mock_s3_client():
    """Mock MinIO S3 client."""
    try:
        from minio import Minio
        client = Minio(
            os.getenv('S3_ENDPOINT', 'localhost:9000'),
            access_key=os.getenv('S3_ACCESS_KEY', 'minioadmin'),
            secret_key=os.getenv('S3_SECRET_KEY', 'minioadmin'),
            secure=False
        )
        yield client
    except (ImportError, Exception):
        pytest.skip("MinIO not available for testing")


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        'DATABASE_URL': TEST_DB_URL,
        'REDIS_URL': f'redis://{os.getenv("REDIS_HOST", "localhost")}:{os.getenv("REDIS_PORT", 6379)}/15',
        'RABBITMQ_URL': f'amqp://{os.getenv("RABBITMQ_USER", "guest")}:{os.getenv("RABBITMQ_PASS", "guest")}@{os.getenv("RABBITMQ_HOST", "localhost")}:{os.getenv("RABBITMQ_PORT", 5672)}/%2F',
        'S3_ENDPOINT': f'http://{os.getenv("S3_ENDPOINT", "localhost:9000")}',
        'S3_ACCESS_KEY': os.getenv('S3_ACCESS_KEY', 'minioadmin'),
        'S3_SECRET_KEY': os.getenv('S3_SECRET_KEY', 'minioadmin'),
        'LOG_LEVEL': 'DEBUG',
    }


@pytest.fixture
def test_client_factory(test_config):
    """Factory for creating test FastAPI clients."""
    try:
        from fastapi.testclient import TestClient
        
        def create_client(app):
            return TestClient(app)
        
        return create_client
    except ImportError:
        pytest.skip("FastAPI not installed")

