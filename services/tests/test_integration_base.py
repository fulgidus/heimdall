"""
Integration Test Base Class

This module provides a base class for integration tests that require
Docker infrastructure (PostgreSQL, Redis, RabbitMQ, MinIO).
"""

import pytest
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import docker
import time


class IntegrationTestBase:
    """Base class for integration tests requiring infrastructure."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_infrastructure(self):
        """Setup infrastructure for integration tests."""
        # Check if Docker containers are running
        try:
            client = docker.from_env()
        except docker.errors.DockerException:
            pytest.skip("Docker not available for integration tests")
            return
        
        required_services = {
            'postgres': 'heimdall-postgres',
            'redis': 'heimdall-redis',
            'rabbitmq': 'heimdall-rabbitmq',
            'minio': 'heimdall-minio',
        }
        
        for service_name, container_name in required_services.items():
            try:
                container = client.containers.get(container_name)
                if container.status != 'running':
                    print(f"Starting {service_name}...")
                    container.start()
                    time.sleep(2)
            except docker.errors.NotFound:
                pytest.skip(
                    f"{service_name} container not found. Run docker-compose up -d"
                )
        
        yield
        
        # Cleanup after tests
        # Don't stop containers, just cleanup data
    
    @pytest.fixture
    def db_session(self):
        """Provide database session for integration tests."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        database_url = os.getenv(
            'TEST_DATABASE_URL',
            'postgresql://heimdall_user:changeme@localhost:5432/heimdall_test'
        )
        
        try:
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            session = Session()
            
            yield session
            
            session.close()
        except Exception as e:
            pytest.skip(f"Database not available: {e}")
    
    @pytest.fixture
    def redis_client(self):
        """Provide Redis client for integration tests."""
        import redis
        
        try:
            client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=1,  # Use separate DB for tests
                decode_responses=True
            )
            client.ping()  # Test connection
            client.flushdb()
            yield client
            client.flushdb()
        except (redis.ConnectionError, redis.TimeoutError):
            pytest.skip("Redis not available for integration tests")
    
    @pytest.fixture
    def rabbitmq_channel(self):
        """Provide RabbitMQ channel for integration tests."""
        import pika
        
        try:
            credentials = pika.PlainCredentials(
                os.getenv('RABBITMQ_USER', 'guest'),
                os.getenv('RABBITMQ_PASS', 'guest')
            )
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=os.getenv('RABBITMQ_HOST', 'localhost'),
                    port=int(os.getenv('RABBITMQ_PORT', 5672)),
                    credentials=credentials
                )
            )
            channel = connection.channel()
            yield channel
            connection.close()
        except pika.exceptions.AMQPConnectionError:
            pytest.skip("RabbitMQ not available for integration tests")
    
    @pytest.fixture
    def minio_client(self):
        """Provide MinIO client for integration tests."""
        from minio import Minio
        
        try:
            client = Minio(
                os.getenv('S3_ENDPOINT', 'localhost:9000'),
                access_key=os.getenv('S3_ACCESS_KEY', 'minioadmin'),
                secret_key=os.getenv('S3_SECRET_KEY', 'minioadmin'),
                secure=False
            )
            # Test connection
            client.list_buckets()
            yield client
        except Exception:
            pytest.skip("MinIO not available for integration tests")
