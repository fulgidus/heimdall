"""
Integration Tests for Service Communication

Tests communication between microservices with actual infrastructure.
"""

import pytest

from services.tests.test_integration_base import IntegrationTestBase


class TestServiceIntegration(IntegrationTestBase):
    """Test communication between microservices."""

    @pytest.mark.integration
    def test_redis_connectivity(self, redis_client):
        """Test Redis connection and basic operations."""
        # Set a test key
        redis_client.set("test_key", "test_value")

        # Get the key back
        value = redis_client.get("test_key")
        assert value == "test_value"

        # Delete the key
        redis_client.delete("test_key")
        assert redis_client.get("test_key") is None

        print("✓ Redis connectivity validated")

    @pytest.mark.integration
    def test_rabbitmq_connectivity(self, rabbitmq_channel):
        """Test RabbitMQ connection and message passing."""
        queue_name = "test_queue"

        # Declare a test queue
        rabbitmq_channel.queue_declare(queue=queue_name, durable=False)

        # Publish a message
        test_message = "test message"
        rabbitmq_channel.basic_publish(exchange="", routing_key=queue_name, body=test_message)

        # Consume the message
        method, properties, body = rabbitmq_channel.basic_get(queue=queue_name, auto_ack=True)

        assert body.decode() == test_message

        # Cleanup
        rabbitmq_channel.queue_delete(queue=queue_name)

        print("✓ RabbitMQ connectivity validated")

    @pytest.mark.integration
    def test_minio_storage_operations(self, minio_client):
        """Test MinIO storage for IQ data."""
        import io

        bucket_name = "test-bucket"
        object_name = "test/data.bin"
        test_data = b"test binary data"

        # Ensure bucket exists
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Upload data
        data_stream = io.BytesIO(test_data)
        minio_client.put_object(bucket_name, object_name, data_stream, length=len(test_data))

        # Verify upload
        stat = minio_client.stat_object(bucket_name, object_name)
        assert stat.size == len(test_data)

        # Download and verify
        response = minio_client.get_object(bucket_name, object_name)
        downloaded_data = response.read()
        assert downloaded_data == test_data

        # Cleanup
        minio_client.remove_object(bucket_name, object_name)

        print("✓ MinIO storage operations validated")

    @pytest.mark.integration
    def test_redis_caching_pattern(self, redis_client):
        """Test Redis caching pattern for inference results."""
        import json

        # Simulate caching an inference result
        cache_key = "inference:task:12345"
        result_data = {
            "latitude": 45.5,
            "longitude": 10.2,
            "confidence": 0.85,
            "timestamp": "2025-10-25T12:00:00Z",
        }

        # Cache with TTL
        redis_client.setex(cache_key, 3600, json.dumps(result_data))  # 1 hour TTL

        # Retrieve from cache
        cached_data = redis_client.get(cache_key)
        assert cached_data is not None

        retrieved_data = json.loads(cached_data)
        assert retrieved_data == result_data

        # Check TTL
        ttl = redis_client.ttl(cache_key)
        assert 0 < ttl <= 3600

        print("✓ Redis caching pattern validated")

    @pytest.mark.integration
    def test_task_queue_pattern(self, rabbitmq_channel):
        """Test task queue pattern for Celery-like workloads."""
        import json

        queue_name = "heimdall.tasks"

        # Declare task queue
        rabbitmq_channel.queue_declare(
            queue=queue_name, durable=True, arguments={"x-max-priority": 10}
        )

        # Publish a task
        task_data = {
            "task": "acquire_iq",
            "args": [145.50, 10.0],
            "kwargs": {"description": "Test acquisition"},
            "priority": 5,
        }

        rabbitmq_channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=json.dumps(task_data),
            properties={"priority": 5},
        )

        # Consume the task
        method, properties, body = rabbitmq_channel.basic_get(queue=queue_name, auto_ack=True)

        received_task = json.loads(body.decode())
        assert received_task == task_data

        # Cleanup
        rabbitmq_channel.queue_delete(queue=queue_name)

        print("✓ Task queue pattern validated")

    @pytest.mark.integration
    def test_minio_bucket_management(self, minio_client):
        """Test MinIO bucket lifecycle management."""
        test_bucket = "test-lifecycle-bucket"

        # Create bucket if it doesn't exist
        if not minio_client.bucket_exists(test_bucket):
            minio_client.make_bucket(test_bucket)

        # Verify bucket exists
        assert minio_client.bucket_exists(test_bucket)

        # List all buckets
        buckets = minio_client.list_buckets()
        bucket_names = [b.name for b in buckets]
        assert test_bucket in bucket_names

        # Cleanup
        minio_client.remove_bucket(test_bucket)

        print("✓ MinIO bucket management validated")

    @pytest.mark.integration
    def test_redis_pub_sub_pattern(self, redis_client):
        """Test Redis pub/sub for real-time updates."""
        channel_name = "heimdall:updates"
        test_message = "Acquisition completed"

        # Create pubsub
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel_name)

        # Skip the subscription message
        pubsub.get_message()

        # Publish a message
        redis_client.publish(channel_name, test_message)

        # Receive the message
        message = pubsub.get_message(timeout=1.0)
        assert message is not None
        assert message["type"] == "message"
        assert message["data"] == test_message

        # Cleanup
        pubsub.unsubscribe(channel_name)
        pubsub.close()

        print("✓ Redis pub/sub pattern validated")


class TestDataPersistence(IntegrationTestBase):
    """Test data persistence across services."""

    @pytest.mark.integration
    def test_measurement_storage_retrieval(self, minio_client):
        """Test storing and retrieving measurements."""
        import io

        import numpy as np

        bucket_name = "heimdall-raw-iq"

        # Ensure bucket exists
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Create test IQ data
        iq_data = np.random.randn(1000, 2).astype(np.float32)

        # Store in MinIO
        object_name = "test/measurement-123.bin"
        data_bytes = io.BytesIO(iq_data.tobytes())
        minio_client.put_object(
            bucket_name, object_name, data_bytes, length=len(data_bytes.getvalue())
        )

        # Retrieve and verify
        response = minio_client.get_object(bucket_name, object_name)
        retrieved_bytes = response.read()
        retrieved_data = np.frombuffer(retrieved_bytes, dtype=np.float32).reshape(-1, 2)

        np.testing.assert_array_equal(iq_data, retrieved_data)

        # Cleanup
        minio_client.remove_object(bucket_name, object_name)

        print("✓ Measurement storage/retrieval validated")
