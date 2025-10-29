"""
E2E Test Fixtures and Configuration
Provides shared setup for end-to-end integration tests.
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any
from httpx import AsyncClient, Client
from sqlalchemy import create_engine, text
from unittest.mock import AsyncMock, patch
import os


@pytest.fixture(scope="session")
def event_loop():
    """Provide event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def db_url():
    """Database connection URL from environment or default."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://heimdall_user:changeme@localhost:5432/heimdall"
    )


@pytest.fixture(scope="session")
def api_base_url():
    """API base URL."""
    return os.getenv("RF_ACQUISITION_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def db_engine(db_url):
    """Create SQLAlchemy engine for database access."""
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def db_connection(db_engine):
    """Get raw database connection."""
    connection = db_engine.connect()
    yield connection
    connection.close()


@pytest.fixture
async def http_client(api_base_url):
    """Async HTTP client for API calls."""
    async with AsyncClient(base_url=api_base_url, timeout=30.0) as client:
        yield client


@pytest.fixture
async def api_client(api_base_url):
    """Async API client (alias for http_client)."""
    async with AsyncClient(base_url=api_base_url, timeout=30.0) as client:
        yield client


@pytest.fixture
def sync_http_client(api_base_url):
    """Sync HTTP client for API calls."""
    with Client(base_url=api_base_url, timeout=30.0) as client:
        yield client


@pytest.fixture
def db_clean(db_connection):
    """Clean up database after test."""
    # Create measurements table if it doesn't exist
    try:
        db_connection.execute(text("""
            CREATE TABLE IF NOT EXISTS measurements (
                id SERIAL PRIMARY KEY,
                task_id VARCHAR(36),
                websdr_id INT,
                frequency_mhz FLOAT,
                snr FLOAT,
                frequency_offset FLOAT,
                timestamp TIMESTAMP,
                data_url VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))
        db_connection.commit()
        print("✓ Measurements table ready")
    except Exception as e:
        print(f"Table creation warning: {e}")
    
    yield
    
    # Delete test data
    try:
        db_connection.execute(
            text("DELETE FROM measurements WHERE created_at > NOW() - INTERVAL '1 minute'")
        )
        db_connection.commit()
    except Exception as e:
        print(f"Cleanup warning: {e}")


class APIHelper:
    """Helper class for common API operations."""
    
    def __init__(self, client: AsyncClient):
        self.client = client
    
    async def trigger_acquisition(
        self,
        frequency_mhz: float = 145.50,
        duration_seconds: float = 2.0,
        description: str = "E2E test acquisition"
    ) -> str:
        """Trigger an acquisition and return task_id."""
        response = await self.client.post(
            "/api/v1/acquisition/acquire",
            json={
                "frequency_mhz": frequency_mhz,
                "duration_seconds": duration_seconds,
                "description": description
            }
        )
        assert response.status_code == 200, f"Acquisition failed: {response.text}"
        return response.json()["task_id"]
    
    async def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get acquisition status."""
        response = await self.client.get(f"/api/v1/acquisition/status/{task_id}")
        assert response.status_code == 200, f"Status check failed: {response.text}"
        return response.json()
    
    async def wait_for_completion(
        self,
        task_id: str,
        timeout_seconds: int = 60,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Poll until acquisition completes (success or failure)."""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Acquisition {task_id} did not complete in {timeout_seconds}s")
            
            status = await self.get_status(task_id)
            state = status.get("status")  # API returns 'status', not 'state'
            
            if state in ["SUCCESS", "FAILED", "CANCELLED"]:
                return status
            
            await asyncio.sleep(poll_interval)
    
    async def get_measurements(self, task_id: str) -> list:
        """Get measurements from completed acquisition."""
        response = await self.client.get(f"/api/v1/acquisition/measurements/{task_id}")
        assert response.status_code == 200, f"Get measurements failed: {response.text}"
        return response.json()["measurements"]


@pytest.fixture
async def api_helper(http_client):
    """Provide API helper."""
    return APIHelper(http_client)


class DatabaseHelper:
    """Helper class for database operations."""
    
    def __init__(self, connection):
        self.connection = connection
    
    def get_measurements_by_task_id(self, task_id: str) -> list:
        """Query measurements for a specific task."""
        result = self.connection.execute(
            text(f"""
            SELECT id, task_id, websdr_id, frequency_mhz, snr, 
                   frequency_offset, timestamp, data_url
            FROM measurements
            WHERE task_id = '{task_id}'
            ORDER BY timestamp
            """)
        )
        return [dict(row) for row in result]
    
    def get_measurement_count(self, task_id: str) -> int:
        """Count measurements for a task."""
        result = self.connection.execute(
            text(f"SELECT COUNT(*) as count FROM measurements WHERE task_id = '{task_id}'")
        )
        return result.scalar()
    
    def verify_measurement_data(self, measurement: Dict) -> bool:
        """Verify measurement has valid data."""
        required_fields = ["id", "websdr_id", "frequency_mhz", "snr", "data_url"]
        return all(field in measurement and measurement[field] is not None 
                  for field in required_fields)


@pytest.fixture
def db_helper(db_connection):
    """Provide database helper."""
    return DatabaseHelper(db_connection)


class MinIOHelper:
    """Helper class for MinIO operations."""
    
    def __init__(self):
        from minio import Minio
        self.client = Minio(
            "localhost:9000",
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
    
    def list_session_files(self, task_id: str, bucket: str = "heimdall-raw-iq") -> list:
        """List all files for a session."""
        prefix = f"sessions/{task_id}/"
        try:
            objects = self.client.list_objects(bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except Exception as e:
            print(f"MinIO list error: {e}")
            return []
    
    def verify_session_files(self, task_id: str, expected_count: int = 7) -> bool:
        """Verify that session has expected number of files."""
        files = self.list_session_files(task_id)
        return len(files) >= expected_count


@pytest.fixture
def minio_helper():
    """Provide MinIO helper."""
    return MinIOHelper()


# Test markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "e2e: end-to-end tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow tests (> 5 seconds)")
