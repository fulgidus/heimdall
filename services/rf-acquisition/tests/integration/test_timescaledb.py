"""Integration tests for TimescaleDB operations."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List


def test_measurement_creation_from_dict():
    """Test creating measurement from dictionary"""
    from src.models.db import Measurement
    
    sample_dict = {
        "websdr_id": 1,
        "frequency_mhz": 144.5,
        "sample_rate_khz": 12.5,
        "samples_count": 125000,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "metrics": {
            "snr_db": 15.5,
            "frequency_offset_hz": 150.0,
            "power_dbm": -75.5,
        }
    }
    
    measurement = Measurement.from_measurement_dict(
        task_id="test-session-001",
        measurement_dict=sample_dict,
        s3_path="s3://bucket/test.npy"
    )
    
    assert measurement.task_id == "test-session-001"
    assert measurement.websdr_id == 1
    assert measurement.frequency_mhz == 144.5
    assert measurement.snr_db == 15.5
    assert measurement.s3_path == "s3://bucket/test.npy"


def test_measurement_to_dict():
    """Test converting measurement to dictionary"""
    from src.models.db import Measurement
    
    sample_dict = {
        "websdr_id": 1,
        "frequency_mhz": 144.5,
        "sample_rate_khz": 12.5,
        "samples_count": 125000,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "metrics": {"snr_db": 15.5}
    }
    
    measurement = Measurement.from_measurement_dict(
        task_id="test-session-001",
        measurement_dict=sample_dict
    )
    
    result_dict = measurement.to_dict()
    
    assert result_dict["task_id"] == "test-session-001"
    assert result_dict["websdr_id"] == 1
    assert result_dict["snr_db"] == 15.5


def test_measurement_missing_required_field():
    """Test error handling for missing required fields"""
    from src.models.db import Measurement
    
    invalid_dict = {
        "websdr_id": 1,
        # Missing frequency_mhz
    }
    
    with pytest.raises(ValueError):
        Measurement.from_measurement_dict(
            task_id="test",
            measurement_dict=invalid_dict
        )


def test_db_manager_init():
    """Test database manager initialization"""
    from src.storage.db_manager import DatabaseManager
    
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    assert manager is not None
    assert manager.engine is not None
    assert manager.SessionLocal is not None


def test_insert_single_measurement():
    """Test inserting a single measurement"""
    from src.storage.db_manager import DatabaseManager
    from src.models.db import Base
    from sqlalchemy.orm import sessionmaker
    
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    Base.metadata.create_all(manager.engine)
    manager.SessionLocal = sessionmaker(bind=manager.engine, expire_on_commit=False)
    
    sample_dict = {
        "websdr_id": 1,
        "frequency_mhz": 144.5,
        "sample_rate_khz": 12.5,
        "samples_count": 125000,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "metrics": {"snr_db": 15.5}
    }
    
    task_id = "test-session-001"
    
    meas_id = manager.insert_measurement(
        task_id=task_id,
        measurement_dict=sample_dict,
        s3_path="s3://bucket/websdr_1.npy"
    )
    
    assert meas_id is not None


def test_bulk_insert_measurements():
    """Test bulk inserting multiple measurements"""
    from src.storage.db_manager import DatabaseManager
    from src.models.db import Base
    from sqlalchemy.orm import sessionmaker
    
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    Base.metadata.create_all(manager.engine)
    manager.SessionLocal = sessionmaker(bind=manager.engine, expire_on_commit=False)
    
    measurements_list = []
    for websdr_id in range(1, 8):
        measurements_list.append({
            "websdr_id": websdr_id,
            "frequency_mhz": 144.5 + (websdr_id * 0.1),
            "sample_rate_khz": 12.5,
            "samples_count": 125000,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "metrics": {
                "snr_db": 10.0 + (websdr_id * 2),
                "frequency_offset_hz": 100.0 * websdr_id,
                "power_dbm": -70.0 - websdr_id,
            }
        })
    
    task_id = "test-session-001"
    
    successful, failed = manager.insert_measurements_bulk(
        task_id=task_id,
        measurements_list=measurements_list
    )
    
    assert successful == 7
    assert failed == 0


def test_get_snr_statistics():
    """Test retrieving SNR statistics"""
    from src.storage.db_manager import DatabaseManager
    from src.models.db import Base
    from sqlalchemy.orm import sessionmaker
    
    manager = DatabaseManager(database_url="sqlite:///:memory:")
    Base.metadata.create_all(manager.engine)
    manager.SessionLocal = sessionmaker(bind=manager.engine, expire_on_commit=False)
    
    measurements_list = []
    for websdr_id in range(1, 8):
        measurements_list.append({
            "websdr_id": websdr_id,
            "frequency_mhz": 144.5,
            "sample_rate_khz": 12.5,
            "samples_count": 125000,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "metrics": {
                "snr_db": 10.0 + (websdr_id * 2),
            }
        })
    
    task_id = "test-session-001"
    
    manager.insert_measurements_bulk(
        task_id=task_id,
        measurements_list=measurements_list
    )
    
    stats = manager.get_snr_statistics(task_id=task_id)
    
    assert len(stats) == 7
    for websdr_id, stat in stats.items():
        assert "avg_snr_db" in stat
        assert "count" in stat
        assert stat["count"] == 1

