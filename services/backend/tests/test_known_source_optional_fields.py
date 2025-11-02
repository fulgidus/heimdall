"""
Test for known source optional fields - ensuring frequency, latitude, longitude can be NULL

This test validates that known sources can be created with optional frequency and position,
as amateur radio stations may not have these details initially known.
"""

from datetime import datetime
from uuid import uuid4

import pytest


def test_known_source_model_with_all_fields():
    """Test that KnownSource model works with all fields provided"""
    from src.models.session import KnownSource

    source = KnownSource(
        id=uuid4(),
        name="Test Beacon",
        description="Test description",
        frequency_hz=145500000,
        latitude=45.0,
        longitude=7.6,
        power_dbm=30.0,
        source_type="beacon",
        is_validated=True,
        error_margin_meters=50.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert source.name == "Test Beacon"
    assert source.frequency_hz == 145500000
    assert source.latitude == 45.0
    assert source.longitude == 7.6
    assert source.is_validated is True


def test_known_source_model_without_frequency():
    """Test that KnownSource model accepts NULL frequency"""
    from src.models.session import KnownSource

    source = KnownSource(
        id=uuid4(),
        name="Unknown Frequency Station",
        description="Amateur station with unknown frequency",
        frequency_hz=None,  # Unknown frequency
        latitude=45.0,
        longitude=7.6,
        power_dbm=None,
        source_type="station",
        is_validated=False,
        error_margin_meters=100.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert source.name == "Unknown Frequency Station"
    assert source.frequency_hz is None
    assert source.latitude == 45.0
    assert source.longitude == 7.6


def test_known_source_model_without_position():
    """Test that KnownSource model accepts NULL latitude/longitude"""
    from src.models.session import KnownSource

    source = KnownSource(
        id=uuid4(),
        name="Unknown Position Station",
        description="Station with unknown position",
        frequency_hz=145500000,
        latitude=None,  # Unknown position
        longitude=None,  # Unknown position
        power_dbm=None,
        source_type="station",
        is_validated=False,
        error_margin_meters=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert source.name == "Unknown Position Station"
    assert source.frequency_hz == 145500000
    assert source.latitude is None
    assert source.longitude is None


def test_known_source_model_minimal():
    """Test that KnownSource model works with only required fields"""
    from src.models.session import KnownSource

    source = KnownSource(
        id=uuid4(),
        name="Minimal Station",
        description=None,
        frequency_hz=None,  # Optional
        latitude=None,  # Optional
        longitude=None,  # Optional
        power_dbm=None,
        source_type=None,
        is_validated=False,
        error_margin_meters=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert source.name == "Minimal Station"
    assert source.frequency_hz is None
    assert source.latitude is None
    assert source.longitude is None
    assert source.is_validated is False


def test_known_source_create_model_with_all_fields():
    """Test that KnownSourceCreate model works with all fields"""
    from src.models.session import KnownSourceCreate

    source_create = KnownSourceCreate(
        name="Test Beacon",
        description="Test description",
        frequency_hz=145500000,
        latitude=45.0,
        longitude=7.6,
        power_dbm=30.0,
        source_type="beacon",
        is_validated=True,
        error_margin_meters=50.0,
    )

    assert source_create.name == "Test Beacon"
    assert source_create.frequency_hz == 145500000
    assert source_create.latitude == 45.0
    assert source_create.longitude == 7.6


def test_known_source_create_model_without_frequency():
    """Test that KnownSourceCreate accepts NULL frequency"""
    from src.models.session import KnownSourceCreate

    source_create = KnownSourceCreate(
        name="Unknown Frequency Station",
        description="Amateur station with unknown frequency",
        frequency_hz=None,  # Optional
        latitude=45.0,
        longitude=7.6,
        source_type="station",
        is_validated=False,
    )

    assert source_create.name == "Unknown Frequency Station"
    assert source_create.frequency_hz is None
    assert source_create.latitude == 45.0
    assert source_create.longitude == 7.6


def test_known_source_create_model_without_position():
    """Test that KnownSourceCreate accepts NULL latitude/longitude"""
    from src.models.session import KnownSourceCreate

    source_create = KnownSourceCreate(
        name="Unknown Position Station",
        description="Station with unknown position",
        frequency_hz=145500000,
        latitude=None,  # Optional
        longitude=None,  # Optional
        source_type="station",
        is_validated=False,
    )

    assert source_create.name == "Unknown Position Station"
    assert source_create.frequency_hz == 145500000
    assert source_create.latitude is None
    assert source_create.longitude is None


def test_known_source_create_model_minimal():
    """Test that KnownSourceCreate works with only name"""
    from src.models.session import KnownSourceCreate

    source_create = KnownSourceCreate(
        name="Minimal Station",
    )

    assert source_create.name == "Minimal Station"
    assert source_create.frequency_hz is None
    assert source_create.latitude is None
    assert source_create.longitude is None
    assert source_create.is_validated is False


def test_known_source_update_model():
    """Test that KnownSourceUpdate works with partial updates"""
    from src.models.session import KnownSourceUpdate

    # Update only frequency
    update1 = KnownSourceUpdate(frequency_hz=145600000)
    assert update1.frequency_hz == 145600000
    assert update1.latitude is None

    # Update only position
    update2 = KnownSourceUpdate(latitude=45.5, longitude=8.0)
    assert update2.latitude == 45.5
    assert update2.longitude == 8.0
    assert update2.frequency_hz is None

    # Update to NULL (clear frequency)
    update3 = KnownSourceUpdate(frequency_hz=None)
    assert update3.frequency_hz is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
