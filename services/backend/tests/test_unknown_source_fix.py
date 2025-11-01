"""
Test for unknown source handling - ensuring NULL source_id is properly supported

This test validates that recording sessions can be created with NULL known_source_id
when the source is unknown, without creating placeholder "Unknown" sources.
"""

from datetime import datetime
from uuid import uuid4

import pytest


def test_recording_session_model_with_null_source():
    """Test that RecordingSession model accepts NULL known_source_id"""
    from src.models.session import RecordingSession

    # Create a session with NULL source_id (unknown source)
    session = RecordingSession(
        id=uuid4(),
        known_source_id=None,  # NULL for unknown sources
        session_name="Test Unknown Source Session",
        session_start=datetime.utcnow(),
        session_end=None,
        duration_seconds=None,
        celery_task_id=None,
        status="pending",
        approval_status="pending",
        notes="Testing unknown source handling",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert session.known_source_id is None
    assert session.session_name == "Test Unknown Source Session"
    assert session.status == "pending"


def test_recording_session_create_model_with_null_source():
    """Test that RecordingSessionCreate model accepts NULL known_source_id"""
    from src.models.session import RecordingSessionCreate

    # Create session request with NULL source_id
    session_create = RecordingSessionCreate(
        known_source_id=None,  # NULL for unknown sources
        session_name="Test Create Unknown",
        frequency_hz=145500000,  # 145.5 MHz
        duration_seconds=60.0,
        notes="Test notes",
    )

    assert session_create.known_source_id is None
    assert session_create.frequency_hz == 145500000


def test_recording_session_with_details_null_source():
    """Test that RecordingSessionWithDetails handles NULL source fields"""
    from src.models.session import RecordingSessionWithDetails

    # Create session with NULL source details
    session = RecordingSessionWithDetails(
        id=uuid4(),
        known_source_id=None,  # NULL for unknown sources
        session_name="Test Session",
        session_start=datetime.utcnow(),
        status="completed",
        approval_status="approved",
        notes="",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        source_name=None,  # NULL when source is unknown
        source_frequency=None,
        source_latitude=None,
        source_longitude=None,
        measurements_count=10,
    )

    assert session.known_source_id is None
    assert session.source_name is None
    assert session.source_frequency is None
    assert session.measurements_count == 10


def test_recording_session_with_known_source():
    """Test that RecordingSession still works with known sources"""
    from src.models.session import RecordingSession

    source_id = uuid4()

    # Create session with a known source
    session = RecordingSession(
        id=uuid4(),
        known_source_id=source_id,  # Valid UUID for known source
        session_name="Test Known Source Session",
        session_start=datetime.utcnow(),
        status="pending",
        approval_status="pending",
        notes="",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert session.known_source_id == source_id
    assert session.session_name == "Test Known Source Session"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
