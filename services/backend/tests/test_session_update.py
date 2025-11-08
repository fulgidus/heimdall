"""
Tests for session update endpoint

Run with: pytest services/backend/tests/test_session_update.py -v

This test validates the session metadata update functionality.
To run these tests, you need:
- PostgreSQL running with heimdall database
- Backend service fixtures (conftest.py)
"""

from uuid import uuid4

import pytest
from fastapi import status


@pytest.mark.skip(reason="Requires running PostgreSQL and backend fixtures")
@pytest.mark.asyncio
async def test_update_session_metadata(client, db_pool):
    """Test updating session metadata"""
    # Create a known source first
    source_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.known_sources
            (id, name, frequency_hz, latitude, longitude, is_validated, error_margin_meters)
            VALUES ($1, 'Test Source', 145500000, 45.0, 9.0, true, 50.0)
            """,
            source_id,
        )

    # Create a session
    session_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.recording_sessions
            (id, known_source_id, session_name, session_start, status, approval_status, notes)
            VALUES ($1, $2, 'Original Name', NOW(), 'completed', 'pending', 'Original notes')
            """,
            session_id,
            source_id,
        )

    # Update the session
    update_data = {
        "session_name": "Updated Session Name",
        "notes": "Updated notes with new information",
        "approval_status": "approved",
    }

    response = await client.patch(f"/api/v1/sessions/{session_id}", json=update_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["session_name"] == "Updated Session Name"
    assert data["notes"] == "Updated notes with new information"
    assert data["approval_status"] == "approved"
    assert data["id"] == str(session_id)

    # Verify in database
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT session_name, notes, approval_status FROM heimdall.recording_sessions WHERE id = $1",
            session_id,
        )

        assert row["session_name"] == "Updated Session Name"
        assert row["notes"] == "Updated notes with new information"
        assert row["approval_status"] == "approved"


@pytest.mark.skip(reason="Requires running PostgreSQL and backend fixtures")
@pytest.mark.asyncio
async def test_update_session_partial(client, db_pool):
    """Test updating only some fields"""
    # Create a known source first
    source_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.known_sources
            (id, name, frequency_hz, latitude, longitude, is_validated, error_margin_meters)
            VALUES ($1, 'Test Source', 145500000, 45.0, 9.0, true, 50.0)
            """,
            source_id,
        )

    # Create a session
    session_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.recording_sessions
            (id, known_source_id, session_name, session_start, status, approval_status, notes)
            VALUES ($1, $2, 'Original Name', NOW(), 'completed', 'pending', 'Original notes')
            """,
            session_id,
            source_id,
        )

    # Update only the notes
    update_data = {"notes": "Just updating the notes"}

    response = await client.patch(f"/api/v1/sessions/{session_id}", json=update_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Name and approval should remain unchanged
    assert data["session_name"] == "Original Name"
    assert data["approval_status"] == "pending"
    # Notes should be updated
    assert data["notes"] == "Just updating the notes"


@pytest.mark.skip(reason="Requires running PostgreSQL and backend fixtures")
@pytest.mark.asyncio
async def test_update_session_not_found(client):
    """Test updating a non-existent session"""
    fake_id = uuid4()
    update_data = {"session_name": "New Name"}

    response = await client.patch(f"/api/v1/sessions/{fake_id}", json=update_data)

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.skip(reason="Requires running PostgreSQL and backend fixtures")
@pytest.mark.asyncio
async def test_update_session_empty_update(client, db_pool):
    """Test updating with no fields should fail"""
    # Create a known source first
    source_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.known_sources
            (id, name, frequency_hz, latitude, longitude, is_validated, error_margin_meters)
            VALUES ($1, 'Test Source', 145500000, 45.0, 9.0, true, 50.0)
            """,
            source_id,
        )

    # Create a session
    session_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.recording_sessions
            (id, known_source_id, session_name, session_start, status, approval_status)
            VALUES ($1, $2, 'Original Name', NOW(), 'completed', 'pending')
            """,
            session_id,
            source_id,
        )

    # Try to update with no fields
    update_data = {}

    response = await client.patch(f"/api/v1/sessions/{session_id}", json=update_data)

    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.skip(reason="Requires running PostgreSQL and backend fixtures")
@pytest.mark.asyncio
async def test_update_session_invalid_approval_status(client, db_pool):
    """Test updating with invalid approval status should fail"""
    # Create a known source first
    source_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.known_sources
            (id, name, frequency_hz, latitude, longitude, is_validated, error_margin_meters)
            VALUES ($1, 'Test Source', 145500000, 45.0, 9.0, true, 50.0)
            """,
            source_id,
        )

    # Create a session
    session_id = uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO heimdall.recording_sessions
            (id, known_source_id, session_name, session_start, status, approval_status)
            VALUES ($1, $2, 'Original Name', NOW(), 'completed', 'pending')
            """,
            session_id,
            source_id,
        )

    # Try to update with invalid approval status
    update_data = {"approval_status": "invalid_status"}

    response = await client.patch(f"/api/v1/sessions/{session_id}", json=update_data)

    # Should fail validation
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
