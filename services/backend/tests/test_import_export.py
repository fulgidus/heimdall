"""Tests for import/export functionality."""

import pytest
from datetime import datetime
from src.models.import_export import (
    CreatorInfo,
    ExportMetadata,
    SectionSizes,
    ExportedSource,
    ExportedWebSDR,
    ExportedSession,
    ExportRequest,
    ImportRequest,
    HeimdallFile,
    ExportSections,
)


def test_creator_info_validation():
    """Test CreatorInfo model validation."""
    creator = CreatorInfo(username="testuser", name="Test User")
    assert creator.username == "testuser"
    assert creator.name == "Test User"


def test_export_metadata_structure():
    """Test ExportMetadata structure and defaults."""
    creator = CreatorInfo(username="admin", name="Admin User")
    metadata = ExportMetadata(creator=creator)
    
    assert metadata.version == "1.0"
    assert metadata.creator.username == "admin"
    assert isinstance(metadata.created_at, datetime)
    assert isinstance(metadata.section_sizes, SectionSizes)


def test_exported_source_model():
    """Test ExportedSource model."""
    source = ExportedSource(
        id="550e8400-e29b-41d4-a716-446655440000",
        name="Test Beacon",
        description="Test source for validation",
        frequency_hz=145500000,
        latitude=45.5,
        longitude=9.2,
        power_dbm=10.0,
        source_type="beacon",
        is_validated=True,
        error_margin_meters=30.0,
        created_at="2025-10-31T10:00:00Z",
        updated_at="2025-10-31T10:00:00Z",
    )
    
    assert source.name == "Test Beacon"
    assert source.frequency_hz == 145500000
    assert source.is_validated is True


def test_exported_websdr_model():
    """Test ExportedWebSDR model."""
    websdr = ExportedWebSDR(
        id="660e8400-e29b-41d4-a716-446655440000",
        name="Test WebSDR",
        url="http://test.websdr.org:8901",
        location_description="Test Location, Italy",
        latitude=45.5,
        longitude=9.2,
        altitude_meters=100.0,
        country="Italy",
        operator="Test Operator",
        is_active=True,
        timeout_seconds=30,
        retry_count=3,
        created_at="2025-10-31T10:00:00Z",
        updated_at="2025-10-31T10:00:00Z",
    )
    
    assert websdr.name == "Test WebSDR"
    assert websdr.country == "Italy"
    assert websdr.is_active is True


def test_heimdall_file_structure():
    """Test complete HeimdallFile structure."""
    creator = CreatorInfo(username="testuser")
    metadata = ExportMetadata(creator=creator)
    sections = ExportSections()
    
    file = HeimdallFile(metadata=metadata, sections=sections)
    
    assert file.metadata.creator.username == "testuser"
    assert file.sections is not None


def test_export_request_validation():
    """Test ExportRequest validation."""
    creator = CreatorInfo(username="admin", name="Admin")
    request = ExportRequest(
        creator=creator,
        description="Test export",
        include_sources=True,
        include_websdrs=True,
        include_sessions=False,
    )
    
    assert request.include_sources is True
    assert request.include_websdrs is True
    assert request.include_sessions is False


def test_import_request_validation():
    """Test ImportRequest validation."""
    creator = CreatorInfo(username="admin")
    metadata = ExportMetadata(creator=creator)
    sections = ExportSections()
    file = HeimdallFile(metadata=metadata, sections=sections)
    
    request = ImportRequest(
        heimdall_file=file,
        import_sources=True,
        import_websdrs=True,
        overwrite_existing=False,
    )
    
    assert request.import_sources is True
    assert request.overwrite_existing is False


def test_section_sizes_calculation():
    """Test SectionSizes calculation."""
    sizes = SectionSizes(
        settings=256,
        sources=2048,
        websdrs=1536,
        sessions=0,
        training_model=0,
        inference_model=0,
    )
    
    assert sizes.settings == 256
    assert sizes.sources == 2048
    assert sizes.websdrs == 1536


def test_heimdall_file_serialization():
    """Test that HeimdallFile can be serialized to JSON."""
    creator = CreatorInfo(username="testuser", name="Test User")
    metadata = ExportMetadata(creator=creator, description="Test export")
    
    source = ExportedSource(
        id="550e8400-e29b-41d4-a716-446655440000",
        name="Test Source",
        frequency_hz=145500000,
        latitude=45.5,
        longitude=9.2,
        is_validated=True,
        created_at="2025-10-31T10:00:00Z",
        updated_at="2025-10-31T10:00:00Z",
    )
    
    sections = ExportSections(sources=[source])
    file = HeimdallFile(metadata=metadata, sections=sections)
    
    # Should be able to serialize to JSON
    json_str = file.model_dump_json()
    assert json_str is not None
    assert len(json_str) > 0
    assert "Test Source" in json_str
    assert "testuser" in json_str
