"""Tests for import/export router."""

import pytest
from datetime import datetime
from uuid import uuid4
import json

from src.models.import_export import (
    ExportRequest,
    ImportRequest,
    HeimdallFile,
    HeimdallMetadata,
    HeimdallSections,
    CreatorInfo,
    UserSettings,
    ExportedSource,
    ExportedWebSDR,
)


def test_export_request_validation():
    """Test ExportRequest model validation."""
    creator = CreatorInfo(username="testuser", name="Test User")
    request = ExportRequest(
        creator=creator,
        description="Test export",
        include_settings=True,
        include_sources=True,
        include_websdrs=False,
    )
    
    assert request.creator.username == "testuser"
    assert request.include_settings is True
    assert request.include_websdrs is False


def test_heimdall_file_structure():
    """Test HeimdallFile model structure."""
    creator = CreatorInfo(username="testuser", name="Test User")
    metadata = HeimdallMetadata(
        version="1.0",
        created_at=datetime.utcnow(),
        creator=creator,
    )
    
    sections = HeimdallSections(
        settings=UserSettings(
            theme="dark",
            language="en",
            auto_approve_sessions=False,
        )
    )
    
    heimdall_file = HeimdallFile(metadata=metadata, sections=sections)
    
    assert heimdall_file.metadata.version == "1.0"
    assert heimdall_file.sections.settings.theme == "dark"


def test_exported_source_model():
    """Test ExportedSource model."""
    source = ExportedSource(
        id=uuid4(),
        name="Test Source",
        description="A test RF source",
        frequency_hz=145500000,
        latitude=45.0,
        longitude=9.0,
        is_validated=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    assert source.name == "Test Source"
    assert source.frequency_hz == 145500000


def test_exported_websdr_model():
    """Test ExportedWebSDR model."""
    websdr = ExportedWebSDR(
        id=uuid4(),
        name="Test WebSDR",
        url="http://example.com:8901",
        latitude=45.0,
        longitude=9.0,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    
    assert websdr.name == "Test WebSDR"
    assert websdr.url == "http://example.com:8901"


def test_heimdall_file_serialization():
    """Test that HeimdallFile can be serialized to JSON."""
    creator = CreatorInfo(username="testuser", name="Test User")
    metadata = HeimdallMetadata(
        version="1.0",
        created_at=datetime.utcnow(),
        creator=creator,
    )
    
    sections = HeimdallSections(
        settings=UserSettings(
            theme="dark",
            language="en",
        ),
        sources=[
            ExportedSource(
                id=uuid4(),
                name="Source 1",
                frequency_hz=145500000,
                latitude=45.0,
                longitude=9.0,
                is_validated=False,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        ],
    )
    
    heimdall_file = HeimdallFile(metadata=metadata, sections=sections)
    
    # Serialize to JSON
    json_str = json.dumps(heimdall_file.model_dump(), default=str)
    assert json_str is not None
    assert "testuser" in json_str
    assert "Source 1" in json_str


def test_import_request_validation():
    """Test ImportRequest model validation."""
    creator = CreatorInfo(username="testuser", name="Test User")
    metadata = HeimdallMetadata(
        version="1.0",
        created_at=datetime.utcnow(),
        creator=creator,
    )
    sections = HeimdallSections()
    
    file_content = HeimdallFile(metadata=metadata, sections=sections)
    
    request = ImportRequest(
        file_content=file_content,
        import_settings=True,
        import_sources=False,
        overwrite_existing=True,
    )
    
    assert request.import_settings is True
    assert request.import_sources is False
    assert request.overwrite_existing is True


def test_section_sizes_calculation():
    """Test that section sizes are calculated correctly."""
    from src.models.import_export import SectionSizes
    
    sizes = SectionSizes(
        settings=1024,
        sources=2048,
        websdrs=512,
    )
    
    assert sizes.settings == 1024
    assert sizes.sources == 2048
    assert sizes.websdrs == 512
