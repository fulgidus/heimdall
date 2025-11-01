"""Import/Export models for .heimdall file format."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field


class CreatorInfo(BaseModel):
    """Information about the file creator."""
    username: str = Field(..., description="Username of the creator")
    name: str = Field(..., description="Full name of the creator")


class SectionSizes(BaseModel):
    """Sizes of each section in bytes."""
    settings: int = Field(default=0, description="Settings section size in bytes")
    sources: int = Field(default=0, description="Sources section size in bytes")
    websdrs: int = Field(default=0, description="WebSDRs section size in bytes")
    sessions: int = Field(default=0, description="Sessions section size in bytes")
    training_model: int = Field(default=0, description="Training model section size in bytes")
    inference_model: int = Field(default=0, description="Inference model section size in bytes")


class HeimdallMetadata(BaseModel):
    """Metadata for .heimdall file."""
    version: str = Field(default="1.0", description="File format version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    creator: CreatorInfo = Field(..., description="Information about the file creator")
    section_sizes: SectionSizes = Field(default_factory=SectionSizes, description="Size of each section")
    description: Optional[str] = Field(None, description="Optional description of the export")


class UserSettings(BaseModel):
    """User settings section."""
    theme: str = Field(default="dark", description="UI theme preference")
    language: str = Field(default="en", description="UI language")
    default_frequency_mhz: Optional[float] = Field(None, description="Default frequency in MHz")
    default_duration_seconds: Optional[float] = Field(None, description="Default acquisition duration")
    map_center_lat: Optional[float] = Field(None, description="Default map center latitude")
    map_center_lon: Optional[float] = Field(None, description="Default map center longitude")
    map_zoom: Optional[int] = Field(None, description="Default map zoom level")
    auto_approve_sessions: bool = Field(default=False, description="Auto-approve recording sessions")
    notification_enabled: bool = Field(default=True, description="Enable notifications")
    advanced_mode: bool = Field(default=False, description="Enable advanced features")


class ExportedSource(BaseModel):
    """Exported known source."""
    id: UUID
    name: str
    description: Optional[str] = None
    frequency_hz: int
    latitude: float
    longitude: float
    power_dbm: Optional[float] = None
    source_type: Optional[str] = None
    is_validated: bool = False
    error_margin_meters: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class ExportedWebSDR(BaseModel):
    """Exported WebSDR station."""
    id: UUID
    name: str
    url: str
    country: Optional[str] = None
    latitude: float
    longitude: float
    frequency_min_hz: Optional[int] = None
    frequency_max_hz: Optional[int] = None
    is_active: bool
    api_type: Optional[str] = None
    rate_limit_ms: Optional[int] = None
    timeout_seconds: Optional[int] = None
    retry_count: Optional[int] = None
    admin_email: Optional[str] = None
    location_description: Optional[str] = None
    altitude_asl: Optional[int] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ExportedSession(BaseModel):
    """Exported recording session with measurements."""
    id: UUID
    known_source_id: UUID
    session_name: str
    session_start: datetime
    session_end: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    celery_task_id: Optional[str] = None
    status: str
    approval_status: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    measurements_count: int = 0
    source_name: Optional[str] = None
    source_frequency: Optional[int] = None


class ExportedModel(BaseModel):
    """Exported ML model (training or inference)."""
    id: UUID
    model_name: str
    model_type: Optional[str] = None
    training_dataset_id: Optional[UUID] = None
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[int] = None
    onnx_model_location: Optional[str] = None
    pytorch_model_location: Optional[str] = None
    accuracy_meters: Optional[float] = None
    accuracy_sigma_meters: Optional[float] = None
    loss_value: Optional[float] = None
    epoch: Optional[int] = None
    is_active: bool
    is_production: bool
    created_at: datetime
    updated_at: datetime
    model_data: Optional[str] = None  # Base64 encoded model file


class HeimdallSections(BaseModel):
    """Optional sections that can be included in export."""
    settings: Optional[UserSettings] = None
    sources: Optional[List[ExportedSource]] = None
    websdrs: Optional[List[ExportedWebSDR]] = None
    sessions: Optional[List[ExportedSession]] = None
    training_model: Optional[ExportedModel] = None
    inference_model: Optional[ExportedModel] = None


class HeimdallFile(BaseModel):
    """Complete .heimdall file structure."""
    metadata: HeimdallMetadata
    sections: HeimdallSections


class ExportRequest(BaseModel):
    """Request to export data."""
    creator: CreatorInfo
    description: Optional[str] = None
    include_settings: bool = Field(default=True, description="Include user settings")
    include_sources: bool = Field(default=True, description="Include known sources")
    include_websdrs: bool = Field(default=True, description="Include WebSDR stations")
    include_sessions: bool = Field(default=False, description="Include recording sessions")
    include_training_model: bool = Field(default=False, description="Include training model")
    include_inference_model: bool = Field(default=False, description="Include inference model")
    session_ids: Optional[List[UUID]] = Field(None, description="Specific session IDs to export (if include_sessions=True)")


class ImportRequest(BaseModel):
    """Request to import data from .heimdall file."""
    file_content: HeimdallFile
    import_settings: bool = Field(default=True, description="Import user settings")
    import_sources: bool = Field(default=True, description="Import known sources")
    import_websdrs: bool = Field(default=True, description="Import WebSDR stations")
    import_sessions: bool = Field(default=False, description="Import recording sessions")
    import_training_model: bool = Field(default=False, description="Import training model")
    import_inference_model: bool = Field(default=False, description="Import inference model")
    overwrite_existing: bool = Field(default=False, description="Overwrite existing items with same ID/name")


class ImportResult(BaseModel):
    """Result of import operation."""
    success: bool
    message: str
    imported_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of imported items per section"
    )
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")


class ExportMetadataResponse(BaseModel):
    """Response for export metadata preview."""
    available_sources_count: int
    available_websdrs_count: int
    available_sessions_count: int
    has_training_model: bool
    has_inference_model: bool
    estimated_size_bytes: int
