"""Import/Export data models for Heimdall SDR.

Defines the .heimdall file format for saving and restoring system state.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CreatorInfo(BaseModel):
    """Information about who created the export."""

    username: str = Field(..., description="Username of the creator")
    name: str | None = Field(None, description="Full name of the creator")


class SectionSizes(BaseModel):
    """Byte sizes of each section in the export."""

    settings: int = 0
    sources: int = 0
    websdrs: int = 0
    sessions: int = 0
    sample_sets: int = 0
    models: int = 0


class ExportMetadata(BaseModel):
    """Metadata about the exported file."""

    version: str = Field(default="1.0", description="File format version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Export timestamp")
    creator: CreatorInfo
    section_sizes: SectionSizes = Field(default_factory=SectionSizes)
    description: str | None = Field(None, description="Optional description of export")


class ExportedSource(BaseModel):
    """Known RF source for export."""

    id: str
    name: str
    description: str | None = None
    frequency_hz: int
    latitude: float
    longitude: float
    power_dbm: float | None = None
    source_type: str | None = None
    is_validated: bool = False
    error_margin_meters: float | None = None
    created_at: str
    updated_at: str


class ExportedWebSDR(BaseModel):
    """WebSDR configuration for export."""

    id: str
    name: str
    url: str
    location_description: str | None = None
    latitude: float
    longitude: float
    altitude_meters: float | None = None
    country: str | None = None
    operator: str | None = None
    is_active: bool = True
    timeout_seconds: int = 30
    retry_count: int = 3
    created_at: str
    updated_at: str


class ExportedSession(BaseModel):
    """Recording session for export."""

    id: str
    known_source_id: str
    session_name: str
    session_start: str
    session_end: str | None = None
    duration_seconds: float | None = None
    celery_task_id: str | None = None
    status: str
    approval_status: str = "pending"
    notes: str | None = None
    created_at: str
    updated_at: str
    measurements_count: int = 0


class ExportedModel(BaseModel):
    """ML model metadata for export."""

    id: str
    model_name: str
    version: int
    model_type: str
    created_at: str
    # ONNX model binary encoded as base64
    onnx_model_base64: str | None = None
    # Model metrics and configuration
    accuracy_meters: float | None = None
    hyperparameters: dict | None = None
    training_metrics: dict | None = None


class UserSettings(BaseModel):
    """User settings for export."""

    theme: str = "light"
    default_frequency_mhz: float = 145.5
    default_duration_seconds: float = 10.0
    auto_approve_sessions: bool = False
    # Add more settings as needed


class ExportedSampleSet(BaseModel):
    """Synthetic dataset (sample set) for export."""

    id: str
    name: str
    description: str | None = None
    num_samples: int
    config: dict | None = None
    quality_metrics: dict | None = None
    created_at: str
    # Sample data is stored separately due to size
    samples: list[dict] | None = None


class ExportSections(BaseModel):
    """All exportable data sections."""

    settings: UserSettings | None = None
    sources: list[ExportedSource] | None = None
    websdrs: list[ExportedWebSDR] | None = None
    sessions: list[ExportedSession] | None = None
    sample_sets: list[ExportedSampleSet] | None = None
    models: list[ExportedModel] | None = None


class HeimdallFile(BaseModel):
    """Complete .heimdall file structure."""

    metadata: ExportMetadata
    sections: ExportSections


class ExportRequest(BaseModel):
    """Request to export data."""

    creator: CreatorInfo
    description: str | None = None
    include_settings: bool = False
    include_sources: bool = True
    include_websdrs: bool = True
    include_sessions: bool = False
    # Sample sets (synthetic datasets) - list of IDs or None for none
    sample_set_ids: list[str] | None = None
    # Models - list of IDs or None for none
    model_ids: list[str] | None = None


class ImportRequest(BaseModel):
    """Request to import data."""

    heimdall_file: HeimdallFile
    import_settings: bool = False
    import_sources: bool = True
    import_websdrs: bool = True
    import_sessions: bool = False
    import_sample_sets: bool = False
    import_models: bool = False
    overwrite_existing: bool = False


class ExportResponse(BaseModel):
    """Response containing the exported .heimdall file."""

    file: HeimdallFile
    size_bytes: int


class ImportResponse(BaseModel):
    """Response after importing data."""

    success: bool
    message: str
    imported_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class AvailableSampleSet(BaseModel):
    """Available sample set for export."""

    id: str
    name: str
    num_samples: int
    created_at: str


class AvailableModel(BaseModel):
    """Available model for export."""

    id: str
    model_name: str
    version: int
    created_at: str
    has_onnx: bool


class MetadataResponse(BaseModel):
    """Response with available data for export."""

    sources_count: int = 0
    websdrs_count: int = 0
    sessions_count: int = 0
    sample_sets: list[AvailableSampleSet] = Field(default_factory=list)
    models: list[AvailableModel] = Field(default_factory=list)
    estimated_sizes: SectionSizes = Field(default_factory=SectionSizes)
