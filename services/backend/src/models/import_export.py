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
    audio_library: int = 0


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


class IQData(BaseModel):
    """IQ data for a single receiver."""
    
    receiver_id: str  # e.g., "RX_000"
    iq_data_base64: str  # numpy array encoded as base64
    

class ExportedIQSample(BaseModel):
    """Complete IQ sample data from synthetic_iq_samples table."""
    
    id: str
    sample_idx: int
    timestamp: str
    tx_lat: float
    tx_lon: float
    tx_alt: float
    tx_power_dbm: float
    frequency_hz: int
    num_receivers: int
    gdop: float | None = None
    mean_snr_db: float | None = None
    overall_confidence: float | None = None
    receivers_metadata: list[dict]  # JSONB array of receiver configs
    iq_metadata: dict  # JSONB (duration_ms, sample_rate_hz, center_frequency_hz)
    iq_storage_paths: dict  # JSONB (mapping receiver_id -> S3 path)
    iq_data: list[IQData]  # Actual IQ binary data in base64
    created_at: str


class ExportedFeature(BaseModel):
    """Complete feature data from measurement_features table."""
    
    recording_session_id: str
    timestamp: str
    receiver_features: list[dict]  # JSONB array
    tx_latitude: float | None = None
    tx_longitude: float | None = None
    tx_altitude_m: float | None = None
    tx_power_dbm: float | None = None
    tx_known: bool = False
    extraction_metadata: dict  # JSONB
    overall_confidence: float
    mean_snr_db: float | None = None
    num_receivers_detected: int | None = None
    gdop: float | None = None
    extraction_failed: bool = False
    error_message: str | None = None
    created_at: str
    num_receivers_in_sample: int | None = None


class ExportedSampleSet(BaseModel):
    """Synthetic dataset (sample set) for export with complete data."""

    id: str
    name: str
    description: str | None = None
    num_samples: int
    config: dict | None = None
    quality_metrics: dict | None = None
    created_at: str
    # Complete feature data from measurement_features
    features: list[ExportedFeature] | None = None
    # Complete IQ data from synthetic_iq_samples
    iq_samples: list[ExportedIQSample] | None = None
    # Range information for partial exports
    num_exported_features: int | None = None
    num_exported_iq_samples: int | None = None
    export_range: dict | None = None  # {"offset": int, "limit": int}


class ExportedAudioChunk(BaseModel):
    """Audio chunk for export (1-second preprocessed audio)."""

    id: str
    chunk_index: int
    duration_seconds: float
    sample_rate: int
    num_samples: int
    file_size_bytes: int
    original_offset_seconds: float
    rms_amplitude: float | None = None
    created_at: str
    # Audio data encoded as base64
    audio_data_base64: str | None = None


class ExportedAudioLibrary(BaseModel):
    """Audio library entry for export."""

    id: str
    filename: str
    category: str
    tags: list[str] | None = None
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    audio_format: str
    processing_status: str
    total_chunks: int
    enabled: bool
    created_at: str
    updated_at: str
    # Associated audio chunks
    chunks: list[ExportedAudioChunk] | None = None


class ExportSections(BaseModel):
    """All exportable data sections."""

    settings: UserSettings | None = None
    sources: list[ExportedSource] | None = None
    websdrs: list[ExportedWebSDR] | None = None
    sessions: list[ExportedSession] | None = None
    sample_sets: list[ExportedSampleSet] | None = None
    models: list[ExportedModel] | None = None
    audio_library: list[ExportedAudioLibrary] | None = None


class HeimdallFile(BaseModel):
    """Complete .heimdall file structure."""

    metadata: ExportMetadata
    sections: ExportSections


class SampleSetExportConfig(BaseModel):
    """Configuration for exporting a sample set with range selection."""

    dataset_id: str
    sample_offset: int = Field(default=0, ge=0, description="Starting offset for samples")
    sample_limit: int | None = Field(
        default=None, ge=1, description="Number of samples to export (None = all)"
    )
    include_iq_data: bool = Field(
        default=True, description="Include binary IQ data (can be very large, ~16MB per sample)"
    )


class ExportRequest(BaseModel):
    """Request to export data."""

    creator: CreatorInfo
    description: str | None = None
    include_settings: bool = False
    include_sources: bool = True
    include_websdrs: bool = True
    include_sessions: bool = False
    # Sample sets (synthetic datasets) - now with range support
    sample_set_configs: list[SampleSetExportConfig] | None = None
    # Models - list of IDs or None for none
    model_ids: list[str] | None = None
    # Audio library - list of IDs or None for none
    audio_library_ids: list[str] | None = None


class ImportRequest(BaseModel):
    """Request to import data."""

    heimdall_file: HeimdallFile
    import_settings: bool = False
    import_sources: bool = True
    import_websdrs: bool = True
    import_sessions: bool = False
    import_sample_sets: bool = False
    import_models: bool = False
    import_audio_library: bool = False
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
    num_samples: int  # Number of feature samples
    num_iq_samples: int = 0  # Number of IQ samples (raw RF data)
    created_at: str
    estimated_size_bytes: int = Field(
        default=0, description="Estimated size of full dataset in bytes (features + IQ data)"
    )
    estimated_size_per_feature: int = Field(
        default=5600, description="Estimated size per feature sample in bytes"
    )
    estimated_size_per_iq: int = Field(
        default=13_000_000, description="Estimated size per IQ sample in bytes (~13MB)"
    )


class AvailableModel(BaseModel):
    """Available model for export."""

    id: str
    model_name: str
    version: int
    created_at: str
    has_onnx: bool


class AvailableAudioLibrary(BaseModel):
    """Available audio library entry for export."""

    id: str
    filename: str
    category: str
    duration_seconds: float
    total_chunks: int
    file_size_bytes: int  # Original audio file size
    chunks_total_bytes: int  # Total size of preprocessed chunks in MinIO
    created_at: str


class MetadataResponse(BaseModel):
    """Response with available data for export."""

    sources_count: int = 0
    websdrs_count: int = 0
    sessions_count: int = 0
    sample_sets: list[AvailableSampleSet] = Field(default_factory=list)
    models: list[AvailableModel] = Field(default_factory=list)
    audio_library: list[AvailableAudioLibrary] = Field(default_factory=list)
    estimated_sizes: SectionSizes = Field(default_factory=SectionSizes)
