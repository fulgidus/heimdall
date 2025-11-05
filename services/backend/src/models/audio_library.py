"""
Pydantic models for audio library feature.

Supports uploading and managing realistic audio samples (voice, music, documentaries)
for more realistic training data generation instead of synthetic formant synthesis.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger(__name__)


class AudioCategory(str, Enum):
    """Audio sample categories for organization."""
    
    VOICE = "voice"
    MUSIC = "music"
    DOCUMENTARY = "documentary"
    CONFERENCE = "conference"
    CUSTOM = "custom"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    
    @classmethod
    def from_filename(cls, filename: str) -> Optional["AudioFormat"]:
        """Extract audio format from filename extension."""
        ext = filename.lower().split('.')[-1]
        try:
            return cls(ext)
        except ValueError:
            return None
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return list of supported file extensions."""
        return [fmt.value for fmt in cls]


class AudioMetadata(BaseModel):
    """
    Metadata for a single audio file in the library.
    
    Stored in metadata.json catalog in MinIO.
    """
    
    id: str = Field(..., description="Unique identifier (UUID)")
    filename: str = Field(..., description="Original filename")
    category: AudioCategory = Field(..., description="Audio category")
    format: AudioFormat = Field(..., description="Audio format")
    
    # File properties
    duration_seconds: float = Field(..., description="Audio duration in seconds", gt=0)
    sample_rate: int = Field(..., description="Sample rate in Hz")
    channels: int = Field(..., description="Number of audio channels (1=mono, 2=stereo)")
    bitrate_kbps: Optional[int] = Field(None, description="Bitrate in kbps (for compressed formats)")
    file_size_bytes: int = Field(..., description="File size in bytes", gt=0)
    
    # Storage
    minio_bucket: str = Field(default="heimdall-audio-library", description="MinIO bucket name")
    minio_key: str = Field(..., description="Object key in MinIO (category/uuid.ext)")
    
    # Management
    enabled: bool = Field(default=True, description="Whether file is active (soft delete)")
    tags: List[str] = Field(default_factory=list, description="User-defined tags for filtering")
    
    # Timestamps
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Ensure filename has valid extension."""
        if not any(v.lower().endswith(f".{ext}") for ext in AudioFormat.supported_extensions()):
            raise ValueError(f"Filename must end with one of: {AudioFormat.supported_extensions()}")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are non-empty strings."""
        return [tag.strip() for tag in v if tag.strip()]
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "voice_sample_01.wav",
                "category": "voice",
                "format": "wav",
                "duration_seconds": 12.5,
                "sample_rate": 44100,
                "channels": 1,
                "bitrate_kbps": None,
                "file_size_bytes": 1102500,
                "minio_bucket": "heimdall-audio-library",
                "minio_key": "voice/550e8400-e29b-41d4-a716-446655440000.wav",
                "enabled": True,
                "tags": ["italian", "male", "test"],
                "uploaded_at": "2025-11-05T10:30:00Z",
                "updated_at": "2025-11-05T10:30:00Z"
            }
        }


class AudioUploadRequest(BaseModel):
    """Request model for audio file upload."""
    
    category: AudioCategory = Field(..., description="Audio category")
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are non-empty strings."""
        return [tag.strip() for tag in v if tag.strip()]
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "voice",
                "tags": ["italian", "male", "test"]
            }
        }


class AudioUploadResponse(BaseModel):
    """Response model for successful audio upload."""
    
    metadata: AudioMetadata = Field(..., description="Uploaded audio metadata")
    message: str = Field(default="Audio file uploaded successfully")
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": AudioMetadata.Config.json_schema_extra["example"],
                "message": "Audio file uploaded successfully"
            }
        }


class AudioListRequest(BaseModel):
    """Request model for listing audio files with filters."""
    
    category: Optional[AudioCategory] = Field(None, description="Filter by category")
    enabled: Optional[bool] = Field(None, description="Filter by enabled status (None=all, True=enabled, False=disabled)")
    tags: List[str] = Field(default_factory=list, description="Filter by tags (AND logic)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "voice",
                "enabled": True,
                "tags": ["italian"]
            }
        }


class AudioListResponse(BaseModel):
    """Response model for audio file listing."""
    
    total: int = Field(..., description="Total files matching filters")
    files: List[AudioMetadata] = Field(..., description="Audio file metadata list")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 1,
                "files": [AudioMetadata.Config.json_schema_extra["example"]]
            }
        }


class AudioUpdateRequest(BaseModel):
    """Request model for updating audio metadata."""
    
    filename: Optional[str] = Field(None, description="New filename (must keep same extension)")
    category: Optional[AudioCategory] = Field(None, description="New category")
    tags: Optional[List[str]] = Field(None, description="New tags (replaces existing)")
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        """Ensure filename has valid extension if provided."""
        if v and not any(v.lower().endswith(f".{ext}") for ext in AudioFormat.supported_extensions()):
            raise ValueError(f"Filename must end with one of: {AudioFormat.supported_extensions()}")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Ensure tags are non-empty strings."""
        if v is not None:
            return [tag.strip() for tag in v if tag.strip()]
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "voice_sample_renamed.wav",
                "category": "documentary",
                "tags": ["italian", "radio", "archive"]
            }
        }


class AudioLibraryStats(BaseModel):
    """Statistics about the audio library."""
    
    total_files: int = Field(..., description="Total files in library")
    enabled_files: int = Field(..., description="Enabled files")
    disabled_files: int = Field(..., description="Disabled files")
    total_duration_seconds: float = Field(..., description="Total audio duration")
    total_size_bytes: int = Field(..., description="Total storage used")
    
    files_by_category: dict[AudioCategory, int] = Field(..., description="File count per category")
    files_by_format: dict[AudioFormat, int] = Field(..., description="File count per format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 10,
                "enabled_files": 8,
                "disabled_files": 2,
                "total_duration_seconds": 125.6,
                "total_size_bytes": 11025000,
                "files_by_category": {"voice": 5, "music": 3, "documentary": 2},
                "files_by_format": {"wav": 6, "mp3": 3, "flac": 1}
            }
        }


class CategoryWeights(BaseModel):
    """
    Category weights for proportional sampling during training.
    
    Each weight is a float between 0.0 and 1.0 representing the probability
    of selecting that category when generating training samples.
    
    Example: {voice: 0.4, music: 0.3, documentary: 0.2, conference: 0.1, custom: 0.0}
    """
    
    voice: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for voice samples")
    music: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for music samples")
    documentary: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for documentary samples")
    conference: float = Field(default=0.1, ge=0.0, le=1.0, description="Weight for conference samples")
    custom: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight for custom samples")
    
    @field_validator('voice', 'music', 'documentary', 'conference', 'custom')
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure weight is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return v
    
    def normalize(self) -> "CategoryWeights":
        """
        Normalize weights to sum to 1.0 (probability distribution).
        
        Returns a new CategoryWeights instance with normalized values.
        If all weights are 0, returns uniform distribution.
        """
        total = self.voice + self.music + self.documentary + self.conference + self.custom
        
        if total == 0.0:
            # Uniform distribution
            return CategoryWeights(
                voice=0.2,
                music=0.2,
                documentary=0.2,
                conference=0.2,
                custom=0.2
            )
        
        return CategoryWeights(
            voice=self.voice / total,
            music=self.music / total,
            documentary=self.documentary / total,
            conference=self.conference / total,
            custom=self.custom / total
        )
    
    def to_dict(self) -> dict[AudioCategory, float]:
        """Convert to dictionary mapping AudioCategory to weight."""
        return {
            AudioCategory.VOICE: self.voice,
            AudioCategory.MUSIC: self.music,
            AudioCategory.DOCUMENTARY: self.documentary,
            AudioCategory.CONFERENCE: self.conference,
            AudioCategory.CUSTOM: self.custom
        }
    
    class Config:
        json_schema_extra = {
            "example": {
                "voice": 0.4,
                "music": 0.3,
                "documentary": 0.2,
                "conference": 0.1,
                "custom": 0.0
            }
        }


class AudioLibraryEmptyError(Exception):
    """Raised when audio library is empty and fallback is disabled."""
    
    def __init__(self, category: Optional[AudioCategory] = None):
        if category:
            super().__init__(f"Audio library is empty for category: {category}")
        else:
            super().__init__("Audio library is empty")
        self.category = category
