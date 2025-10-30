"""WebSDR configuration and data models."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, HttpUrl, field_serializer


class WebSDRConfig(BaseModel):
    """Configuration for a single WebSDR receiver."""
    
    id: int = Field(..., description="Unique identifier for the WebSDR")
    name: str = Field(..., description="Friendly name (e.g., 'F5LEN Toulouse')")
    url: HttpUrl = Field(..., description="Base URL of WebSDR (e.g., http://websdr.f5len.net:8901)")
    location_name: str = Field(..., description="Location description")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    is_active: bool = Field(default=True, description="Whether this receiver is currently active")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "F5LEN Toulouse",
                "url": "http://websdr.f5len.net:8901",
                "location_name": "Toulouse, France",
                "latitude": 43.5,
                "longitude": 1.4,
                "is_active": True,
                "timeout_seconds": 30,
                "retry_count": 3
            }
        }


class IQDataPoint(BaseModel):
    """Single IQ sample."""
    i: float = Field(..., description="In-phase component")
    q: float = Field(..., description="Quadrature component")


class AcquisitionRequest(BaseModel):
    """Request to acquire IQ data from WebSDRs."""
    
    frequency_mhz: float = Field(..., gt=0, description="Frequency in MHz")
    duration_seconds: float = Field(..., gt=0, le=300, description="Duration in seconds (max 5 minutes)")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Acquisition start time (UTC)")
    websdrs: Optional[list[int]] = Field(default=None, description="Specific WebSDR IDs to use (None = all active)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "frequency_mhz": 145.5,
                "duration_seconds": 10,
                "start_time": "2025-10-22T10:00:00Z",
                "websdrs": None
            }
        }


class SignalMetrics(BaseModel):
    """Computed signal metrics for a measurement."""
    
    snr_db: float = Field(..., description="Signal-to-Noise Ratio in dB")
    psd_dbm: float = Field(..., description="Power Spectral Density in dBm/Hz")
    frequency_offset_hz: float = Field(..., description="Frequency offset from target in Hz")
    signal_power_dbm: float = Field(..., description="Signal power in dBm")
    noise_power_dbm: float = Field(..., description="Noise power in dBm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "snr_db": 15.5,
                "psd_dbm": -80.2,
                "frequency_offset_hz": 50.0,
                "signal_power_dbm": -50.0,
                "noise_power_dbm": -65.5
            }
        }


class MeasurementRecord(BaseModel):
    """Single measurement from one WebSDR receiver."""
    
    websdrs_id: int = Field(..., description="Reference to WebSDR receiver")
    frequency_mhz: float = Field(..., description="Target frequency in MHz")
    sample_rate_khz: float = Field(default=12.5, description="Sample rate in kHz")
    samples_count: int = Field(..., description="Total number of IQ samples")
    timestamp_utc: datetime = Field(..., description="Timestamp of measurement (UTC)")
    metrics: SignalMetrics = Field(..., description="Computed signal metrics")
    iq_data_path: str = Field(..., description="Path to IQ data in MinIO (e.g., s3://bucket/key)")
    metadata_json: Optional[dict] = Field(default=None, description="Additional metadata")


class AcquisitionTaskResponse(BaseModel):
    """Response when triggering an acquisition task."""
    
    task_id: str = Field(..., description="Celery task ID for tracking")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Human-readable message")
    frequency_mhz: float = Field(..., description="Requested frequency")
    websdrs_count: int = Field(..., description="Number of WebSDRs being used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "c2f8e4a0-9d5f-4c3b-a1e2-7f9c8b6d5a4e",
                "status": "PENDING",
                "message": "Acquisition task queued for 7 WebSDR receivers",
                "frequency_mhz": 145.5,
                "websdrs_count": 7
            }
        }


class AcquisitionStatusResponse(BaseModel):
    """Status of an ongoing acquisition task."""
    
    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Task status (PENDING, PROGRESS, SUCCESS, FAILURE, REVOKED)")
    progress: float = Field(default=0.0, ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Status message")
    measurements_collected: int = Field(default=0, description="Number of successful measurements")
    errors: Optional[list[str]] = Field(default=None, description="Error messages if any")
    result: Optional[dict] = Field(default=None, description="Result data when complete")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "c2f8e4a0-9d5f-4c3b-a1e2-7f9c8b6d5a4e",
                "status": "PROGRESS",
                "progress": 57.14,
                "message": "Fetching from 4/7 WebSDRs",
                "measurements_collected": 4,
                "errors": None,
                "result": None
            }
        }


class WebSDRFetcherConfig(BaseModel):
    """Configuration for WebSDR fetcher behavior."""
    
    timeout_seconds: int = Field(default=30, description="Individual request timeout")
    retry_count: int = Field(default=3, description="Number of retries")
    concurrent_requests: int = Field(default=7, description="Max concurrent requests")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")


# ============================================================================
# CRUD Models for WebSDR Management
# ============================================================================

class WebSDRCreateRequest(BaseModel):
    """Request to create a new WebSDR station."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Unique station name")
    url: str = Field(..., description="WebSDR base URL (e.g., http://websdr.example.com:8073)")
    latitude: float = Field(..., ge=-90, le=90, description="GPS latitude")
    longitude: float = Field(..., ge=-180, le=180, description="GPS longitude")
    location_description: Optional[str] = Field(None, max_length=255, description="Human-readable location")
    country: Optional[str] = Field("Italy", max_length=100, description="Country name")
    admin_email: Optional[str] = Field(None, max_length=255, description="Administrator email")
    altitude_asl: Optional[int] = Field(None, description="Altitude above sea level (meters)")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Connection timeout")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retry attempts")
    is_active: bool = Field(True, description="Whether station is active")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "IW2MXM Milano",
                "url": "http://websdr.iw2mxm.it:8073",
                "latitude": 45.464,
                "longitude": 9.188,
                "location_description": "Milano, Italy",
                "country": "Italy",
                "admin_email": "admin@iw2mxm.it",
                "altitude_asl": 120,
                "timeout_seconds": 30,
                "retry_count": 3,
                "is_active": True
            }
        }


class WebSDRUpdateRequest(BaseModel):
    """Request to update an existing WebSDR station."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Station name")
    url: Optional[str] = Field(None, description="WebSDR base URL")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="GPS latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="GPS longitude")
    location_description: Optional[str] = Field(None, max_length=255, description="Location description")
    country: Optional[str] = Field(None, max_length=100, description="Country name")
    admin_email: Optional[str] = Field(None, max_length=255, description="Administrator email")
    altitude_asl: Optional[int] = Field(None, description="Altitude ASL (meters)")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=300, description="Connection timeout")
    retry_count: Optional[int] = Field(None, ge=0, le=10, description="Retry attempts")
    is_active: Optional[bool] = Field(None, description="Active status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "IW2MXM Milano - Updated",
                "is_active": True,
                "timeout_seconds": 45
            }
        }


class WebSDRFetchInfoRequest(BaseModel):
    """Request to fetch WebSDR information from URL."""
    
    url: str = Field(..., description="WebSDR base URL to fetch info from")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "http://websdr.example.com:8073"
            }
        }


class WebSDRFetchInfoResponse(BaseModel):
    """Response containing WebSDR information fetched from URL."""
    
    receiver_name: Optional[str] = Field(None, description="Receiver name from status.json")
    location: Optional[str] = Field(None, description="Location from status.json")
    latitude: Optional[float] = Field(None, description="GPS latitude")
    longitude: Optional[float] = Field(None, description="GPS longitude")
    altitude_asl: Optional[int] = Field(None, description="Altitude ASL in meters")
    admin_email: Optional[str] = Field(None, description="Administrator email")
    frequency_min_hz: Optional[int] = Field(None, description="Minimum frequency in Hz")
    frequency_max_hz: Optional[int] = Field(None, description="Maximum frequency in Hz")
    sdr_count: int = Field(default=0, description="Number of SDR devices")
    profile_count: int = Field(default=0, description="Total number of profiles")
    success: bool = Field(..., description="Whether fetch was successful")
    error_message: Optional[str] = Field(None, description="Error message if fetch failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "receiver_name": "IW2MXM Milano",
                "location": "Milano, Italy",
                "latitude": 45.464,
                "longitude": 9.188,
                "altitude_asl": 120,
                "admin_email": "admin@example.com",
                "frequency_min_hz": 144000000,
                "frequency_max_hz": 146000000,
                "sdr_count": 2,
                "profile_count": 4,
                "success": True,
                "error_message": None
            }
        }


class WebSDRResponse(BaseModel):
    """Response model for WebSDR station."""
    
    id: UUID = Field(..., description="Station UUID")
    name: str = Field(..., description="Station name")
    url: str = Field(..., description="WebSDR base URL")
    latitude: float = Field(..., description="GPS latitude")
    longitude: float = Field(..., description="GPS longitude")
    location_description: Optional[str] = Field(None, description="Location description")
    country: Optional[str] = Field(None, description="Country")
    admin_email: Optional[str] = Field(None, description="Administrator email")
    altitude_asl: Optional[int] = Field(None, description="Altitude ASL")
    frequency_min_hz: Optional[int] = Field(None, description="Minimum frequency in Hz")
    frequency_max_hz: Optional[int] = Field(None, description="Maximum frequency in Hz")
    timeout_seconds: int = Field(default=30, description="Connection timeout")
    retry_count: Optional[int] = Field(default=None, description="Retry count")
    is_active: bool = Field(..., description="Active status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @field_serializer('id')
    def serialize_uuid(self, value: UUID, _info) -> str:
        """Convert UUID to string for JSON serialization."""
        return str(value)
    
    class Config:
        from_attributes = True  # Enable ORM mode
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "IW2MXM Milano",
                "url": "http://websdr.iw2mxm.it:8073",
                "latitude": 45.464,
                "longitude": 9.188,
                "location_description": "Milano, Italy",
                "country": "Italy",
                "admin_email": "admin@iw2mxm.it",
                "altitude_asl": 120,
                "frequency_min_hz": 144000000,
                "frequency_max_hz": 146000000,
                "timeout_seconds": 30,
                "retry_count": 3,
                "is_active": True,
                "created_at": "2025-10-27T10:00:00Z",
                "updated_at": "2025-10-27T10:00:00Z"
            }
        }
