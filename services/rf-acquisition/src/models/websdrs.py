"""WebSDR configuration and data models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


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
