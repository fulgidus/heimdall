"""SQLAlchemy ORM models for TimescaleDB storage."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class WebSDRStation(Base):
    """
    WebSDR receiver station configuration.

    Each row represents a configured WebSDR receiver with geographic location
    and operational parameters for RF data acquisition.
    """

    __tablename__ = "websdr_stations"
    __table_args__ = {"schema": "heimdall"}

    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))

    # Station identification
    name = Column(String(255), nullable=False, unique=True)
    url = Column(String(512), nullable=False)
    country = Column(String(100), nullable=True)

    # Geographic location (required for triangulation)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # Frequency capabilities
    frequency_min_hz = Column(BigInteger, nullable=True)
    frequency_max_hz = Column(BigInteger, nullable=True)

    # Operational parameters
    is_active = Column(Boolean, default=True, nullable=False)
    api_type = Column(String(50), default="http", nullable=True)
    rate_limit_ms = Column(Integer, default=1000, nullable=True)
    timeout_seconds = Column(Integer, default=30, nullable=True)
    retry_count = Column(Integer, default=3, nullable=True)

    # Extended metadata (from health-check)
    admin_email = Column(String(255), nullable=True)
    location_description = Column(Text, nullable=True)
    altitude_asl = Column(Integer, nullable=True)

    # Additional metadata
    notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<WebSDRStation(id={self.id}, name='{self.name}', "
            f"lat={self.latitude}, lon={self.longitude}, "
            f"active={self.is_active})>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert WebSDR station to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "url": self.url,
            "location_name": f"{self.name}, {self.country}" if self.country else self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "is_active": self.is_active,
            "timeout_seconds": self.timeout_seconds,
            "frequency_min_hz": self.frequency_min_hz,
            "frequency_max_hz": self.frequency_max_hz,
        }


class SDRProfile(Base):
    """
    SDR receiver profile representing a specific frequency/mode configuration.

    Populated from WebSDR health-check JSON (receiver.sdrs[].profiles[]).
    One WebSDR station can have multiple SDR receivers, each with multiple profiles.
    """

    __tablename__ = "sdr_profiles"
    __table_args__ = {"schema": "heimdall"}

    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))

    # Foreign key to websdr_stations
    websdr_station_id = Column(PG_UUID(as_uuid=True), nullable=False, index=True)

    # SDR identification (from health-check JSON)
    sdr_name = Column(String(50), nullable=False)  # e.g., "A)", "B)", "C)"
    sdr_type = Column(String(100), nullable=True)  # e.g., "RtlSdrSource"

    # Profile details
    profile_name = Column(String(255), nullable=False)  # e.g., "2m [144.00-146.00 Mhz]"
    center_freq_hz = Column(BigInteger, nullable=False)
    sample_rate_hz = Column(Integer, nullable=False)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<SDRProfile(id={self.id}, sdr={self.sdr_name}, "
            f"profile='{self.profile_name}', "
            f"freq={self.center_freq_hz}Hz)>"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert SDR profile to dictionary."""
        return {
            "id": str(self.id),
            "websdr_station_id": str(self.websdr_station_id),
            "sdr_name": self.sdr_name,
            "sdr_type": self.sdr_type,
            "profile_name": self.profile_name,
            "center_freq_hz": self.center_freq_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "is_active": self.is_active,
        }


class Measurement(Base):
    """
    Time-series measurement record optimized for TimescaleDB.

    Each row represents a measurement from one WebSDR receiver at a specific time.
    Supports fast queries on frequency, time, and receiver ID.
    """

    __tablename__ = "measurements"

    # Primary key (TimescaleDB hypertable)
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Acquisition metadata
    task_id = Column(String(36), nullable=False, index=True)
    websdr_id = Column(Integer, nullable=False, index=True)

    # Signal parameters
    frequency_mhz = Column(DOUBLE_PRECISION, nullable=False)
    sample_rate_khz = Column(DOUBLE_PRECISION, nullable=False)
    samples_count = Column(Integer, nullable=False)

    # Timestamp (TimescaleDB time dimension)
    timestamp_utc = Column(
        DateTime(timezone=True), nullable=False, index=True, default=datetime.utcnow
    )

    # Computed metrics
    snr_db = Column(DOUBLE_PRECISION, nullable=True)
    frequency_offset_hz = Column(DOUBLE_PRECISION, nullable=True)
    power_dbm = Column(DOUBLE_PRECISION, nullable=True)

    # Storage reference
    s3_path = Column(Text, nullable=True)

    # Compound indexes for common queries
    __table_args__ = (
        Index("idx_measurements_websdr_time", "websdr_id", "timestamp_utc"),
        Index("idx_measurements_task_time", "task_id", "timestamp_utc"),
        Index("idx_measurements_frequency", "frequency_mhz", "timestamp_utc"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Measurement(id={self.id}, task_id={self.task_id}, "
            f"websdr_id={self.websdr_id}, snr={self.snr_db}dB, "
            f"timestamp={self.timestamp_utc})>"
        )

    @classmethod
    def from_measurement_dict(
        cls,
        task_id: str,
        measurement_dict: dict[str, Any],
        s3_path: str | None = None,
    ) -> "Measurement":
        """
        Create a Measurement instance from measurement dictionary.

        Args:
            task_id: Acquisition task ID
            measurement_dict: Dictionary containing measurement data
                Expected keys:
                  - websdr_id (int)
                  - frequency_mhz (float)
                  - sample_rate_khz (float)
                  - samples_count (int)
                  - timestamp_utc (str or datetime)
                  - metrics (dict with snr_db, frequency_offset_hz, power_dbm)
            s3_path: Optional S3 path where IQ data is stored

        Returns:
            Measurement instance

        Raises:
            ValueError: If required fields are missing
            TypeError: If types cannot be converted
        """
        try:
            # Extract and validate required fields
            websdr_id = int(measurement_dict.get("websdr_id"))
            frequency_mhz = float(measurement_dict.get("frequency_mhz"))
            sample_rate_khz = float(measurement_dict.get("sample_rate_khz"))
            samples_count = int(measurement_dict.get("samples_count"))

            # Handle timestamp
            timestamp_str = measurement_dict.get("timestamp_utc")
            if isinstance(timestamp_str, str):
                # Parse ISO format datetime
                if "T" in timestamp_str:
                    timestamp_utc = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp_utc = datetime.fromisoformat(timestamp_str)
            else:
                timestamp_utc = timestamp_str or datetime.utcnow()

            # Extract metrics
            metrics = measurement_dict.get("metrics", {})
            snr_db = metrics.get("snr_db")
            frequency_offset_hz = metrics.get("frequency_offset_hz")
            power_dbm = metrics.get("power_dbm")

            # Convert to float if present
            if snr_db is not None:
                snr_db = float(snr_db)
            if frequency_offset_hz is not None:
                frequency_offset_hz = float(frequency_offset_hz)
            if power_dbm is not None:
                power_dbm = float(power_dbm)

            return cls(
                task_id=task_id,
                websdr_id=websdr_id,
                frequency_mhz=frequency_mhz,
                sample_rate_khz=sample_rate_khz,
                samples_count=samples_count,
                timestamp_utc=timestamp_utc,
                snr_db=snr_db,
                frequency_offset_hz=frequency_offset_hz,
                power_dbm=power_dbm,
                s3_path=s3_path,
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to create Measurement from dict: {str(e)}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert measurement to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "websdr_id": self.websdr_id,
            "frequency_mhz": self.frequency_mhz,
            "sample_rate_khz": self.sample_rate_khz,
            "samples_count": self.samples_count,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "snr_db": self.snr_db,
            "frequency_offset_hz": self.frequency_offset_hz,
            "power_dbm": self.power_dbm,
            "s3_path": self.s3_path,
        }
