"""SQLAlchemy ORM models for TimescaleDB storage."""

import uuid
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

    # Primary key (TimescaleDB hypertable) - composite key with timestamp
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=text("uuid_generate_v4()"),
    )

    # Foreign key to WebSDR station
    websdr_station_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)

    # Signal parameters
    frequency_hz = Column(BigInteger, nullable=False)
    signal_strength_db = Column(DOUBLE_PRECISION, nullable=True)
    snr_db = Column(DOUBLE_PRECISION, nullable=True)
    frequency_offset_hz = Column(Integer, nullable=True)

    # IQ data metadata
    iq_data_location = Column(String(512), nullable=True)
    iq_data_format = Column(String(50), nullable=True)
    iq_sample_rate = Column(Integer, nullable=True)
    iq_samples_count = Column(Integer, nullable=True)

    # Additional fields
    duration_seconds = Column(DOUBLE_PRECISION, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True, default=datetime.utcnow)

    # Compound indexes for common queries (existing from database)
    __table_args__ = (
        Index("idx_measurements_frequency", "frequency_hz", "timestamp"),
        Index("idx_measurements_time", "timestamp"),
        Index("idx_measurements_websdr_station", "websdr_station_id", "timestamp"),
    )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Measurement(id={self.id}, "
            f"websdr_station_id={self.websdr_station_id}, snr={self.snr_db}dB, "
            f"timestamp={self.timestamp})>"
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
                Expected keys matching database schema:
                  - websdr_id (int) - will be mapped to websdr_station_id via lookup
                  - frequency_hz (int)
                  - iq_sample_rate (int)
                  - iq_samples_count (int)
                  - timestamp (str or datetime)
                  - snr_db, frequency_offset_hz, signal_strength_db (floats)
                  - iq_data_location (str)
                  - duration_seconds (float)
            s3_path: Optional S3 path where IQ data is stored

        Returns:
            Measurement instance

        Raises:
            ValueError: If required fields are missing
            TypeError: If types cannot be converted
        """
        try:
            # Handle timestamp
            timestamp_str = measurement_dict.get("timestamp")
            if isinstance(timestamp_str, str):
                # Parse ISO format datetime
                if "T" in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = timestamp_str or datetime.now()

            return cls(
                timestamp=timestamp,
                id=str(uuid.uuid4()),
                websdr_station_id=measurement_dict.get("websdr_station_id"),  # UUID from station lookup
                frequency_hz=int(measurement_dict.get("frequency_hz", 0)),
                signal_strength_db=float(measurement_dict.get("signal_strength_db")) if measurement_dict.get("signal_strength_db") is not None else None,
                snr_db=float(measurement_dict.get("snr_db")) if measurement_dict.get("snr_db") is not None else None,
                frequency_offset_hz=int(measurement_dict.get("frequency_offset_hz", 0)),
                iq_data_location=measurement_dict.get("iq_data_location") or s3_path,
                iq_data_format=measurement_dict.get("iq_data_format", "npy"),
                iq_sample_rate=int(measurement_dict.get("iq_sample_rate", 0)),
                iq_samples_count=int(measurement_dict.get("iq_samples_count", 0)),
                duration_seconds=float(measurement_dict.get("duration_seconds")) if measurement_dict.get("duration_seconds") is not None else None,
                notes=measurement_dict.get("notes"),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to create Measurement from dict: {str(e)}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert measurement to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "websdr_station_id": self.websdr_station_id,
            "frequency_hz": self.frequency_hz,
            "signal_strength_db": self.signal_strength_db,
            "snr_db": self.snr_db,
            "frequency_offset_hz": self.frequency_offset_hz,
            "iq_data_location": self.iq_data_location,
            "iq_data_format": self.iq_data_format,
            "iq_sample_rate": self.iq_sample_rate,
            "iq_samples_count": self.iq_samples_count,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
