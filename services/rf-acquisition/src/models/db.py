"""SQLAlchemy ORM models for TimescaleDB storage."""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, BigInteger, Index
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION

Base = declarative_base()


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
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=datetime.utcnow
    )
    
    # Computed metrics
    snr_db = Column(DOUBLE_PRECISION, nullable=True)
    frequency_offset_hz = Column(DOUBLE_PRECISION, nullable=True)
    power_dbm = Column(DOUBLE_PRECISION, nullable=True)
    
    # Storage reference
    s3_path = Column(Text, nullable=True)
    
    # Compound indexes for common queries
    __table_args__ = (
        Index('idx_measurements_websdr_time', 'websdr_id', 'timestamp_utc'),
        Index('idx_measurements_task_time', 'task_id', 'timestamp_utc'),
        Index('idx_measurements_frequency', 'frequency_mhz', 'timestamp_utc'),
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
        measurement_dict: Dict[str, Any],
        s3_path: Optional[str] = None,
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
                    timestamp_utc = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
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
            raise ValueError(
                f"Failed to create Measurement from dict: {str(e)}"
            ) from e
    
    def to_dict(self) -> Dict[str, Any]:
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
