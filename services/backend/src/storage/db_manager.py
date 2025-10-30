"""Database management utilities for TimescaleDB operations."""

import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator, Tuple
from datetime import datetime, timedelta

from sqlalchemy import create_engine, select, func, and_, desc
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

try:
    from ..config import settings
except ImportError:
    # For testing
    from config import settings

try:
    from ..models.db import Measurement, WebSDRStation, SDRProfile, Base
except ImportError:
    # For testing
    from models.db import Measurement, WebSDRStation, SDRProfile, Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations for TimescaleDB."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager."""
        self.database_url = database_url or settings.database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine."""
        try:
            # Build connect_args based on database URL
            connect_args = {}
            poolclass = NullPool  # Default: no connection pooling
            
            if "postgresql" in self.database_url or "postgres" in self.database_url:
                connect_args = {"options": "-c timezone=utc"}
            elif "sqlite" in self.database_url:
                connect_args = {"check_same_thread": False}
                # SQLite in-memory needs StaticPool to maintain the connection
                if ":memory:" in self.database_url:
                    from sqlalchemy.pool import StaticPool
                    poolclass = StaticPool
            
            self.engine = create_engine(
                self.database_url,
                echo=False,
                poolclass=poolclass,
                connect_args=connect_args if connect_args else {}
            )
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=False
            )
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def create_tables(self) -> bool:
        """Create all tables in the database."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            return False
    
    def check_connection(self) -> bool:
        """Check if database connection is working."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(select(1))
                result.close()
            logger.debug("Database connection check successful")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database session."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error, rolling back: {e}")
            raise
        finally:
            session.close()
    
    def insert_measurement(
        self,
        task_id: str,
        measurement_dict: Dict[str, Any],
        s3_path: Optional[str] = None
    ) -> Optional[int]:
        """Insert a single measurement into the database."""
        try:
            with self.get_session() as session:
                measurement = Measurement.from_measurement_dict(
                    task_id=task_id,
                    measurement_dict=measurement_dict,
                    s3_path=s3_path
                )
                session.add(measurement)
                session.flush()
                meas_id = measurement.id
                logger.debug(f"Inserted measurement {meas_id} for task {task_id}")
                return meas_id
        except IntegrityError as e:
            logger.warning(f"Integrity error inserting measurement: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inserting measurement: {e}")
            return None
    
    def insert_measurements_bulk(
        self,
        task_id: str,
        measurements_list: List[Dict[str, Any]],
        s3_paths: Optional[Dict[int, str]] = None
    ) -> Tuple[int, int]:
        """Bulk insert measurements into the database."""
        successful = 0
        failed = 0
        
        try:
            with self.get_session() as session:
                for measurement_dict in measurements_list:
                    try:
                        websdr_id = measurement_dict.get("websdr_id")
                        s3_path = s3_paths.get(websdr_id) if s3_paths else None
                        
                        measurement = Measurement.from_measurement_dict(
                            task_id=task_id,
                            measurement_dict=measurement_dict,
                            s3_path=s3_path
                        )
                        session.add(measurement)
                        successful += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Skipping invalid measurement for WebSDR "
                            f"{measurement_dict.get('websdr_id')}: {e}"
                        )
                        failed += 1
                
                session.commit()
                logger.info(
                    f"Bulk insert completed: {successful} successful, {failed} failed"
                )
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            failed += len(measurements_list) - successful
        
        return successful, failed
    
    def get_recent_measurements(
        self,
        task_id: Optional[str] = None,
        websdr_id: Optional[int] = None,
        limit: int = 100,
        hours_back: int = 24
    ) -> List[Measurement]:
        """Get recent measurements from the database."""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
                
                query = select(Measurement).where(
                    Measurement.timestamp_utc >= cutoff_time
                )
                
                if task_id:
                    query = query.where(Measurement.task_id == task_id)
                if websdr_id:
                    query = query.where(Measurement.websdr_id == websdr_id)
                
                query = query.order_by(desc(Measurement.timestamp_utc)).limit(limit)
                results = session.execute(query).scalars().all()
                logger.debug(f"Retrieved {len(results)} recent measurements")
                return results
        except Exception as e:
            logger.error(f"Error retrieving recent measurements: {e}")
            return []
    
    def get_session_measurements(self, task_id: str) -> Dict[int, List[Measurement]]:
        """Get all measurements for a specific session/task."""
        try:
            with self.get_session() as session:
                query = select(Measurement).where(
                    Measurement.task_id == task_id
                ).order_by(Measurement.websdr_id, desc(Measurement.timestamp_utc))
                
                results = session.execute(query).scalars().all()
                
                # Group by websdr_id
                grouped = {}
                for measurement in results:
                    if measurement.websdr_id not in grouped:
                        grouped[measurement.websdr_id] = []
                    grouped[measurement.websdr_id].append(measurement)
                
                logger.debug(
                    f"Retrieved {len(results)} measurements for task {task_id} "
                    f"from {len(grouped)} WebSDRs"
                )
                return grouped
        except Exception as e:
            logger.error(f"Error retrieving session measurements: {e}")
            return {}
    
    def get_snr_statistics(
        self,
        task_id: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[int, Dict[str, float]]:
        """Get SNR statistics grouped by WebSDR."""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
                
                query = select(
                    Measurement.websdr_id,
                    func.avg(Measurement.snr_db).label("avg_snr"),
                    func.min(Measurement.snr_db).label("min_snr"),
                    func.max(Measurement.snr_db).label("max_snr"),
                    func.count(Measurement.id).label("count")
                ).where(Measurement.timestamp_utc >= cutoff_time)
                
                if task_id:
                    query = query.where(Measurement.task_id == task_id)
                
                query = query.group_by(Measurement.websdr_id)
                results = session.execute(query).all()
                
                stats = {}
                for row in results:
                    stats[row.websdr_id] = {
                        "avg_snr_db": float(row.avg_snr) if row.avg_snr else None,
                        "min_snr_db": float(row.min_snr) if row.min_snr else None,
                        "max_snr_db": float(row.max_snr) if row.max_snr else None,
                        "count": row.count
                    }
                
                logger.debug(f"Retrieved SNR statistics for {len(stats)} WebSDRs")
                return stats
        except Exception as e:
            logger.error(f"Error retrieving SNR statistics: {e}")
            return {}
    
    def get_all_websdrs(self) -> List[WebSDRStation]:
        """Get all WebSDR stations from the database."""
        try:
            with self.get_session() as session:
                query = select(WebSDRStation).order_by(WebSDRStation.name)
                results = session.execute(query).scalars().all()
                logger.debug(f"Retrieved {len(results)} WebSDR stations")
                return list(results)
        except Exception as e:
            logger.error(f"Error retrieving WebSDR stations: {e}")
            return []
    
    def get_active_websdrs(self) -> List[WebSDRStation]:
        """Get only active WebSDR stations from the database."""
        try:
            with self.get_session() as session:
                query = select(WebSDRStation).where(
                    WebSDRStation.is_active == True
                ).order_by(WebSDRStation.name)
                results = session.execute(query).scalars().all()
                logger.debug(f"Retrieved {len(results)} active WebSDR stations")
                return list(results)
        except Exception as e:
            logger.error(f"Error retrieving active WebSDR stations: {e}")
            return []
    
    def get_websdr_by_name(self, name: str) -> Optional[WebSDRStation]:
        """Get a WebSDR station by name."""
        try:
            with self.get_session() as session:
                query = select(WebSDRStation).where(WebSDRStation.name == name)
                result = session.execute(query).scalar_one_or_none()
                return result
        except Exception as e:
            logger.error(f"Error retrieving WebSDR by name '{name}': {e}")
            return None
    
    def upsert_websdr_from_health_check(
        self, 
        name: str,
        url: str,
        health_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Insert or update WebSDR station from health-check JSON.
        
        Args:
            name: Station name (unique identifier)
            url: Base URL of the WebSDR
            health_data: Parsed JSON from health-check endpoint
                Expected structure:
                {
                    "receiver": {
                        "name": str,
                        "admin": str (email),
                        "gps": {"lat": float, "lon": float},
                        "asl": int,
                        "location": str
                    },
                    "sdrs": [
                        {
                            "name": str (e.g., "A)"),
                            "type": str,
                            "profiles": [
                                {
                                    "name": str,
                                    "center_freq": int (Hz),
                                    "sample_rate": int (Hz)
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                }
        
        Returns:
            WebSDR station UUID as string if successful, None otherwise
        """
        try:
            with self.get_session() as session:
                # Extract receiver info
                receiver = health_data.get("receiver", {})
                gps = receiver.get("gps", {})
                
                latitude = gps.get("lat")
                longitude = gps.get("lon")
                
                if latitude is None or longitude is None:
                    logger.error(f"Missing GPS coordinates for {name}")
                    return None
                
                # Check if station exists
                existing = session.execute(
                    select(WebSDRStation).where(WebSDRStation.name == name)
                ).scalar_one_or_none()
                
                if existing:
                    # Update existing station
                    existing.url = url
                    existing.latitude = float(latitude)
                    existing.longitude = float(longitude)
                    existing.admin_email = receiver.get("admin")
                    existing.location_description = receiver.get("location")
                    existing.altitude_asl = receiver.get("asl")
                    existing.updated_at = datetime.utcnow()
                    station = existing
                    logger.info(f"Updated WebSDR station: {name}")
                else:
                    # Create new station
                    station = WebSDRStation(
                        name=name,
                        url=url,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        admin_email=receiver.get("admin"),
                        location_description=receiver.get("location"),
                        altitude_asl=receiver.get("asl"),
                        is_active=True
                    )
                    session.add(station)
                    session.flush()
                    logger.info(f"Created new WebSDR station: {name}")
                
                station_id = station.id
                
                # Process SDR profiles and collect frequency ranges
                sdrs = health_data.get("sdrs", [])
                all_frequencies = []
                
                for sdr in sdrs:
                    sdr_name = sdr.get("name", "Unknown")
                    sdr_type = sdr.get("type")
                    profiles = sdr.get("profiles", [])
                    
                    for profile in profiles:
                        profile_name = profile.get("name", "Unknown")
                        center_freq = profile.get("center_freq")
                        sample_rate = profile.get("sample_rate")
                        
                        if center_freq is None or sample_rate is None:
                            logger.warning(
                                f"Skipping profile '{profile_name}' - missing freq/sample_rate"
                            )
                            continue
                        
                        all_frequencies.append(int(center_freq))
                        
                        # Check if profile exists
                        existing_profile = session.execute(
                            select(SDRProfile).where(
                                and_(
                                    SDRProfile.websdr_station_id == station_id,
                                    SDRProfile.sdr_name == sdr_name,
                                    SDRProfile.profile_name == profile_name
                                )
                            )
                        ).scalar_one_or_none()
                        
                        if existing_profile:
                            # Update existing profile
                            existing_profile.center_freq_hz = int(center_freq)
                            existing_profile.sample_rate_hz = int(sample_rate)
                            existing_profile.sdr_type = sdr_type
                            existing_profile.updated_at = datetime.utcnow()
                        else:
                            # Create new profile
                            new_profile = SDRProfile(
                                websdr_station_id=station_id,
                                sdr_name=sdr_name,
                                sdr_type=sdr_type,
                                profile_name=profile_name,
                                center_freq_hz=int(center_freq),
                                sample_rate_hz=int(sample_rate),
                                is_active=True
                            )
                            session.add(new_profile)
                
                # Update frequency ranges based on collected frequencies
                if all_frequencies:
                    station.frequency_min_hz = min(all_frequencies)
                    station.frequency_max_hz = max(all_frequencies)
                    logger.info(
                        f"Updated frequency range: {station.frequency_min_hz} - {station.frequency_max_hz} Hz"
                    )
                
                session.commit()
                logger.info(
                    f"Successfully upserted WebSDR {name} with {len(sdrs)} SDR profiles"
                )
                return str(station_id)
                
        except Exception as e:
            logger.error(f"Error upserting WebSDR from health-check: {e}")
            return None
    
    def get_sdr_profiles(self, websdr_station_id: str) -> List[SDRProfile]:
        """Get all SDR profiles for a specific WebSDR station."""
        try:
            with self.get_session() as session:
                query = select(SDRProfile).where(
                    SDRProfile.websdr_station_id == websdr_station_id
                ).order_by(SDRProfile.sdr_name, SDRProfile.center_freq_hz)
                results = session.execute(query).scalars().all()
                logger.debug(f"Retrieved {len(results)} SDR profiles for station {websdr_station_id}")
                return list(results)
        except Exception as e:
            logger.error(f"Error retrieving SDR profiles: {e}")
            return []
    
    def create_websdr(
        self,
        name: str,
        url: str,
        latitude: float,
        longitude: float,
        location_description: Optional[str] = None,
        country: Optional[str] = "Italy",
        admin_email: Optional[str] = None,
        altitude_asl: Optional[int] = None,
        timeout_seconds: int = 30,
        retry_count: int = 3,
        is_active: bool = True
    ) -> Optional[WebSDRStation]:
        """
        Create a new WebSDR station.
        
        Args:
            name: Unique station name
            url: WebSDR base URL
            latitude: GPS latitude (-90 to 90)
            longitude: GPS longitude (-180 to 180)
            location_description: Human-readable location
            country: Country name
            admin_email: Administrator email
            altitude_asl: Altitude above sea level (meters)
            timeout_seconds: Connection timeout
            retry_count: Number of retry attempts
            is_active: Whether station is active
        
        Returns:
            Created WebSDRStation object or None on error
        """
        try:
            with self.get_session() as session:
                # Check if name already exists
                existing = session.execute(
                    select(WebSDRStation).where(WebSDRStation.name == name)
                ).scalar_one_or_none()
                
                if existing:
                    logger.error(f"WebSDR with name '{name}' already exists")
                    return None
                
                # Create new station
                new_station = WebSDRStation(
                    name=name,
                    url=url,
                    latitude=latitude,
                    longitude=longitude,
                    location_description=location_description,
                    country=country,
                    admin_email=admin_email,
                    altitude_asl=altitude_asl,
                    timeout_seconds=timeout_seconds,
                    retry_count=retry_count,
                    is_active=is_active
                )
                
                session.add(new_station)
                session.commit()
                session.refresh(new_station)
                
                logger.info(f"Created WebSDR station: {name}")
                return new_station
                
        except Exception as e:
            logger.error(f"Error creating WebSDR station: {e}")
            return None
    
    def update_websdr(
        self,
        station_id: str,
        **kwargs
    ) -> Optional[WebSDRStation]:
        """
        Update an existing WebSDR station.
        
        Args:
            station_id: UUID of the station to update
            **kwargs: Fields to update (name, url, latitude, longitude, etc.)
        
        Returns:
            Updated WebSDRStation object or None on error
        """
        try:
            with self.get_session() as session:
                station = session.get(WebSDRStation, station_id)
                
                if not station:
                    logger.error(f"WebSDR station {station_id} not found")
                    return None
                
                # If updating name, check for uniqueness
                if 'name' in kwargs and kwargs['name'] != station.name:
                    existing = session.execute(
                        select(WebSDRStation).where(WebSDRStation.name == kwargs['name'])
                    ).scalar_one_or_none()
                    
                    if existing:
                        logger.error(f"WebSDR with name '{kwargs['name']}' already exists")
                        return None
                
                # Update allowed fields
                allowed_fields = {
                    'name', 'url', 'latitude', 'longitude', 'location_description',
                    'country', 'admin_email', 'altitude_asl', 'timeout_seconds',
                    'retry_count', 'is_active'
                }
                
                for key, value in kwargs.items():
                    if key in allowed_fields:
                        setattr(station, key, value)
                
                session.commit()
                session.refresh(station)
                
                logger.info(f"Updated WebSDR station: {station.name}")
                return station
                
        except Exception as e:
            logger.error(f"Error updating WebSDR station: {e}")
            return None
    
    def delete_websdr(self, station_id: str, soft_delete: bool = True) -> bool:
        """
        Delete a WebSDR station (soft delete by default).
        
        Args:
            station_id: UUID of the station to delete
            soft_delete: If True, set is_active=False; if False, hard delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                station = session.get(WebSDRStation, station_id)
                
                if not station:
                    logger.error(f"WebSDR station {station_id} not found")
                    return False
                
                if soft_delete:
                    # Soft delete: just deactivate
                    station.is_active = False
                    session.commit()
                    logger.info(f"Soft deleted (deactivated) WebSDR station: {station.name}")
                else:
                    # Hard delete: remove from database
                    session.delete(station)
                    session.commit()
                    logger.info(f"Hard deleted WebSDR station: {station.name}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error deleting WebSDR station: {e}")
            return False
    
    def get_websdr_by_id(self, station_id: str) -> Optional[WebSDRStation]:
        """Get a WebSDR station by UUID."""
        try:
            with self.get_session() as session:
                station = session.get(WebSDRStation, station_id)
                if station:
                    # Detach from session to avoid lazy loading issues
                    session.expunge(station)
                return station
        except Exception as e:
            logger.error(f"Error retrieving WebSDR by ID '{station_id}': {e}")
            return None
    
    def close(self) -> None:
        """Close database engine and cleanup resources."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database engine closed")
        except Exception as e:
            logger.error(f"Error closing database engine: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db_manager() -> None:
    """Reset global database manager (useful for testing)."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = None
