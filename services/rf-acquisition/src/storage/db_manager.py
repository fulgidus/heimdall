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
    from ..models.db import Measurement, Base
except ImportError:
    # For testing
    from models.db import Measurement, Base

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
