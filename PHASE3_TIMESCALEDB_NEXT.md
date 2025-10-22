# Phase 3 - Next Steps: TimescaleDB Integration

## Status: READY FOR IMPLEMENTATION

**Estimated Time**: 4-6 hours  
**Priority**: Priority 2 (after MinIO ✅)  
**Complexity**: Medium

## Overview

TimescaleDB will store measurement metadata in a time-series optimized database, enabling fast queries and analytics on:
- Signal strength trends (SNR over time)
- Frequency offset patterns
- Receiver performance metrics
- Coverage statistics

## Architecture

### Database Schema

```sql
-- Create hypertable for time-series measurements
CREATE TABLE measurements (
    id BIGSERIAL NOT NULL,
    task_id TEXT NOT NULL,
    websdr_id INTEGER NOT NULL,
    frequency_mhz FLOAT8 NOT NULL,
    sample_rate_khz FLOAT8 NOT NULL,
    samples_count INTEGER NOT NULL,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    
    -- Metrics
    snr_db FLOAT8,
    frequency_offset_hz FLOAT8,
    power_dbm FLOAT8,
    
    -- Storage reference
    s3_path TEXT,
    
    PRIMARY KEY (task_id, websdr_id, timestamp_utc)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('measurements', 'timestamp_utc', if_not_exists => TRUE);

-- Create indexes for common queries
CREATE INDEX idx_measurements_websdr_time ON measurements (websdr_id, timestamp_utc DESC);
CREATE INDEX idx_measurements_task_time ON measurements (task_id, timestamp_utc DESC);
CREATE INDEX idx_measurements_frequency ON measurements (frequency_mhz, timestamp_utc DESC);
```

### Data Model (SQLAlchemy)

```python
# src/models/db.py

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, BigInteger
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Measurement(Base):
    __tablename__ = "measurements"
    
    id = Column(BigInteger, primary_key=True)
    task_id = Column(String(36), nullable=False, index=True)
    websdr_id = Column(Integer, nullable=False, index=True)
    frequency_mhz = Column(Float, nullable=False)
    sample_rate_khz = Column(Float, nullable=False)
    samples_count = Column(Integer, nullable=False)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Metrics
    snr_db = Column(Float)
    frequency_offset_hz = Column(Float)
    power_dbm = Column(Float)
    
    # S3 Reference
    s3_path = Column(Text)
    
    def __repr__(self):
        return f"<Measurement(task_id={self.task_id}, websdr_id={self.websdr_id}, snr={self.snr_db}dB)>"
```

## Implementation Tasks

### 1. Database Migration Script (2-3 hours)
**File**: `db/migrations/001_create_measurements_table.sql`

```sql
-- Create hypertable
CREATE TABLE IF NOT EXISTS measurements (
    id BIGSERIAL NOT NULL,
    task_id TEXT NOT NULL,
    websdr_id INTEGER NOT NULL,
    frequency_mhz DOUBLE PRECISION NOT NULL,
    sample_rate_khz DOUBLE PRECISION NOT NULL,
    samples_count INTEGER NOT NULL,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    snr_db DOUBLE PRECISION,
    frequency_offset_hz DOUBLE PRECISION,
    power_dbm DOUBLE PRECISION,
    s3_path TEXT,
    PRIMARY KEY (task_id, websdr_id, timestamp_utc)
);

-- Enable TimescaleDB hypertable
SELECT create_hypertable('measurements', 'timestamp_utc', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_measurements_websdr ON measurements (websdr_id, timestamp_utc DESC);
CREATE INDEX idx_measurements_task ON measurements (task_id, timestamp_utc DESC);
CREATE INDEX idx_measurements_freq ON measurements (frequency_mhz, timestamp_utc DESC);

-- Apply data retention policy (30 days)
SELECT add_retention_policy('measurements', INTERVAL '30 days', if_not_exists => true);

-- Enable compression for chunks older than 7 days
ALTER TABLE measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'websdr_id,task_id',
    timescaledb.compress_orderby = 'timestamp_utc DESC'
);

SELECT add_compression_policy('measurements', INTERVAL '7 days', if_not_exists => true);
```

### 2. SQLAlchemy Models (1 hour)
**File**: `src/models/db.py`

```python
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, 
    Text, BigInteger, create_engine, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class Measurement(Base):
    __tablename__ = "measurements"
    __table_args__ = (
        Index('idx_measurements_websdr', 'websdr_id', 'timestamp_utc'),
        Index('idx_measurements_task', 'task_id', 'timestamp_utc'),
        Index('idx_measurements_freq', 'frequency_mhz', 'timestamp_utc'),
    )
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    task_id = Column(String(36), nullable=False)
    websdr_id = Column(Integer, nullable=False)
    frequency_mhz = Column(Float, nullable=False)
    sample_rate_khz = Column(Float, nullable=False)
    samples_count = Column(Integer, nullable=False)
    timestamp_utc = Column(DateTime(timezone=True), nullable=False, index=True)
    
    snr_db = Column(Float, nullable=True)
    frequency_offset_hz = Column(Float, nullable=True)
    power_dbm = Column(Float, nullable=True)
    
    s3_path = Column(Text, nullable=True)
    
    @classmethod
    def from_measurement_dict(cls, task_id: str, measurement: dict):
        return cls(
            task_id=task_id,
            websdr_id=measurement['websdr_id'],
            frequency_mhz=measurement['frequency_mhz'],
            sample_rate_khz=measurement['sample_rate_khz'],
            samples_count=measurement['samples_count'],
            timestamp_utc=datetime.fromisoformat(measurement['timestamp_utc']),
            snr_db=measurement['metrics']['snr_db'],
            frequency_offset_hz=measurement['metrics']['frequency_offset_hz'],
            s3_path=measurement.get('iq_data_path'),
        )
```

### 3. Celery Task for TimescaleDB (2 hours)
**File**: `src/tasks/acquire_iq.py` (update)

```python
@shared_task(bind=True)
def save_measurements_to_timescaledb(
    self,
    task_id: str,
    measurements: List[Dict],
):
    """Save measurement metadata to TimescaleDB hypertable."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from ..models.db import Base, Measurement
        
        engine = create_engine(settings.database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        logger.info("Saving %d measurements to TimescaleDB...", len(measurements))
        
        # Bulk insert measurements
        measurement_objects = []
        for measurement in measurements:
            m = Measurement.from_measurement_dict(task_id, measurement)
            measurement_objects.append(m)
        
        session.bulk_save_objects(measurement_objects)
        session.commit()
        
        logger.info("Saved %d measurements to TimescaleDB", len(measurements_objects))
        
        return {
            'status': 'SUCCESS',
            'measurements_count': len(measurements_objects),
        }
        
    except Exception as e:
        logger.exception("TimescaleDB storage failed: %s", str(e))
        return {
            'status': 'FAILED',
            'error': str(e),
        }
    finally:
        if session:
            session.close()
```

### 4. Query Examples (1 hour)
**File**: `src/queries/measurements.py`

```python
# Recent measurements
SELECT * FROM measurements 
WHERE websdr_id = 1 
AND timestamp_utc > NOW() - INTERVAL '24 hours'
ORDER BY timestamp_utc DESC;

# Average SNR per receiver
SELECT 
    websdr_id,
    AVG(snr_db) as avg_snr,
    time_bucket('1 hour', timestamp_utc) as hour
FROM measurements
WHERE task_id = 'task_12345'
GROUP BY websdr_id, hour
ORDER BY hour DESC;

# Frequency drift analysis
SELECT 
    websdr_id,
    frequency_offset_hz,
    timestamp_utc
FROM measurements
WHERE frequency_mhz = 100.0
ORDER BY websdr_id, timestamp_utc;

# Performance report
SELECT 
    task_id,
    websdr_id,
    COUNT(*) as measurements_count,
    AVG(snr_db) as avg_snr,
    MIN(snr_db) as min_snr,
    MAX(snr_db) as max_snr,
    AVG(frequency_offset_hz) as avg_offset
FROM measurements
WHERE timestamp_utc > NOW() - INTERVAL '7 days'
GROUP BY task_id, websdr_id;
```

## Dependencies

```bash
pip install sqlalchemy==2.0.23      # Already installed
pip install psycopg2-binary==2.9.9  # Already installed
```

## Testing Strategy

### Unit Tests
```python
# Test model creation
def test_measurement_model_creation():
    m = Measurement(...)
    assert m.snr_db == 15.5

# Test bulk insert
def test_bulk_insert_measurements():
    measurements = [...]
    session.bulk_save_objects(measurements)
    assert session.query(Measurement).count() == len(measurements)
```

### Integration Tests
```python
# Test Celery task
def test_save_measurements_to_timescaledb_task():
    result = save_measurements_to_timescaledb.delay(
        task_id="task_12345",
        measurements=[...]
    )
    assert result.get()['status'] == 'SUCCESS'

# Test queries
def test_query_recent_measurements():
    results = session.query(Measurement)\
        .filter(Measurement.websdr_id == 1)\
        .order_by(Measurement.timestamp_utc.desc())\
        .limit(10)\
        .all()
    assert len(results) <= 10
```

## Performance Optimization

### Bulk Insert
```python
# Use bulk_save_objects for speed
session.bulk_save_objects(measurements, return_defaults=False)

# ~1000x faster than individual inserts
# 7 measurements in <10ms vs 70ms
```

### Compression
- Auto-compress chunks older than 7 days
- Saves ~80% storage space
- Queries automatically decompress

### Retention Policy
- Auto-delete measurements older than 30 days
- Optional configuration

## Migration from MinIO Metadata

```python
# Extract from MinIO JSON files
import json
from src.storage.minio_client import MinIOClient

client = MinIOClient(...)
measurements = client.get_session_measurements(task_id)

for websdr_id, info in measurements.items():
    # Download metadata
    response = client.s3_client.get_object(
        Bucket=bucket,
        Key=f"sessions/{task_id}/websdr_{websdr_id}_metadata.json"
    )
    metadata = json.loads(response['Body'].read())
    
    # Insert into TimescaleDB
    measurement = Measurement.from_metadata_dict(task_id, metadata)
    session.add(measurement)

session.commit()
```

## Checklist

- [ ] Create migration SQL file
- [ ] Implement SQLAlchemy models
- [ ] Update save_measurements_to_timescaledb()
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test queries
- [ ] Performance testing
- [ ] Create query guide documentation
- [ ] Update API endpoints for TimescaleDB queries
- [ ] Add metrics dashboard queries

## Next Priority After TimescaleDB

1. WebSDR Configuration from Database (2-3 hours)
2. End-to-End Integration Testing (4-5 hours)
3. Metrics Dashboard/Queries (3-4 hours)
4. Deployment & CI/CD (2-3 hours)

**Total Phase 3 Estimated Time**: 15-20 hours (60-70% complete)
