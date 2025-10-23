# Data Schema Documentation

## Overview

Complete documentation of Heimdall database schema with tables, relationships, and constraints.

## Tables

### tasks

Stores RF acquisition task requests from users.

```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    frequencies FLOAT8[] NOT NULL,
    duration_seconds INTEGER NOT NULL,
    bandwidth_hz INTEGER NOT NULL,
    name VARCHAR(255),
    priority VARCHAR(20) DEFAULT 'normal',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    CONSTRAINT valid_status CHECK (status IN ('submitted', 'processing', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_created_at ON tasks(created_at DESC);
CREATE INDEX idx_tasks_user_id ON tasks(user_id) WHERE user_id IS NOT NULL;
```

**Fields**:
- `id`: Unique task identifier
- `user_id`: User who submitted the task
- `status`: Current task status
- `frequencies`: Array of target frequencies in MHz
- `duration_seconds`: Acquisition duration
- `bandwidth_hz`: Bandwidth in Hz
- `name`: Human-readable name
- `priority`: Task priority (low, normal, high)
- `created_at`: Creation timestamp
- `started_at`: Processing start time
- `completed_at`: Completion timestamp
- `error_message`: Error details if failed

---

### signal_measurements

Time-series table storing individual signal measurements from WebSDR stations.

```sql
CREATE TABLE signal_measurements (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    station_name VARCHAR(255) NOT NULL,
    frequency_mhz FLOAT8 NOT NULL,
    signal_strength_dbm FLOAT8,
    bearing_degrees FLOAT8,
    bearing_uncertainty_degrees FLOAT8,
    arrival_time_us BIGINT,
    signal_quality FLOAT8,
    snr_db FLOAT8,
    measurement_error_code VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_frequency CHECK (frequency_mhz > 0),
    CONSTRAINT valid_bearing CHECK (bearing_degrees IS NULL OR (bearing_degrees >= 0 AND bearing_degrees < 360)),
    CONSTRAINT valid_quality CHECK (signal_quality IS NULL OR (signal_quality >= 0 AND signal_quality <= 1))
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('signal_measurements', 'created_at', if_not_exists => TRUE);

-- Compression policy (keep 1 week uncompressed, compress older)
SELECT add_compression_policy('signal_measurements', INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_measurements_task_id ON signal_measurements(task_id);
CREATE INDEX idx_measurements_station_freq ON signal_measurements(station_name, frequency_mhz);
CREATE INDEX idx_measurements_created_at ON signal_measurements(created_at DESC);
```

**Fields**:
- `id`: Unique measurement ID
- `task_id`: Reference to parent task
- `station_name`: WebSDR station name
- `frequency_mhz`: Measured frequency
- `signal_strength_dbm`: Signal power in dBm
- `bearing_degrees`: Direction to source (0-360°)
- `bearing_uncertainty_degrees`: Confidence in bearing
- `arrival_time_us`: Microsecond timestamp
- `signal_quality`: Quality score (0-1)
- `snr_db`: Signal-to-noise ratio
- `measurement_error_code`: Any error codes

---

### task_results

Localization results for completed tasks.

```sql
CREATE TABLE task_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL UNIQUE REFERENCES tasks(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    latitude FLOAT8,
    longitude FLOAT8,
    altitude_m FLOAT8,
    latitude_uncertainty_m FLOAT8,
    longitude_uncertainty_m FLOAT8,
    confidence FLOAT8,
    algorithm_version VARCHAR(50),
    processing_time_ms INTEGER,
    stations_used INTEGER,
    measurements_used INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_latitude CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90)),
    CONSTRAINT valid_longitude CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180)),
    CONSTRAINT valid_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX idx_results_task_id ON task_results(task_id);
CREATE INDEX idx_results_confidence ON task_results(confidence DESC);
CREATE INDEX idx_results_created_at ON task_results(created_at DESC);
```

**Fields**:
- `id`: Unique result ID
- `task_id`: Reference to task
- `status`: Result status (completed, error, partial)
- `latitude`, `longitude`, `altitude_m`: Computed location
- `latitude_uncertainty_m`, `longitude_uncertainty_m`: Confidence bounds
- `confidence`: Confidence score (0-1)
- `algorithm_version`: ML model version used
- `processing_time_ms`: Time to compute result
- `stations_used`: Number of stations in triangulation
- `measurements_used`: Number of measurements used

---

### websdr_stations

Configuration and status of WebSDR network.

```sql
CREATE TABLE websdr_stations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL,
    latitude FLOAT8 NOT NULL,
    longitude FLOAT8 NOT NULL,
    altitude_m FLOAT8,
    frequency_min_mhz FLOAT8,
    frequency_max_mhz FLOAT8,
    status VARCHAR(50) DEFAULT 'active',
    last_check_time TIMESTAMP WITH TIME ZONE,
    uptime_percentage FLOAT8,
    is_trusted BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_stations_status ON websdr_stations(status);
```

**Fields**:
- `id`: Station identifier
- `name`: Human-readable station name
- `url`: WebSDR URL endpoint
- `latitude`, `longitude`: Station location
- `frequency_min_mhz`, `frequency_max_mhz`: Frequency range
- `status`: online/offline/maintenance
- `uptime_percentage`: Historical availability
- `is_trusted`: Verification status

---

### ml_models

ML model registry and versioning.

```sql
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100),
    architecture TEXT,
    input_shape VARCHAR(100),
    output_shape VARCHAR(100),
    accuracy_test FLOAT8,
    accuracy_validation FLOAT8,
    parameters_count BIGINT,
    model_size_mb FLOAT8,
    pytorch_path VARCHAR(255),
    onnx_path VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trained_at TIMESTAMP WITH TIME ZONE,
    is_production BOOLEAN DEFAULT FALSE,
    UNIQUE(name, version)
);

CREATE INDEX idx_models_name ON ml_models(name);
CREATE INDEX idx_models_production ON ml_models(is_production);
```

**Fields**:
- `id`: Unique model ID
- `name`: Model name
- `version`: Version identifier
- `model_type`: CNN, RNN, Transformer, etc.
- `accuracy_*`: Test/validation accuracy
- `pytorch_path`, `onnx_path`: Model file paths
- `is_production`: Whether model is in production

---

## Relationships

```
tasks (1) ──────────── (N) signal_measurements
       └──────────────── (1) task_results

websdr_stations (referenced by) signal_measurements (via station_name)

ml_models (referenced by) tasks (via algorithm_version in results)
```

## Data Types

| Type                     | Usage                   | Example                                |
| ------------------------ | ----------------------- | -------------------------------------- |
| UUID                     | Unique IDs              | `550e8400-e29b-41d4-a716-446655440000` |
| BIGSERIAL                | Auto-increment IDs      | Time-series indexes                    |
| FLOAT8                   | High-precision decimals | Coordinates, signal strength           |
| INTEGER                  | Whole numbers           | Duration, counts                       |
| VARCHAR                  | Text                    | Names, URLs, status                    |
| TIMESTAMP WITH TIME ZONE | Events                  | Task creation, completion              |
| BOOLEAN                  | Flags                   | Active status, trusted                 |
| ARRAY                    | Multiple values         | Frequencies array                      |
| TEXT                     | Large text              | Algorithm details, notes               |

## Queries

### Get Recent Results

```sql
SELECT 
    t.id,
    t.frequencies,
    tr.latitude,
    tr.longitude,
    tr.confidence
FROM tasks t
JOIN task_results tr ON t.id = tr.task_id
WHERE t.created_at > NOW() - INTERVAL '24 hours'
ORDER BY t.created_at DESC
LIMIT 10;
```

### Find High-Confidence Results

```sql
SELECT 
    tr.latitude,
    tr.longitude,
    COUNT(*) as count
FROM task_results tr
WHERE tr.confidence > 0.95
  AND tr.created_at > NOW() - INTERVAL '7 days'
GROUP BY ROUND(tr.latitude, 2), ROUND(tr.longitude, 2)
ORDER BY count DESC;
```

### Analyze Station Performance

```sql
SELECT 
    station_name,
    COUNT(*) as measurements,
    AVG(signal_strength_dbm) as avg_strength,
    AVG(signal_quality) as avg_quality
FROM signal_measurements
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY station_name
ORDER BY avg_quality DESC;
```

### Database Statistics

```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_live_tup as row_count
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Maintenance

### Vacuuming

```sql
-- Remove dead rows
VACUUM ANALYZE signal_measurements;

-- Full vacuum (locks table)
VACUUM FULL signal_measurements;
```

### Reindexing

```sql
-- Rebuild index
REINDEX INDEX idx_measurements_created_at;

-- Rebuild all indexes
REINDEX TABLE signal_measurements;
```

---

**Last Updated**: October 2025

**Related**: [Architecture Guide](./ARCHITECTURE.md) | [Installation Guide](./installation.md)
