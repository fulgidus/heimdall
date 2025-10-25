-- TimescaleDB Hypertable Migration: Create measurements table
-- Version: 001
-- Description: Creates the measurements hypertable optimized for time-series data storage
-- Enable TimescaleDB extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create base table for measurements
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
    PRIMARY KEY (id, timestamp_utc)
);

-- Convert to hypertable with time dimension
SELECT
    create_hypertable(
        'measurements',
        'timestamp_utc',
        if_not_exists = > TRUE,
        chunk_time_interval = > INTERVAL '1 day'
    );

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_measurements_websdr_time ON measurements (websdr_id, timestamp_utc DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_task_time ON measurements (task_id, timestamp_utc DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_frequency ON measurements (frequency_mhz, timestamp_utc DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_task_websdr ON measurements (task_id, websdr_id, timestamp_utc DESC);

-- Create composite index for session queries
CREATE INDEX IF NOT EXISTS idx_measurements_task_id_only ON measurements (task_id);

-- Enable compression for older data (optional, set retention policy)
-- This will compress data older than 7 days to save storage
ALTER TABLE
    measurements
SET
    (
        timescaledb.compress,
        timescaledb.compress_orderby = 'timestamp_utc DESC'
    );

-- Set compression policy: compress data older than 7 days
SELECT
    add_compression_policy(
        'measurements',
        INTERVAL '7 days',
        if_not_exists = > TRUE
    );

-- Set continuous aggregate for daily statistics (optional)
-- This pre-computes daily aggregates for faster queries
CREATE MATERIALIZED VIEW IF NOT EXISTS measurements_daily AS
SELECT
    DATE_TRUNC('day', timestamp_utc) AS day,
    task_id,
    websdr_id,
    frequency_mhz,
    COUNT(*) as measurement_count,
    AVG(snr_db) as avg_snr_db,
    MIN(snr_db) as min_snr_db,
    MAX(snr_db) as max_snr_db,
    AVG(frequency_offset_hz) as avg_frequency_offset_hz,
    AVG(power_dbm) as avg_power_dbm
FROM
    measurements
GROUP BY
    DATE_TRUNC('day', timestamp_utc),
    task_id,
    websdr_id,
    frequency_mhz WITH DATA;

-- Create index on continuous aggregate
CREATE INDEX IF NOT EXISTS idx_measurements_daily_task_time ON measurements_daily (task_id, day DESC);

-- Set data retention policy: keep data for 30 days, delete older
SELECT
    add_retention_policy(
        'measurements',
        INTERVAL '30 days',
        if_not_exists = > TRUE
    );