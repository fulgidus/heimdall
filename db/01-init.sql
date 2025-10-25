-- Heimdall SDR - PostgreSQL Initialization Script
-- This script initializes the database schema with TimescaleDB extensions
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema
CREATE SCHEMA IF NOT EXISTS heimdall;

SET
    search_path TO heimdall,
    public;

-- Table: websdr_stations - Configuration of available WebSDR receivers
CREATE TABLE IF NOT EXISTS websdr_stations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    url VARCHAR(512) NOT NULL,
    country VARCHAR(100),
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    frequency_min_hz BIGINT,
    frequency_max_hz BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    api_type VARCHAR(50) DEFAULT 'http',
    rate_limit_ms INT DEFAULT 1000,
    timeout_seconds INT DEFAULT 30,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_frequencies CHECK (frequency_min_hz <= frequency_max_hz)
);

CREATE INDEX idx_websdr_stations_active ON websdr_stations(is_active);

-- Table: known_sources - Radio sources used for training and validation
CREATE TABLE IF NOT EXISTS known_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    frequency_hz BIGINT NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    power_dbm FLOAT,
    source_type VARCHAR(100),
    is_validated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_known_sources_frequency ON known_sources(frequency_hz);

CREATE INDEX idx_known_sources_validated ON known_sources(is_validated);

-- Table: measurements - Time-series IQ measurements from WebSDR stations
CREATE TABLE IF NOT EXISTS measurements (
    websdr_station_id UUID NOT NULL REFERENCES websdr_stations(id) ON DELETE RESTRICT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    id UUID DEFAULT uuid_generate_v4(),
    frequency_hz BIGINT NOT NULL,
    signal_strength_db FLOAT,
    snr_db FLOAT,
    frequency_offset_hz INT,
    iq_data_location VARCHAR(512),
    iq_data_format VARCHAR(50),
    iq_sample_rate INT,
    iq_samples_count INT,
    duration_seconds FLOAT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert measurements table to TimescaleDB hypertable
SELECT
    create_hypertable(
        'measurements',
        'timestamp',
        if_not_exists => TRUE
    );

CREATE INDEX idx_measurements_websdr_station ON measurements(websdr_station_id, timestamp DESC);

CREATE INDEX idx_measurements_frequency ON measurements(frequency_hz, timestamp DESC);

CREATE INDEX idx_measurements_time ON measurements(timestamp DESC);

-- Chunk size: 1 day for optimal query performance
SELECT
    set_chunk_time_interval('measurements', INTERVAL '1 day');

-- Table: recording_sessions - Human-assisted data collection sessions
CREATE TABLE IF NOT EXISTS recording_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    known_source_id UUID NOT NULL REFERENCES known_sources(id) ON DELETE RESTRICT,
    session_name VARCHAR(255) NOT NULL,
    session_start TIMESTAMP WITH TIME ZONE NOT NULL,
    session_end TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,
    celery_task_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    approval_status VARCHAR(50) DEFAULT 'pending',
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_status CHECK (
        status IN ('pending', 'in_progress', 'completed', 'failed')
    )
);

CREATE INDEX idx_recording_sessions_source ON recording_sessions(known_source_id);

CREATE INDEX idx_recording_sessions_status ON recording_sessions(status);

CREATE INDEX idx_recording_sessions_approval ON recording_sessions(approval_status);

CREATE INDEX idx_recording_sessions_start_time ON recording_sessions(session_start DESC);

-- Table: training_datasets - Collections of approved measurements for model training
CREATE TABLE IF NOT EXISTS training_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    measurement_count INT DEFAULT 0,
    frequency_range_min_hz BIGINT,
    frequency_range_max_hz BIGINT,
    geographic_region VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_training_datasets_active ON training_datasets(is_active);

-- Table: dataset_measurements - Join table for many-to-many relationship
CREATE TABLE IF NOT EXISTS dataset_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES training_datasets(id) ON DELETE CASCADE,
    measurement_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(dataset_id, measurement_id)
);

CREATE INDEX idx_dataset_measurements_dataset ON dataset_measurements(dataset_id);

CREATE INDEX idx_dataset_measurements_measurement ON dataset_measurements(measurement_id);

-- Table: models - Trained ML models metadata
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100),
    training_dataset_id UUID REFERENCES training_datasets(id) ON DELETE
    SET
        NULL,
        mlflow_run_id VARCHAR(255),
        mlflow_experiment_id INT,
        onnx_model_location VARCHAR(512),
        pytorch_model_location VARCHAR(512),
        accuracy_meters FLOAT,
        accuracy_sigma_meters FLOAT,
        loss_value FLOAT,
        epoch INT,
        hyperparameters JSONB,
        performance_metrics JSONB,
        is_active BOOLEAN DEFAULT FALSE,
        is_production BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_models_active ON models(is_active);

CREATE INDEX idx_models_production ON models(is_production);

CREATE INDEX idx_models_mlflow_run ON models(mlflow_run_id);

-- Table: inference_requests - Track inference API calls for monitoring
CREATE TABLE IF NOT EXISTS inference_requests (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    id UUID DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
    frequency_hz BIGINT,
    position_lat FLOAT,
    position_lon FLOAT,
    uncertainty_m FLOAT,
    cache_hit BOOLEAN DEFAULT FALSE,
    processing_time_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert inference_requests to TimescaleDB hypertable
SELECT
    create_hypertable(
        'inference_requests',
        'timestamp',
        if_not_exists => TRUE
    );

CREATE INDEX idx_inference_requests_model ON inference_requests(model_id, timestamp DESC);

CREATE INDEX idx_inference_requests_time ON inference_requests(timestamp DESC);

-- Grants
GRANT CONNECT ON DATABASE heimdall TO heimdall_user;

GRANT USAGE ON SCHEMA heimdall TO heimdall_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA heimdall TO heimdall_user;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA heimdall TO heimdall_user;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA heimdall TO heimdall_user;

-- Comments for documentation
COMMENT ON TABLE websdr_stations IS 'WebSDR receiver stations across Europe';

COMMENT ON TABLE known_sources IS 'Radio sources used for training and localization';

COMMENT ON TABLE measurements IS 'IQ measurements from WebSDR receivers (TimescaleDB hypertable)';

COMMENT ON TABLE recording_sessions IS 'Human-assisted data collection sessions';

COMMENT ON TABLE training_datasets IS 'Collections of measurements for model training';

COMMENT ON TABLE models IS 'Trained ML models metadata and performance';

COMMENT ON TABLE inference_requests IS 'Inference API call tracking (TimescaleDB hypertable)';

-- Table: websdrs_uptime_history - Track WebSDR online/offline status over time
CREATE TABLE IF NOT EXISTS websdrs_uptime_history (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    id UUID DEFAULT uuid_generate_v4(),
    websdr_id INT NOT NULL,
    websdr_name VARCHAR(255),
    status VARCHAR(20) NOT NULL CHECK (status IN ('online', 'offline')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
SELECT
    create_hypertable(
        'websdrs_uptime_history',
        'timestamp',
        if_not_exists => TRUE
    );

CREATE INDEX idx_websdrs_uptime_history_websdr_time ON websdrs_uptime_history(websdr_id, timestamp DESC);

CREATE INDEX idx_websdrs_uptime_history_timestamp ON websdrs_uptime_history(timestamp DESC);

COMMENT ON TABLE websdrs_uptime_history IS 'TimescaleDB hypertable for WebSDR uptime history';