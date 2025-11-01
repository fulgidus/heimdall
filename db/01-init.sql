-- Heimdall SDR - PostgreSQL Initialization Script (CORRECTED)
-- Quick initialization to create basic tables for tests
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" CASCADE;

-- Create schema
CREATE SCHEMA IF NOT EXISTS heimdall;

SET
    search_path TO heimdall,
    public;

-- Table: websdr_stations
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
    rate_limit_ms INTEGER DEFAULT 1000,
    timeout_seconds INTEGER DEFAULT 30,
    retry_count INTEGER DEFAULT 3,
    admin_email VARCHAR(255),
    location_description TEXT,
    altitude_asl INTEGER,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: known_sources
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
    error_margin_meters FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: measurements (will become hypertable)
CREATE TABLE IF NOT EXISTS measurements (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    websdr_station_id UUID REFERENCES websdr_stations(id) ON DELETE CASCADE,
    recording_session_id UUID REFERENCES recording_sessions(id) ON DELETE CASCADE,
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

-- Convert to hypertable
SELECT
    create_hypertable(
        'measurements',
        'timestamp',
        if_not_exists => TRUE
    );

CREATE INDEX IF NOT EXISTS idx_measurements_websdr_station ON measurements(websdr_station_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_recording_session ON measurements(recording_session_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_frequency ON measurements(frequency_hz, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_time ON measurements(timestamp DESC);

-- Table: recording_sessions
CREATE TABLE IF NOT EXISTS recording_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    known_source_id UUID REFERENCES known_sources(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    session_start TIMESTAMP WITH TIME ZONE NOT NULL,
    session_end TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,
    celery_task_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    approval_status VARCHAR(50) DEFAULT 'pending',
    uncertainty_meters FLOAT,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: training_datasets
CREATE TABLE IF NOT EXISTS training_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    measurement_count INT DEFAULT 0,
    frequency_range_min_hz BIGINT,
    frequency_range_max_hz BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: dataset_measurements
CREATE TABLE IF NOT EXISTS dataset_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES training_datasets(id) ON DELETE CASCADE,
    measurement_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(dataset_id, measurement_id)
);

-- Table: models
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
        is_active BOOLEAN DEFAULT FALSE,
        is_production BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table: inference_requests
CREATE TABLE IF NOT EXISTS inference_requests (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    frequency_hz BIGINT,
    position_lat FLOAT,
    position_lon FLOAT,
    uncertainty_m FLOAT,
    cache_hit BOOLEAN DEFAULT FALSE,
    processing_time_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (timestamp, id)
);

-- Convert to hypertable
SELECT
    create_hypertable(
        'inference_requests',
        'timestamp',
        if_not_exists => TRUE
    );

CREATE INDEX IF NOT EXISTS idx_inference_requests_model ON inference_requests(model_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_inference_requests_time ON inference_requests(timestamp DESC);

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
        if_not_exists = > TRUE
    );

CREATE INDEX IF NOT EXISTS idx_websdrs_uptime_history_websdr_time ON websdrs_uptime_history(websdr_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_websdrs_uptime_history_timestamp ON websdrs_uptime_history(timestamp DESC);

-- Grants
GRANT CONNECT ON DATABASE heimdall TO heimdall_user;

GRANT USAGE ON SCHEMA heimdall TO heimdall_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA heimdall TO heimdall_user;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA heimdall TO heimdall_user;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA heimdall TO heimdall_user;

-- Seed Italian WebSDR stations
INSERT INTO
    websdr_stations (
        name,
        url,
        country,
        latitude,
        longitude,
        is_active,
        timeout_seconds,
        retry_count,
        location_description
    )
VALUES
    (
        'Aquila di Giaveno',
        'http://sdr1.ik1jns.it:8076',
        'Italy',
        45.03,
        7.27,
        TRUE,
        30,
        3,
        'Aquila, Giaveno, Turin'
    ),
    (
        'Montanaro',
        'http://cbfenis.ddns.net:43510',
        'Italy',
        45.234,
        7.857,
        TRUE,
        30,
        3,
        'Montanaro, Italy'
    ),
    (
        'Torino',
        'http://vst-aero.it:8073',
        'Italy',
        45.044,
        7.672,
        TRUE,
        30,
        3,
        'Torino, Italy'
    ),
    (
        'Coazze',
        'http://94.247.189.130:8076',
        'Italy',
        45.03,
        7.27,
        TRUE,
        30,
        3,
        'Coazze, Italy'
    ),
    (
        'Passo del Giovi',
        'http://iz1mlt.ddns.net:8074',
        'Italy',
        44.561,
        8.956,
        TRUE,
        30,
        3,
        'Passo del Giovi, Italy'
    ),
    (
        'Genova',
        'http://iq1zw.ddns.net:42154',
        'Italy',
        44.395,
        8.956,
        TRUE,
        30,
        3,
        'Genova, Italy'
    ),
    (
        'Milano - Baggio',
        'http://iu2mch.duckdns.org:8073',
        'Italy',
        45.478,
        9.123,
        TRUE,
        30,
        3,
        'Milano (Baggio), Italy'
    ) ON CONFLICT (name) DO NOTHING;