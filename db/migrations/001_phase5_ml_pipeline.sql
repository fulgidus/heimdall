-- Phase 5: ML Training Pipeline Database Schema Extensions
-- Migration 001: Add synthetic datasets, trained models metadata, evaluations, terrain cache

-- Set search path
SET search_path TO heimdall, public;

-- ============================================================================
-- SYNTHETIC DATASETS TABLE
-- ============================================================================
-- Stores metadata for synthetically generated training datasets
CREATE TABLE IF NOT EXISTS synthetic_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    
    -- Sample counts
    num_samples INTEGER NOT NULL,
    train_count INTEGER,
    val_count INTEGER,
    test_count INTEGER,
    
    -- Generation configuration
    config JSONB NOT NULL,  -- includes: inside/outside ratio, TX types, frequency ranges
    
    -- Quality metrics from generation
    quality_metrics JSONB,  -- includes: avg SNR, receivers/session, GDOP stats
    
    -- Storage location
    storage_table VARCHAR(100) DEFAULT 'synthetic_training_samples',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Link to job that created it
    created_by_job_id UUID REFERENCES training_jobs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_synthetic_datasets_created ON synthetic_datasets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_synthetic_datasets_name ON synthetic_datasets(name);

-- ============================================================================
-- SYNTHETIC TRAINING SAMPLES TABLE (Hypertable)
-- ============================================================================
-- Stores individual synthetic training samples
CREATE TABLE IF NOT EXISTS synthetic_training_samples (
    timestamp TIMESTAMPTZ NOT NULL,
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES synthetic_datasets(id) ON DELETE CASCADE,
    
    -- Transmitter ground truth
    tx_lat FLOAT NOT NULL,
    tx_lon FLOAT NOT NULL,
    tx_power_dbm FLOAT NOT NULL,
    frequency_hz BIGINT NOT NULL,
    
    -- Per-receiver measurements (variable length, stored as JSONB array)
    receivers JSONB NOT NULL,  -- [{rx_id, lat, lon, snr, psd, freq_offset, signal_present}, ...]
    
    -- Geometry quality
    gdop FLOAT,  -- Geometric Dilution of Precision
    num_receivers INTEGER,  -- Number of receivers with signal
    
    -- Split assignment
    split VARCHAR(10) NOT NULL CHECK (split IN ('train', 'val', 'test')),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (timestamp, id)
);

-- Convert to hypertable for efficient time-series queries
SELECT create_hypertable(
    'synthetic_training_samples',
    'timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_synthetic_samples_dataset ON synthetic_training_samples(dataset_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_synthetic_samples_split ON synthetic_training_samples(dataset_id, split);

-- ============================================================================
-- TRAINED MODELS METADATA (Extends existing 'models' table)
-- ============================================================================
-- Add columns to existing models table for Phase 5 requirements
DO $$
BEGIN
    -- Add version column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'models' 
        AND column_name = 'version'
    ) THEN
        ALTER TABLE models ADD COLUMN version INTEGER DEFAULT 1;
    END IF;
    
    -- Add hyperparameters JSONB if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'models' 
        AND column_name = 'hyperparameters'
    ) THEN
        ALTER TABLE models ADD COLUMN hyperparameters JSONB;
    END IF;
    
    -- Add training_metrics JSONB if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'models' 
        AND column_name = 'training_metrics'
    ) THEN
        ALTER TABLE models ADD COLUMN training_metrics JSONB;
    END IF;
    
    -- Add test_metrics JSONB if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'models' 
        AND column_name = 'test_metrics'
    ) THEN
        ALTER TABLE models ADD COLUMN test_metrics JSONB;
    END IF;
    
    -- Add synthetic_dataset_id reference if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'models' 
        AND column_name = 'synthetic_dataset_id'
    ) THEN
        ALTER TABLE models ADD COLUMN synthetic_dataset_id UUID REFERENCES synthetic_datasets(id) ON DELETE SET NULL;
    END IF;
    
    -- Add trained_by_job_id reference if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'models' 
        AND column_name = 'trained_by_job_id'
    ) THEN
        ALTER TABLE models ADD COLUMN trained_by_job_id UUID REFERENCES training_jobs(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Create unique constraint on (model_name, version)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'models_name_version_unique'
    ) THEN
        ALTER TABLE models ADD CONSTRAINT models_name_version_unique UNIQUE (model_name, version);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_models_synthetic_dataset ON models(synthetic_dataset_id);
CREATE INDEX IF NOT EXISTS idx_models_trained_by_job ON models(trained_by_job_id);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active) WHERE is_active = TRUE;

-- ============================================================================
-- MODEL EVALUATIONS TABLE
-- ============================================================================
-- Stores evaluation results for trained models
CREATE TABLE IF NOT EXISTS model_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES synthetic_datasets(id) ON DELETE SET NULL,
    
    -- Evaluation metrics
    metrics JSONB NOT NULL,  -- includes: median_error, p68, p95, calibration, etc.
    
    -- Visualization paths (stored in MinIO)
    visualization_paths JSONB,  -- {error_histogram, calibration_curve, error_heatmap, etc.}
    
    -- Timestamps
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Link to job that ran evaluation
    evaluated_by_job_id UUID REFERENCES training_jobs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_model_evaluations_model ON model_evaluations(model_id, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_evaluations_dataset ON model_evaluations(dataset_id);

-- ============================================================================
-- TERRAIN TILES CACHE TABLE
-- ============================================================================
-- Caches downloaded SRTM terrain tiles
CREATE TABLE IF NOT EXISTS terrain_tiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Tile identification
    tile_name VARCHAR(50) NOT NULL UNIQUE,  -- e.g., 'N44E007', 'N45E008'
    lat_min INTEGER NOT NULL,
    lat_max INTEGER NOT NULL,
    lon_min INTEGER NOT NULL,
    lon_max INTEGER NOT NULL,
    
    -- Storage
    minio_bucket VARCHAR(100) DEFAULT 'terrain',
    minio_path VARCHAR(255) NOT NULL,  -- path in MinIO
    file_size_bytes BIGINT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'downloading', 'ready', 'failed')),
    error_message TEXT,
    
    -- Metadata
    source_url TEXT,  -- original SRTM download URL
    checksum_sha256 VARCHAR(64),
    
    -- Timestamps
    downloaded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_terrain_tiles_status ON terrain_tiles(status);
CREATE INDEX IF NOT EXISTS idx_terrain_tiles_name ON terrain_tiles(tile_name);
CREATE INDEX IF NOT EXISTS idx_terrain_tiles_bounds ON terrain_tiles(lat_min, lat_max, lon_min, lon_max);

-- ============================================================================
-- TRAINING JOB TYPE EXTENSION
-- ============================================================================
-- Add job_type column to training_jobs for different operation types
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'training_jobs' 
        AND column_name = 'job_type'
    ) THEN
        ALTER TABLE training_jobs ADD COLUMN job_type VARCHAR(50) DEFAULT 'training' 
            CHECK (job_type IN ('training', 'terrain_download', 'synthetic_generation', 'model_export', 'model_evaluation'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_training_jobs_type ON training_jobs(job_type, created_at DESC);

-- ============================================================================
-- GRANTS
-- ============================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA heimdall TO heimdall_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA heimdall TO heimdall_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA heimdall TO heimdall_user;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE synthetic_datasets IS 'Metadata for synthetically generated training datasets';
COMMENT ON TABLE synthetic_training_samples IS 'Individual synthetic training samples with per-receiver measurements';
COMMENT ON TABLE model_evaluations IS 'Evaluation results for trained models';
COMMENT ON TABLE terrain_tiles IS 'Cache of downloaded SRTM terrain elevation tiles';
COMMENT ON COLUMN training_jobs.job_type IS 'Type of job: training, terrain_download, synthetic_generation, model_export, model_evaluation';
