-- Migration 022: Add IQ-Raw Dataset Type Support
-- Adds support for raw IQ datasets for CNN training with random receiver geometry
-- per sample, while maintaining backward compatibility with feature-based datasets

-- Set search path
SET search_path TO heimdall, public;

-- ============================================================================
-- ADD DATASET TYPE TO SYNTHETIC_DATASETS
-- ============================================================================
-- Add dataset_type enum to distinguish between feature-based and iq-raw datasets
DO $$
BEGIN
    -- Create dataset_type enum if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'dataset_type_enum') THEN
        CREATE TYPE dataset_type_enum AS ENUM ('feature_based', 'iq_raw');
    END IF;
END$$;

-- Add dataset_type column (default to 'feature_based' for backward compatibility)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'synthetic_datasets' 
        AND column_name = 'dataset_type'
    ) THEN
        ALTER TABLE synthetic_datasets 
        ADD COLUMN dataset_type dataset_type_enum DEFAULT 'feature_based' NOT NULL;
        
        COMMENT ON COLUMN synthetic_datasets.dataset_type IS 
            'Dataset type: feature_based (mel-spectrograms, MFCCs) or iq_raw (raw IQ samples for CNN)';
    END IF;
END$$;

-- Add index on dataset_type for efficient filtering
CREATE INDEX IF NOT EXISTS idx_synthetic_datasets_type 
ON synthetic_datasets(dataset_type, created_at DESC);

-- ============================================================================
-- SYNTHETIC IQ SAMPLES TABLE (Hypertable)
-- ============================================================================
-- Stores metadata and MinIO pointers for raw IQ samples
-- Each sample can have different number of receivers (5-10, randomized per sample)
CREATE TABLE IF NOT EXISTS synthetic_iq_samples (
    timestamp TIMESTAMPTZ NOT NULL,
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES synthetic_datasets(id) ON DELETE CASCADE,
    sample_idx INTEGER NOT NULL,  -- Sample index within dataset (for ordering)
    
    -- Transmitter ground truth
    tx_lat FLOAT NOT NULL,
    tx_lon FLOAT NOT NULL,
    tx_alt FLOAT NOT NULL,  -- Altitude in meters ASL
    tx_power_dbm FLOAT NOT NULL,
    frequency_hz BIGINT NOT NULL,
    
    -- Receiver configuration (JSONB array)
    -- [{rx_id, lat, lon, alt, distance_km, snr_db, signal_present}, ...]
    receivers_metadata JSONB NOT NULL,
    num_receivers INTEGER NOT NULL,  -- Number of receivers (5-10)
    
    -- Quality metrics
    gdop FLOAT,  -- Geometric Dilution of Precision
    mean_snr_db FLOAT,
    overall_confidence FLOAT,
    
    -- IQ generation parameters
    iq_metadata JSONB NOT NULL,  -- {sample_rate_hz, duration_ms, center_frequency_hz}
    
    -- MinIO storage paths (JSONB map: {rx_id: minio_path})
    -- Example: {"rx_0": "synthetic/dataset-uuid/sample-0/rx_0.npy", ...}
    iq_storage_paths JSONB NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (timestamp, id),
    UNIQUE (dataset_id, sample_idx)  -- Ensure sample_idx is unique within dataset
);

-- Convert to hypertable for efficient time-series queries
SELECT create_hypertable(
    'synthetic_iq_samples',
    'timestamp',
    if_not_exists => TRUE
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_synthetic_iq_dataset 
ON synthetic_iq_samples(dataset_id, sample_idx);

CREATE INDEX IF NOT EXISTS idx_synthetic_iq_quality 
ON synthetic_iq_samples(dataset_id, gdop, mean_snr_db);

CREATE INDEX IF NOT EXISTS idx_synthetic_iq_timestamp 
ON synthetic_iq_samples(timestamp DESC);

-- Add comment
COMMENT ON TABLE synthetic_iq_samples IS 
    'Raw IQ samples for CNN training with random receiver geometry per sample (5-10 receivers)';

-- ============================================================================
-- UPDATE MEASUREMENT_FEATURES TO SUPPORT VARIABLE RECEIVERS
-- ============================================================================
-- Add num_receivers_in_sample field to track actual receiver count per sample
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'heimdall' 
        AND table_name = 'measurement_features' 
        AND column_name = 'num_receivers_in_sample'
    ) THEN
        ALTER TABLE measurement_features 
        ADD COLUMN num_receivers_in_sample INTEGER;
        
        COMMENT ON COLUMN measurement_features.num_receivers_in_sample IS 
            'Number of receivers in this sample (for variable-receiver datasets)';
    END IF;
END$$;

-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

-- View: IQ Dataset Summary
CREATE OR REPLACE VIEW v_iq_dataset_summary AS
SELECT 
    sd.id AS dataset_id,
    sd.name AS dataset_name,
    sd.dataset_type,
    sd.num_samples AS total_samples,
    COUNT(iq.id) AS iq_samples_saved,
    AVG(iq.num_receivers) AS avg_receivers_per_sample,
    AVG(iq.gdop) AS avg_gdop,
    AVG(iq.mean_snr_db) AS avg_snr_db,
    sd.created_at
FROM synthetic_datasets sd
LEFT JOIN synthetic_iq_samples iq ON sd.id = iq.dataset_id
WHERE sd.dataset_type = 'iq_raw'
GROUP BY sd.id, sd.name, sd.dataset_type, sd.num_samples, sd.created_at;

COMMENT ON VIEW v_iq_dataset_summary IS 
    'Summary statistics for IQ-raw datasets';

-- View: Combined Dataset Overview (both types)
CREATE OR REPLACE VIEW v_all_datasets_overview AS
SELECT 
    id,
    name,
    dataset_type,
    num_samples,
    config,
    quality_metrics,
    created_at,
    CASE 
        WHEN dataset_type = 'feature_based' THEN 'Feature-based (Mel-spec/MFCC)'
        WHEN dataset_type = 'iq_raw' THEN 'Raw IQ for CNN'
        ELSE 'Unknown'
    END AS type_description
FROM synthetic_datasets
ORDER BY created_at DESC;

COMMENT ON VIEW v_all_datasets_overview IS 
    'Unified view of all synthetic datasets (feature-based and IQ-raw)';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- This migration adds:
-- 1. dataset_type field to synthetic_datasets (feature_based vs iq_raw)
-- 2. synthetic_iq_samples table for raw IQ metadata + MinIO pointers
-- 3. Support for variable receiver count (5-10 per sample)
-- 4. Views for easy querying of IQ datasets
