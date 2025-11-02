-- Migration: Corrected measurement_features table for multi-receiver localization
-- Date: 2025-11-02
-- Purpose: Store correlated features from multiple receivers for ML training

BEGIN;

-- Create measurement_features table (CORRECTED)
CREATE TABLE IF NOT EXISTS heimdall.measurement_features (
    -- Primary key: recording_session_id (groups all receivers together)
    recording_session_id UUID PRIMARY KEY,

    timestamp TIMESTAMPTZ NOT NULL,

    -- Feature array: N JSONB objects (one per receiver that detected signal)
    -- N varies: 2-7 receivers depending on signal strength and geometry
    -- Each JSONB contains all 72 features (18 base × 4 aggregations)
    receiver_features JSONB[] NOT NULL,

    -- Ground truth for supervised learning
    -- For synthetic data: known TX position
    -- For real recordings: NULL (to be estimated by model) or known beacon
    tx_latitude DOUBLE PRECISION,
    tx_longitude DOUBLE PRECISION,
    tx_altitude_m DOUBLE PRECISION,
    tx_power_dbm DOUBLE PRECISION,
    tx_known BOOLEAN DEFAULT FALSE,  -- TRUE if ground truth is available

    -- Extraction metadata
    extraction_metadata JSONB NOT NULL,
    -- Expected keys:
    --   - extraction_method: 'synthetic' | 'recorded'
    --   - iq_duration_ms: float (total IQ duration)
    --   - sample_rate_hz: int
    --   - num_chunks: int (number of chunks processed)
    --   - chunk_duration_ms: float
    --   - synthetic_dataset_id: UUID (only for synthetic)

    -- Quality metrics (for quick filtering)
    overall_confidence FLOAT NOT NULL CHECK (overall_confidence >= 0 AND overall_confidence <= 1),
    mean_snr_db FLOAT,
    num_receivers_detected INT CHECK (num_receivers_detected >= 0 AND num_receivers_detected <= 7),

    -- GDOP (Geometric Dilution of Precision) for geometry quality
    gdop FLOAT,

    -- Error handling (for real recordings only)
    extraction_failed BOOLEAN DEFAULT FALSE,
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Foreign key to recording_sessions (NULL for synthetic data)
    CONSTRAINT fk_measurement_features_session
        FOREIGN KEY (recording_session_id)
        REFERENCES heimdall.recording_sessions(id)
        ON DELETE CASCADE
);

-- Indexes for common queries

-- Filter by confidence score
CREATE INDEX idx_measurement_features_confidence
    ON heimdall.measurement_features(overall_confidence DESC);

-- Filter by SNR
CREATE INDEX idx_measurement_features_snr
    ON heimdall.measurement_features(mean_snr_db DESC);

-- Filter by GDOP (geometry quality)
CREATE INDEX idx_measurement_features_gdop
    ON heimdall.measurement_features(gdop ASC)
    WHERE gdop IS NOT NULL;

-- Find failed extractions (for debugging)
CREATE INDEX idx_measurement_features_failed
    ON heimdall.measurement_features(extraction_failed)
    WHERE extraction_failed = TRUE;

-- Find by timestamp
CREATE INDEX idx_measurement_features_timestamp
    ON heimdall.measurement_features(timestamp DESC);

-- Find by number of receivers
CREATE INDEX idx_measurement_features_num_receivers
    ON heimdall.measurement_features(num_receivers_detected DESC);

-- Find samples with known ground truth (for supervised training)
CREATE INDEX idx_measurement_features_ground_truth
    ON heimdall.measurement_features(tx_known)
    WHERE tx_known = TRUE;

-- Find by extraction method
CREATE INDEX idx_measurement_features_method
    ON heimdall.measurement_features((extraction_metadata->>'extraction_method'));

-- Comments for documentation
COMMENT ON TABLE heimdall.measurement_features IS
    'Extracted RF features from multi-receiver IQ samples. Used for ML localization training. Each row represents ONE localization sample with features from multiple receivers.';

COMMENT ON COLUMN heimdall.measurement_features.recording_session_id IS
    'Primary key. For real recordings: references recording_sessions.id. For synthetic: generated UUID.';

COMMENT ON COLUMN heimdall.measurement_features.receiver_features IS
    'Array of JSONB objects (2-7 receivers). Each contains 72 features (18 base × mean/std/min/max). All receivers in this array captured the SAME transmission event.';

COMMENT ON COLUMN heimdall.measurement_features.tx_latitude IS
    'Ground truth TX latitude. Known for synthetic data and beacons. NULL for unknown real recordings.';

COMMENT ON COLUMN heimdall.measurement_features.tx_longitude IS
    'Ground truth TX longitude. Known for synthetic data and beacons. NULL for unknown real recordings.';

COMMENT ON COLUMN heimdall.measurement_features.tx_known IS
    'TRUE if ground truth position is available (synthetic or known beacon). Used to split supervised/unsupervised training data.';

COMMENT ON COLUMN heimdall.measurement_features.overall_confidence IS
    'Confidence score (0-1) based on SNR, detection rate, and spectral clarity across all receivers.';

COMMENT ON COLUMN heimdall.measurement_features.gdop IS
    'Geometric Dilution of Precision. Lower = better receiver geometry for localization. NULL if <3 receivers.';

COMMENT ON COLUMN heimdall.measurement_features.extraction_failed IS
    'TRUE if extraction failed (only for real recordings). Synthetic failures are skipped.';

COMMIT;
