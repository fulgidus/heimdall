-- Migration: Add measurement_features table for RF feature extraction
-- Date: 2025-11-02
-- Purpose: Store extracted features from IQ samples (synthetic and real)

BEGIN;

-- Create measurement_features table
CREATE TABLE IF NOT EXISTS heimdall.measurement_features (
    timestamp TIMESTAMPTZ NOT NULL,
    measurement_id UUID NOT NULL,

    -- Feature array: 7 JSONB objects (one per receiver)
    -- Each JSONB contains all 72 features (18 base × 4 aggregations)
    receiver_features JSONB[] NOT NULL,

    -- Extraction metadata
    extraction_metadata JSONB NOT NULL,
    -- Expected keys:
    --   - extraction_method: 'synthetic' | 'recorded'
    --   - iq_duration_ms: float (total IQ duration)
    --   - sample_rate_hz: int
    --   - num_chunks: int (number of chunks processed)
    --   - chunk_duration_ms: float

    -- Quality metrics (for quick filtering)
    overall_confidence FLOAT NOT NULL CHECK (overall_confidence >= 0 AND overall_confidence <= 1),
    mean_snr_db FLOAT,
    num_receivers_detected INT CHECK (num_receivers_detected >= 0 AND num_receivers_detected <= 7),

    -- Error handling (for real recordings only)
    extraction_failed BOOLEAN DEFAULT FALSE,
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Primary key
    PRIMARY KEY (timestamp, measurement_id),

    -- Foreign key to measurements table (composite key)
    CONSTRAINT fk_measurement_features_measurement
        FOREIGN KEY (timestamp, measurement_id)
        REFERENCES heimdall.measurements(timestamp, id)
        ON DELETE CASCADE
);

-- Indexes for common queries

-- Filter by confidence score
CREATE INDEX idx_measurement_features_confidence
    ON heimdall.measurement_features(overall_confidence DESC);

-- Filter by SNR
CREATE INDEX idx_measurement_features_snr
    ON heimdall.measurement_features(mean_snr_db DESC);

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

-- Comments for documentation
COMMENT ON TABLE heimdall.measurement_features IS
    'Extracted RF features from IQ samples. Used for ML training.';

COMMENT ON COLUMN heimdall.measurement_features.receiver_features IS
    'Array of JSONB objects, one per receiver. Each contains 72 features (18 base × mean/std/min/max).';

COMMENT ON COLUMN heimdall.measurement_features.overall_confidence IS
    'Confidence score (0-1) based on SNR, detection rate, and spectral clarity.';

COMMENT ON COLUMN heimdall.measurement_features.extraction_failed IS
    'TRUE if extraction failed (only for real recordings). Synthetic failures are skipped.';

COMMIT;
