# Step 1: Database Schema for Feature Storage

## Objective

Create the PostgreSQL table `measurement_features` to store extracted RF features from both synthetic and real IQ samples.

## Context

Currently, Heimdall stores:
- Raw IQ samples in MinIO (for recording sessions)
- High-level features (SNR, PSD) directly in synthetic samples

We need a unified storage for **extracted features** (72 per receiver × 7 receivers = 504 total).

## Implementation

### 1. Create Migration File

**File**: `db/migrations/010-measurement-features-table.sql`

```sql
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

    -- Foreign key to measurements table
    CONSTRAINT fk_measurement_features_measurement
        FOREIGN KEY (measurement_id)
        REFERENCES heimdall.measurements(id)
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
```

### 2. Expected JSONB Structure

**Example `receiver_features` entry** (one of 7 in the array):

```json
{
  "rx_id": "Torino",
  "rx_lat": 45.044,
  "rx_lon": 7.672,
  "signal_present": true,

  "rssi": {"mean": -65.2, "std": 2.1, "min": -68.5, "max": -62.0},
  "snr": {"mean": 22.5, "std": 1.8, "min": 19.0, "max": 25.0},
  "noise_floor": {"mean": -87.7, "std": 0.5, "min": -88.5, "max": -87.0},

  "frequency_offset": {"mean": -15.2, "std": 5.3, "min": -25.0, "max": -8.0},
  "bandwidth": {"mean": 12500.0, "std": 800.0, "min": 11200.0, "max": 13500.0},
  "psd": {"mean": -155.3, "std": 1.2, "min": -157.0, "max": -153.0},
  "spectral_centroid": {"mean": 144000050.0, "std": 20.0, "min": 144000020.0, "max": 144000080.0},
  "spectral_rolloff": {"mean": 144012500.0, "std": 500.0, "min": 144011800.0, "max": 144013200.0},

  "envelope_mean": {"mean": 0.65, "std": 0.08, "min": 0.52, "max": 0.78},
  "envelope_std": {"mean": 0.15, "std": 0.03, "min": 0.11, "max": 0.19},
  "envelope_max": {"mean": 0.95, "std": 0.05, "min": 0.88, "max": 1.0},
  "peak_to_avg_ratio": {"mean": 6.2, "std": 0.8, "min": 5.1, "max": 7.5},
  "zero_crossing_rate": {"mean": 0.42, "std": 0.05, "min": 0.35, "max": 0.48},

  "multipath_delay_spread_us": {"mean": 2.5, "std": 0.8, "min": 1.2, "max": 4.1},
  "coherence_bandwidth_khz": {"mean": 80.0, "std": 25.0, "min": 48.8, "max": 166.7},

  "delay_spread_confidence": {"mean": 0.85, "std": 0.10, "min": 0.65, "max": 0.95}
}
```

**Example `extraction_metadata`**:

```json
{
  "extraction_method": "synthetic",
  "iq_duration_ms": 1000.0,
  "sample_rate_hz": 200000,
  "num_chunks": 5,
  "chunk_duration_ms": 200.0
}
```

### 3. Run Migration

Execute the migration:

```bash
# Via Docker
DOCKER_HOST="" docker exec heimdall-postgres psql -U heimdall_user -d heimdall -f /migrations/010-measurement-features-table.sql

# Or manually
psql -h localhost -U heimdall_user -d heimdall -f db/migrations/010-measurement-features-table.sql
```

## Verification

### 1. Check Table Exists

```sql
\dt heimdall.measurement_features
```

Expected output: Table should exist with correct schema.

### 2. Check Indexes

```sql
\di heimdall.idx_measurement_features*
```

Expected: 5 indexes created.

### 3. Test Insert

```sql
INSERT INTO heimdall.measurement_features (
    timestamp, measurement_id, receiver_features, extraction_metadata,
    overall_confidence, mean_snr_db, num_receivers_detected
)
VALUES (
    NOW(),
    gen_random_uuid(),
    ARRAY['{"rx_id": "Test", "snr": {"mean": 20.0}}'::jsonb],
    '{"extraction_method": "test"}'::jsonb,
    0.85,
    20.0,
    1
);
```

Expected: Insert succeeds without errors.

### 4. Test Query

```sql
SELECT * FROM heimdall.measurement_features LIMIT 1;
```

Expected: Returns the test row.

### 5. Clean Up Test Data

```sql
DELETE FROM heimdall.measurement_features WHERE extraction_metadata->>'extraction_method' = 'test';
```

## Success Criteria

- ✅ Table `heimdall.measurement_features` exists
- ✅ All 5 indexes created
- ✅ Foreign key constraint to `measurements` works
- ✅ Check constraints on `overall_confidence` and `num_receivers_detected` work
- ✅ JSONB array column accepts feature data
- ✅ Test insert/query/delete successful

## Next Step

Proceed to **`02-feature-extractor-core.md`** to implement the feature extraction logic.
