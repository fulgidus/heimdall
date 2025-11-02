# Step 1-FIX: Correct Database Schema for Multi-Receiver Localization

## ⚠️ CRITICAL FIX REQUIRED

**Problem**: The initial schema in `01-database-schema.md` stores features **per measurement** (single receiver), making multi-receiver localization **impossible**.

**Root Cause**: ML localization requires **correlated features from multiple receivers simultaneously** (TDOA, RSS-based trilateration). The current schema breaks this correlation.

## Why This Matters

**Localization works like this**:
```
TX transmits at 144.5 MHz
    ↓
SDR Torino:  RSSI=-65dB, delay=0.5ms
SDR Milano:  RSSI=-68dB, delay=0.8ms
SDR Bologna: RSSI=-70dB, delay=1.2ms
SDR Genova:  RSSI=-72dB, delay=1.5ms
    ↓
ML Model: Combines all 4 signals → Estimates TX position (lat, lon)
```

**Without correlation**: You have 4 separate records with no relationship → **Cannot train localization model**.

## Corrected Schema

### 1. Drop Old Table (if already created)

```sql
-- ONLY run this if you already created the table from prompt 01
DROP TABLE IF EXISTS heimdall.measurement_features CASCADE;
```

### 2. Create Corrected Table

**File**: `db/migrations/010-measurement-features-table-corrected.sql`

```sql
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
```

## Key Changes from Original Schema

| Aspect | Original (WRONG) | Corrected (RIGHT) |
|--------|------------------|-------------------|
| **Primary Key** | `(timestamp, measurement_id)` | `recording_session_id` |
| **Scope** | Single measurement (1 receiver) | Entire session (2-7 receivers) |
| **Foreign Key** | `measurements(id)` | `recording_sessions(id)` |
| **Ground Truth** | ❌ Missing | ✅ `tx_latitude`, `tx_longitude`, `tx_known` |
| **GDOP** | ❌ Missing | ✅ `gdop` (geometry quality) |
| **Use Case** | ❌ Cannot do localization | ✅ Ready for ML training |

## Expected JSONB Structure (UNCHANGED)

**Example `receiver_features` entry** (one of 2-7 in the array):

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

**Example `extraction_metadata` for synthetic**:

```json
{
  "extraction_method": "synthetic",
  "iq_duration_ms": 1000.0,
  "sample_rate_hz": 200000,
  "num_chunks": 5,
  "chunk_duration_ms": 200.0,
  "synthetic_dataset_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example `extraction_metadata` for real recording**:

```json
{
  "extraction_method": "recorded",
  "iq_duration_ms": 1000.0,
  "sample_rate_hz": 200000,
  "num_chunks": 5,
  "chunk_duration_ms": 200.0,
  "recording_session_id": "660e8400-e29b-41d4-a716-446655440001"
}
```

## Run Migration

Execute the corrected migration:

```bash
# Via Docker
DOCKER_HOST="" docker exec heimdall-postgres psql -U heimdall_user -d heimdall -f /migrations/010-measurement-features-table-corrected.sql

# Or manually
psql -h localhost -U heimdall_user -d heimdall -f db/migrations/010-measurement-features-table-corrected.sql
```

## Verification

### 1. Check Table Exists

```sql
\d heimdall.measurement_features
```

Expected output:
```
Column                  | Type             | Nullable | Default
------------------------+------------------+----------+---------
recording_session_id    | uuid             | not null |
timestamp               | timestamptz      | not null |
receiver_features       | jsonb[]          | not null |
tx_latitude             | double precision |          |
tx_longitude            | double precision |          |
tx_altitude_m           | double precision |          |
tx_power_dbm            | double precision |          |
tx_known                | boolean          |          | false
extraction_metadata     | jsonb            | not null |
overall_confidence      | double precision | not null |
mean_snr_db             | double precision |          |
num_receivers_detected  | integer          |          |
gdop                    | double precision |          |
extraction_failed       | boolean          |          | false
error_message           | text             |          |
created_at              | timestamptz      |          | now()

Indexes:
    "measurement_features_pkey" PRIMARY KEY, btree (recording_session_id)
```

### 2. Check Indexes

```sql
\di heimdall.idx_measurement_features*
```

Expected: 8 indexes created.

### 3. Test Insert (Synthetic Sample with 4 Receivers)

```sql
INSERT INTO heimdall.measurement_features (
    recording_session_id, timestamp, receiver_features,
    tx_latitude, tx_longitude, tx_power_dbm, tx_known,
    extraction_metadata, overall_confidence, mean_snr_db,
    num_receivers_detected, gdop
)
VALUES (
    gen_random_uuid(),
    NOW(),
    ARRAY[
        '{"rx_id": "Torino", "rx_lat": 45.044, "rx_lon": 7.672, "signal_present": true, "snr": {"mean": 22.5}}'::jsonb,
        '{"rx_id": "Milano", "rx_lat": 45.464, "rx_lon": 9.188, "signal_present": true, "snr": {"mean": 20.0}}'::jsonb,
        '{"rx_id": "Bologna", "rx_lat": 44.494, "rx_lon": 11.342, "signal_present": true, "snr": {"mean": 18.5}}'::jsonb,
        '{"rx_id": "Genova", "rx_lat": 44.407, "rx_lon": 8.934, "signal_present": true, "snr": {"mean": 16.0}}'::jsonb
    ],
    45.123,  -- TX latitude (ground truth)
    7.456,   -- TX longitude (ground truth)
    33.0,    -- TX power
    TRUE,    -- Ground truth known (synthetic)
    '{"extraction_method": "synthetic", "iq_duration_ms": 1000.0, "sample_rate_hz": 200000, "num_chunks": 5}'::jsonb,
    0.85,    -- Overall confidence
    19.25,   -- Mean SNR across 4 receivers
    4,       -- 4 receivers detected
    8.5      -- GDOP (good geometry)
);
```

Expected: Insert succeeds without errors.

### 4. Test Query (Verify Multi-Receiver Structure)

```sql
SELECT
    recording_session_id,
    jsonb_array_length(receiver_features) as num_receivers,
    tx_latitude,
    tx_longitude,
    tx_known,
    gdop,
    receiver_features[1]->>'rx_id' as first_receiver,
    receiver_features[4]->>'rx_id' as fourth_receiver
FROM heimdall.measurement_features
LIMIT 1;
```

Expected output:
```
recording_session_id | num_receivers | tx_latitude | tx_longitude | tx_known | gdop | first_receiver | fourth_receiver
---------------------+---------------+-------------+--------------+----------+------+----------------+-----------------
<uuid>              | 4             | 45.123      | 7.456        | true     | 8.5  | Torino         | Genova
```

### 5. Clean Up Test Data

```sql
DELETE FROM heimdall.measurement_features
WHERE extraction_metadata->>'extraction_method' = 'test';
```

## Impact on ML Training

### Before (BROKEN):
```python
# Each receiver is a separate sample ❌
sample_1 = {
    'features': [torino_features],  # Only Torino
    'tx_position': ???  # Cannot determine
}
sample_2 = {
    'features': [milano_features],  # Only Milano
    'tx_position': ???  # Cannot determine
}
# Impossible to train localization model
```

### After (CORRECT):
```python
# All receivers for one transmission event ✅
sample = {
    'features': [
        torino_features,   # Receiver 1
        milano_features,   # Receiver 2
        bologna_features,  # Receiver 3
        genova_features    # Receiver 4
    ],
    'tx_position': (45.123, 7.456)  # Ground truth
}

# Model learns: features from 4 receivers → TX position
model.fit(X=receiver_features, y=tx_position)
```

## Success Criteria

- ✅ Table `heimdall.measurement_features` uses `recording_session_id` as PRIMARY KEY
- ✅ All 8 indexes created
- ✅ Foreign key constraint to `recording_sessions` works
- ✅ Ground truth columns (`tx_latitude`, `tx_longitude`, `tx_known`) present
- ✅ GDOP column added for geometry quality
- ✅ JSONB array accepts multi-receiver feature data
- ✅ Test insert/query/delete successful with 4 receivers
- ✅ Schema supports ML localization training

## Next Step

After applying this fix:
- **DO NOT run** the original `01-database-schema.md` migration
- Proceed to **`02-feature-extractor-core.md`** (no changes needed)
- The corrected schema will be used in **`04-synthetic-pipeline.md`** and **`05-real-pipeline.md`** (already updated)
