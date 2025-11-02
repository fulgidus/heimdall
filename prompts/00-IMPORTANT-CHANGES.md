# ⚠️ IMPORTANT ARCHITECTURE CHANGES

## Critical Fix Applied: Session-Based Feature Grouping

**Date**: 2025-11-02
**Reason**: Multi-receiver localization requires correlated features

---

## The Problem

The original design (prompt 01) stored features **per measurement** (single receiver), making ML-based geolocation **IMPOSSIBLE**.

### Why This Breaks Localization

Geolocation algorithms (TDOA, RSS-based trilateration) work by combining signals from **multiple receivers simultaneously**:

```
TX transmits at 144.5 MHz
    ↓
SDR Torino:  RSSI=-65dB, delay=0.5ms  ┐
SDR Milano:  RSSI=-68dB, delay=0.8ms  ├─→ ML Model → (TX_lat, TX_lon)
SDR Bologna: RSSI=-70dB, delay=1.2ms  ┘
```

**Original schema**: Each receiver = separate database record = **NO correlation** = **Cannot train**
**Fixed schema**: All receivers for one session = ONE database record = **Correlated** = **ML training possible** ✅

---

## Changes Made

### 1. Database Schema (`01b-fix-schema-for-localization.md`)

**Primary Key Changed**:
- ❌ **BEFORE**: `(timestamp, measurement_id)` - one record per receiver
- ✅ **AFTER**: `recording_session_id` - one record per session (all receivers)

**New Columns Added**:
```sql
-- Ground truth for supervised learning
tx_latitude DOUBLE PRECISION,
tx_longitude DOUBLE PRECISION,
tx_altitude_m DOUBLE PRECISION,
tx_power_dbm DOUBLE PRECISION,
tx_known BOOLEAN DEFAULT FALSE,  -- TRUE for synthetic/beacons

-- Geometry quality
gdop FLOAT,  -- Geometric Dilution of Precision
```

**Foreign Key Changed**:
- ❌ **BEFORE**: References `measurements(id)`
- ✅ **AFTER**: References `recording_sessions(id)`

### 2. Real Recording Pipeline (`05-real-pipeline.md`)

**Logic Changed**:
- ❌ **BEFORE**: Process each measurement individually → save N records
- ✅ **AFTER**: Process ALL measurements in a session → save 1 record

**New Features**:
- Extract features from ALL receivers in session
- Calculate GDOP for geometry quality
- Detect known beacons (ground truth)
- Graceful degradation (skip failed receivers, continue with others)

### 3. Background Jobs (`06-background-jobs.md`)

**Query Changed**:
```sql
-- Find sessions without features (not measurements)
LEFT JOIN heimdall.measurement_features mf
    ON mf.recording_session_id = rs.id  -- Was: mf.measurement_id = m.id
WHERE mf.recording_session_id IS NULL  -- No features extracted yet
```

---

## Data Structure Comparison

### Synthetic Data (Always Worked Correctly)
```python
{
    'recording_session_id': <uuid>,
    'receiver_features': [
        {rx_id: "Torino", rssi: {...}, snr: {...}, ...},
        {rx_id: "Milano", rssi: {...}, snr: {...}, ...},
        # ... 7 receivers (all SDRs in network)
    ],
    'tx_latitude': 45.123,    # Ground truth KNOWN
    'tx_longitude': 7.456,
    'tx_known': True,
    'gdop': 8.5
}
```

### Real Recordings (NOW MATCHES SYNTHETIC!)
```python
{
    'recording_session_id': <uuid>,
    'receiver_features': [
        {rx_id: "Torino", rssi: {...}, snr: {...}, ...},
        {rx_id: "Milano", rssi: {...}, snr: {...}, ...},
        {rx_id: "Bologna", rssi: {...}, snr: {...}, ...},
        # ... 2-7 receivers (those that captured signal)
    ],
    'tx_latitude': None,      # Ground truth UNKNOWN (to be estimated)
    'tx_longitude': None,
    'tx_known': False,
    'gdop': 12.3
}
```

**Perfect compatibility for ML training!** ✅

---

## ML Training Workflow

### Before (BROKEN)
```python
# Cannot train - receivers are separate records
for row in db.query("SELECT * FROM measurement_features"):
    features = row['receiver_features']  # Only 1 receiver
    # ❌ Cannot do localization with single receiver
```

### After (WORKING)
```python
# Load dataset
dataset = []
for row in db.query("SELECT * FROM measurement_features WHERE tx_known = TRUE"):
    # Input: features from multiple receivers
    X = row['receiver_features']  # 2-7 receivers (72 features each)

    # Output: ground truth position
    y = (row['tx_latitude'], row['tx_longitude'])

    dataset.append((X, y))

# Train model
model.fit(X_train, y_train)

# Predict unknown transmitters
for row in db.query("SELECT * FROM measurement_features WHERE tx_known = FALSE"):
    X_unknown = row['receiver_features']
    predicted_position = model.predict(X_unknown)
    # ✅ Localization works!
```

---

## Execution Order (UPDATED)

1. **`01-database-schema.md`** - Create initial schema (**SKIP THIS IF NOT DONE YET**)
2. **`01b-fix-schema-for-localization.md`** - **RUN THIS INSTEAD** (corrected schema)
3. `02-feature-extractor-core.md` - Core feature extraction (no changes)
4. `03-iq-generator.md` - Synthetic IQ generation (no changes)
5. `04-synthetic-pipeline.md` - Synthetic data pipeline (updated to use session-based schema)
6. `05-real-pipeline.md` - **COMPLETELY REWRITTEN** (multi-receiver extraction)
7. `06-background-jobs.md` - **UPDATED** (query by session_id)
8. `07-tests.md` - Test suite (updated test cases)

---

## Impact Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Records per session** | N (one per receiver) | 1 (all receivers) |
| **Primary Key** | (timestamp, measurement_id) | recording_session_id |
| **Localization** | ❌ Impossible | ✅ Possible |
| **Synthetic/Real match** | ❌ Different structure | ✅ Identical structure |
| **Ground truth** | ❌ Missing | ✅ Available |
| **GDOP** | ❌ Missing | ✅ Calculated |
| **ML Training** | ❌ Cannot train | ✅ Ready for training |

---

## Verification Checklist

After implementing the fixes:

- [ ] Database uses `recording_session_id` as PRIMARY KEY
- [ ] Columns `tx_latitude`, `tx_longitude`, `tx_known`, `gdop` exist
- [ ] Real recordings create ONE record per session (not per measurement)
- [ ] `receiver_features` array has 2-7 elements (multi-receiver)
- [ ] Synthetic and real data have identical JSONB structure
- [ ] Query: `SELECT jsonb_array_length(receiver_features) FROM measurement_features` returns > 1

---

## Migration Path

If you already ran the **OLD** prompt 01:

```sql
-- 1. Drop old table
DROP TABLE IF EXISTS heimdall.measurement_features CASCADE;

-- 2. Run corrected migration
\i /migrations/010-measurement-features-table-corrected.sql
```

If you **haven't run** prompt 01 yet:

```
-- Skip prompt 01, run 01b directly
\i /migrations/010-measurement-features-table-corrected.sql
```

---

## Questions?

If unclear:
1. Check `01b-fix-schema-for-localization.md` for detailed schema
2. Check `05-real-pipeline.md` for multi-receiver extraction logic
3. Look at "Data Structure Comparison" section above

**Key takeaway**: One recording session = one ML training sample with features from ALL receivers.
