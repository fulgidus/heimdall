# Step 4: IQ Generation Integration - Implementation Summary

## Overview

Successfully integrated IQ sample generation and feature extraction into the synthetic data pipeline. The system now generates realistic IQ samples, extracts features, and saves them to the database for ML training.

## Architecture

```
Celery Task (Sync)
    ↓
asyncio.run() [async/sync boundary]
    ↓
generate_synthetic_data_with_iq() [async orchestrator]
    ↓
ProcessPoolExecutor (up to 24 workers)
    ↓
_generate_single_sample() [worker function - one per sample]
    ↓
SyntheticIQGenerator → IQ samples (1000ms @ 200kHz, all 7 receivers)
    ↓
RFFeatureExtractor → features (5 chunks × 200ms, aggregated stats)
    ↓
Return: features, metadata, quality_metrics, iq_samples, tx_position
    ↓
Quality Filtering (SNR, receivers, GDOP)
    ↓
save_features_to_db() → measurement_features table
save_iq_to_minio() → heimdall-synthetic-iq bucket (first 100 samples only)
```

## Key Components

### 1. Worker Function: `_generate_single_sample()`

**Location**: `services/training/src/data/synthetic_generator.py` (module level)

**Purpose**: Generate a single synthetic sample with IQ data and features

**Process**:
1. Generate random TX position (70% inside network, 30% outside)
2. Calculate propagation for each receiver (FSPL, terrain, fading)
3. Generate IQ samples using `SyntheticIQGenerator`:
   - Duration: 1000ms
   - Sample rate: 200 kHz
   - Effects: multipath, Rayleigh fading, AWGN
4. Extract features using `RFFeatureExtractor`:
   - 5 chunks of 200ms each
   - 18 base features per chunk
   - Aggregated: mean, std, min, max (72 features total)
5. Calculate quality metrics (SNR, confidence, GDOP)
6. Return all data for validation and storage

**Key Features**:
- Module-level function (required for pickle serialization)
- Deterministic with seed for reproducibility
- Independent execution (no shared state)

### 2. Orchestrator: `generate_synthetic_data_with_iq()`

**Location**: `services/training/src/data/synthetic_generator.py`

**Purpose**: Orchestrate parallel generation and save results

**Process**:
1. Setup multiprocessing (up to 24 workers)
2. Prepare arguments for all samples
3. Submit to ProcessPoolExecutor
4. Collect results as they complete
5. Apply quality filters:
   - Min SNR: 3 dB (configurable)
   - Min receivers: 3 (configurable)
   - Max GDOP: 10 (configurable)
6. Save valid samples to database
7. Save first 100 IQ samples to MinIO
8. Return statistics

**Quality Validation**:
- Rejects samples with insufficient SNR
- Rejects samples with too few receivers
- Rejects samples with poor geometry (high GDOP)
- Expected success rate: 50-95% depending on parameters

### 3. Database Storage: `save_features_to_db()`

**Location**: `services/training/src/data/synthetic_generator.py`

**Purpose**: Save features to `measurement_features` table

**Schema**: `heimdall.measurement_features`
- `recording_session_id`: UUID (primary key, generated for synthetic)
- `receiver_features`: JSONB[] (array of feature objects, one per receiver)
- `tx_latitude`, `tx_longitude`, `tx_altitude_m`: Ground truth position
- `tx_known`: TRUE (for synthetic data)
- `extraction_metadata`: JSONB (method, sample rate, duration, etc.)
- `overall_confidence`: FLOAT (0-1)
- `mean_snr_db`: FLOAT
- `num_receivers_detected`: INT (2-7)
- `gdop`: FLOAT (geometric dilution of precision)

**Features per Receiver** (72 total):
```
Base Features (18):
- rssi_dbm, snr_db, noise_floor_dbm
- frequency_offset_hz, bandwidth_hz, psd_dbm_per_hz
- spectral_centroid_hz, spectral_rolloff_hz
- envelope_mean, envelope_std, envelope_max
- peak_to_avg_ratio_db, zero_crossing_rate
- multipath_delay_spread_us, coherence_bandwidth_khz
- delay_spread_confidence, confidence_score
- signal_present (boolean)

Aggregations per Feature (×4):
- mean, std, min, max
```

### 4. IQ Storage: `save_iq_to_minio()`

**Location**: `services/training/src/data/synthetic_generator.py`

**Purpose**: Save raw IQ samples for debugging

**Storage**:
- Bucket: `heimdall-synthetic-iq`
- Path: `synthetic/{dataset_id}/{sample_idx}/{rx_id}.npy`
- Format: NumPy `.npy` (complex64)
- Samples saved: First 100 only (debugging/validation)

**Why First 100?**:
- Full dataset would be ~112 MB per 100 samples
- 10k samples = 11.2 GB (excessive)
- First 100 provides sufficient validation data
- Features are what matter for training

### 5. Celery Task: `generate_synthetic_data_task()`

**Location**: `services/training/src/tasks/training_task.py`

**Purpose**: Celery task wrapper for synthetic generation

**Changes**:
- Added async/sync boundary with `asyncio.run()`
- Created async database connection with `asyncpg`
- Calls `generate_synthetic_data_with_iq()`
- Updates dataset metadata
- Returns stats with IQ samples saved

## Performance

### Expected Performance (24-core CPU)

| Samples | Time (est.) | Memory | Disk (Features) | Disk (IQ) |
|---------|-------------|--------|-----------------|-----------|
| 100     | ~5 sec      | 2 GB   | 500 KB          | 11 MB     |
| 1,000   | ~30 sec     | 2 GB   | 5 MB            | 112 MB    |
| 10,000  | ~3 min      | 3 GB   | 50 MB           | 112 MB*   |
| 100,000 | ~45-60 min  | 4 GB   | 500 MB          | 112 MB*   |

*IQ storage capped at first 100 samples

### Bottlenecks

1. **IQ Generation**: CPU-bound (FFT, signal processing)
2. **Feature Extraction**: CPU-bound (FFT, autocorrelation)
3. **Database Insertion**: I/O-bound (can be batched)
4. **MinIO Upload**: Network-bound (but only 100 samples)

### Optimizations Applied

- ✅ Multiprocessing with ProcessPoolExecutor
- ✅ Up to 24 workers (configurable)
- ✅ Quality filtering before storage
- ✅ Minimal IQ storage (first 100 only)
- ⏳ Database batching (could be improved)

## Testing

### Unit Tests

**test_iq_generator.py**: Tests IQ generation
- ✅ Initialization
- ✅ Sample shape and type
- ✅ Multipath effects
- ✅ Rayleigh fading
- ✅ Reproducibility with seed

**test_feature_extractor_basic.py**: Tests feature extraction
- ✅ Initialization
- ✅ Feature extraction from clean signal
- ✅ Chunked extraction with aggregation
- ✅ Confidence calculation

**test_synthetic_iq_integration.py**: Integration tests
- ✅ Single sample generation
- ✅ Reproducibility with seed
- ✅ Quality validation

### Manual Verification Steps

1. **Start Infrastructure**:
   ```bash
   docker-compose up -d postgres rabbitmq redis minio
   ```

2. **Run Celery Worker**:
   ```bash
   docker-compose up training
   ```

3. **Submit Generation Job** (via Frontend or API):
   - Navigate to Training Dashboard
   - Click "Generate Synthetic Data"
   - Set parameters:
     - Name: "Test IQ Generation"
     - Samples: 1000
     - Frequency: 144 MHz
     - Min SNR: 3 dB
     - Min Receivers: 3
     - Max GDOP: 50

4. **Verify Database**:
   ```sql
   -- Check features were saved
   SELECT COUNT(*) FROM heimdall.measurement_features;

   -- Check feature structure
   SELECT
       recording_session_id,
       array_length(receiver_features, 1) as num_receivers,
       overall_confidence,
       mean_snr_db,
       num_receivers_detected,
       gdop
   FROM heimdall.measurement_features
   LIMIT 5;

   -- Check one receiver's features
   SELECT
       receiver_features[1]->>'rx_id' as rx_id,
       receiver_features[1]->'snr_db'->>'mean' as snr_mean,
       receiver_features[1]->'rssi_dbm'->>'mean' as rssi_mean
   FROM heimdall.measurement_features
   LIMIT 1;
   ```

5. **Verify MinIO**:
   ```bash
   # List IQ samples
   docker exec heimdall-minio mc ls myminio/heimdall-synthetic-iq/synthetic/
   
   # Should show 100 directories (sample_0 to sample_99)
   # Each with 7 .npy files (one per receiver)
   ```

6. **Check Logs**:
   ```bash
   docker-compose logs training | grep "Generation complete"
   ```
   Expected output:
   ```
   Generation complete: 950 samples, 100 IQ samples saved to MinIO
   ```

## Success Criteria

- ✅ IQ generation integrated into synthetic pipeline
- ✅ Features extracted and saved to `measurement_features` table
- ✅ First 100 IQ samples saved to MinIO
- ✅ Multiprocessing uses up to 24 workers
- ✅ 50-95% success rate for realistic parameters
- ✅ All existing tests still pass
- ⏳ Dashboard shows completed datasets with sample counts (requires frontend testing)

## Known Issues / Limitations

1. **Database Connection**: Currently creates new async connection per task
   - Could be pooled for better performance
   - Not critical for current workload

2. **Progress Callbacks**: Async callback in Celery task
   - Works but could be cleaner
   - Consider event-based progress tracking

3. **Error Handling**: Basic try/catch in worker
   - Individual sample failures are logged but not tracked
   - Could add failure statistics

4. **Memory Usage**: ~3-4 GB for large datasets
   - Could be reduced with streaming writes
   - Not critical for current infrastructure

## Next Steps

1. **Test with Frontend**: Verify UI integration
2. **Performance Benchmarking**: Measure actual performance vs. estimates
3. **Real Pipeline** (Step 5): Implement feature extraction for real recordings
4. **Dataset Management**: Add dataset versioning and cleanup
5. **Monitoring**: Add metrics for generation pipeline

## References

- Problem Statement: `step4-synthetic-iq-integration.md`
- Database Schema: `db/migrations/010-measurement-features-table.sql`
- IQ Generator: `services/training/src/data/iq_generator.py`
- Feature Extractor: `services/training/src/data/rf_feature_extractor.py`
- Synthetic Generator: `services/training/src/data/synthetic_generator.py`
- Training Task: `services/training/src/tasks/training_task.py`
