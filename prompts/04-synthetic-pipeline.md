# Step 4: Integrate IQ Generation into Synthetic Pipeline

## Objective

Update the synthetic data generation pipeline to:
1. Generate realistic IQ samples using `SyntheticIQGenerator`
2. Extract features using `RFFeatureExtractor`
3. Save features to `measurement_features` table
4. Save first 100 IQ samples to MinIO for debugging
5. Use multiprocessing (up to 24 workers) for parallel generation

## Context

Currently, `synthetic_generator.py` generates high-level features directly. We need to:
- Generate 1000ms IQ samples → extract features → save features
- Store raw IQ for first 100 samples only (debugging)
- Use same feature extraction as real recordings (consistency)

## Implementation

### 1. Update Synthetic Generator

**File**: `services/training/src/data/synthetic_generator.py`

Add imports:

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import io

from .iq_generator import SyntheticIQGenerator, IQSample
from .feature_extractor import RFFeatureExtractor
```

Add helper function for multiprocessing:

```python
def _generate_single_sample(args):
    """
    Generate a single synthetic sample (for multiprocessing).

    This function must be at module level for pickle serialization.

    Args:
        args: Tuple of (sample_index, receivers_config, tx_config, config, seed)

    Returns:
        Tuple of (receiver_features, extraction_metadata, quality_metrics, iq_samples_dict)
    """
    sample_idx, receivers_config, tx_config, config, base_seed = args

    # Import inside worker (needed for multiprocessing)
    import numpy as np
    from data.iq_generator import SyntheticIQGenerator
    from data.feature_extractor import RFFeatureExtractor
    from data.propagation import calculate_path_loss, calculate_receivers_signal

    # Create unique seed for this sample
    sample_seed = base_seed + sample_idx if base_seed else None

    # Initialize generators
    iq_generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,  # 200 kHz
        duration_ms=1000.0,       # 1 second
        seed=sample_seed
    )
    feature_extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    # Generate random TX position
    rng = np.random.default_rng(sample_seed)

    # ... (use existing TX position generation logic from synthetic_generator.py)
    # Generate tx_lat, tx_lon, tx_power_dbm, frequency_mhz, etc.

    # Calculate signal at each receiver (use existing logic)
    receiver_signals = calculate_receivers_signal(
        tx_lat, tx_lon, tx_power_dbm, frequency_mhz, receivers_config
    )

    # Generate IQ samples for each receiver
    iq_samples = {}
    receiver_features_list = []

    for receiver in receivers_config:
        rx_id = receiver['name']
        rx_lat = receiver['latitude']
        rx_lon = receiver['longitude']

        # Get signal parameters from propagation model
        signal_params = receiver_signals[rx_id]

        # Generate IQ sample
        iq_sample = iq_generator.generate_iq_sample(
            center_frequency_hz=frequency_mhz * 1e6,
            signal_power_dbm=signal_params['rssi_dbm'],
            noise_floor_dbm=signal_params['noise_floor_dbm'],
            snr_db=signal_params['snr_db'],
            frequency_offset_hz=rng.uniform(-50, 50),  # Doppler/osc drift
            bandwidth_hz=12500.0,  # FM signal bandwidth
            rx_id=rx_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            timestamp=float(sample_idx)
        )

        # Store IQ sample (for first 100 samples only)
        if sample_idx < 100:
            iq_samples[rx_id] = iq_sample

        # Extract features (chunked: 1000ms → 5×200ms with aggregation)
        features_dict = feature_extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        receiver_features_list.append(features_dict)

    # Calculate overall quality metrics
    snr_values = [f['snr']['mean'] for f in receiver_features_list if f['signal_present']]
    num_receivers_detected = sum(1 for f in receiver_features_list if f['signal_present'])
    mean_snr_db = float(np.mean(snr_values)) if snr_values else 0.0

    # Calculate overall confidence (weighted average)
    confidence_scores = [f.get('delay_spread_confidence', {}).get('mean', 0.8)
                        for f in receiver_features_list if f['signal_present']]
    overall_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

    extraction_metadata = {
        'extraction_method': 'synthetic',
        'iq_duration_ms': 1000.0,
        'sample_rate_hz': 200_000,
        'num_chunks': 5,
        'chunk_duration_ms': 200.0,
        'generated_at': sample_idx
    }

    quality_metrics = {
        'overall_confidence': overall_confidence,
        'mean_snr_db': mean_snr_db,
        'num_receivers_detected': num_receivers_detected
    }

    return (receiver_features_list, extraction_metadata, quality_metrics, iq_samples)
```

Update `generate_synthetic_data()` function:

```python
async def generate_synthetic_data(
    dataset_id: UUID,
    num_samples: int,
    receivers_config: list,
    tx_config: dict,
    config: dict,
    pool,
    progress_callback=None,
    seed: Optional[int] = None
) -> dict:
    """
    Generate synthetic training data with IQ samples and feature extraction.

    Args:
        dataset_id: Dataset UUID
        num_samples: Number of samples to generate
        receivers_config: List of receiver configurations
        tx_config: Transmitter configuration
        config: Generation parameters (frequency, power, SNR thresholds, etc.)
        pool: Database connection pool
        progress_callback: Optional callback for progress updates
        seed: Random seed for reproducibility

    Returns:
        dict with generation statistics
    """
    logger.info(f"Starting synthetic data generation: {num_samples} samples")

    # Extract generation parameters
    min_snr_db = config.get('min_snr_db', 3.0)
    min_receivers = config.get('min_receivers', 3)
    max_gdop = config.get('max_gdop', 10.0)

    # Determine number of workers (up to 24)
    num_workers = min(24, mp.cpu_count())
    logger.info(f"Using {num_workers} worker processes for parallel generation")

    # Prepare arguments for parallel processing
    args_list = [
        (i, receivers_config, tx_config, config, seed)
        for i in range(num_samples)
    ]

    # Generate samples in parallel
    generated_samples = []
    iq_samples_to_save = {}  # First 100 IQ samples

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_single_sample, args): args[0]
                  for args in args_list}

        completed = 0

        for future in as_completed(futures):
            sample_idx = futures[future]

            try:
                receiver_features, extraction_metadata, quality_metrics, iq_samples = future.result()

                # Validation checks
                if quality_metrics['num_receivers_detected'] < min_receivers:
                    logger.debug(f"Sample {sample_idx}: rejected (receivers={quality_metrics['num_receivers_detected']} < {min_receivers})")
                    continue

                if quality_metrics['mean_snr_db'] < min_snr_db:
                    logger.debug(f"Sample {sample_idx}: rejected (SNR={quality_metrics['mean_snr_db']:.1f} < {min_snr_db})")
                    continue

                # Store sample
                generated_samples.append({
                    'sample_idx': sample_idx,
                    'receiver_features': receiver_features,
                    'extraction_metadata': extraction_metadata,
                    'quality_metrics': quality_metrics
                })

                # Store IQ samples (first 100 only)
                if sample_idx < 100 and iq_samples:
                    iq_samples_to_save[sample_idx] = iq_samples

                completed += 1

                # Progress update
                if progress_callback and completed % 100 == 0:
                    await progress_callback(completed, num_samples)
                    logger.info(f"Progress: {completed}/{num_samples} samples generated")

            except Exception as e:
                logger.error(f"Error generating sample {sample_idx}: {e}")
                continue

    logger.info(f"Generated {len(generated_samples)} valid samples (success rate: {len(generated_samples)/num_samples*100:.1f}%)")

    # Save features to database
    await save_features_to_db(dataset_id, generated_samples, pool)

    # Save IQ samples to MinIO (first 100 only)
    if iq_samples_to_save:
        await save_iq_to_minio(dataset_id, iq_samples_to_save)

    return {
        'total_generated': len(generated_samples),
        'total_attempted': num_samples,
        'success_rate': len(generated_samples) / num_samples if num_samples > 0 else 0,
        'iq_samples_saved': len(iq_samples_to_save)
    }
```

Add database save function:

```python
async def save_features_to_db(
    dataset_id: UUID,
    samples: list[dict],
    pool
) -> None:
    """
    Save extracted features to measurement_features table.

    Args:
        dataset_id: Dataset UUID
        samples: List of sample dicts with receiver_features, metadata, quality_metrics
        pool: Database connection pool
    """
    async with pool.acquire() as conn:
        # Create measurement records first
        insert_measurement_query = text("""
            INSERT INTO heimdall.measurements (
                id, websdr_station_id, frequency_hz, signal_strength_db,
                iq_data_location, created_at
            )
            VALUES (
                :id, :websdr_station_id, :frequency_hz, :signal_strength_db,
                :iq_data_location, NOW()
            )
        """)

        insert_features_query = text("""
            INSERT INTO heimdall.measurement_features (
                timestamp, measurement_id, receiver_features, extraction_metadata,
                overall_confidence, mean_snr_db, num_receivers_detected,
                extraction_failed, created_at
            )
            VALUES (
                NOW(), :measurement_id, CAST(:receiver_features AS jsonb),
                CAST(:extraction_metadata AS jsonb),
                :overall_confidence, :mean_snr_db, :num_receivers_detected,
                FALSE, NOW()
            )
        """)

        for sample in samples:
            # Create measurement ID
            measurement_id = uuid.uuid4()

            # Get first receiver for station reference
            first_rx = sample['receiver_features'][0]

            # Insert measurement record
            await conn.execute(
                insert_measurement_query,
                {
                    'id': measurement_id,
                    'websdr_station_id': first_rx['rx_id'],
                    'frequency_hz': int(first_rx.get('spectral_centroid', {}).get('mean', 144_000_000)),
                    'signal_strength_db': first_rx.get('rssi', {}).get('mean', -999),
                    'iq_data_location': f"synthetic/{dataset_id}/{sample['sample_idx']}.bin" if sample['sample_idx'] < 100 else None
                }
            )

            # Insert features
            await conn.execute(
                insert_features_query,
                {
                    'measurement_id': measurement_id,
                    'receiver_features': json.dumps(sample['receiver_features']),
                    'extraction_metadata': json.dumps(sample['extraction_metadata']),
                    'overall_confidence': sample['quality_metrics']['overall_confidence'],
                    'mean_snr_db': sample['quality_metrics']['mean_snr_db'],
                    'num_receivers_detected': sample['quality_metrics']['num_receivers_detected']
                }
            )
```

Add MinIO save function:

```python
async def save_iq_to_minio(
    dataset_id: UUID,
    iq_samples_dict: dict[int, dict[str, IQSample]]
) -> None:
    """
    Save IQ samples to MinIO.

    Args:
        dataset_id: Dataset UUID
        iq_samples_dict: Dict mapping sample_idx to dict of {rx_id: IQSample}
    """
    from storage.minio_client import get_minio_client

    minio_client = get_minio_client()
    bucket_name = "heimdall-synthetic-iq"

    # Ensure bucket exists
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    logger.info(f"Saving {len(iq_samples_dict)} IQ samples to MinIO")

    for sample_idx, receivers_iq in iq_samples_dict.items():
        for rx_id, iq_sample in receivers_iq.items():
            # Binary format: complex64 array
            buffer = io.BytesIO()
            np.save(buffer, iq_sample.samples)
            buffer.seek(0)

            object_name = f"synthetic/{dataset_id}/{sample_idx}/{rx_id}.npy"

            minio_client.put_object(
                bucket_name,
                object_name,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )

    logger.info(f"Saved {len(iq_samples_dict)} IQ samples to MinIO bucket '{bucket_name}'")
```

### 2. Update Celery Task

**File**: `services/training/src/tasks/training_task.py`

Update `start_synthetic_generation()` to call new function:

```python
@celery_app.task(name='src.tasks.training_task.start_synthetic_generation')
def start_synthetic_generation(job_id: str):
    """Start synthetic data generation with IQ samples."""
    logger.info(f"Starting synthetic generation task for job {job_id}")

    # ... (existing job lookup logic)

    # Call updated generate_synthetic_data()
    stats = await generate_synthetic_data(
        dataset_id=dataset_id,
        num_samples=config['num_samples'],
        receivers_config=receivers_config,
        tx_config=tx_config,
        config=config,
        pool=pool,
        progress_callback=update_progress,
        seed=config.get('seed')
    )

    logger.info(f"Generation complete: {stats['total_generated']} samples, "
               f"{stats['iq_samples_saved']} IQ samples saved to MinIO")
```

## Verification

### 1. Run Tests

```bash
# Test feature extraction
DOCKER_HOST="" docker exec heimdall-training pytest src/tests/test_feature_extractor_basic.py -v

# Test IQ generation
DOCKER_HOST="" docker exec heimdall-training pytest src/tests/test_iq_generator.py -v
```

### 2. Generate Test Dataset

From Training Dashboard UI:
1. Click "Generate Synthetic Data"
2. Set parameters:
   - Name: "Test IQ Generation"
   - Samples: 1000
   - Frequency: 144 MHz
   - Min SNR: 3 dB
   - Min Receivers: 3
   - Max GDOP: 50
3. Submit

Expected: Job completes successfully with ~900-950 samples (95% success rate).

### 3. Verify Database

```sql
-- Check features were saved
SELECT COUNT(*) FROM heimdall.measurement_features;

-- Check feature structure
SELECT
    measurement_id,
    jsonb_array_length(receiver_features) as num_receivers,
    overall_confidence,
    mean_snr_db,
    num_receivers_detected
FROM heimdall.measurement_features
LIMIT 5;

-- Check one receiver's features
SELECT
    receiver_features[1]->>'rx_id' as rx_id,
    receiver_features[1]->'snr'->>'mean' as snr_mean,
    receiver_features[1]->'rssi'->>'mean' as rssi_mean
FROM heimdall.measurement_features
LIMIT 1;
```

Expected:
- 900-950 rows in `measurement_features`
- 7 receivers per sample
- Confidence scores 0.6-0.95
- SNR values realistic (3-30 dB)

### 4. Verify MinIO

```bash
# List IQ samples
DOCKER_HOST="" docker exec heimdall-minio mc ls myminio/heimdall-synthetic-iq/synthetic/
```

Expected: 100 sample directories with 7 `.npy` files each (one per receiver).

### 5. Check Logs

```bash
DOCKER_HOST="" docker compose logs training | grep "Generation complete"
```

Expected:
```
Generation complete: 950 samples, 100 IQ samples saved to MinIO
```

## Performance Expectations

- **10k samples**: ~2-3 minutes on 24-core CPU
- **100k samples**: ~20-30 minutes
- **Memory usage**: ~2-4 GB (parallel processing overhead)
- **Disk usage**: ~5 KB per feature row, ~1.6 MB per IQ sample (100 samples × 7 RX = 112 MB)

## Success Criteria

- ✅ IQ generation integrated into synthetic pipeline
- ✅ Features extracted and saved to `measurement_features` table
- ✅ First 100 IQ samples saved to MinIO
- ✅ Multiprocessing uses up to 24 workers
- ✅ 95%+ success rate for realistic parameters
- ✅ All tests pass
- ✅ Dashboard shows completed datasets with sample counts

## Next Step

Proceed to **`05-real-pipeline.md`** to implement feature extraction for real recording sessions.
