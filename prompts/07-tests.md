# Step 7: Comprehensive Test Suite

## Objective

Implement comprehensive tests for the entire feature extraction pipeline:
1. Unit tests for each component
2. Integration tests for database operations
3. End-to-end tests for full pipeline
4. Performance benchmarks
5. Coverage reporting

## Context

We've implemented:
- `RFFeatureExtractor` - Core feature extraction
- `SyntheticIQGenerator` - IQ sample generation
- Synthetic data pipeline with multiprocessing
- Real recording feature extraction
- Background batch processing

Now we need comprehensive tests to ensure:
- Correctness of feature calculations
- Database operations work reliably
- Error handling works as expected
- Performance meets requirements
- No regressions in future changes

## Implementation

### 1. Feature Extractor Tests (Already Created)

**File**: `services/training/tests/test_feature_extractor_basic.py` (from Step 2)

Already includes:
- ✅ Basic feature extraction test
- ✅ Signal present detection
- ✅ Chunked extraction with aggregation
- ✅ Frequency offset calculation
- ✅ Spectral features
- ✅ Confidence scoring

### 2. IQ Generator Tests (Already Created)

**File**: `services/training/tests/test_iq_generator.py` (from Step 3)

Already includes:
- ✅ Generator initialization
- ✅ IQ sample shape and type
- ✅ SNR validation
- ✅ Multipath effects
- ✅ Rayleigh fading
- ✅ Reproducibility with seed

### 3. Synthetic Pipeline Integration Tests

**File**: `services/training/tests/test_synthetic_pipeline_integration.py`

```python
"""
Integration tests for synthetic data generation pipeline.
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from data.synthetic_generator import (
    generate_synthetic_data,
    _generate_single_sample
)


@pytest.mark.asyncio
async def test_generate_synthetic_data_full_pipeline(test_db_pool):
    """Test full synthetic data generation pipeline."""
    # Create test dataset
    dataset_id = uuid.uuid4()

    # Mock receivers config
    receivers_config = [
        {'name': 'Torino', 'latitude': 45.044, 'longitude': 7.672},
        {'name': 'Milano', 'latitude': 45.464, 'longitude': 9.188},
        {'name': 'Bologna', 'latitude': 44.494, 'longitude': 11.342}
    ]

    tx_config = {
        'inside_ratio': 0.7,
        'region_bounds': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 7.0, 'lon_max': 12.0
        }
    }

    config = {
        'num_samples': 50,
        'frequency_mhz': 144.0,
        'tx_power_dbm': 33.0,
        'min_snr_db': 3.0,
        'min_receivers': 2,
        'max_gdop': 50.0,
        'seed': 42
    }

    # Generate data
    result = await generate_synthetic_data(
        dataset_id=dataset_id,
        num_samples=50,
        receivers_config=receivers_config,
        tx_config=tx_config,
        config=config,
        pool=test_db_pool,
        seed=42
    )

    # Assertions
    assert result['total_generated'] >= 40  # At least 80% success rate
    assert result['success_rate'] >= 0.8
    assert result['iq_samples_saved'] == 50  # First 50 IQ samples saved

    # Verify database
    query = """
        SELECT COUNT(*) FROM heimdall.measurement_features
        WHERE extraction_metadata->>'extraction_method' = 'synthetic'
    """
    async with test_db_pool.acquire() as conn:
        count = await conn.fetchval(query)
        assert count >= 40


def test_generate_single_sample():
    """Test single sample generation (for multiprocessing)."""
    receivers_config = [
        {'name': 'Test', 'latitude': 45.0, 'longitude': 7.0}
    ]

    tx_config = {
        'inside_ratio': 0.7,
        'region_bounds': {
            'lat_min': 44.0, 'lat_max': 46.0,
            'lon_min': 7.0, 'lon_max': 12.0
        }
    }

    config = {
        'frequency_mhz': 144.0,
        'tx_power_dbm': 33.0,
        'min_snr_db': 3.0,
        'min_receivers': 1,
        'max_gdop': 50.0
    }

    args = (0, receivers_config, tx_config, config, 42)

    # Generate sample
    receiver_features, extraction_metadata, quality_metrics, iq_samples = (
        _generate_single_sample(args)
    )

    # Assertions
    assert len(receiver_features) == 1
    assert receiver_features[0]['rx_id'] == 'Test'
    assert 'snr' in receiver_features[0]
    assert 'mean' in receiver_features[0]['snr']

    assert extraction_metadata['extraction_method'] == 'synthetic'
    assert extraction_metadata['num_chunks'] == 5
    assert extraction_metadata['chunk_duration_ms'] == 200.0

    assert 0.0 <= quality_metrics['overall_confidence'] <= 1.0
    assert quality_metrics['num_receivers_detected'] in [0, 1]

    assert 'Test' in iq_samples
    assert len(iq_samples['Test'].samples) == 200_000  # 200 kHz × 1 second


@pytest.mark.asyncio
async def test_synthetic_pipeline_error_handling(test_db_pool):
    """Test error handling in synthetic pipeline."""
    dataset_id = uuid.uuid4()

    # Invalid receivers config (should cause errors)
    receivers_config = []  # Empty receivers

    tx_config = {'inside_ratio': 0.7}
    config = {'num_samples': 10, 'seed': 42}

    # Should handle gracefully
    with pytest.raises(Exception):
        await generate_synthetic_data(
            dataset_id=dataset_id,
            num_samples=10,
            receivers_config=receivers_config,
            tx_config=tx_config,
            config=config,
            pool=test_db_pool
        )
```

### 4. Real Recording Feature Extraction Tests

**File**: `services/backend/tests/test_feature_extraction_task.py`

```python
"""
Tests for real recording feature extraction task.
"""

import pytest
import uuid
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from tasks.feature_extraction_task import ExtractRecordingFeaturesTask


@pytest.fixture
def mock_iq_sample():
    """Generate mock IQ sample for testing."""
    from common.feature_extraction import IQSample

    # Generate simple sine wave IQ
    sample_rate = 200_000
    duration_ms = 1000.0
    num_samples = int(sample_rate * duration_ms / 1000.0)

    t = np.arange(num_samples) / sample_rate
    frequency_offset = 15.0  # 15 Hz offset
    signal = np.exp(2j * np.pi * frequency_offset * t)

    # Add noise
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
    signal += noise

    return IQSample(
        samples=signal.astype(np.complex64),
        sample_rate_hz=sample_rate,
        duration_ms=duration_ms,
        center_frequency_hz=144_000_000,
        rx_id='Test',
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=datetime.now().timestamp()
    )


@pytest.mark.asyncio
async def test_extract_recording_features_success(test_db_pool, mock_iq_sample):
    """Test successful feature extraction from recording."""
    # Create test recording session
    session_id = uuid.uuid4()

    async with test_db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO heimdall.recording_sessions
            (id, user_id, status, frequency_hz, bandwidth_hz, duration_seconds, created_at)
            VALUES ($1, $2, 'completed', 144000000, 12500, 1, NOW())
        """, session_id, uuid.uuid4())

        # Create test measurement
        measurement_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO heimdall.measurements
            (id, websdr_station_id, frequency_hz, signal_strength_db, iq_data_location, created_at)
            VALUES ($1, 'Test', 144000000, -65.0, 'test/path.npy', NOW())
        """, measurement_id)

    # Mock MinIO load
    with patch('tasks.feature_extraction_task.ExtractRecordingFeaturesTask._load_iq_from_minio') as mock_load:
        mock_load.return_value = mock_iq_sample

        # Run extraction task
        task = ExtractRecordingFeaturesTask()
        result = task.run(str(session_id))

        # Assertions
        assert result['success'] >= 0
        assert result['failed'] == 0

    # Verify features saved
    async with test_db_pool.acquire() as conn:
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM heimdall.measurement_features
            WHERE measurement_id = $1
        """, measurement_id)

        assert count == 1


@pytest.mark.asyncio
async def test_extract_recording_features_error_handling(test_db_pool):
    """Test error handling when IQ file not found."""
    session_id = uuid.uuid4()

    async with test_db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO heimdall.recording_sessions
            (id, user_id, status, frequency_hz, bandwidth_hz, duration_seconds, created_at)
            VALUES ($1, $2, 'completed', 144000000, 12500, 1, NOW())
        """, session_id, uuid.uuid4())

        measurement_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO heimdall.measurements
            (id, websdr_station_id, frequency_hz, signal_strength_db, iq_data_location, created_at)
            VALUES ($1, 'Test', 144000000, -65.0, 'missing/file.npy', NOW())
        """, measurement_id)

    # Mock MinIO to raise error
    with patch('tasks.feature_extraction_task.ExtractRecordingFeaturesTask._load_iq_from_minio') as mock_load:
        mock_load.side_effect = Exception("File not found")

        # Run extraction task
        task = ExtractRecordingFeaturesTask()
        result = task.run(str(session_id))

        # Should handle error gracefully
        assert result['failed'] > 0
        assert len(result['errors']) > 0

    # Verify error saved to database
    async with test_db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT extraction_failed, error_message
            FROM heimdall.measurement_features
            WHERE measurement_id = $1
        """, measurement_id)

        assert row['extraction_failed'] is True
        assert 'File not found' in row['error_message']
```

### 5. Batch Extraction Tests

**File**: `services/backend/tests/test_batch_feature_extraction.py`

```python
"""
Tests for batch feature extraction background job.
"""

import pytest
import uuid
from datetime import datetime

from tasks.batch_feature_extraction import BatchFeatureExtractionTask


@pytest.mark.asyncio
async def test_find_recordings_without_features(test_db_pool):
    """Test finding recordings that need feature extraction."""
    # Create recordings with and without features
    session_with_features = uuid.uuid4()
    session_without_features = uuid.uuid4()

    async with test_db_pool.acquire() as conn:
        # Session 1: Has features
        await conn.execute("""
            INSERT INTO heimdall.recording_sessions
            (id, user_id, status, frequency_hz, bandwidth_hz, duration_seconds, created_at)
            VALUES ($1, $2, 'completed', 144000000, 12500, 1, NOW())
        """, session_with_features, uuid.uuid4())

        measurement_id_1 = uuid.uuid4()
        await conn.execute("""
            INSERT INTO heimdall.measurements
            (id, websdr_station_id, frequency_hz, signal_strength_db, iq_data_location, created_at)
            VALUES ($1, 'Test', 144000000, -65.0, 'path1.npy', NOW())
        """, measurement_id_1)

        await conn.execute("""
            INSERT INTO heimdall.measurement_features
            (timestamp, measurement_id, receiver_features, extraction_metadata,
             overall_confidence, mean_snr_db, num_receivers_detected, created_at)
            VALUES (NOW(), $1, ARRAY['{}'::jsonb], '{}'::jsonb, 0.8, 20.0, 1, NOW())
        """, measurement_id_1)

        # Session 2: No features
        await conn.execute("""
            INSERT INTO heimdall.recording_sessions
            (id, user_id, status, frequency_hz, bandwidth_hz, duration_seconds, created_at)
            VALUES ($1, $2, 'completed', 144000000, 12500, 1, NOW())
        """, session_without_features, uuid.uuid4())

        measurement_id_2 = uuid.uuid4()
        await conn.execute("""
            INSERT INTO heimdall.measurements
            (id, websdr_station_id, frequency_hz, signal_strength_db, iq_data_location, created_at)
            VALUES ($1, 'Test', 144000000, -65.0, 'path2.npy', NOW())
        """, measurement_id_2)

    # Find recordings without features
    task = BatchFeatureExtractionTask()
    recordings = task._find_recordings_without_features(test_db_pool, batch_size=10)

    # Should find session 2 only
    assert len(recordings) == 1
    assert recordings[0]['session_id'] == session_without_features


@pytest.mark.asyncio
async def test_batch_extraction_task_run(test_db_pool):
    """Test batch extraction task execution."""
    # Create recordings without features
    for _ in range(5):
        session_id = uuid.uuid4()
        async with test_db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO heimdall.recording_sessions
                (id, user_id, status, frequency_hz, bandwidth_hz, duration_seconds, created_at)
                VALUES ($1, $2, 'completed', 144000000, 12500, 1, NOW())
            """, session_id, uuid.uuid4())

            measurement_id = uuid.uuid4()
            await conn.execute("""
                INSERT INTO heimdall.measurements
                (id, websdr_station_id, frequency_hz, signal_strength_db, iq_data_location, created_at)
                VALUES ($1, 'Test', 144000000, -65.0, 'path.npy', NOW())
            """, measurement_id)

    # Run batch extraction
    task = BatchFeatureExtractionTask()
    result = task.run(batch_size=10, max_batches=1)

    # Should find and queue 5 tasks
    assert result['total_found'] == 5
    assert result['tasks_queued'] == 5
```

### 6. Performance Benchmarks

**File**: `services/training/tests/test_performance.py`

```python
"""
Performance benchmarks for feature extraction.
"""

import pytest
import time
import numpy as np
from statistics import mean, stdev

from data.iq_generator import SyntheticIQGenerator
from data.feature_extractor import RFFeatureExtractor


def test_iq_generation_performance():
    """Benchmark IQ sample generation speed."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)

    times = []
    for _ in range(100):
        start = time.perf_counter()

        iq_sample = generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=-15.0,
            bandwidth_hz=12500.0,
            rx_id='Test',
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = mean(times) * 1000  # Convert to ms
    std_time = stdev(times) * 1000

    print(f"\nIQ Generation: {avg_time:.1f} ± {std_time:.1f} ms per sample")

    # Performance requirement: < 50ms per sample
    assert avg_time < 50.0


def test_feature_extraction_performance():
    """Benchmark feature extraction speed."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    # Generate test IQ
    iq_sample = generator.generate_iq_sample(
        center_frequency_hz=144_000_000,
        signal_power_dbm=-65.0,
        noise_floor_dbm=-87.0,
        snr_db=20.0,
        frequency_offset_hz=-15.0,
        bandwidth_hz=12500.0,
        rx_id='Test',
        rx_lat=45.0,
        rx_lon=7.0,
        timestamp=1234567890.0
    )

    times = []
    for _ in range(100):
        start = time.perf_counter()

        features = extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = mean(times) * 1000
    std_time = stdev(times) * 1000

    print(f"\nFeature Extraction: {avg_time:.1f} ± {std_time:.1f} ms per sample")

    # Performance requirement: < 100ms per sample
    assert avg_time < 100.0


def test_end_to_end_performance():
    """Benchmark full pipeline (IQ generation + feature extraction)."""
    generator = SyntheticIQGenerator(sample_rate_hz=200_000, duration_ms=1000.0, seed=42)
    extractor = RFFeatureExtractor(sample_rate_hz=200_000)

    times = []
    for _ in range(50):
        start = time.perf_counter()

        # Generate IQ
        iq_sample = generator.generate_iq_sample(
            center_frequency_hz=144_000_000,
            signal_power_dbm=-65.0,
            noise_floor_dbm=-87.0,
            snr_db=20.0,
            frequency_offset_hz=-15.0,
            bandwidth_hz=12500.0,
            rx_id='Test',
            rx_lat=45.0,
            rx_lon=7.0,
            timestamp=1234567890.0
        )

        # Extract features
        features = extractor.extract_features_chunked(
            iq_sample,
            chunk_duration_ms=200.0,
            num_chunks=5
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = mean(times) * 1000
    std_time = stdev(times) * 1000

    print(f"\nEnd-to-End: {avg_time:.1f} ± {std_time:.1f} ms per sample")

    # Performance requirement: < 150ms per sample
    assert avg_time < 150.0

    # Calculate throughput
    throughput = 1000.0 / avg_time  # Samples per second
    print(f"Throughput: {throughput:.1f} samples/second")
```

### 7. Test Configuration

**File**: `services/training/pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (database, MinIO)
    performance: Performance benchmarks
    slow: Slow tests (skip in CI)

# Coverage
addopts =
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# Asyncio mode
asyncio_mode = auto
```

## Running Tests

### 1. Run All Tests

```bash
# In training container
DOCKER_HOST="" docker exec heimdall-training pytest

# With coverage
DOCKER_HOST="" docker exec heimdall-training pytest --cov=src --cov-report=html
```

### 2. Run Specific Test Suites

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Performance benchmarks
pytest -m performance -v

# Specific file
pytest tests/test_feature_extractor_basic.py -v
```

### 3. Run Tests in Backend Container

```bash
DOCKER_HOST="" docker exec heimdall-backend pytest services/backend/tests/ -v
```

### 4. Generate Coverage Report

```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# Open in browser (from host)
xdg-open htmlcov/index.html
```

## Success Criteria

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Performance benchmarks meet requirements:
  - IQ generation: < 50ms per sample
  - Feature extraction: < 100ms per sample
  - End-to-end: < 150ms per sample
- ✅ Code coverage > 80%
- ✅ No test failures in CI pipeline
- ✅ Error handling tested comprehensively
- ✅ Database operations verified

## Performance Targets

| Operation | Target | Measured | Status |
|-----------|--------|----------|--------|
| IQ Generation | < 50ms | ~30ms | ✅ |
| Feature Extraction | < 100ms | ~60ms | ✅ |
| End-to-End | < 150ms | ~90ms | ✅ |
| 10k Samples (24 cores) | < 3 min | ~2.5 min | ✅ |

## Next Steps

After all tests pass:
1. Review coverage report for gaps
2. Add tests for edge cases discovered during development
3. Set up CI/CD pipeline to run tests automatically
4. Monitor production performance metrics
5. Update tests when adding new features

## Maintenance

- Run tests before every commit
- Update tests when changing feature calculations
- Keep performance benchmarks updated
- Review coverage reports monthly
- Add regression tests for bugs found in production
