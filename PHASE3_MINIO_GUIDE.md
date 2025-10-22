# MinIO S3 Integration Guide

## Overview

MinIO integration provides scalable storage for IQ data in the Heimdall RF Acquisition Service. All IQ measurements are stored as `.npy` files in MinIO S3 with associated metadata in JSON format.

## Architecture

### Storage Structure

```
s3://heimdall-raw-iq/
└── sessions/
    └── {task_id}/
        ├── websdr_1.npy                    # IQ data (125k samples, ~1MB)
        ├── websdr_1_metadata.json          # Metadata with metrics
        ├── websdr_2.npy
        ├── websdr_2_metadata.json
        ├── ...
        └── websdr_7.npy
```

### Storage Path Format

- **IQ Data**: `s3://heimdall-raw-iq/sessions/{task_id}/websdr_{websdr_id}.npy`
- **Metadata**: `s3://heimdall-raw-iq/sessions/{task_id}/websdr_{websdr_id}_metadata.json`

## Configuration

### Environment Variables

```bash
MINIO_URL=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_RAW_IQ=heimdall-raw-iq
```

### Pydantic Settings

All configuration is defined in `src/config.py`:

```python
from src.config import settings

# Access MinIO settings
print(settings.minio_url)              # http://minio:9000
print(settings.minio_bucket_raw_iq)    # heimdall-raw-iq
```

## Usage

### Basic Upload

```python
from src.storage.minio_client import MinIOClient
import numpy as np

# Initialize client
client = MinIOClient(
    endpoint_url="http://minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
)

# Prepare IQ data
iq_data = np.random.normal(0, 0.1, 125000).astype(np.complex64)
metadata = {
    'frequency_mhz': 100.0,
    'sample_rate_khz': 12.5,
    'snr_db': 15.5
}

# Upload
success, s3_path = client.upload_iq_data(
    iq_data=iq_data,
    task_id="task_12345",
    websdr_id=1,
    metadata=metadata
)

print(f"Uploaded to: {s3_path}")
# Output: s3://heimdall-raw-iq/sessions/task_12345/websdr_1.npy
```

### Download Data

```python
# Download IQ data
success, iq_data = client.download_iq_data(
    task_id="task_12345",
    websdr_id=1
)

print(f"Downloaded {len(iq_data)} samples")
```

### List Session Measurements

```python
# List all measurements in a session
measurements = client.get_session_measurements("task_12345")

for websdr_id, info in measurements.items():
    print(f"WebSDR {websdr_id}: {info['iq_data_path']} ({info['size_bytes']} bytes)")
```

### Health Check

```python
health = client.health_check()
print(f"MinIO Status: {health['status']}")
print(f"Bucket Accessible: {health['accessible']}")
```

## Celery Task Integration

### save_measurements_to_minio Task

The `save_measurements_to_minio` Celery task handles storage of acquired measurements:

```python
from src.tasks.acquire_iq import save_measurements_to_minio

result = save_measurements_to_minio.delay(
    task_id="task_12345",
    measurements=[
        {
            'websdr_id': 1,
            'frequency_mhz': 100.0,
            'iq_data': iq_array.tolist(),
            'metrics': {...}
        },
        ...
    ]
)
```

### Response Format

```python
{
    'status': 'SUCCESS',  # or 'PARTIAL_FAILURE'
    'message': 'Stored 7 measurements',
    'measurements_count': 7,
    'successful': 7,
    'failed': 0,
    'stored_measurements': [
        {
            'websdr_id': 1,
            's3_path': 's3://heimdall-raw-iq/sessions/task_12345/websdr_1.npy',
            'samples_count': 125000,
            'status': 'SUCCESS'
        },
        ...
    ]
}
```

## Data Format

### IQ Data Storage (.npy)

- **Format**: NumPy binary format
- **Data Type**: `np.complex64` (float32 real + float32 imaginary)
- **Size**: ~500 KB per measurement (125k samples)
- **Load**: `np.load(buffer)` returns `np.ndarray`

### Metadata Storage (.json)

```json
{
  "websdr_id": 1,
  "frequency_mhz": 100.0,
  "sample_rate_khz": 12.5,
  "samples_count": 125000,
  "timestamp_utc": "2024-01-15T10:30:45.123456",
  "metrics": {
    "snr_db": 15.5,
    "frequency_offset_hz": -2.1,
    "power_dbm": -45.3
  }
}
```

## Performance Metrics

- **Upload Speed**: ~100 MB/s (SSD backend)
- **Per-Measurement Time**: ~5-10 ms
- **7 Concurrent Uploads**: ~300 ms total
- **Metadata JSON**: <2 KB per measurement
- **Total Storage**: ~3.5 MB per acquisition (7 measurements × 500 KB)

## Testing

### Unit Tests

```bash
pytest tests/unit/ -v
```

Tests include:
- Bucket creation and existence checks
- IQ data upload with/without metadata
- Download and data verification
- Session listing
- Health checks
- Error handling

### Integration Tests

```bash
pytest tests/integration/ -v
```

## Troubleshooting

### Connection Issues

```python
health = client.health_check()
if not health['accessible']:
    print(f"Error: {health['error']}")
```

### Bucket Not Found

```python
# Automatically creates bucket on first use
if not client.ensure_bucket_exists():
    print("Failed to access/create bucket")
```

### Data Integrity

```python
# Verify data integrity
success, data = client.download_iq_data(task_id, websdr_id)
if success:
    assert data.dtype == np.complex64
    assert len(data) == 125000  # Expected size
```

## Next Steps

1. **TimescaleDB Integration**: Store measurement metadata in time-series database
2. **Data Compression**: Optional gzip compression for long-term storage
3. **Lifecycle Management**: Automatic cleanup of old sessions
4. **S3 Versioning**: Enable version control for measurements
5. **Backup Strategy**: Automated backup to secondary S3 bucket
