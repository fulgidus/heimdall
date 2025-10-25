# Quick Start: TimescaleDB Integration

## Setup

### 1. Environment Variables
```bash
export DATABASE_URL="postgresql://heimdall_user:password@postgres:5432/heimdall"
export MINIO_URL="http://minio:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
```

### 2. Initialize Database
```bash
# Using DatabaseManager directly
from src.storage.db_manager import get_db_manager
db = get_db_manager()
db.create_tables()  # Creates all tables including hypertable
```

## Usage Examples

### Store Measurements
```python
from src.storage.db_manager import get_db_manager

db = get_db_manager()

measurement = {
    "websdr_id": 1,
    "frequency_mhz": 144.5,
    "sample_rate_khz": 12.5,
    "samples_count": 125000,
    "timestamp_utc": "2024-01-01T12:00:00.000Z",
    "metrics": {
        "snr_db": 15.5,
        "frequency_offset_hz": 150.0,
        "power_dbm": -75.5
    }
}

# Single insert
measurement_id = db.insert_measurement(
    task_id="acq-001",
    measurement_dict=measurement,
    s3_path="s3://bucket/websdr_1.npy"
)

# Bulk insert
measurements = [...]  # List of measurement dicts
successful, failed = db.insert_measurements_bulk(
    task_id="acq-001",
    measurements_list=measurements
)
```

### Query Recent Data
```python
# Get last 24 hours for a WebSDR
recent = db.get_recent_measurements(
    task_id="acq-001",
    websdr_id=1,
    hours_back=24,
    limit=100
)

for m in recent:
    print(f"{m.timestamp_utc}: SNR={m.snr_db}dB")
```

### Get Session Data
```python
# All measurements from an acquisition session
session_data = db.get_session_measurements(task_id="acq-001")

for websdr_id, measurements in session_data.items():
    print(f"WebSDR {websdr_id}: {len(measurements)} measurements")
```

### Analyze Performance
```python
# SNR statistics per WebSDR
stats = db.get_snr_statistics(task_id="acq-001", hours_back=24)

for websdr_id, stat in stats.items():
    print(f"WebSDR {websdr_id}:")
    print(f"  Avg SNR: {stat['avg_snr_db']:.1f} dB")
    print(f"  Min SNR: {stat['min_snr_db']:.1f} dB")
    print(f"  Max SNR: {stat['max_snr_db']:.1f} dB")
    print(f"  Count: {stat['count']}")
```

### Frequency Drift Analysis
```python
drift = db.get_frequency_drift_analysis(
    task_id="acq-001",
    websdr_id=1
)

for point in drift:
    print(f"{point['timestamp']}: offset={point['frequency_offset_hz']:.1f}Hz")
```

## Celery Task

### In Acquisition Handler
```python
from celery import chain
from src.tasks.acquire_iq import acquire_iq, save_measurements_to_minio, save_measurements_to_timescaledb

# Chain IQ acquisition → MinIO storage → DB storage
workflow = chain(
    acquire_iq.s(
        frequency_mhz=144.5,
        duration_seconds=10.0,
        start_time_iso=datetime.utcnow().isoformat(),
        websdrs_config_list=websdrs,
        sample_rate_khz=12.5
    ),
    save_measurements_to_minio.s(),  # Store IQ data to S3
    save_measurements_to_timescaledb.s()  # Store metrics to DB
)

result = workflow.apply_async()
```

## API Endpoints (Future)

```bash
# Recent measurements
GET /api/measurements/recent?task_id=acq-001&hours_back=24&limit=100

# Session data
GET /api/measurements/session/acq-001

# SNR statistics  
GET /api/statistics/snr?task_id=acq-001

# Frequency drift
GET /api/analysis/frequency-drift?task_id=acq-001&websdr_id=1
```

## Maintenance

### Clean Old Data
```python
# Delete measurements older than 30 days
deleted_count = db.delete_old_measurements(days_old=30)
print(f"Deleted {deleted_count} old measurements")
```

### Check Connection
```python
if db.check_connection():
    print("Database is healthy")
else:
    print("Database connection failed")
```

## Error Handling

```python
try:
    successful, failed = db.insert_measurements_bulk(
        task_id="acq-001",
        measurements_list=measurements
    )
    print(f"Stored: {successful}, Errors: {failed}")
except Exception as e:
    print(f"Database error: {e}")
```

## Testing

### Run Tests
```bash
cd services/rf-acquisition

# TimescaleDB tests
pytest tests/integration/test_timescaledb.py -v

# All tests
pytest tests/ -v

# Specific test
pytest tests/integration/test_timescaledb.py::test_bulk_insert_measurements -v
```

### Test with PostgreSQL
```bash
export DATABASE_URL="postgresql://localhost/heimdall_test"
pytest tests/integration/test_timescaledb.py -v
```

## Troubleshooting

### Import Errors
```python
# If getting import errors, check:
from src.models.db import Measurement, Base
from src.storage.db_manager import DatabaseManager

print("Imports OK")
```

### Connection Issues
```python
from src.storage.db_manager import DatabaseManager

db = DatabaseManager()
if not db.check_connection():
    print(f"Cannot connect to: {db.database_url}")
    print("Check DATABASE_URL environment variable")
```

### SQLite vs PostgreSQL
```python
# SQLite (testing)
db = DatabaseManager(database_url="sqlite:///:memory:")

# PostgreSQL (production)
db = DatabaseManager(database_url="postgresql://user:pass@host/db")
```

---

For detailed documentation see:
- `PHASE3_TIMESCALEDB_STATUS.md` - Full implementation details
- `db/migrations/001_create_measurements_table.sql` - Database schema
- `src/storage/db_manager.py` - API documentation
