# Storage Management & Cleanup System

## 🎯 Overview

This document describes the automated storage cleanup and monitoring system designed to prevent disk space leaks from orphaned MinIO files.

## 🔴 Problem Solved

**Root Cause**: The `delete_synthetic_dataset()` endpoint only deleted database records but never cleaned up MinIO files, leading to **244GB of orphaned data** and a **100% full disk** (456GB/466GB).

**Solution**: Multi-layer approach combining:
1. Fixed deletion endpoints to clean MinIO immediately
2. Periodic cleanup task to remove old orphans
3. Prometheus monitoring with Grafana alerting
4. Storage health status tracking

## ✅ Implemented Features

### 1. Fixed Dataset Deletion (Phase 1)
**File Modified**: `/services/backend/src/routers/training.py` (lines 1787-1817)

**Changes**:
- Added MinIO client initialization in `delete_synthetic_dataset()`
- Now calls `delete_dataset_iq_data()` before database deletion
- Logs cleanup results (successful/failed deletions)
- This fixes the root cause of 95% of the disk space leak

**Code Reference**: `services/backend/src/routers/training.py:1787`

### 2. Lifecycle Cleanup Task (Phase 2)
**File Created**: `/services/backend/src/tasks/minio_lifecycle.py` (484 lines)

**Features**:
- **Automatic orphan cleanup** - Celery periodic task runs daily at 3 AM
- **Age-based filtering** - Only deletes files older than 30-60 days (configurable per bucket)
- **Batch deletion** - Processes 1000 files per batch to avoid memory issues
- **Dry-run mode** - Test without actual deletion (`dry_run=True`)
- **Multi-layer validation** - Database check + age check + confirmed orphan
- **Comprehensive logging** - Detailed cleanup summaries with success/failure tracking

**Task Functions**:
```python
# Main cleanup task (runs daily)
@shared_task(name="tasks.minio_lifecycle.cleanup_orphan_files")
def cleanup_orphan_files(dry_run: bool = False) -> Dict

# Storage statistics (runs hourly)
@shared_task(name="tasks.minio_lifecycle.get_storage_stats")
def get_storage_stats() -> Dict
```

**Celery Beat Schedule**: `/services/backend/src/main.py:98-109`
```python
"minio-lifecycle-cleanup": {
    "task": "tasks.minio_lifecycle.cleanup_orphan_files",
    "schedule": 86400.0,  # Every 24 hours
    "kwargs": {"dry_run": False}
},
"minio-storage-stats": {
    "task": "tasks.minio_lifecycle.get_storage_stats",
    "schedule": 3600.0,  # Every hour
}
```

### 3. Prometheus Monitoring (Phase 3)
**Files Created**:
- `/services/backend/src/monitoring/storage_metrics.py` (230 lines)
- `/services/backend/src/monitoring/__init__.py`
- `/services/backend/src/routers/metrics.py` (40 lines)

**Prometheus Metrics Exposed**:
```
heimdall_storage_disk_usage_gb{bucket="..."}       # Total disk space used
heimdall_storage_bucket_size_gb{bucket="..."}      # Size of each bucket
heimdall_storage_orphan_files{bucket="..."}        # Number of orphans
heimdall_storage_orphan_size_gb{bucket="..."}      # Size of orphans
heimdall_storage_total_objects{bucket="..."}       # Total object count
heimdall_storage_referenced_objects{bucket="..."}  # Referenced objects
```

**API Endpoints**:
- `GET /metrics` - Prometheus metrics in text format (for scraping)
- `GET /metrics/storage` - Human-readable storage health status

**Health Status Thresholds**:
- **Healthy**: <10% orphaned data
- **Warning**: 10-25% orphaned data
- **Critical**: >25% orphaned data

**Initialization**: `/services/backend/src/main.py:147-153`
```python
# Initialize Prometheus storage metrics on startup
from .monitoring.storage_metrics import init_storage_metrics
init_storage_metrics()
```

## 📊 Monitored Buckets

| Bucket Name | Min Age (days) | Batch Size | Description |
|-------------|----------------|------------|-------------|
| `heimdall-synthetic-iq` | 30 | 1000 | Synthetic IQ data for training |
| `heimdall-audio-chunks` | 30 | 1000 | Preprocessed audio chunks |
| `heimdall-raw-iq` | 60 | 1000 | Raw IQ from WebSDR sessions |

**Configuration**: `/services/backend/src/tasks/minio_lifecycle.py:26-45`

## 🚀 Usage

### Manual Cleanup (Dry Run)
```python
from services.backend.src.tasks.minio_lifecycle import cleanup_orphan_files

# Test what would be deleted (no actual deletion)
result = cleanup_orphan_files.apply_async(kwargs={"dry_run": True})
print(result.get())
```

### Manual Cleanup (Production)
```python
# Actually delete orphans
result = cleanup_orphan_files.apply_async(kwargs={"dry_run": False})
print(result.get())
```

### Get Storage Statistics
```python
from services.backend.src.tasks.minio_lifecycle import get_storage_stats

result = get_storage_stats.apply_async()
stats = result.get()
print(f"Total orphans: {stats['buckets']['heimdall-synthetic-iq']['orphan_objects']}")
print(f"Orphan size: {stats['buckets']['heimdall-synthetic-iq']['orphan_size_gb']} GB")
```

### Check Storage Health via API
```bash
# Get Prometheus metrics
curl http://localhost:8001/metrics

# Get human-readable health status
curl http://localhost:8001/metrics/storage
```

## 🔍 How It Works

### Orphan Detection Logic
```python
1. Query database for referenced files:
   - synthetic_iq_samples.dataset_id + sample_idx + receiver_idx
   - audio_chunks.minio_path
   - measurements.iq_data_location

2. List all files in MinIO bucket

3. Calculate orphans = (MinIO files) - (DB referenced files)

4. Filter by age:
   - Only delete if last_modified > min_age_days

5. Delete in batches of 1000 to avoid memory issues
```

**Database Queries**:
- `_get_referenced_synthetic_iq_files()` - Lines 307-318
- `_get_referenced_audio_chunks()` - Lines 321-330
- `_get_referenced_raw_iq_files()` - Lines 333-343

### Metrics Update Flow
```
Celery Task (cleanup_orphan_files or get_storage_stats)
    ↓
Calculate storage statistics per bucket
    ↓
update_storage_metrics(stats)
    ↓
Update Prometheus Gauge metrics
    ↓
Prometheus scrapes /metrics endpoint
    ↓
Grafana displays dashboards + triggers alerts
```

## 🛡️ Safety Features

1. **Multi-layer validation**:
   - File must not exist in database
   - File must be older than `min_age_days`
   - Confirmed as orphan before deletion

2. **Batch processing**:
   - Deletes 1000 files per batch
   - Avoids memory exhaustion
   - Tracks success/failure per batch

3. **Dry-run mode**:
   - Test cleanup logic without actual deletion
   - Returns what *would* be deleted
   - Perfect for validation

4. **Comprehensive logging**:
   - Every deletion logged with result
   - Failed deletions tracked separately
   - Execution time and summary statistics

5. **Retry logic**:
   - Celery task retries on failure (max 3 attempts)
   - 5-minute backoff between retries

## 📈 Monitoring & Alerts

### Grafana Dashboard Setup (Future)
```yaml
# Example alert rule
- alert: StorageOrphansHigh
  expr: heimdall_storage_orphan_size_gb > 50
  for: 1h
  labels:
    severity: warning
  annotations:
    summary: "High orphaned storage detected"
    description: "{{ $labels.bucket }} has {{ $value }}GB of orphaned files"

- alert: DiskSpaceCritical
  expr: sum(heimdall_storage_disk_usage_gb) > 400
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Disk space critical"
    description: "Total MinIO usage is {{ $value }}GB (>400GB threshold)"
```

### Prometheus Scrape Config
```yaml
scrape_configs:
  - job_name: 'heimdall-backend'
    static_configs:
      - targets: ['backend:8001']
    metrics_path: '/metrics'
    scrape_interval: 60s
```

## 🧪 Testing

**Test File**: `/home/fulgidus/Documents/Projects/heimdall/test_storage_cleanup_system.py`

**Run Tests**:
```bash
# Run all storage cleanup tests
pytest test_storage_cleanup_system.py -v

# Run specific test
pytest test_storage_cleanup_system.py::test_lifecycle_cleanup_registered_in_celery_beat -v
```

**Test Coverage**:
- ✅ Celery Beat schedule registration
- ✅ Metrics initialization
- ✅ Metrics update from stats
- ✅ Prometheus /metrics endpoint
- ✅ Storage health endpoint
- ✅ Lifecycle configuration
- ✅ Dataset deletion includes MinIO cleanup

## 🔧 Configuration

### Environment Variables
```bash
# MinIO connection (from .env)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_REGION=us-east-1

# Celery broker (for task scheduling)
CELERY_BROKER_URL=amqp://guest:guest@rabbitmq:5672//
```

### Lifecycle Policy Tuning
Edit `/services/backend/src/tasks/minio_lifecycle.py:26-45`:

```python
LIFECYCLE_CONFIG = {
    "heimdall-synthetic-iq": {
        "enabled": True,
        "min_age_days": 30,  # Increase for more safety
        "batch_size": 1000,  # Decrease if memory constrained
        "description": "..."
    }
}
```

### Celery Beat Schedule Tuning
Edit `/services/backend/src/main.py:98-109`:

```python
"minio-lifecycle-cleanup": {
    "schedule": 86400.0,  # Change to run more/less frequently
    "kwargs": {"dry_run": False}  # Set True to disable deletion
}
```

## 📦 Dependencies Added

**File**: `/services/requirements/base.txt:29`
```
prometheus-client==0.19.0
```

**Install**:
```bash
cd services
pip install -r requirements/base.txt
```

## 🐛 Troubleshooting

### Issue: Cleanup not running
**Check**:
```bash
# Verify Celery Beat is running
docker-compose ps celery-beat

# Check Celery Beat schedule
docker-compose exec backend python -c "from src.main import celery_app; print(celery_app.conf.beat_schedule)"

# Check task logs
docker-compose logs -f celery-worker | grep lifecycle
```

### Issue: Metrics not updating
**Check**:
```bash
# Verify metrics endpoint responds
curl http://localhost:8001/metrics | grep heimdall_storage

# Check metrics initialization in logs
docker-compose logs backend | grep "storage metrics"

# Manually trigger stats task
docker-compose exec backend python -c "
from src.tasks.minio_lifecycle import get_storage_stats
result = get_storage_stats()
print(result)
"
```

### Issue: Too many orphans detected
**Investigate**:
```bash
# Get detailed statistics
curl http://localhost:8001/metrics/storage | jq

# Run dry-run to see what would be deleted
docker-compose exec backend python -c "
from src.tasks.minio_lifecycle import cleanup_orphan_files
result = cleanup_orphan_files(dry_run=True)
print(result)
"
```

## 🔮 Future Enhancements (Phase 4+)

### Phase 4: Database Registry
- Create `minio_object_registry` table for file tracking
- PostgreSQL NOTIFY triggers for event-driven cleanup
- Backend listener for real-time cleanup notifications

### Phase 5: Advanced Monitoring
- Grafana dashboards with pre-configured alerts
- Disk space trend analysis
- Orphan growth rate tracking
- Automated capacity planning

### Phase 6: S3 Lifecycle Policies
- Native MinIO lifecycle rules
- Automatic expiration of old objects
- Glacier-style archival for compliance

## 📚 Related Documentation

- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Disk space issues
- [ARCHITECTURE.md](./ARCHITECTURE.md) - MinIO integration
- [DEVELOPMENT.md](./DEVELOPMENT.md) - Local development setup

## 🎓 Key Lessons Learned

1. **Root Cause**: Always clean up storage in the same transaction as database deletion
2. **Defense in Depth**: Multiple cleanup strategies (immediate + periodic + monitoring)
3. **Safety First**: Age-based filtering + dry-run mode prevent accidental deletion
4. **Observability**: Metrics are essential for proactive disk space management
5. **Batch Processing**: Large-scale deletions must be batched to avoid memory issues

## 📧 Support

For questions or issues:
- Open GitHub issue: https://github.com/fulgidus/heimdall/issues
- Contact: alessio.corsi@gmail.com

---

**Last Updated**: 2025-11-07  
**Status**: ✅ Phases 1-3 Complete (Deletion Fix + Lifecycle + Monitoring)  
**Next**: Phase 4 (Database Registry) or Phase 5 (Grafana Dashboards)
