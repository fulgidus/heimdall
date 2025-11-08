# Batch Feature Extraction - Verification Guide

## Overview

This guide provides step-by-step instructions to verify the batch feature extraction implementation.

## Prerequisites

- Docker and Docker Compose installed
- Repository cloned and in working directory
- `.env` file configured (or use defaults)

## Verification Steps

### 1. Start Infrastructure

```bash
# Start all services
docker compose up -d

# Wait for services to be healthy (30-60 seconds)
docker compose ps

# Check logs
docker compose logs -f backend | grep "batch feature"
```

### 2. Verify Celery Beat Schedule

Check that the batch extraction task is scheduled:

```bash
# Inspect Celery Beat schedule
docker compose exec backend celery -A src.main:celery_app inspect scheduled

# Expected output should include:
# - 'backend.tasks.batch_feature_extraction'
# - scheduled every 300 seconds (5 minutes)
```

### 3. Test Admin Endpoints

#### Get Coverage Statistics

```bash
curl -X GET http://localhost:8001/api/v1/admin/features/stats

# Expected response:
# {
#   "total_recordings": 0,
#   "recordings_with_features": 0,
#   "recordings_without_features": 0,
#   "total_measurements": 0,
#   "coverage_percent": 0.0
# }
```

#### Manually Trigger Batch Extraction

```bash
curl -X POST http://localhost:8001/api/v1/admin/features/batch-extract \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 10, "max_batches": 2}'

# Expected response:
# {
#   "task_id": "abc123...",
#   "message": "Batch extraction started (batch_size=10, max_batches=2)",
#   "status": "queued"
# }
```

#### Trigger Full Backfill (use with caution)

```bash
curl -X POST http://localhost:8001/api/v1/admin/features/backfill-all

# Expected response:
# {
#   "task_id": "def456...",
#   "message": "Full backfill started - this may take hours",
#   "status": "queued"
# }
```

### 4. Monitor Batch Processing

Watch the backend logs for batch extraction activity:

```bash
docker compose logs -f backend | grep -E "(batch feature|Backfill)"

# Expected logs (every 5 minutes):
# Starting batch feature extraction (batch_size=50, max_batches=5)
# Found 0 recordings without features
# No recordings without features
```

### 5. Create Test Data

To test with actual recordings:

```bash
# 1. Create a recording session
curl -X POST http://localhost:8001/api/v1/acquisition/acquire \
  -H "Content-Type: application/json" \
  -d '{
    "frequency_mhz": 145.5,
    "duration_seconds": 10,
    "start_time": "'$(date -u +%Y-%m-%dT%H:%M:%S)'"
  }'

# 2. Wait for acquisition to complete (~70 seconds)

# 3. Check stats again
curl http://localhost:8001/api/v1/admin/features/stats

# Should show:
# - total_recordings: 1
# - recordings_without_features: 1 (if extraction hasn't run yet)

# 4. Wait 5 minutes for automatic batch extraction
# OR trigger manually:
curl -X POST http://localhost:8001/api/v1/admin/features/batch-extract

# 5. Check stats again after extraction completes
# Should show:
# - recordings_with_features: 1
# - coverage_percent: 100.0
```

### 6. Verify Database Queries

Connect to PostgreSQL and verify the LEFT JOIN query:

```bash
docker compose exec postgres psql -U heimdall_user -d heimdall

# Run the query manually:
SELECT
    rs.id as session_id,
    rs.created_at,
    rs.status,
    COUNT(m.id) as num_measurements,
    mf.recording_session_id as has_features
FROM heimdall.recording_sessions rs
LEFT JOIN heimdall.measurements m
    ON m.created_at >= rs.session_start
    AND (rs.session_end IS NULL OR m.created_at <= rs.session_end)
    AND m.iq_data_location IS NOT NULL
LEFT JOIN heimdall.measurement_features mf
    ON mf.recording_session_id = rs.id
WHERE rs.status = 'completed'
GROUP BY rs.id, rs.created_at, rs.status, mf.recording_session_id
ORDER BY rs.created_at DESC
LIMIT 10;
```

### 7. Run Tests

```bash
# Run unit tests
docker compose exec backend pytest tests/unit/test_batch_feature_extraction.py -v

# Run integration tests
docker compose exec backend pytest tests/integration/test_admin_endpoints.py -v

# Expected: All tests pass
```

## Success Criteria

✅ Celery Beat schedule includes batch-feature-extraction task (300s interval)  
✅ Admin endpoints respond correctly to all requests  
✅ Coverage statistics endpoint returns valid data  
✅ Manual batch extraction queues tasks successfully  
✅ Automatic batch extraction runs every 5 minutes  
✅ LEFT JOIN query correctly identifies recordings without features  
✅ Feature extraction tasks are queued and processed  
✅ All tests pass

## Troubleshooting

### Issue: Batch extraction not running automatically

**Solution**: Check Celery Beat logs:
```bash
docker compose logs backend | grep -i "beat"
```

### Issue: Admin endpoints return 500 error

**Solution**: Check database connection and pool initialization:
```bash
docker compose logs backend | grep -E "(pool|database)"
```

### Issue: No recordings found without features

**Solution**: Verify recordings exist:
```bash
docker compose exec postgres psql -U heimdall_user -d heimdall \
  -c "SELECT COUNT(*) FROM heimdall.recording_sessions WHERE status = 'completed';"
```

### Issue: Tasks not being queued

**Solution**: Check RabbitMQ connectivity:
```bash
docker compose logs backend | grep -i "rabbitmq\|celery"
```

## Performance Expectations

- **Query execution**: <100ms for LEFT JOIN
- **Task queuing**: <10ms per task
- **Batch processing**: ~1 second for 50 recordings
- **Memory usage**: ~50 MB per batch
- **CPU usage**: Minimal (database-bound)

## Architecture Notes

### LEFT JOIN Query Logic

The query finds recordings without features by:
1. Joining `recording_sessions` with `measurements` (recordings)
2. LEFT JOIN with `measurement_features` (extractions)
3. Filtering WHERE `mf.recording_session_id IS NULL` (no features)
4. Only considering completed recordings with IQ data

### Rate Limiting

- **Per run**: Max 250 recordings (50 batch_size × 5 max_batches)
- **Per 5 minutes**: 250 recordings
- **Per hour**: ~3000 recordings
- **Safety limit**: 50,000 recordings (backfill)

### Celery Beat Schedule

The beat schedule in `main.py` defines:
```python
"batch-feature-extraction": {
    "task": "backend.tasks.batch_feature_extraction",
    "schedule": 300.0,  # Every 5 minutes
    "kwargs": {
        "batch_size": 50,
        "max_batches": 5
    }
}
```

This ensures:
- Periodic execution without manual intervention
- Gradual backfill of historical data
- System resource protection via rate limiting
