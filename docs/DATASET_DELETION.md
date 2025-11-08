# Dataset and Job Deletion with Storage Cleanup

## Overview

When datasets or training jobs are deleted from Heimdall, the system now automatically cleans up associated data files stored in MinIO object storage. This prevents orphaned data from accumulating and consuming storage space.

## Problem Statement

Previously, deleting a dataset or job would only remove database records:
- Dataset metadata from `synthetic_datasets` table
- Sample records from `synthetic_training_samples` and `synthetic_iq_samples` tables
- Job records from `training_jobs` table

However, the actual IQ data files stored in MinIO buckets remained, creating "orphaned" data that:
- Consumed storage space
- Could not be easily identified or cleaned up
- Had no database association

## Solution

The system now performs comprehensive cleanup when datasets or jobs are deleted:

### Dataset Deletion

When a dataset is deleted via `DELETE /api/v1/jobs/synthetic/datasets/{dataset_id}`:

1. **Check dataset type**: Only `iq_raw` datasets have IQ files in MinIO
2. **List MinIO objects**: Find all files matching the dataset ID pattern
3. **Batch delete files**: Delete files in batches of 1000 (S3 API limit)
4. **Delete database records**: Remove dataset and sample records (CASCADE)
5. **Log results**: Record files deleted, space freed, any errors

### Job Deletion

When a job is deleted via `DELETE /api/v1/jobs/synthetic/{job_id}`:

1. **Cancel running job**: If job is still running, terminate it
2. **Find associated datasets**: Query for datasets with `created_by_job_id = job_id`
3. **Delete each dataset**: For each dataset, perform full cleanup (as above)
4. **Delete job record**: Remove job from database
5. **Return statistics**: Report datasets and files deleted

The job deletion endpoint accepts an optional `delete_dataset` parameter:
- `delete_dataset=false` (default): Delete only job record, keep datasets (sets `created_by_job_id = NULL`) ⚡ **SAFE DEFAULT**
- `delete_dataset=true`: Delete job and all associated datasets (explicit opt-in required)

### Dataset Deletion Safety Features

To prevent accidental data loss, the system includes several safety mechanisms:

1. **Safe Defaults**: Datasets are preserved by default unless explicitly requested
2. **Active Model Protection**: Datasets used by active models cannot be deleted (returns 409 Conflict)
3. **Frontend Confirmation**: UI requires explicit checkbox to enable dataset deletion
4. **Clear Warnings**: API and UI display prominent warnings about permanent data loss
5. **Error Messages**: When deletion is blocked, error messages include which models are using the dataset

## API Reference

### Delete Dataset

```http
DELETE /api/v1/jobs/synthetic/datasets/{dataset_id}
```

**Response**: 204 No Content

**Behavior**:
- Deletes dataset record from database (CASCADE to samples)
- If `iq_raw` dataset: Deletes all IQ files from MinIO bucket `heimdall-synthetic-iq`
- **Blocked if dataset is used by active models** (returns 409 Conflict instead of 204)
- Logs cleanup statistics and any errors
- Returns success even if MinIO cleanup fails (best effort)

**Safety Checks**:
- Before deletion, checks if any active model (is_active=TRUE) references this dataset
- If blocked, returns HTTP 409 with model names

**Example**:
```bash
curl -X DELETE http://localhost:8002/api/v1/jobs/synthetic/datasets/550e8400-e29b-41d4-a716-446655440000
```

### Delete Job

```http
DELETE /api/v1/jobs/synthetic/{job_id}?delete_dataset={true|false}
```

**Parameters**:
- `delete_dataset` (optional, default: `false`): Whether to cascade delete to datasets
  - ⚡ **Default behavior**: Job deleted, datasets preserved (safe)
  - `true`: Job and datasets deleted (requires explicit opt-in)

**Response**: 200 OK
```json
{
  "message": "Synthetic generation job {job_id} deleted successfully",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "datasets_deleted": 1,
  "minio_files_deleted": 150
}
```

**Behavior**:
- Cancels job if still running
- By default (`delete_dataset=false`): Preserves datasets, only deletes job record
- If `delete_dataset=true`: Finds and deletes all associated datasets with MinIO cleanup
  - **Blocked if datasets are used by active models** (returns 409 Conflict)
  - Error message includes which models are blocking deletion
- Deletes job record from database
- Returns statistics of deleted resources

**Safety Checks**:
- Before deleting datasets, checks if any active model references them
- If blocked, returns HTTP 409 with model names: `"Cannot delete dataset 'my_dataset': it is currently used by 2 active model(s): localization_v1, localization_v2. Please deactivate or delete these models first, or uncheck 'delete_dataset'."`

**Examples**:
```bash
# Delete job but keep datasets (DEFAULT - safe behavior)
curl -X DELETE http://localhost:8002/api/v1/jobs/synthetic/550e8400-e29b-41d4-a716-446655440000

# Explicitly delete job and all datasets it created (opt-in)
curl -X DELETE "http://localhost:8002/api/v1/jobs/synthetic/550e8400-e29b-41d4-a716-446655440000?delete_dataset=true"

# If datasets are used by active models, returns 409 Conflict:
# {
#   "detail": "Cannot delete dataset 'synthetic_v1': it is currently used by 1 active model(s): localization_v2. ..."
# }
```

## Storage Patterns

MinIO objects are stored with the following patterns:

### IQ Raw Datasets
- Bucket: `heimdall-synthetic-iq`
- Path: `synthetic/{dataset_id}/sample-{N}/rx_{N}.npy`
- Example: `synthetic/550e8400-e29b-41d4-a716-446655440000/sample-0/rx_0.npy`

### Real Recording Sessions
- Bucket: `heimdall-raw-iq`
- Path: `sessions/{session_id}/websdr_{N}.npy`
- Example: `sessions/7c3f9e42-1234-5678-90ab-cdef12345678/websdr_1.npy`

The deletion logic searches for objects using the dataset ID prefix and deletes all matches.

## Error Handling

The system uses "best effort" cleanup:

### MinIO Unavailable
If MinIO is unreachable during deletion:
- Error is logged with full details
- Database deletion proceeds normally
- Response indicates partial cleanup
- Orphaned files can be cleaned up later manually or via maintenance script

### Partial Deletion Failure
If some files fail to delete:
- Successful deletions are committed
- Failed files are logged individually
- Overall operation still succeeds
- Response includes counts of successful and failed deletions

### Database Failure
If database deletion fails:
- Transaction is rolled back
- MinIO files remain (can be deleted in retry)
- Error response returned to client

### Active Model Protection (409 Conflict)
If attempting to delete a dataset used by active models:
- Deletion is blocked
- HTTP 409 Conflict returned
- Error message includes model names
- User must either:
  - Deactivate or delete the blocking models first
  - Choose not to delete the dataset (for job deletion, uncheck `delete_dataset`)

## Logging

All deletion operations are logged with structured logging:

### Dataset Deletion
```json
{
  "event": "synthetic_dataset_deleted",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "dataset_name": "test_dataset",
  "dataset_type": "iq_raw",
  "minio_cleanup_success": true,
  "minio_files_deleted": 150
}
```

### Job Deletion
```json
{
  "event": "synthetic_job_deleted",
  "job_id": "7c3f9e42-1234-5678-90ab-cdef12345678",
  "datasets_deleted": 1,
  "minio_files_deleted": 150
}
```

### MinIO Cleanup
```json
{
  "event": "minio_cleanup_complete",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "files_deleted": 150
}
```

## Testing

### Manual Testing

A manual test script is provided for verification:

```bash
# Ensure services are running
make dev-up

# Run manual test
python scripts/test_dataset_deletion_manual.py
```

The script will:
1. Create a test `iq_raw` dataset
2. Wait for generation to complete
3. Verify IQ files exist in MinIO
4. Delete the dataset
5. Verify MinIO files are removed
6. Repeat for job deletion with cascade

### Integration Testing

For automated testing, see `services/training/tests/test_dataset_deletion.py`:

```bash
# Run deletion tests (requires infrastructure)
pytest services/training/tests/test_dataset_deletion.py
```

## Monitoring

Track deletion operations through logs and metrics:

### Key Metrics
- Total datasets deleted per day
- Total storage freed (bytes)
- MinIO cleanup success rate
- Average deletion time

### Log Queries

Find deletion operations in logs:
```bash
# Dataset deletions
grep "synthetic_dataset_deleted" /var/log/heimdall/training.log

# MinIO cleanup operations
grep "minio_cleanup" /var/log/heimdall/training.log

# Cleanup failures
grep "minio_cleanup_failed" /var/log/heimdall/training.log
```

## Maintenance

### Manual Cleanup of Orphaned Files

If MinIO files become orphaned (e.g., due to service downtime during deletion):

```sql
-- List all dataset IDs in database
SELECT id FROM heimdall.synthetic_datasets;
```

```bash
# Compare with MinIO bucket contents
aws s3 ls s3://heimdall-synthetic-iq/synthetic/ --recursive --endpoint-url http://minio:9000

# Delete orphaned files
aws s3 rm s3://heimdall-synthetic-iq/synthetic/{orphaned-id}/ --recursive --endpoint-url http://minio:9000
```

### Storage Audit

Periodically audit storage to identify orphaned data:

```sql
-- Find datasets with no associated samples (anomaly)
SELECT d.id, d.name, d.num_samples
FROM heimdall.synthetic_datasets d
LEFT JOIN heimdall.synthetic_iq_samples s ON s.dataset_id = d.id
WHERE d.dataset_type = 'iq_raw'
GROUP BY d.id, d.name, d.num_samples
HAVING COUNT(s.id) = 0;
```

## Future Enhancements

Potential improvements to the deletion system:

1. **Soft Delete**: Mark datasets as deleted but retain for recovery period
2. **Async Cleanup**: Queue MinIO deletion as background task for large datasets
3. **Retention Policies**: Automatic cleanup of old datasets based on age/usage
4. **Storage Quota**: Enforce per-user or per-project storage limits
5. **Cleanup Dashboard**: UI to track storage usage and cleanup operations
6. **Audit Logging**: Track who deleted what and when for compliance

## Changelog

### 2025-11-06: Safety Improvements
- Changed default `delete_dataset` from `true` to `false` (BREAKING CHANGE for safety)
- Added active model protection (409 Conflict when datasets used by active models)
- Enhanced frontend with confirmation dialog and explicit checkbox
- Added comprehensive error messages with model names
- Created test suite for safety features

### 2025-11-06: Initial Implementation
- Added MinIO cleanup on dataset deletion
- Added cascade deletion for jobs
- Implemented batch deletion (1000 objects per request)
- Added structured logging

## Related Documentation

- [MinIO Storage Architecture](./MINIO_STORAGE.md)
- [Database Schema](./DATABASE_SCHEMA.md)
- [Training API Reference](./API_TRAINING.md)
