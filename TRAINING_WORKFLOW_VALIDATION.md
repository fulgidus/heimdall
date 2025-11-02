# Training Workflow Validation Guide

This document provides step-by-step instructions to validate the end-to-end training workflow implementation.

## Prerequisites

1. **Docker services running**:
   ```bash
   docker-compose up -d
   ```

2. **Verify all services are healthy**:
   ```bash
   docker-compose ps
   ```
   
   Expected: postgres, rabbitmq, redis, minio, backend, training should all be running.

3. **Database migrations applied**:
   - The schema should include: `synthetic_datasets`, `synthetic_training_samples`, `training_jobs`, `training_metrics`, `models` tables

## Step 1: Generate Synthetic Training Data

Before training, we need synthetic data. Use the API to create a dataset:

```bash
curl -X POST http://localhost:8001/api/v1/training/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_dataset_10k",
    "description": "Test dataset with 10k samples",
    "num_samples": 10000,
    "inside_ratio": 0.7,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "frequency_mhz": 145.0,
    "tx_power_dbm": 37.0,
    "min_snr_db": 3.0,
    "min_receivers": 3,
    "max_gdop": 10.0
  }'
```

**Expected Response**:
```json
{
  "job_id": "...",
  "status": "pending",
  "created_at": "...",
  "status_url": "/api/v1/training/jobs/{job_id}"
}
```

**Monitor generation progress**:
```bash
curl http://localhost:8001/api/v1/training/jobs/{job_id}
```

Wait until status is "completed". Note the `dataset_id` from the response.

**Verify dataset created**:
```bash
curl http://localhost:8001/api/v1/training/synthetic/datasets
```

## Step 2: Start Training Job

Submit a training job with the dataset_id from Step 1:

```bash
curl -X POST http://localhost:8001/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "triangulation_test_run",
    "description": "Test run of triangulation model training",
    "config": {
      "dataset_id": "DATASET_ID_FROM_STEP_1",
      "model_architecture": "triangulation",
      "batch_size": 32,
      "epochs": 50,
      "learning_rate": 0.001,
      "weight_decay": 0.0001,
      "dropout_rate": 0.2,
      "early_stop_patience": 20,
      "early_stop_delta": 0.001,
      "max_grad_norm": 1.0,
      "lr_scheduler": "cosine",
      "accelerator": "cpu",
      "max_gdop": 5.0
    }
  }'
```

**Expected Response**:
```json
{
  "id": "...",
  "job_name": "triangulation_test_run",
  "status": "queued",
  "created_at": "...",
  "config": {...},
  "total_epochs": 50,
  "celery_task_id": "..."
}
```

Note the `id` (job_id) from the response.

## Step 3: Monitor Training Progress

### Get job status and recent metrics:
```bash
curl http://localhost:8001/api/v1/training/jobs/{job_id}
```

**Expected Fields**:
- `status`: "running" → "completed" (or "failed")
- `current_epoch`: Increases from 0 to 50
- `progress_percent`: 0 to 100
- `train_loss`: Should decrease over epochs
- `val_loss`: Should decrease over epochs
- `train_accuracy`: Actually RMSE in meters (should decrease)
- `val_accuracy`: Actually RMSE in meters (should decrease)
- `learning_rate`: Should decrease with cosine annealing
- `recent_metrics`: Array of last 10 epochs

### Get detailed metrics history:
```bash
curl http://localhost:8001/api/v1/training/jobs/{job_id}/metrics?limit=100
```

**Expected**: Array of epoch metrics showing training progression.

### Monitor Celery worker logs:
```bash
docker-compose logs -f backend
```

Look for:
- "Starting training job {job_id}"
- "Epoch X/Y: train_loss=..., val_loss=..., train_rmse=...m, val_rmse=...m"
- "Saved best model checkpoint at epoch X"
- "Training job {job_id} completed successfully"

## Step 4: Verify Checkpoints in MinIO

### Access MinIO UI:
Open http://localhost:9001 in browser (credentials: minioadmin/minioadmin)

### Check `models` bucket:
- Navigate to `checkpoints/{job_id}/`
- Should contain:
  - `best_model.pth` - Best validation loss checkpoint
  - `final_model.pth` - Final epoch checkpoint
  - `epoch_10.pth`, `epoch_20.pth`, etc. - Periodic checkpoints

### Verify checkpoint content:
```bash
# Download checkpoint
curl http://localhost:9000/models/checkpoints/{job_id}/best_model.pth \
  -u minioadmin:minioadmin \
  --output best_model.pth

# Check file size
ls -lh best_model.pth
```

**Expected**: File should be several MB (model weights + optimizer state).

## Step 5: Verify Model Metadata in Database

### Check models table:
```bash
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT id, model_name, model_type, accuracy_meters, loss_value, epoch, is_active, trained_by_job_id FROM heimdall.models ORDER BY created_at DESC LIMIT 1;"
```

**Expected Output**:
```
             id              |         model_name          | model_type | accuracy_meters | loss_value | epoch | is_active | trained_by_job_id
-----------------------------+-----------------------------+------------+-----------------+------------+-------+-----------+-------------------
 ...                         | triangulation_job_{job_id}  | triangulation | <100.0      | ...        | ...   | f         | {job_id}
```

### Check training_jobs table:
```bash
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT id, job_name, status, current_epoch, total_epochs, best_epoch, best_val_loss FROM heimdall.training_jobs WHERE id = '{job_id}';"
```

**Expected**: Status = "completed", best_epoch set, best_val_loss recorded.

### Check training_metrics table:
```bash
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT epoch, train_loss, val_loss, train_accuracy, val_accuracy FROM heimdall.training_metrics WHERE training_job_id = '{job_id}' ORDER BY epoch LIMIT 10;"
```

**Expected**: Multiple rows with decreasing loss values.

## Step 6: List All Training Jobs and Models

### List all training jobs:
```bash
curl http://localhost:8001/api/v1/training/jobs
```

### List all trained models:
```bash
curl http://localhost:8001/api/v1/training/models
```

### Filter for active models only:
```bash
curl "http://localhost:8001/api/v1/training/models?active_only=true"
```

## Success Criteria Validation

### ✅ Training completes successfully:
- Job status transitions: pending → queued → running → completed
- No errors in logs

### ✅ Validation RMSE < 100m on GDOP<5 subset:
Check the model's `accuracy_sigma_meters` field:
```bash
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT accuracy_sigma_meters FROM heimdall.models WHERE trained_by_job_id = '{job_id}';"
```

### ✅ Checkpoints saved to MinIO:
- `best_model.pth` exists
- `final_model.pth` exists
- Periodic checkpoints exist

### ✅ Job status transitions correctly:
- Database shows correct status progression
- Metrics logged for each epoch

### ✅ Training curves accessible via REST API:
- `/api/v1/training/jobs/{job_id}` returns job details
- `/api/v1/training/jobs/{job_id}/metrics` returns epoch-by-epoch metrics

## Troubleshooting

### Training fails with "dataset_id not found":
- Ensure synthetic data generation completed successfully in Step 1
- Check dataset_id is valid UUID

### Training fails with CUDA errors:
- Set `accelerator: "cpu"` in config
- Or ensure GPU drivers installed if using GPU

### MinIO checkpoints not found:
- Check MinIO service is running: `docker-compose ps minio`
- Verify credentials in backend config match MinIO

### Celery task not running:
- Check RabbitMQ is running: `docker-compose ps rabbitmq`
- Check backend service logs: `docker-compose logs backend`
- Verify Celery worker is running

### Database connection errors:
- Check PostgreSQL is running: `docker-compose ps postgres`
- Verify database credentials in config

## Performance Benchmarks

Expected performance on 10k samples:
- **Training time**: ~5-10 minutes (CPU), ~2-3 minutes (GPU)
- **Memory usage**: ~500MB-1GB
- **Final validation RMSE**: 50-100m (GDOP<5 subset)

Expected performance on 50k samples:
- **Training time**: ~20-30 minutes (CPU), ~10-15 minutes (GPU)
- **Memory usage**: ~1-2GB
- **Final validation RMSE**: 30-80m (GDOP<5 subset)

## API Testing with pytest

Run integration tests:
```bash
cd services/backend
pytest tests/integration/test_training_workflow.py -v
```

**Expected**: All tests pass (or skip if dependencies unavailable).

## Notes

- The training implementation uses real PyTorch training, not simulations
- RMSE is stored in `train_accuracy` and `val_accuracy` fields (legacy naming)
- `accuracy_meters` field stores overall validation RMSE
- `accuracy_sigma_meters` field stores RMSE for GDOP<5 subset (success criterion)
- Early stopping activates after 20 epochs without improvement
- Gradient clipping prevents exploding gradients
- Cosine annealing gradually reduces learning rate
