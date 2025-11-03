# Prompt 02: Phase 5 - Test Existing Training Pipeline

## Context
GPU is now configured (Prompt 01 complete). Backend training implementation is 80% complete. Before building API/frontend, validate the core pipeline works end-to-end.

## Current Implementation Status

### ✅ Already Implemented
1. **Model Architecture** (`services/training/src/models/localization_net.py`):
   - ConvNeXt-Large backbone (200M params, pretrained ImageNet)
   - Dual-head: position [lat, lon] + uncertainty [sigma_x, sigma_y]
   - Input: mel-spectrogram (3, 128, 32)

2. **PyTorch Lightning Module** (`services/training/src/models/lightning_module.py`):
   - Complete training loop with validation
   - GaussianNLLLoss for uncertainty-aware training
   - Adam optimizer + CosineAnnealing scheduler
   - MLflow logging integration

3. **Dataset** (`services/training/src/data/dataset.py`):
   - HeimdallDataset with lazy loading
   - Mel-spectrogram feature extraction from IQ data
   - Database integration (PostgreSQL + MinIO)

4. **ONNX Export** (`services/training/src/onnx_export.py`):
   - PyTorch → ONNX conversion with validation
   - MinIO upload + MLflow Model Registry
   - Performance benchmarking (PyTorch vs ONNX)

5. **Training Script** (`services/training/src/train.py`):
   - TrainingPipeline orchestration class
   - CLI argument parsing
   - Checkpoint callbacks (ModelCheckpoint, EarlyStopping)

6. **Synthetic Data Generation** (Celery task):
   - `generate_synthetic_data_task` in `training_task.py`
   - IQ sample generation with RF propagation
   - Feature extraction and database storage

### Database Schema
- `heimdall.synthetic_datasets`: Dataset metadata
- `heimdall.synthetic_training_samples`: Feature samples (TimescaleDB hypertable)
- `heimdall.training_jobs`: Job tracking
- `heimdall.training_metrics`: Per-epoch metrics (TimescaleDB)
- `heimdall.models`: Trained model registry

## Architectural Decision: Test Strategy

### Phase A: Generate Small Synthetic Dataset
**Goal**: Create 1000 samples for quick training test
**Why**: Full 10k samples take ~2 hours to generate

**Database Check**:
```sql
-- Check existing datasets
SELECT id, name, num_samples, created_at 
FROM heimdall.synthetic_datasets 
ORDER BY created_at DESC LIMIT 5;

-- Count samples per dataset
SELECT dataset_id, COUNT(*) as samples 
FROM heimdall.synthetic_training_samples 
GROUP BY dataset_id;
```

**Options**:
1. If usable dataset exists (≥1000 samples): Skip to Phase B
2. If no dataset: Trigger synthetic generation via API or Celery

**Trigger Generation** (choose one method):

**Method 1: Direct Celery Task** (faster for testing):
```python
# In Python shell inside training container
from src.tasks.training_task import generate_synthetic_data_task
import uuid

job_id = str(uuid.uuid4())
task = generate_synthetic_data_task.delay(job_id)
print(f"Task ID: {task.id}, Job ID: {job_id}")
```

**Method 2: Via API** (if endpoint exists):
```bash
curl -X POST http://localhost:8002/v1/training/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Dataset 1k",
    "num_samples": 1000,
    "description": "Small dataset for pipeline testing",
    "frequency_mhz": 145.0,
    "tx_power_dbm": 37.0,
    "min_snr_db": 3.0
  }'
```

### Phase B: Run Training for 5 Epochs
**Goal**: Validate training loop, checkpointing, MLflow logging

**Training Command** (inside container or via script):
```bash
# Option 1: Direct Python execution
docker exec -it heimdall-training python3 /app/services/training/src/train.py \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --validation_split 0.2 \
  --accelerator gpu \
  --devices 1

# Option 2: Using existing train.py at project root (if configured)
docker exec -it heimdall-training python3 /app/services/training/train.py \
  --epochs 5 \
  --batch_size 16
```

**Expected Outputs**:
1. **Console**: PyTorch Lightning training progress bars
2. **MLflow**: Run created in `heimdall-localization` experiment
3. **Checkpoints**: Saved to `/tmp/heimdall_checkpoints/` (or configured path)
4. **Database**: Metrics inserted into `training_metrics` table

**Monitor Progress**:
```sql
-- Check training metrics
SELECT epoch, train_loss, val_loss, train_accuracy, val_accuracy 
FROM heimdall.training_metrics 
WHERE training_job_id = '<job_id>' 
ORDER BY epoch DESC 
LIMIT 10;
```

### Phase C: Test ONNX Export
**Goal**: Validate model export and inference

**Export Command**:
```python
# After training completes, find best checkpoint
from src.onnx_export import export_and_register_model

# Get checkpoint path from training output or database
checkpoint_path = "/tmp/heimdall_checkpoints/best.ckpt"
model_name = "localization-net-test"
version = "1.0.0-test"

export_and_register_model(
    checkpoint_path=checkpoint_path,
    model_name=model_name,
    version=version,
    onnx_filename=f"{model_name}-{version}.onnx"
)
```

**Validation**:
1. Check MinIO bucket `heimdall-models` for ONNX file
2. Check MLflow Model Registry for registered model
3. Test inference with sample data

### Phase D: Test Checkpoint Resume
**Goal**: Validate pause/resume functionality

**Steps**:
1. Start training for 10 epochs
2. Interrupt after epoch 3 (Ctrl+C or kill process)
3. Resume from checkpoint:
```bash
docker exec -it heimdall-training python3 /app/services/training/src/train.py \
  --checkpoint /tmp/heimdall_checkpoints/last.ckpt \
  --resume_training \
  --epochs 10
```

**Expected**: Training continues from epoch 4

## Validation Criteria

### Must Pass
- [ ] Synthetic generation creates samples in database
- [ ] Training runs for 5 epochs without errors
- [ ] Loss decreases over epochs (validation working)
- [ ] Checkpoints saved to disk/MinIO
- [ ] MLflow run created with metrics
- [ ] ONNX export succeeds
- [ ] ONNX model registered in MLflow
- [ ] Inference works with ONNX model (latency <500ms)
- [ ] Resume training works from checkpoint

### Performance Targets
- Training throughput: ≥10 samples/sec on GPU
- GPU memory usage: <20GB (RTX 3090 has 24GB)
- Checkpoint save time: <10 seconds
- ONNX export time: <30 seconds
- ONNX inference: 1.5-2.5x faster than PyTorch

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution**: Reduce `batch_size` from 32 to 16 or 8

### Issue: Dataset loader fails
**Check**: 
- MinIO connectivity
- PostgreSQL feature samples exist
- Feature extraction completed successfully

### Issue: MLflow connection fails
**Check**: MinIO (MLflow backend) health
**Fallback**: Training works without MLflow, just no logging

### Issue: ONNX export fails
**Common cause**: Dynamic axes or unsupported operations
**Debug**: Check PyTorch version compatibility with ONNX opset

## Non-Breaking Requirement
All tests run in training container only. No changes to other services. System remains operational throughout testing.

## Success Criteria
When all validation criteria pass, Phase 5 backend is confirmed working. Proceed to **Prompt 03: Build Training API Endpoints**.

## Deliverables
Document in session notes:
- Dataset ID and sample count used
- Training job ID and final metrics (epoch 5 loss/accuracy)
- MLflow run ID
- ONNX model path in MinIO
- Any issues encountered and solutions
- Performance measurements (throughput, memory, latency)
