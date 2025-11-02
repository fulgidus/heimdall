# Pause/Resume Training Jobs - Test Plan

**Feature**: Pause and resume ML training jobs at epoch boundaries  
**Date**: 2025-11-02  
**Status**: Implementation Complete, Ready for Testing

---

## 🎯 Test Objectives

Verify that training jobs can be:
1. Paused gracefully at epoch boundaries
2. Resume from the exact state where they were paused
3. Continue training without loss of progress
4. Handle edge cases (pause during different states, multiple pause/resume cycles)

---

## ✅ Prerequisites

### Database Migration Applied
```bash
# Verify migration is applied
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT column_name FROM information_schema.columns 
   WHERE table_name = 'training_jobs' AND column_name = 'pause_checkpoint_path';"

# Should show:
#       column_name      
# -----------------------
#  pause_checkpoint_path
```

### Services Running
```bash
docker compose ps backend training

# Both should be 'healthy' or 'Up'
```

### Test Dataset Available
```bash
# Check for datasets with samples
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT id, name, num_samples FROM heimdall.synthetic_datasets 
   WHERE num_samples > 0 LIMIT 5;"
```

---

## 🧪 Test Cases

### Test 1: Basic Pause/Resume Flow ⭐

**Objective**: Verify basic pause and resume functionality works end-to-end.

**Steps**:

1. **Create Training Job**
   ```bash
   # Via Frontend UI (http://localhost:3000/training)
   # Or via API:
   curl -X POST http://localhost:8001/v1/training/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "job_name": "Pause Resume Test",
       "config": {
         "dataset_id": "efedcfec-61b5-41a6-b29f-10255a1bc39e",
         "epochs": 20,
         "batch_size": 16,
         "learning_rate": 0.001
       }
     }'
   ```

2. **Wait for Job to Start Running**
   - Check UI: Status should change to "running"
   - Wait for at least epoch 3-4

3. **Pause the Job**
   ```bash
   # Note the job_id from step 1
   curl -X POST http://localhost:8001/v1/training/jobs/{JOB_ID}/pause
   
   # Expected response:
   # {
   #   "status": "paused",
   #   "message": "Training will pause after completing the current epoch"
   # }
   ```

4. **Verify Pause Checkpoint**
   ```bash
   # Check database for pause checkpoint path
   docker exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
     "SELECT id, status, current_epoch, pause_checkpoint_path 
      FROM heimdall.training_jobs WHERE id = '{JOB_ID}';"
   
   # Should show:
   # - status: 'paused'
   # - pause_checkpoint_path: s3://models/checkpoints/{JOB_ID}/pause_checkpoint.pth
   ```

5. **Resume the Job**
   ```bash
   curl -X POST http://localhost:8001/v1/training/jobs/{JOB_ID}/resume
   
   # Expected response:
   # {
   #   "status": "queued",
   #   "celery_task_id": "...",
   #   "message": "Training resumed from epoch X"
   # }
   ```

6. **Verify Training Continues**
   - Watch UI: Status should change from "queued" → "running"
   - Verify epochs continue from where paused (e.g., if paused at epoch 5, should resume at epoch 6)
   - Check that training metrics continue to improve

**Expected Results**:
✅ Job pauses at epoch boundary  
✅ Pause checkpoint is saved to MinIO  
✅ Job resumes from correct epoch  
✅ Training continues without data loss  
✅ WebSocket updates work for pause/resume events

**Pass/Fail**: ___________

---

### Test 2: Pause During Different Epochs

**Objective**: Verify pause works at various points in training.

**Test Cases**:
- Pause at epoch 2 (early)
- Pause at epoch 10 (middle)
- Pause at epoch 18 (near end)

**Expected Results**: All pauses should work correctly regardless of epoch.

**Pass/Fail**: ___________

---

### Test 3: Multiple Pause/Resume Cycles

**Objective**: Verify job can be paused and resumed multiple times.

**Steps**:
1. Create job with 30 epochs
2. Pause at epoch 5 → Resume
3. Pause at epoch 12 → Resume
4. Pause at epoch 20 → Resume
5. Let complete to epoch 30

**Expected Results**: 
✅ Each pause/resume cycle works  
✅ Training state is preserved across multiple cycles  
✅ Final model is valid

**Pass/Fail**: ___________

---

### Test 4: Edge Cases

#### 4.1: Pause Before First Epoch Completes
**Steps**: Try to pause immediately after job starts (epoch 0)  
**Expected**: Should pause after epoch 1 completes

#### 4.2: Pause Non-Running Job
**Steps**: Try to pause a 'pending', 'completed', or 'failed' job  
**Expected**: HTTP 400 error "Cannot pause job in status '{status}'"

#### 4.3: Pause Synthetic Data Job
**Steps**: Try to pause a synthetic data generation job (total_epochs = 0)  
**Expected**: HTTP 400 error "Cannot pause synthetic data generation jobs"

#### 4.4: Resume Non-Paused Job
**Steps**: Try to resume a 'running' or 'completed' job  
**Expected**: HTTP 400 error "Cannot resume job in status '{status}'"

#### 4.5: Resume Without Checkpoint
**Steps**: Manually delete pause_checkpoint_path from DB, then try to resume  
**Expected**: HTTP 400 error "No pause checkpoint found"

**Pass/Fail**: ___________

---

### Test 5: UI/UX Verification

**Objective**: Verify frontend buttons and status display work correctly.

**Checks**:
- ✅ Pause button appears only for running training jobs (not synthetic data)
- ✅ Pause button shows pause icon (⏸)
- ✅ Resume button appears only for paused jobs
- ✅ Resume button shows play icon (▶️)
- ✅ Delete button is disabled for paused jobs
- ✅ Cancel button not shown for paused jobs
- ✅ Status badge shows 'paused' with dark variant and pause icon
- ✅ Confirmation dialogs appear before pause/resume actions
- ✅ WebSocket updates reflect status changes in real-time

**Pass/Fail**: ___________

---

### Test 6: Training State Preservation

**Objective**: Verify all training state is correctly preserved and restored.

**Verification After Resume**:
```python
# Check training logs show:
# - "Resuming from pause checkpoint: s3://models/checkpoints/.../pause_checkpoint.pth"
# - "Resumed from epoch X, best_val_loss=Y.ZZZZ"
# - Training loop continues from correct epoch
```

**State to Verify**:
- ✅ Model weights restored
- ✅ Optimizer state restored (momentum, etc.)
- ✅ Learning rate scheduler state restored
- ✅ Best validation loss preserved
- ✅ Best epoch number preserved
- ✅ Early stopping patience counter preserved
- ✅ Epoch counter correct

**Pass/Fail**: ___________

---

### Test 7: Performance & Timing

**Objective**: Verify pause/resume doesn't cause significant overhead.

**Measurements**:
- Time to pause: Should complete within one epoch duration
- Checkpoint save time: Should be < 10 seconds
- Resume startup time: Should be < 30 seconds
- Training performance after resume: Should match pre-pause performance

**Pass/Fail**: ___________

---

## 🔍 Automated Test Script

A Python test script is available to automate the basic flow:

```bash
# Install httpx if not already available
pip install httpx

# Run automated test
python scripts/test_pause_resume.py
```

**Expected Output**:
```
======================================================================
Testing Pause/Resume Functionality
======================================================================

1️⃣  Creating training job...
✅ Created job: <job_id>

2️⃣  Waiting for job to start running...
✅ Job is running at epoch 1

3️⃣  Waiting for epoch 2 or later...
✅ Reached epoch 2

4️⃣  Pausing training job...
✅ Pause requested: Training will pause after completing the current epoch

5️⃣  Waiting for job to pause...
✅ Job paused at epoch 3

6️⃣  Verifying pause checkpoint...
✅ Job status verified as 'paused'

7️⃣  Resuming training job...
✅ Resume requested: Training resumed from epoch 3

8️⃣  Waiting for job to resume running...
✅ Job resumed at epoch 3

9️⃣  Verifying training continues...
✅ Training progressed from epoch 3 to 4

======================================================================
✅ ALL TESTS PASSED - Pause/Resume functionality works!
======================================================================
```

---

## 📊 Test Results Summary

| Test Case | Status | Notes |
|-----------|--------|-------|
| Test 1: Basic Flow | ☐ Pass / ☐ Fail | |
| Test 2: Different Epochs | ☐ Pass / ☐ Fail | |
| Test 3: Multiple Cycles | ☐ Pass / ☐ Fail | |
| Test 4: Edge Cases | ☐ Pass / ☐ Fail | |
| Test 5: UI/UX | ☐ Pass / ☐ Fail | |
| Test 6: State Preservation | ☐ Pass / ☐ Fail | |
| Test 7: Performance | ☐ Pass / ☐ Fail | |

**Overall Result**: ☐ **PASS** / ☐ **FAIL**

**Tested By**: _______________  
**Date**: _______________  
**Environment**: _______________

---

## 🐛 Known Issues / Limitations

1. **Pause Timing**: Pause happens at epoch boundary, not immediately
   - This is by design to avoid data corruption
   - User receives message: "Training will pause after completing the current epoch"

2. **Scope**: Only applies to training jobs
   - Synthetic data generation jobs cannot be paused (total_epochs = 0)
   - This is intentional as data generation is typically faster

3. **Checkpoint Storage**: Pause checkpoint stored separately from best model checkpoint
   - Located at: `s3://models/checkpoints/{job_id}/pause_checkpoint.pth`
   - Automatically cleared when job resumes successfully

---

## 📝 Implementation Files

**Backend**:
- `db/migrations/020-add-pause-resume-training.sql` - Database schema
- `services/backend/src/models/training.py:25` - PAUSED status enum
- `services/backend/src/routers/training.py:424-594` - API endpoints
- `services/training/src/tasks/training_task.py:177-213` - Resume logic
- `services/training/src/tasks/training_task.py:452-503` - Pause detection

**Frontend**:
- `frontend/src/services/api/training.ts:19` - TypeScript types
- `frontend/src/services/api/training.ts:122-130` - API functions
- `frontend/src/pages/TrainingDashboard.tsx:246-268` - Event handlers
- `frontend/src/pages/TrainingDashboard.tsx:375` - Status badge
- `frontend/src/pages/TrainingDashboard.tsx:560-590` - Action buttons

---

## 🎓 References

- [Training Architecture](../../ARCHITECTURE.md#training-pipeline)
- [API Documentation](../../API.md#training-endpoints)
- [Development Guide](../../DEVELOPMENT.md)
