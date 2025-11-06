# Dataset Expansion Debug Guide

## Problem Summary
Users report two issues when expanding datasets:
1. **Generation stops after 5 samples instead of generating the requested amount (e.g., 1000)**
2. **New samples overwrite existing dataset instead of appending**

## Investigation Summary

### Code Analysis Completed

#### Generation Loop (`synthetic_generator.py` lines 1545-1847)
- **Target samples**: Correctly set to `num_samples` parameter (requested NEW samples, not including existing)
- **Loop condition**: `while valid_samples_collected < target_valid_samples and total_attempted < max_attempts`
- **Batch processing**: Saves incrementally with commits after each batch
- **Early termination**: Only stops if:
  - Target reached (`valid_samples_collected >= target_valid_samples`)
  - Max attempts exhausted (`total_attempted >= max_attempts` where `max_attempts = num_samples * 20`)
  - 5 consecutive batches with 0 valid samples (`consecutive_failures >= 5`)
  - Job manually cancelled

#### Database Logic (`save_features_to_db`, lines 1937-1998)
- **INSERT strategy**: Uses `INSERT INTO` (not UPDATE/DELETE) → Should append correctly
- **Primary key**: `recording_session_id = uuid.uuid4()` (unique for each sample)
- **Foreign key**: `dataset_id` (same for all samples in dataset, including expansions)
- **Table**: `heimdall.measurement_features` (for feature_based datasets)

#### Expansion Flow (`training_task.py` lines 1221-1552)
- **Dataset reuse**: When `expand_dataset_id` is set, REUSES existing dataset (line 1331)
- **Offset tracking**: `samples_offset` = current sample count (line 1233)
- **Progress display**: Shows cumulative count: `samples_offset + current` (line 1394)
- **Final count**: Queries actual count from database (lines 1527-1544)

### Expected Behavior
1. User has dataset with 5 samples
2. User requests expansion with 1000 additional samples
3. System should:
   - Set `samples_offset = 5`
   - Generate 1000 NEW samples (`target_valid_samples = 1000`)
   - INSERT all 1000 into `measurement_features` with same `dataset_id`
   - Count total: `SELECT COUNT(*) FROM measurement_features WHERE dataset_id = :id` → Returns 1005
   - Update: `UPDATE synthetic_datasets SET num_samples = 1005 WHERE id = :id`

### Potential Root Causes

#### Hypothesis 1: Low Success Rate
If validation parameters are too strict, the generation might fail to produce valid samples:
- **GDOP filter**: `max_gdop` (default 150, relaxed to 200 for random receivers)
- **SNR filter**: `min_snr_db` (default 3.0)
- **Receiver filter**: `min_receivers` (default 3)

If success rate is < 5%, generation might stop at `max_attempts` before reaching target.

**Detection**: Look for log messages:
```
Batch X complete: Y valid samples | Total: Z/1000 valid (W attempted, A.B% success rate)
```

If success rate is consistently < 10%, parameters are too strict.

#### Hypothesis 2: Database Transaction Issues
Batch commits might be failing silently or rolling back:
- **Symptom**: Logs show samples saved, but database count is lower
- **Cause**: Transaction rollback on error, foreign key violation, or connection issues

**Detection**: Look for:
```
[DB COUNT DEBUG] Expected cumulative: 150, Actual in DB: 5
```

If expected ≠ actual, commits are failing.

#### Hypothesis 3: Frontend Display Issue
Generation completes correctly, but frontend shows stale data:
- **Symptom**: Logs show 1000 samples generated and saved, database has 1005, but UI shows 5
- **Cause**: WebSocket event not received, cache not invalidated, or state not updated

**Detection**: Check browser console and compare with:
```
Published dataset update event: {dataset_id} with 1005 samples
```

## Debug Instrumentation Added

### In `synthetic_generator.py`

#### Generation Start (line 1543-1544)
```
[GENERATION DEBUG] Target: 1000 valid samples, max attempts: 20000
[GENERATION DEBUG] batch_size: 1, use_gpu: False, dataset_type: feature_based
```

#### After Each Batch (line 1754)
```
[BATCH DEBUG] Batch 1 results: valid_in_batch=5, valid_samples_collected=5, target=1000
```

#### Database Save (lines 1774-1776)
```
[DB SAVE DEBUG] About to save 5 feature samples to dataset_id=abc-123
[DB SAVE DEBUG] Saved 5 feature samples to database (5 total)
```

#### Database Count Update (lines 1799-1800)
```
[DB COUNT DEBUG] Dataset sample count updated: 10 samples (type: feature_based, batch: 2)
[DB COUNT DEBUG] Expected cumulative: 10, Actual in DB: 10
```

#### Progress Callback (lines 1833-1838)
```
[PROGRESS DEBUG] Calling progress_callback with valid=10, target=1000, attempted=12
[PROGRESS DEBUG] Callback completed successfully
```

### In `training_task.py`

#### Generation Complete (lines 1515-1516)
```
[GENERATION COMPLETE DEBUG] Generation stats: {'total_generated': 1000, 'total_attempted': 1234, ...}
[GENERATION COMPLETE DEBUG] total_generated=1000, reached_target=True, stopped_reason=target_reached
```

#### Final Count Query (lines 1527-1528, 1544)
```
[FINAL COUNT DEBUG] Querying final sample count for dataset_id=abc-123, dataset_type=feature_based
[FINAL COUNT DEBUG] samples_offset=5, stats_generated=1000
[FINAL COUNT DEBUG] Query returned actual_count=1005 from database
```

## Testing Instructions

### Test Case 1: Small Expansion with High Success Rate
```bash
# 1. Create initial dataset with 5 samples
curl -X POST http://localhost:8000/api/training/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-expansion-base",
    "description": "Base dataset for expansion test",
    "num_samples": 5,
    "frequency_mhz": 145.0,
    "tx_power_dbm": 5.0,
    "min_snr_db": 3.0,
    "min_receivers": 3,
    "max_gdop": 150,
    "dataset_type": "feature_based",
    "use_random_receivers": false,
    "use_gpu": false
  }'

# Wait for job to complete and note dataset_id

# 2. Expand dataset with 10 more samples
curl -X POST http://localhost:8000/api/training/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-expansion-extended",
    "description": "Expanded dataset",
    "num_samples": 10,
    "expand_dataset_id": "<dataset_id_from_step_1>",
    "dataset_type": "feature_based",
    "use_gpu": false
  }'

# 3. Monitor logs
docker-compose logs -f training | grep "DEBUG\]"

# 4. Verify final count
psql -U postgres -d heimdall -c "SELECT id, name, num_samples FROM heimdall.synthetic_datasets WHERE name LIKE 'test-expansion%';"
psql -U postgres -d heimdall -c "SELECT COUNT(*) FROM heimdall.measurement_features WHERE dataset_id = '<dataset_id>';"
```

**Expected Output**:
- Logs show: `[BATCH DEBUG] ... valid_samples_collected=10, target=10`
- Logs show: `[DB COUNT DEBUG] ... Actual in DB: 15` (5 original + 10 new)
- Logs show: `[FINAL COUNT DEBUG] ... actual_count=15`
- Database query returns: `num_samples=15`
- Feature count query returns: `15`

### Test Case 2: Large Expansion (Reproduce Issue)
```bash
# Use same steps as Test Case 1, but with num_samples: 1000 in step 2
# This should reproduce the reported issue if it exists
```

### Test Case 3: Low Success Rate Scenario
```bash
# Create expansion with VERY strict parameters to force low success rate
curl -X POST http://localhost:8000/api/training/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-strict-params",
    "description": "Test with strict validation",
    "num_samples": 100,
    "expand_dataset_id": "<existing_dataset_id>",
    "min_snr_db": 30.0,  # VERY high SNR requirement
    "min_receivers": 7,   # ALL receivers must detect
    "max_gdop": 10.0,     # VERY low GDOP requirement
    "dataset_type": "feature_based",
    "use_gpu": false
  }'

# Monitor for:
# - Low success rate in batch logs
# - "Max attempts reached" or "Too many consecutive failures" messages
```

**Expected**: Generation stops early with clear reason in logs.

## Log Analysis Checklist

When investigating an expansion issue, check these key log patterns:

### 1. Did generation start correctly?
```
✓ Expanding dataset {id} (current samples: 5)
✓ [GENERATION DEBUG] Target: 1000 valid samples, max attempts: 20000
```

### 2. Are batches processing?
```
✓ Batch 1 complete: X valid samples | Total: X/1000 valid (Y attempted, Z% success rate)
✓ [BATCH DEBUG] Batch 1 results: valid_in_batch=X, valid_samples_collected=X, target=1000
```

### 3. Are samples being saved to database?
```
✓ [DB SAVE DEBUG] About to save X feature samples to dataset_id=...
✓ [DB SAVE DEBUG] Saved X feature samples to database (X total)
```

### 4. Do database counts match expectations?
```
✓ [DB COUNT DEBUG] Expected cumulative: X, Actual in DB: X
```
**❌ RED FLAG**: If Expected ≠ Actual, commits are failing!

### 5. Did generation reach target?
```
✓ ✓ Target reached: 1000/1000 valid samples generated
✓ [GENERATION COMPLETE DEBUG] reached_target=True, stopped_reason=target_reached
```
**❌ RED FLAG**: If stopped_reason is NOT 'target_reached', investigate why.

### 6. Does final count include old + new samples?
```
✓ [FINAL COUNT DEBUG] samples_offset=5, stats_generated=1000
✓ [FINAL COUNT DEBUG] Query returned actual_count=1005 from database
```
**❌ RED FLAG**: If actual_count ≠ (samples_offset + stats_generated), data was lost!

## Quick Diagnosis Commands

```bash
# View all debug logs for a generation job
docker-compose logs training 2>&1 | grep "DEBUG\]" | less

# Check dataset sample counts
docker-compose exec postgres psql -U postgres -d heimdall -c \
  "SELECT id, name, num_samples, dataset_type FROM heimdall.synthetic_datasets ORDER BY created_at DESC LIMIT 10;"

# Verify actual feature count for a dataset
docker-compose exec postgres psql -U postgres -d heimdall -c \
  "SELECT dataset_id, COUNT(*) as actual_samples FROM heimdall.measurement_features GROUP BY dataset_id ORDER BY dataset_id;"

# Compare metadata vs actual counts
docker-compose exec postgres psql -U postgres -d heimdall -c \
  "SELECT d.id, d.name, d.num_samples as metadata_count, COUNT(f.id) as actual_count 
   FROM heimdall.synthetic_datasets d 
   LEFT JOIN heimdall.measurement_features f ON f.dataset_id = d.id 
   WHERE d.dataset_type = 'feature_based' 
   GROUP BY d.id, d.name, d.num_samples;"

# Check for any failed transactions or errors in logs
docker-compose logs training 2>&1 | grep -i "error\|failed\|rollback" | tail -50
```

## Next Steps

1. **Run Test Case 1** to verify instrumentation is working
2. **Run Test Case 2** to reproduce the reported issue
3. **Analyze logs** using the checklist above
4. **Report findings**:
   - If logs show `reached_target=False` with low success rate → Parameters too strict
   - If logs show `Expected ≠ Actual` in DB count → Transaction/commit issue
   - If logs show correct counts but UI is wrong → Frontend issue
   - If logs show generation stops early without clear reason → Unknown bug (deeper investigation needed)

## Related Files

- `services/training/src/data/synthetic_generator.py` - Core generation logic
- `services/training/src/tasks/training_task.py` - Celery task orchestration
- `services/training/src/api/synthetic.py` - REST API endpoint
- `db/migrations/013_create_measurement_features.sql` - Database schema

## Rollback Instructions

If you need to remove the debug logging after investigation:

```bash
# Revert changes
git diff services/training/src/data/synthetic_generator.py services/training/src/tasks/training_task.py
git checkout services/training/src/data/synthetic_generator.py services/training/src/tasks/training_task.py

# Or keep it (recommended) - the logging is non-intrusive and helpful for future debugging
```
