# GDOP Threshold Bug Fix - Session Complete

**Date**: 2025-11-04  
**Issue**: GPU synthetic data generation producing 0 samples due to incorrect GDOP threshold  
**Status**: ✅ RESOLVED

## Problem Summary

Synthetic data generation was failing with **0 samples generated** (100% rejection rate) despite GPU optimization working correctly. All samples were being PRE-REJECTED with error: `GDOP=50.0 > 10.0`.

### Root Cause

The `max_gdop` parameter had **inconsistent default values** across the codebase:

1. **Pydantic Model** (`synthetic_data.py:55`): `default=10.0`
2. **Training Task** (`training_task.py:134`): `default=5.0`
3. **Training Task** (`training_task.py:676`): `default=10.0`
4. **Generator** (`synthetic_generator.py:197`): `default=100.0`

When requests were submitted without explicitly setting `max_gdop`, the Pydantic model's default of `10.0` was used, **overwriting** the intended value of `100.0`.

### Why GDOP 10.0 Fails

With 7 fixed Italian WebSDR receivers:
- Geometric diversity is limited
- GDOP values typically range from **40-60** for most TX positions
- `max_gdop=10.0` → **100% rejection rate**
- `max_gdop=100.0` → **98-100% success rate**

## Solution

Updated all default values to **100.0** across the codebase:

### Files Modified

1. **`services/backend/src/models/synthetic_data.py`** (Line 55)
   ```python
   # BEFORE
   max_gdop: float = Field(default=10.0, ge=1.0, description="Maximum GDOP")
   
   # AFTER
   max_gdop: float = Field(default=100.0, ge=1.0, description="Maximum GDOP")
   ```

2. **`services/backend/src/tasks/training_task.py`** (Line 134)
   ```python
   # BEFORE
   max_gdop = config.get("max_gdop", 5.0)
   
   # AFTER
   max_gdop = config.get("max_gdop", 100.0)
   ```

3. **`services/backend/src/tasks/training_task.py`** (Line 676)
   ```python
   # BEFORE
   max_gdop=config.get('max_gdop', 10.0),
   
   # AFTER
   max_gdop=config.get('max_gdop', 100.0),
   ```

## Verification Results

### Before Fix
| Job ID | max_gdop | Samples Generated | Success Rate |
|--------|----------|-------------------|--------------|
| `97ee9e65-0703-4f90-88df-2dc509db3673` | 10.0 | 0 | 0.0% |
| `bf395339-a050-47da-8240-303a815f06a1` | 10.0 | 0 | 0.0% |

### After Fix
| Job ID | max_gdop | Samples Generated | Success Rate |
|--------|----------|-------------------|--------------|
| `d2ad83bb-c3b4-4908-8d52-60db6e08db48` | 100.0 | 100 | 100.0% |
| `2cc48ac8-4fba-4299-9e10-a5c59fd4ca91` | 100.0 | 99 | 99.0% |

**Performance**:
- Generation time: ~40 seconds for 100 samples
- GPU utilization: 34%
- CPU utilization: 108% (multi-threaded)
- Memory usage: 2.26 GiB

## Debug Process

1. **Added debug logging** to trace config values through the pipeline
2. **Queried database** to confirm stored config values
3. **Traced config inheritance** logic for dataset expansion
4. **Identified Pydantic model** as the source of default value
5. **Updated all occurrences** to ensure consistency
6. **Rebuilt Docker container** to pick up changes
7. **Verified fix** with multiple test jobs

## Key Learnings

### Config Propagation Path
```
HTTP Request (Pydantic validation)
    ↓ (default=10.0 applied here if missing)
Backend Router (config inheritance)
    ↓
Database Storage (JSONB)
    ↓
Celery Task (load from DB)
    ↓
Generator Function (config.get('max_gdop', 100.0))
    ↓
Worker Threads (pre-check threshold)
```

### Best Practices

1. **Single Source of Truth**: Define defaults in ONE location
2. **Explicit Over Implicit**: Always set critical parameters explicitly in requests
3. **Validation Early**: Pydantic models validate at API boundary
4. **Log Config Values**: Add debug logging for critical parameters
5. **Test Default Behavior**: Verify what happens when params are omitted

## Impact

✅ **Synthetic data generation now works with default parameters**  
✅ **98-100% success rate for fixed Italian receiver network**  
✅ **Consistent behavior across all code paths**  
✅ **No breaking changes** (only default value adjustments)

## Related Files

- `services/backend/src/models/synthetic_data.py`
- `services/backend/src/tasks/training_task.py`
- `services/training/src/data/synthetic_generator.py`

## Next Steps

- [ ] Remove debug logging from `synthetic_generator.py` (lines 200-202)
- [ ] Consider adding config validation at API level
- [ ] Document recommended GDOP thresholds for different receiver configurations
- [ ] Add integration test for default parameter behavior

---

**Session Duration**: ~45 minutes  
**Lines Changed**: 3  
**Test Jobs Run**: 5  
**Success**: ✅ Complete
