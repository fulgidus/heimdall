# Audio Library Fix Summary

## Problem Description (Italian)
C'è un problema con la generazione dei sample. In pratica dovrebbero usare gli audio di media library per generare il contenuto del segnale che poi va alterato per darlo ai ricevitori della costellazione di ogni sample ma per qualche motivo reverta alla generazione dei suoni non basata su file.

## Problem Description (English)
There is a problem with sample generation. In practice, they should use audio from the media library to generate the signal content that is then modified for the receivers in each sample's constellation, but for some reason it reverts to non-file-based sound generation.

## Root Cause
The issue was a **configuration default mismatch** across multiple components in the codebase. While the training service had `use_audio_library: bool = True` as the intended default in its configuration, the API model and frontend were defaulting to `False`, which caused samples to be generated using synthetic formant tones instead of real audio from the media library.

## Architecture Flow
```
Frontend Request → Backend API → Database (Job Config) → Celery Task → IQ Generator
```

When a user submitted a dataset generation request:
1. Frontend sent request with `use_audio_library` default from dialog
2. Backend API received request with model default
3. Request serialized to database with model default value
4. Celery task loaded config from database
5. IQ Generator initialized with loaded config value

**Problem**: At step 1-2, the default was `False`, so this propagated through the entire chain.

## Files Modified

### 1. Backend API Model
**File**: `services/backend/src/models/synthetic_data.py`
**Line**: 94-95
**Change**:
```python
# Before
use_audio_library: bool = Field(
    default=False,  # ❌ WRONG
    ...
)

# After
use_audio_library: bool = Field(
    default=True,  # ✅ CORRECT
    ...
)
```

### 2. Frontend Dialog (2 locations)
**File**: `frontend/src/pages/Training/components/SyntheticTab/GenerateDataDialog.tsx`
**Lines**: 71, 141
**Change**:
```typescript
// Before
use_audio_library: false,  // ❌ WRONG

// After
use_audio_library: true,  // ✅ CORRECT
```

### 3. Training Service API
**File**: `services/training/src/api/synthetic.py`
**Line**: 345
**Change**:
```python
# Before
use_audio_library: bool = False  # ❌ WRONG

# After
use_audio_library: bool = True  # ✅ CORRECT
```

### 4. IQ Generator
**File**: `services/training/src/data/iq_generator.py`
**Line**: 81
**Change**:
```python
# Before
use_audio_library: bool = False,  # ❌ WRONG

# After
use_audio_library: bool = True,  # ✅ CORRECT
```

### 5. Synthetic Generator (2 locations)
**File**: `services/training/src/data/synthetic_generator.py`
**Lines**: 265, 587
**Change**:
```python
# Before
use_audio_library = config.get('use_audio_library', False)  # ❌ WRONG

# After
use_audio_library = config.get('use_audio_library', True)  # ✅ CORRECT
```

## Impact Analysis

### Before Fix
- All generated samples used **synthetic formant tones** (programmatically generated)
- Voice characteristics were artificial (3 formants, simple envelopes)
- No real-world audio diversity

### After Fix
- All generated samples use **real audio from media library**
- Voice characteristics match actual radio transmissions
- Realistic audio diversity (voice, music, documentaries, conferences)
- Fallback to formant synthesis if library is empty or fails

## Testing Impact

**Good News**: No tests need to be updated! 

The following test files were checked and already explicitly set `use_audio_library=True`:
- `services/training/tests/test_iq_generator.py` (line 262)
- `test_training_pipeline_integration.py` (line 48)

These tests will continue to work correctly because they explicitly override the default.

## Backward Compatibility

### Existing Datasets
Existing datasets generated before this fix will **NOT be affected**. They were generated with `use_audio_library=False` and that value is stored in their configuration.

### New Datasets
New datasets generated after this fix will **use audio library by default**. Users can still explicitly set `use_audio_library=false` in the frontend checkbox if they want formant synthesis.

## Verification Steps

To verify the fix works:

1. **Frontend Test**
   - Open Training page → Synthetic Data tab
   - Click "Generate Dataset"
   - Verify "Use Audio Library" checkbox is **checked by default**

2. **Backend Test**
   - Create a new dataset generation request without specifying `use_audio_library`
   - Check job config in database: `use_audio_library` should be `true`

3. **Runtime Test**
   - Generate a small dataset (e.g., 100 samples)
   - Check logs for: `voice_audio_generated_from_library` messages
   - Verify no `audio_library_fallback` warnings (unless library is empty)

4. **Audio Verification**
   - Download IQ samples from generated dataset
   - Analyze spectrograms - should show realistic speech patterns, not just 3 formants

## Configuration Hierarchy

The final value of `use_audio_library` is determined by (in priority order):

1. **Explicit user selection** (Frontend checkbox) → Highest priority
2. **API request parameter** → If provided in REST call
3. **Default from model** → Now `True` ✅
4. **Fallback in generator** → Now `True` ✅

## Related Configuration

The fix also ensures consistency with:
- `audio_library_fallback: bool = True` (unchanged, correct default)
  - If audio library fails or is empty, fallback to formant synthesis
  - This prevents generation failure if library is not populated

## Additional Notes

### Why This Matters
Real audio from the media library provides:
- **Better ML training**: Model learns from realistic signal variations
- **More robust localization**: Handles real-world audio characteristics
- **Realistic testing**: Validation against actual radio transmission patterns

### Future Enhancements
Consider adding:
- Audio library usage statistics to dataset metadata
- Warning if audio library is empty during generation
- Category selection for audio library (currently uses random selection)

## Commit Hash
Commit: `1040dd7`
Branch: `copilot/fix-sample-generation-issue`

## Related Issues
- Audio library integration (Phase 5)
- Training pipeline configuration (config.py)
- Sample generation flow (synthetic_generator.py)
