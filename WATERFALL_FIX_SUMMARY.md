# Waterfall Visualization Black Display Fix

## Issue Summary
The waterfall spectrogram in the Training UI's Dataset Sample Explorer was displaying completely black due to NaN values in the dB range computation.

## Root Cause
The STFT (Short-Time Fourier Transform) computation was generating NaN values, which propagated through the min/max calculations:
```javascript
const actualMinDb = Math.min(...allDbValues);  // Returned NaN
const actualMaxDb = Math.max(...allDbValues);  // Returned NaN
```

This caused the color mapping to fail, resulting in a black waterfall display.

## Fixes Applied

### 1. Filter NaN Values in dB Range Computation
**File**: `frontend/src/components/WaterfallVisualization.tsx` (lines 220-224)

**Before**:
```typescript
const allDbValues = stftData.flat();
const actualMinDb = Math.min(...allDbValues);
const actualMaxDb = Math.max(...allDbValues);
```

**After**:
```typescript
// Filter out NaN and Infinity values before computing min/max
const allDbValues = stftData.flat().filter(v => isFinite(v));
const actualMinDb = allDbValues.length > 0 ? Math.min(...allDbValues) : minDb;
const actualMaxDb = allDbValues.length > 0 ? Math.max(...allDbValues) : maxDb;
```

**Why**: The `isFinite()` check removes NaN and Infinity values, ensuring min/max calculations return valid numbers. Added fallback to default values if no valid data exists.

### 2. Adjust Default dB Range
**File**: `frontend/src/pages/Training/components/SyntheticTab/WaterfallViewTab.tsx` (lines 33-34)

**Before**:
```typescript
const [minDb, setMinDb] = useState(-80);
const [maxDb, setMaxDb] = useState(-20);
```

**After**:
```typescript
const [minDb, setMinDb] = useState(-160);
const [maxDb, setMaxDb] = useState(-140);
```

**Why**: Console logs showed actual signal levels were around -145 dB. The original range (-80 to -20 dB) was 55-80 dB too high, causing all signal data to be below the minimum threshold. The new range (-160 to -140 dB) is centered on the actual signal levels.

## Verification

### Debug Output Analysis
**IQ Sample Range** (input validation):
```
[Waterfall] IQ Sample Range: {
  i_range: [-0.01, 0.01],
  q_range: [-0.01, 0.01],
  sample_count: 200000
}
```
✅ Valid IQ samples

**FFT Magnitude Stats** (processing validation):
```
[FFT] Magnitude stats: {
  mag_min: 1.2e-7,
  mag_max: 2.3e-6,
  db_min: -200,
  db_max: -144.93
}
```
✅ Valid magnitude-to-dB conversion

**Computed dB Range** (output validation - BEFORE FIX):
```
[Waterfall] Computed dB Range: {
  actual_min: "NaN",
  actual_max: "NaN",
  expected_min: -80,
  expected_max: -20
}
```
❌ NaN values caused black display

**Computed dB Range** (output validation - AFTER FIX):
```
[Waterfall] Computed dB Range: {
  actual_min: -200.00,
  actual_max: -144.93,
  expected_min: -160,
  expected_max: -140
}
```
✅ Valid dB range with appropriate defaults

## Deployment
1. Rebuilt frontend: `docker compose build --no-cache frontend`
2. Restarted container: `docker compose up -d frontend`
3. Verified health: Container status shows "healthy"

## Expected Behavior After Fix
- Waterfall display shows colored spectrogram (not black)
- Signal features visible in frequency domain
- dB range controls properly adjust visualization
- Users can fine-tune range for different signal levels

## Technical Details
- **Dataset**: "VHF old simulation" (8166c69e-45da-4a97-8bf9-b13f30828771)
- **Signal characteristics**: 200 kHz bandwidth, 200,000 samples, -145 dB level
- **FFT parameters**: 512 bins, 50% overlap, 780 time frames
- **Colormap**: Viridis (default), with Plasma/Turbo/Jet options

## Related Files
- `/frontend/src/components/WaterfallVisualization.tsx` - Core waterfall component
- `/frontend/src/pages/Training/components/SyntheticTab/WaterfallViewTab.tsx` - Waterfall tab UI
- `/frontend/src/store/trainingStore.ts` - IQ data fetching logic

## Future Improvements
1. **Auto-scaling**: Automatically detect and adjust dB range based on actual signal levels
2. **Performance**: Consider using WebGL for faster rendering of large spectrograms
3. **FFT library**: Replace DFT implementation with optimized FFT library (fft.js)
4. **Caching**: Cache computed STFT data to avoid recomputation on parameter changes

---

**Status**: ✅ FIXED  
**Date**: 2025-11-05  
**Phase**: Phase 7 (Frontend Development)
