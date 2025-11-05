# 🌊 Waterfall Visualization Enhancement - Testing Guide

**Date**: 2025-11-05  
**Status**: ✅ Implementation Complete - Ready for Testing

---

## 📋 What Was Implemented

### Core Fixes (Phase 2)
1. **FFT Performance Fix**: Replaced naive O(N²) DFT with `fft.js` library (O(N log N))
2. **dB Range Fix**: Changed defaults from -160/-140 dB → **-80/-20 dB** (matches normalized IQ signals)
3. **Float32Array Validation**: Added runtime checks with error alerts

### New Features (Phase 3)
4. **Auto-scaling**: Uses 5th-95th percentile for intelligent dB range adjustment
5. **Progress Indicator**: ProgressBar shows STFT computation progress
6. **Settings Persistence**: LocalStorage saves user preferences
7. **Web Worker Support**: Framework ready for offloading FFT to background thread
8. **Statistics Display**: Badges showing actual/auto-scaled dB ranges

---

## 🎯 How to Test

### Prerequisites
- ✅ Frontend dev server running on `http://localhost:3001`
- ✅ Backend API running (Docker containers healthy)
- ✅ Test dataset available: `6e9e6129-d38b-4565-927f-5f65f2bf7aae`

### Test Steps

#### 1. Navigate to Training UI
```
URL: http://localhost:3001/training
```

1. Click on **"Synthetic Datasets"** tab
2. Find dataset with ID `6e9e6129-d38b-4565-927f-5f65f2bf7aae` (50 samples, FM voice modulation)
3. Click on any sample row to open the details panel
4. Click on the **"Waterfall"** tab

#### 2. Test Basic Waterfall Display

**Expected Results**:
- ✅ Waterfall should display **colorful FM signal structure** (NOT black)
- ✅ Computation should complete in **<1 second** for 50 samples
- ✅ Progress bar should appear briefly during computation
- ✅ Frequency axis labels should show correct MHz values
- ✅ Signal should be centered around the center frequency

**What to Look For**:
- FM voice modulation should appear as **vertical streaks** with varying intensity
- Colors should range from dark (low power) to bright (high power)
- No black/empty regions (this was the original bug)

#### 3. Test Auto-Scaling Feature

**Test Procedure**:
1. **Auto-scale ON** (default):
   - Check the green badge: "Auto: X to Y dB"
   - Should show a narrow range based on actual signal statistics
   - Signal should fill most of the color range

2. **Auto-scale OFF**:
   - Toggle the "Auto-scale dB" switch to OFF
   - Green badge changes to gray "Manual: -80 to -20 dB"
   - Signal may appear dimmer or over-saturated depending on actual power levels

3. **Manual Adjustment**:
   - With auto-scale OFF, change dB range manually
   - Try: Min=-100, Max=0 (wider range → less contrast)
   - Try: Min=-70, Max=-30 (narrow range → more contrast)

**Expected Results**:
- ✅ Auto-scaled range should be narrower than manual (e.g., -65 to -35 dB vs -80 to -20 dB)
- ✅ Manual dB inputs should be **disabled** when auto-scale is ON
- ✅ Badge should update to reflect current mode

#### 4. Test Settings Persistence

**Test Procedure**:
1. Change settings:
   - FFT Size: 1024
   - Overlap: 75%
   - Colormap: Plasma
   - Auto-scale: OFF
   - Min dB: -90, Max dB: -30

2. **Reload the page** (F5 or Ctrl+R)

3. Navigate back to Training → Synthetic Datasets → Sample → Waterfall

**Expected Results**:
- ✅ All settings should be **restored** from localStorage
- ✅ Waterfall should render with the saved settings

**Verify**:
```javascript
// Open browser console (F12) and run:
JSON.parse(localStorage.getItem('heimdall_waterfall_settings'))
```
Should show your saved settings.

#### 5. Test Reset to Defaults

**Test Procedure**:
1. Change several settings (FFT size, colormap, dB range, etc.)
2. Click **"Reset to Defaults"** button

**Expected Results**:
- ✅ FFT Size: 512
- ✅ Overlap: 50%
- ✅ Colormap: Viridis
- ✅ dB Range: -80 to -20
- ✅ Auto-scale: ON
- ✅ Use Web Worker: ON
- ✅ localStorage entry removed

#### 6. Test Different Parameters

**FFT Size**:
- 256: Fewer frequency bins, coarser resolution, faster
- 512: Default, balanced
- 1024: More frequency bins, finer resolution, slower
- 2048: Very high resolution, slowest

**Expected**: Larger FFT sizes show more detail in the frequency domain.

**Overlap**:
- 25%: Faster, less temporal smoothing
- 50%: Default, balanced
- 75%: Slower, smoother time evolution

**Expected**: Higher overlap creates smoother vertical transitions.

**Colormaps**:
- Viridis: Purple → Green → Yellow (perceptually uniform)
- Plasma: Purple → Orange → Yellow (high contrast)
- Turbo: Blue → Green → Yellow → Red (rainbow-like)
- Jet: Blue → Cyan → Green → Yellow → Red (classic rainbow)

**Expected**: Each colormap should clearly distinguish signal from noise.

#### 7. Test Progress Indicator

**Test Procedure**:
1. Select FFT Size: **2048** (largest)
2. Select Overlap: **75%** (highest)
3. Switch to a different receiver or sample

**Expected Results**:
- ✅ Blue progress bar should appear at the top
- ✅ Percentage should update during computation (e.g., "Computing STFT... 45%")
- ✅ Canvas should be slightly transparent (opacity: 0.5) during computation
- ✅ Progress bar should disappear when complete

#### 8. Test Statistics Display

**Look at the badges at the bottom right**:

- **Green Badge** (if auto-scale ON): "Auto: -65.3 to -35.7 dB"
  - Shows the 5th-95th percentile range
  
- **Gray Badge** (if auto-scale OFF): "Manual: -80 to -20 dB"
  - Shows user-set range

- **Blue Badge**: "Range: -85.2 to -15.6 dB"
  - Shows absolute min/max values in the STFT data
  - Hover for mean and median values

**Expected**:
- ✅ Blue badge should show wider range than auto-scale range
- ✅ Tooltip on blue badge should show mean and median dB values

---

## 🐛 Known Issues / Limitations

### 1. Web Worker Not Fully Functional
- **Status**: Fallback to main thread works correctly
- **Impact**: For very large datasets (>50 frames), computation happens on main thread
- **Workaround**: Progress indicator still shows progress, UI remains responsive

### 2. TypeScript Build Warnings (Unrelated)
- **Files**: `GenerateDataDialog.tsx`, `SampleDetailsPanel.tsx`, `SampleMapView.tsx`
- **Issue**: Pre-existing type errors not related to waterfall
- **Impact**: None - waterfall functionality unaffected

### 3. Settings Not Loaded on Initial Render
- **Status**: Fixed by using `localStorage.getItem()` in `useState()` initializer
- **Impact**: None if localStorage is available

---

## ✅ Success Criteria Checklist

- [ ] Waterfall displays FM signal structure (NOT black)
- [ ] Computation completes in <1 second for 50-sample dataset
- [ ] Progress bar appears during computation
- [ ] Auto-scale produces better visual results than manual
- [ ] Settings persist across page reloads
- [ ] Reset button restores all defaults
- [ ] All 4 colormaps work correctly
- [ ] Different FFT sizes produce different resolutions
- [ ] Statistics badges show accurate dB ranges
- [ ] Manual dB inputs disabled when auto-scale is ON

---

## 🔍 Debugging Tips

### If Waterfall is Still Black:

1. **Check browser console** (F12 → Console tab):
   ```
   Look for: "[Waterfall] IQ Sample Range:"
   Should show i_range and q_range with non-zero values
   ```

2. **Check STFT statistics**:
   ```
   Look for: "[Waterfall] Computed Statistics:"
   Should show minDb, maxDb, meanDb, medianDb values
   ```

3. **Verify IQ data decoding**:
   - Red alert should appear if Float32Array conversion fails
   - Check network tab for `/api/training/datasets/{id}/samples/{idx}/iq_data/{rx_id}`

### If Settings Don't Persist:

1. **Check localStorage** (F12 → Console):
   ```javascript
   localStorage.getItem('heimdall_waterfall_settings')
   ```
   Should return a JSON string with settings.

2. **Check browser privacy settings**:
   - Ensure cookies and localStorage are allowed
   - Incognito mode may block localStorage

### Performance Issues:

1. **Reduce FFT size**: 1024 → 512 or 256
2. **Reduce overlap**: 75% → 50% or 25%
3. **Enable Web Worker**: Check "Use Web Worker" toggle (though not fully implemented yet)

---

## 📊 Test Dataset Details

**Dataset ID**: `6e9e6129-d38b-4565-927f-5f65f2bf7aae`

**Characteristics**:
- 50 synthetic samples
- FM voice modulation (new signal generation)
- 7 receivers per sample
- Sample rate: ~50 kHz
- Center frequency: ~145.5 MHz (2m band)
- Duration: ~50-100 ms per sample
- SNR range: -10 to +30 dB

**Expected Waterfall Appearance**:
- Vertical streaks (time-varying frequency components)
- Multiple intensity peaks (voice harmonics)
- Symmetry around center frequency (FM modulation)

---

## 📝 Next Steps After Testing

1. **If tests pass**:
   - Mark Phase 7 waterfall enhancement as complete
   - Update CHANGELOG.md with new features
   - Consider full Web Worker implementation (optional)

2. **If tests fail**:
   - Document failure mode (screenshot + console logs)
   - Check specific test case that failed
   - File issue with reproduction steps

---

## 📚 Related Documentation

- [Waterfall Fix Summary](WATERFALL_FIX_SUMMARY.md) - Original bug fix documentation
- [Frontend Architecture](docs/ARCHITECTURE.md) - Overall system design
- [Training UI Documentation](docs/TRAINING.md) - Training interface details

---

**Questions?** Contact alessio.corsi@gmail.com or file an issue on GitHub.
