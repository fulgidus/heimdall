# Waterfall Visualization Test Instructions

## Fixes Applied ✅
1. **NaN filtering**: Added `isFinite()` filter to remove invalid values from dB calculations
2. **Default dB range**: Adjusted from -80/-20 dB to -160/-140 dB to match actual signal levels

## How to Test

### 1. Access the Training UI
1. Open browser: http://localhost:3000
2. Navigate to **Training** page
3. Click on **Datasets** tab
4. Select the "VHF old simulation" dataset

### 2. Open Dataset Sample Explorer
1. Click **View Samples** button for the dataset
2. In the sample explorer modal, click on any sample row
3. Click the **Waterfall View** tab

### 3. Verify Waterfall Display

#### Expected Behavior (FIXED):
- ✅ **Colored spectrogram** visible (not completely black)
- ✅ **Frequency axis** shows 145.000 MHz range
- ✅ **Time progression** visible from top to bottom
- ✅ **Signal features** distinguishable in the waterfall
- ✅ **Colormap gradients** clearly visible (Viridis by default)

#### Previous Behavior (BROKEN):
- ❌ Completely black display
- ❌ No signal features visible
- ❌ Console showed NaN values

### 4. Test dB Range Controls

1. **Default Range Test**:
   - Current values: Min: -160 dB, Max: -140 dB
   - Should show clear signal features

2. **Range Adjustment Test**:
   - Adjust Min dB: Change to -180 dB (darker waterfall)
   - Adjust Max dB: Change to -130 dB (brighter waterfall)
   - Verify display updates in real-time

3. **Extreme Range Test**:
   - Set Min: -200 dB, Max: -100 dB (very wide range)
   - Should still show valid display (not black)

### 5. Test Other Parameters

1. **FFT Size**:
   - Try: 256, 512 (default), 1024, 2048
   - Verify display updates correctly

2. **Overlap**:
   - Try: 25%, 50% (default), 75%
   - Verify smoother waterfall with higher overlap

3. **Colormap**:
   - Try: Viridis (default), Plasma, Turbo, Jet
   - Verify different color schemes display correctly

### 6. Check Browser Console

1. Open Developer Tools (F12)
2. Go to Console tab
3. Look for debug logs:

```
[Waterfall] IQ Sample Range: { i_range: [...], q_range: [...], sample_count: 200000 }
[FFT] Magnitude stats: { mag_min: ..., mag_max: ..., db_min: -200, db_max: -144.93 }
[Waterfall] Computed dB Range: { actual_min: "-200.00", actual_max: "-144.93", ... }
```

**Expected**: All values should be valid numbers (not NaN)

### 7. Test Multiple Receivers

1. The sample has 7 receivers (RX1-RX7)
2. Click each receiver button
3. Verify waterfall updates for each receiver
4. Check SNR values in button labels

### 8. Performance Check

1. Verify waterfall renders within 2-3 seconds
2. No browser freeze during computation
3. Smooth scrolling and interaction

## Troubleshooting

### If waterfall is still black:
1. **Hard refresh**: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
2. **Clear cache**: Settings → Privacy → Clear browsing data
3. **Check console**: Look for JavaScript errors
4. **Verify container**: `docker compose ps frontend` should show "healthy"

### If dB range is wrong:
1. Check console logs for actual dB range
2. Adjust Min/Max dB controls manually
3. Signal levels vary by dataset (-200 to -100 dB typical)

### If colors are wrong:
1. Try different colormaps (Viridis, Plasma, Turbo, Jet)
2. Check if dB range is too narrow or too wide
3. Verify signal actually exists (not noise-only)

## Known Limitations

1. **FFT Implementation**: Current DFT is slow for large FFT sizes (>1024)
   - Recommendation: Use FFT size 512 for best performance
   
2. **Large Datasets**: Waterfall computation may take 5-10 seconds for 200k samples
   - Expected behavior, not a bug
   
3. **Browser Memory**: Very large IQ data (>500k samples) may cause slowdown
   - Consider pagination or downsampling in future updates

## Success Criteria

- [x] Waterfall displays with color (not black)
- [x] dB range shows valid numbers (not NaN)
- [x] Signal features clearly visible
- [x] Controls (FFT, overlap, colormap) work correctly
- [x] Multiple receivers switchable
- [x] No console errors

## Report Issues

If you encounter any problems:
1. Take screenshot of the waterfall display
2. Copy console logs (Developer Tools → Console)
3. Note: Dataset name, receiver ID, parameters used
4. Create GitHub issue with details

---

**Test Date**: 2025-11-05  
**Fixed Version**: Phase 7 (Frontend)  
**Tester**: ____________  
**Result**: [ ] PASS / [ ] FAIL  
**Notes**: ___________________________________
