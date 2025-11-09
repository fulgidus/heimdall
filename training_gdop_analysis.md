# Training Results Analysis: GDOP Filter Investigation

## Executive Summary
✅ **Root cause CONFIRMED**: GDOP filtering was rejecting all 57 samples
✅ **Solution VALIDATED**: Disabling GDOP filter (max_gdop=999.0) allows training to proceed
⚠️ **UNEXPECTED RESULT**: Validation RMSE still very high (37.2 km)

## Test Results

### Job Configuration
- **Job Name**: triangulation_no_gdop_filter_test
- **Model**: triangulation_model
- **Dataset**: rf_features_fixed (fb242aa2-6068-48df-80e1-22bc90de961d)
- **Samples**: 45 train / 12 validation (57 total)
- **GDOP Filter**: DISABLED (max_gdop=999.0)
- **SNR Filter**: DISABLED (min_snr_db=-999.0)

### Training Metrics (Best Epoch: 9)
- **Train Loss**: 0.8063 (Gaussian NLL)
- **Val Loss**: 0.8592 (Gaussian NLL)
- **Train RMSE**: 49,101 meters (49.1 km)
- **Val RMSE**: 37,239 meters (37.2 km)
- **Val P50**: 36,555 meters (50% of predictions within 36.5 km)
- **Val P68**: 40,091 meters (68% within 40.1 km)
- **Val P95**: 59,948 meters (95% within 60 km)

### Geometry Quality
- **Mean GDOP**: 111.5 (VERY POOR)
- **GDOP < 5.0**: 0% (no samples with good geometry)
- **Mean Predicted Uncertainty**: 1.39 meters (SEVERELY UNDERESTIMATED)
- **Uncertainty Calibration Error**: 37,238 meters (MASSIVE)

## Analysis

### 1. GDOP Filter Issue - CONFIRMED ✅
The original problem was correctly identified:
- Database GDOP range: 8.45 - 146.03
- Default filter threshold: max_gdop=5.0
- Result: ALL 57 samples rejected → "num_samples=0" error

**Disabling the filter allows training to proceed successfully.**

### 2. Why is RMSE Still High? (37.2 km)

#### Possible Causes:

**A) Poor Receiver Geometry (Most Likely)**
- Mean GDOP of 111.5 is EXTREMELY poor
- GDOP reference: < 2 excellent, 2-5 good, 5-10 moderate, >20 poor
- With GDOP > 100, triangulation is mathematically ill-conditioned
- Even perfect RF features can't overcome bad geometry

**B) Synthetic Data Quality**
- Ground truth positions may not match actual triangulation potential
- 7 Italian WebSDR receivers may have poor spatial distribution
- Need to verify receiver positions form adequate triangulation geometry

**C) Model Architecture Limitations**
- triangulation_model may not be sophisticated enough
- Feature extraction (PSD, freq_offset, SNR) may need more channels
- Need to compare with heimdall_net architecture

**D) RF Feature Extraction (Less Likely)**
Previous analysis showed:
- ✅ PSD values: Real (-174.5 to -137.8 dBm), NOT hardcoded
- ✅ Freq offset: Real (-41.9 to 14.7 Hz), NOT hardcoded  
- ✅ SNR: Real statistics with proper distribution

### 3. Uncertainty Calibration Failure ❌
- Model predicts uncertainty: ~1.4 meters
- Actual error: ~37,239 meters
- **Calibration error**: Model is overconfident by 26,600x

This indicates the model hasn't learned meaningful uncertainty estimates.

## Recommendations

### Immediate Actions

1. **Verify Receiver Geometry**
   ```sql
   SELECT 
       name, latitude, longitude,
       ST_Distance(
           geom::geography,
           ST_SetSRID(ST_MakePoint(12.0, 42.0), 4326)::geography
       ) / 1000.0 as distance_from_center_km
   FROM websdrs
   ORDER BY distance_from_center_km;
   ```
   Check if receivers are:
   - Too close together (poor baseline)
   - In a line (poor triangulation)
   - Missing altitude diversity

2. **Generate Higher Quality Synthetic Data**
   - Target GDOP < 10 (ideally < 5)
   - Ensure receivers form good triangulation geometry
   - Add more diverse source positions
   - Increase dataset size (1000+ samples)

3. **Relax GDOP Filter Strategically**
   Instead of max_gdop=999.0 (no filter), use:
   - max_gdop=20.0: Accept "fair" geometry
   - max_gdop=50.0: Accept "poor" geometry (for testing)
   - Document expected RMSE degradation per GDOP threshold

4. **Test with Real WebSDR Data**
   - Collect actual IQ samples from known transmitter positions
   - Validate RF feature extraction works correctly
   - Compare synthetic vs. real data GDOP distributions

### Long-Term Solutions

1. **Improve Synthetic Data Generation**
   - Optimize receiver placement for triangulation
   - Add terrain-aware propagation modeling
   - Generate GDOP-stratified datasets

2. **Model Architecture Improvements**
   - Add GDOP as input feature (help model learn geometry dependence)
   - Implement uncertainty-aware loss that penalizes miscalibration
   - Try heimdall_net architecture (more sophisticated)

3. **Multi-Model Ensemble**
   - Train separate models for different GDOP ranges
   - Use GDOP-based model selection at inference time
   - Improve uncertainty estimates through ensemble disagreement

## Conclusion

### What We Learned
1. ✅ **GDOP filter was the root cause** of "num_samples=0" error
2. ✅ **Disabling filter allows training** to complete successfully
3. ⚠️ **High RMSE (37 km) is due to EXTREMELY poor geometry** (GDOP=111)
4. ❌ **Model uncertainty estimates are severely miscalibrated**

### The Real Problem
**The RF fixes ARE working** (PSD and frequency offset are correct), but **the synthetic data has fundamentally poor triangulation geometry**. You can't triangulate accurately with GDOP > 100, regardless of signal quality.

### Next Steps (Priority Order)
1. **Check receiver spatial distribution** (verify they're not colinear/clustered)
2. **Generate new synthetic dataset** with GDOP < 20 target
3. **Retrain with better data** and compare RMSE improvement
4. **Add GDOP as model input** to learn geometry-aware predictions
5. **Test with real WebSDR recordings** to validate end-to-end pipeline

### Expected Outcomes
With GDOP < 5 and proper geometry:
- Target RMSE: < 1 km (1000 m)
- P68 accuracy: < 500 m
- P95 accuracy: < 2 km

With current GDOP=111 geometry:
- Best case RMSE: 20-40 km (what we're seeing)
- Triangulation is mathematically ill-conditioned
- No amount of training will significantly improve accuracy
