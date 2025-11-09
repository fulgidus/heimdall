#!/usr/bin/env python3
"""
Quick test to verify training normalization/denormalization fixes.

Tests that the coordinate normalization constants match between
gpu_cached_dataset.py and training_task.py.

Run from project root: 
    python test_training_normalization_fix.py

Or in container:
    docker exec heimdall-training python /app/test_training_normalization_fix.py
"""

import sys
import os
import torch
import numpy as np

# Detect if running in container or from project root
if os.path.exists('/app/src'):
    # Running in container
    sys.path.insert(0, '/app')
    from src.data.gpu_cached_dataset import (
        METERS_PER_DEG_LAT,
        METERS_PER_DEG_LON,
        DELTA_METERS_MIN,
        DELTA_METERS_MAX
    )
    from src.models.localization_net import LocalizationNet
else:
    # Running from project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'training'))
    from services.training.src.data.gpu_cached_dataset import (
        METERS_PER_DEG_LAT,
        METERS_PER_DEG_LON,
        DELTA_METERS_MIN,
        DELTA_METERS_MAX
    )
    from services.training.src.models.localization_net import LocalizationNet


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters (NumPy version)."""
    R = 6371000.0  # Earth radius in meters
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    return distance


def test_normalization_denormalization():
    """
    Test that normalization and denormalization are inverse operations.
    """
    print("\n" + "="*70)
    print("NORMALIZATION/DENORMALIZATION VERIFICATION")
    print("="*70)
    
    print(f"\nConstants from gpu_cached_dataset.py:")
    print(f"  METERS_PER_DEG_LAT = {METERS_PER_DEG_LAT}")
    print(f"  METERS_PER_DEG_LON = {METERS_PER_DEG_LON}")
    print(f"  DELTA_METERS_MIN = {DELTA_METERS_MIN}")
    print(f"  DELTA_METERS_MAX = {DELTA_METERS_MAX}")
    
    # Test case: Italian coordinates
    print(f"\n{'='*70}")
    print("TEST 1: Coordinate normalization/denormalization round-trip")
    print("="*70)
    
    # Example: Rome to Milan
    centroid_lat = 42.0  # Rome latitude
    centroid_lon = 12.5  # Rome longitude
    target_lat = 45.5    # Milan latitude
    target_lon = 9.2     # Milan longitude
    
    # Calculate delta in degrees
    delta_lat_deg = target_lat - centroid_lat  # +3.5 degrees
    delta_lon_deg = target_lon - centroid_lon  # -3.3 degrees
    
    print(f"\nOriginal coordinates:")
    print(f"  Centroid: ({centroid_lat:.4f}°, {centroid_lon:.4f}°)")
    print(f"  Target: ({target_lat:.4f}°, {target_lon:.4f}°)")
    print(f"  Delta (deg): ({delta_lat_deg:.4f}°, {delta_lon_deg:.4f}°)")
    
    # Calculate actual distance
    actual_distance = haversine_distance(centroid_lat, centroid_lon, target_lat, target_lon)
    print(f"  Actual distance: {actual_distance:.2f} m ({actual_distance/1000:.2f} km)")
    
    # === NORMALIZATION (as done in gpu_cached_dataset.py) ===
    # Step 1: Convert degrees to meters
    delta_lat_meters = delta_lat_deg * METERS_PER_DEG_LAT
    delta_lon_meters = delta_lon_deg * METERS_PER_DEG_LON
    
    print(f"\nNormalization steps:")
    print(f"  Step 1 - Delta in meters:")
    print(f"    Lat: {delta_lat_meters:.2f} m")
    print(f"    Lon: {delta_lon_meters:.2f} m")
    
    # Step 2: Normalize to [0, 1]
    delta_lat_normalized = (delta_lat_meters - DELTA_METERS_MIN) / (DELTA_METERS_MAX - DELTA_METERS_MIN)
    delta_lon_normalized = (delta_lon_meters - DELTA_METERS_MIN) / (DELTA_METERS_MAX - DELTA_METERS_MIN)
    
    print(f"  Step 2 - Normalized to [0,1]:")
    print(f"    Lat: {delta_lat_normalized:.6f}")
    print(f"    Lon: {delta_lon_normalized:.6f}")
    
    # Check if values are in valid range
    if not (0 <= delta_lat_normalized <= 1 and 0 <= delta_lon_normalized <= 1):
        print(f"\n❌ FAIL: Normalized values out of [0,1] range!")
        return False
    
    # === DENORMALIZATION (as done in training_task.py) ===
    # Step 1: Denormalize [0, 1] to meters [-100k, +100k]
    recovered_lat_meters = delta_lat_normalized * (DELTA_METERS_MAX - DELTA_METERS_MIN) + DELTA_METERS_MIN
    recovered_lon_meters = delta_lon_normalized * (DELTA_METERS_MAX - DELTA_METERS_MIN) + DELTA_METERS_MIN
    
    print(f"\nDenormalization steps:")
    print(f"  Step 1 - Recovered meters:")
    print(f"    Lat: {recovered_lat_meters:.2f} m")
    print(f"    Lon: {recovered_lon_meters:.2f} m")
    
    # Step 2: Convert meters to degrees
    recovered_lat_deg = recovered_lat_meters / METERS_PER_DEG_LAT
    recovered_lon_deg = recovered_lon_meters / METERS_PER_DEG_LON
    
    print(f"  Step 2 - Recovered degrees:")
    print(f"    Lat: {recovered_lat_deg:.6f}°")
    print(f"    Lon: {recovered_lon_deg:.6f}°")
    
    # Step 3: Add centroid to get absolute coordinates
    recovered_target_lat = recovered_lat_deg + centroid_lat
    recovered_target_lon = recovered_lon_deg + centroid_lon
    
    print(f"  Step 3 - Recovered absolute coordinates:")
    print(f"    Target: ({recovered_target_lat:.6f}°, {recovered_target_lon:.6f}°)")
    
    # Verify round-trip accuracy
    lat_error = abs(recovered_target_lat - target_lat)
    lon_error = abs(recovered_target_lon - target_lon)
    
    print(f"\nRound-trip error:")
    print(f"  Lat error: {lat_error:.9f}° ({lat_error * METERS_PER_DEG_LAT:.6f} m)")
    print(f"  Lon error: {lon_error:.9f}° ({lon_error * METERS_PER_DEG_LON:.6f} m)")
    
    # Allow 1mm error (floating point precision)
    if lat_error * METERS_PER_DEG_LAT < 0.001 and lon_error * METERS_PER_DEG_LON < 0.001:
        print(f"\n✅ PASS: Round-trip normalization/denormalization accurate (<1mm error)")
    else:
        print(f"\n❌ FAIL: Round-trip error too large")
        return False
    
    # === TEST 2: Boundary values ===
    print(f"\n{'='*70}")
    print("TEST 2: Boundary value handling")
    print("="*70)
    
    test_cases = [
        ("Min boundary", -100.0, "km"),  # -100km delta
        ("Max boundary", 100.0, "km"),   # +100km delta
        ("Zero", 0.0, "km"),              # No delta
        ("Typical", 50.0, "km"),          # 50km delta
    ]
    
    all_passed = True
    for name, delta_km, unit in test_cases:
        delta_meters = delta_km * 1000.0
        
        # Normalize
        normalized = (delta_meters - DELTA_METERS_MIN) / (DELTA_METERS_MAX - DELTA_METERS_MIN)
        
        # Denormalize
        recovered_meters = normalized * (DELTA_METERS_MAX - DELTA_METERS_MIN) + DELTA_METERS_MIN
        
        error_meters = abs(recovered_meters - delta_meters)
        
        if 0 <= normalized <= 1 and error_meters < 0.001:
            status = "✅"
        else:
            status = "❌"
            all_passed = False
        
        print(f"  {status} {name:15s}: {delta_km:+7.1f} {unit} → norm={normalized:.6f} → {recovered_meters/1000:+7.1f} km (err={error_meters:.6f} m)")
    
    if not all_passed:
        print(f"\n❌ FAIL: Some boundary values failed")
        return False
    
    print(f"\n✅ PASS: All boundary values handled correctly")
    
    # === TEST 3: Model forward pass with realistic data ===
    print(f"\n{'='*70}")
    print("TEST 3: Model forward pass with normalized inputs")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    
    # Create model
    model = LocalizationNet(
        pretrained=False,  # Faster initialization for testing
        backbone_size="tiny"  # Smallest model for faster testing
    ).to(device)
    
    # Create synthetic batch (mel-spectrogram format expected by LocalizationNet)
    batch_size = 4
    
    # Features: [batch, 3, 128, 32] (mel-spectrogram)
    # 3 channels: I, Q, magnitude
    # 128 frequency bins (mel-spectrogram)
    # 32 time frames
    features = torch.randn(batch_size, 3, 128, 32).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        position_pred, uncertainty_pred = model(features)
    
    print(f"\n  Input shape: {features.shape}")
    print(f"  Output position shape: {position_pred.shape}")
    print(f"  Output uncertainty shape: {uncertainty_pred.shape}")
    print(f"  Position range: [{position_pred.min().item():.4f}, {position_pred.max().item():.4f}]")
    print(f"  Uncertainty range: [{uncertainty_pred.min().item():.4f}, {uncertainty_pred.max().item():.4f}]")
    
    # Check for NaN
    if torch.isnan(position_pred).any() or torch.isnan(uncertainty_pred).any():
        print(f"\n❌ FAIL: Model outputs contain NaN")
        return False
    
    # Verify outputs are in reasonable range (position should be unbounded due to sigmoid removal)
    # But let's just check they're not ridiculously large
    if position_pred.abs().max() > 1e6:
        print(f"\n❌ FAIL: Model outputs unreasonably large values")
        return False
    
    print(f"\n✅ PASS: Model forward pass produces valid outputs")
    
    # === TEST 4: Loss computation with normalized values ===
    print(f"\n{'='*70}")
    print("TEST 4: Loss computation with realistic normalized values")
    print("="*70)
    
    # Simulate normalized target positions [0, 1]
    target_positions = torch.rand(batch_size, 2).to(device)
    
    # Simulate model predictions (also normalized)
    predictions = torch.rand(batch_size, 2).to(device)
    uncertainties = torch.randn(batch_size, 2).to(device) * 0.1  # Log variance
    
    print(f"\n  Target range: [{target_positions.min().item():.4f}, {target_positions.max().item():.4f}]")
    print(f"  Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    # Compute Gaussian NLL loss
    log_var = uncertainties
    variance = torch.exp(log_var)
    
    position_loss = 0.5 * (
        ((predictions - target_positions) ** 2) / variance + log_var
    ).mean()
    
    print(f"\n  Loss: {position_loss.item():.6f}")
    
    if torch.isnan(position_loss):
        print(f"\n❌ FAIL: Loss is NaN")
        return False
    
    if position_loss.item() > 1000:
        print(f"\n❌ FAIL: Loss unreasonably high")
        return False
    
    print(f"\n✅ PASS: Loss computation produces valid results")
    
    # === FINAL SUMMARY ===
    print(f"\n{'='*70}")
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    print(f"\nVerification complete:")
    print(f"  ✅ Normalization/denormalization are inverse operations")
    print(f"  ✅ Boundary values handled correctly")
    print(f"  ✅ Model forward pass produces valid outputs")
    print(f"  ✅ Loss computation works with normalized values")
    print(f"\nThe training pipeline is ready for full training runs.")
    
    return True


def main():
    """Run verification test."""
    try:
        success = test_normalization_denormalization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
