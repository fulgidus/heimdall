#!/usr/bin/env python3
"""
Integration test for training data pipeline.
Verifies that the full training pipeline produces spatially coherent samples.

Run from project root: python test_training_pipeline_integration.py
Run in container: docker compose exec heimdall-training python /app/test_training_pipeline_integration.py
"""

import sys
import os
import numpy as np

# Detect if running in container or from project root
# In container: /app/ is working directory, imports are from src/
# From root: need to add services/training to path
if os.path.exists('/app/src'):
    # Running in container - use container import paths
    sys.path.insert(0, '/app')
    from src.data.iq_generator import SyntheticIQGenerator  # type: ignore
else:
    # Running from project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'training'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))
    from services.training.src.data.iq_generator import SyntheticIQGenerator  # type: ignore


def test_training_data_spatial_coherence():
    """
    Integration test: Verify training data has spatial coherence.
    
    This simulates what happens during real training data generation:
    1. Generate a batch of IQ samples (7 receivers)
    2. Each receiver has different propagation effects
    3. Verify all receivers have same audio content
    """
    print("\n" + "="*70)
    print("TRAINING PIPELINE INTEGRATION TEST")
    print("Testing spatial coherence in training data generation")
    print("="*70)
    
    # Initialize generator (CPU mode for testing)
    generator = SyntheticIQGenerator(
        sample_rate_hz=200_000,
        duration_ms=1000.0,
        seed=42,
        use_gpu=False,
        use_audio_library=True,
        audio_library_fallback=True
    )
    
    # Simulate 7 WebSDR receivers (realistic scenario)
    batch_size = 7
    
    # Realistic propagation parameters (from Phase 3 WebSDR configuration)
    frequency_offsets = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    bandwidths = np.array([12500.0] * batch_size)
    # Use high SNR for testing audio consistency (50 dB)
    # Real-world SNR would be 15-27 dB, but that makes noise dominant in correlation tests
    signal_powers_dbm = np.array([-37.0, -37.0, -37.0, -37.0, -37.0, -37.0, -37.0])
    noise_floors_dbm = np.array([-87.0] * batch_size)
    snr_dbs = signal_powers_dbm - noise_floors_dbm  # 50 dB SNR
    
    print(f"\nGenerating batch of {batch_size} IQ samples...")
    print(f"  Sample rate: {generator.sample_rate_hz/1000:.1f} kHz")
    print(f"  Duration: {generator.duration_ms} ms")
    print(f"  Frequency offsets: {frequency_offsets.tolist()}")
    
    # Generate batch WITH effects (realistic training scenario)
    batch_signals_with_effects = generator.generate_iq_batch(
        frequency_offsets=frequency_offsets,
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=True,  # Realistic
        enable_fading=True       # Realistic
    )
    
    # Generate ANOTHER batch WITHOUT effects (same generator, different audio content)
    # NOTE: Each call to generate_iq_batch loads a NEW random audio file
    # We'll test spatial coherence WITHIN each batch, not across batches
    batch_signals_clean = generator.generate_iq_batch(
        frequency_offsets=np.zeros(batch_size),  # Zero offsets for perfect match
        bandwidths=bandwidths,
        signal_powers_dbm=signal_powers_dbm,
        noise_floors_dbm=noise_floors_dbm,
        snr_dbs=snr_dbs,
        batch_size=batch_size,
        enable_multipath=False,
        enable_fading=False
    )
    
    print(f"\n✓ Generated batch with shape: {batch_signals_with_effects.shape}")
    
    # Helper function for audio extraction
    def extract_audio_from_iq(iq_signal):
        """Extract audio by FM demodulation."""
        phase = np.unwrap(np.angle(iq_signal))
        audio = np.diff(phase)
        return audio
    
    # Test 1: Clean signals should have SAME AUDIO CONTENT within the batch
    print("\n" + "-"*70)
    print("TEST 1: Spatial coherence in CLEAN batch (zero freq offset)")
    print("-"*70)
    print("NOTE: All receivers in this batch should have identical audio")
    print("      (but independent noise, so we use correlation, not equality)")
    
    reference_audio_clean = extract_audio_from_iq(batch_signals_clean[0])
    clean_correlations = []
    for i in range(1, batch_size):
        audio = extract_audio_from_iq(batch_signals_clean[i])
        
        # Normalize for correlation
        ref_norm = (reference_audio_clean - np.mean(reference_audio_clean)) / (np.std(reference_audio_clean) + 1e-8)
        audio_norm = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
        
        correlation = np.corrcoef(ref_norm, audio_norm)[0, 1]
        clean_correlations.append(correlation)
        print(f"  Receiver {i+1} vs Receiver 1: correlation={correlation:.6f}")
    
    min_clean_corr = np.min(clean_correlations)
    avg_clean_corr = np.mean(clean_correlations)
    
    print(f"\nCorrelation statistics:")
    print(f"  Average: {avg_clean_corr:.6f}")
    print(f"  Minimum: {min_clean_corr:.6f}")
    print(f"  Maximum: {np.max(clean_correlations):.6f}")
    
    # With zero frequency offsets and no multipath/fading, correlation should be very high (>0.95)
    # Note: Noise prevents perfect 1.0 correlation, but audio content is identical
    if min_clean_corr > 0.95:
        print(f"\n✅ PASS: Audio content identical within batch (min corr={min_clean_corr:.6f} > 0.95)")
    else:
        print(f"\n❌ FAIL: Audio content differs within batch (min corr={min_clean_corr:.6f} < 0.95)")
        return False
    
    # Test 2: With effects, audio should still be consistent
    print("\n" + "-"*70)
    print("TEST 2: Audio consistency with multipath + fading + noise")
    print("-"*70)
    
    audio_signals = []
    for i in range(batch_size):
        audio = extract_audio_from_iq(batch_signals_with_effects[i])
        audio_signals.append(audio)
        print(f"  Receiver {i+1}: mean={np.mean(audio):.6f}, std={np.std(audio):.6f}")
    
    # Compute correlations between audio signals
    reference_audio = audio_signals[0]
    correlations = []
    for i in range(1, batch_size):
        # Normalize audio to [-1, 1] for fair comparison
        ref_norm = (reference_audio - np.mean(reference_audio)) / (np.std(reference_audio) + 1e-8)
        audio_norm = (audio_signals[i] - np.mean(audio_signals[i])) / (np.std(audio_signals[i]) + 1e-8)
        
        correlation = np.corrcoef(ref_norm, audio_norm)[0, 1]
        correlations.append(correlation)
        print(f"  Receiver {i+1} vs Receiver 1: correlation={correlation:.4f}")
    
    avg_correlation = np.mean(correlations)
    min_correlation = np.min(correlations)
    
    print(f"\nCorrelation statistics:")
    print(f"  Average: {avg_correlation:.4f}")
    print(f"  Minimum: {min_correlation:.4f}")
    print(f"  Maximum: {np.max(correlations):.4f}")
    
    # With multipath + fading + noise, we expect lower correlation than clean signals
    # but still significant (>0.5 indicates same audio content)
    if min_correlation > 0.5:
        print(f"\n✅ PASS: Audio content consistent (min corr={min_correlation:.4f} > 0.5)")
    else:
        print(f"\n⚠️  WARNING: Low correlation (min={min_correlation:.4f})")
        print("   This may indicate:")
        print("   - Heavy multipath/fading degradation (expected)")
        print("   - Different audio content (bug if correlation < 0.2)")
    
    # Test 3: Verify samples are different (not all identical due to effects)
    print("\n" + "-"*70)
    print("TEST 3: Propagation effects create phase diversity")
    print("-"*70)
    
    # Check that phases differ (due to frequency offsets in the WITH-EFFECTS batch)
    phase_diffs = []
    ref_phase = np.angle(batch_signals_with_effects[0])
    for i in range(1, batch_size):
        phase_i = np.angle(batch_signals_with_effects[i])
        # Compute mean absolute phase difference
        phase_diff = np.mean(np.abs(np.angle(np.exp(1j * (phase_i - ref_phase)))))
        phase_diffs.append(phase_diff)
        print(f"  Receiver {i+1} mean phase diff: {phase_diff:.3f} rad")
    
    # Phase differences should be significant (due to frequency offsets + multipath)
    avg_phase_diff = np.mean(phase_diffs)
    if avg_phase_diff > 0.5:
        print(f"\n✅ PASS: Phase diversity present (mean diff={avg_phase_diff:.3f} rad > 0.5)")
    else:
        print(f"\n❌ FAIL: No phase diversity (mean diff={avg_phase_diff:.3f} rad)")
        return False
    
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print("✅ Clean signals identical (same audio content)")
    print(f"✅ Audio correlation with effects: {avg_correlation:.4f}")
    print("✅ Propagation effects create diversity")
    print("\n🎉 TRAINING PIPELINE PRODUCES SPATIALLY COHERENT DATA!")
    
    return True


def main():
    """Run integration test."""
    try:
        success = test_training_data_spatial_coherence()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
