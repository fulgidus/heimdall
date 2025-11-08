#!/usr/bin/env python3
"""
Verification script to test dataloader selection logic in training pipeline.
This script verifies that models are correctly routed to IQ or feature-based dataloaders.

Usage:
    # Copy script to container
    docker cp verify_dataloader_selection.py heimdall-training:/app/
    
    # Run inside container
    docker exec heimdall-training python /app/verify_dataloader_selection.py
"""

import sys
sys.path.insert(0, '/app')

# Import using the src prefix since training_task.py uses this pattern
from src.models.model_factory import get_model_input_requirements
from src.models.model_registry import MODEL_REGISTRY

def test_dataloader_selection():
    """Test that the dataloader selection logic matches training_task.py lines 211-213."""
    
    print("=" * 80)
    print("DATALOADER SELECTION VERIFICATION")
    print("=" * 80)
    print()
    
    # Test models representing different categories
    test_models = [
        ("triangulation_model", "Feature-based (backward compat)"),
        ("heimdall_net", "Multi-modal (should use IQ dataloader)"),
        ("iq_resnet18", "IQ raw model"),
        ("localization_net_resnet18", "Spectrogram model"),
        ("iq_transformer", "IQ transformer model"),
        ("iq_vggnet", "IQ VGG model"),
        ("heimdall_net_v2", "Multi-modal v2 (should use IQ dataloader)"),
    ]
    
    results = []
    
    for model_id, description in test_models:
        if model_id not in MODEL_REGISTRY:
            print(f"⚠️  SKIP: {model_id} - Not in registry")
            continue
            
        # Get model requirements
        model_requirements = get_model_input_requirements(model_id)
        
        # Apply training_task.py lines 211-213 logic
        requires_iq = model_requirements.get("iq_raw", False)
        requires_spectrogram = model_requirements.get("spectrogram", False)
        use_iq_dataloader = requires_iq or requires_spectrogram
        
        # Determine dataloader type
        dataloader_type = "🎵 IQ/Spectrogram" if use_iq_dataloader else "📊 Feature-based"
        
        # Get model info from registry
        model_info = MODEL_REGISTRY[model_id]
        data_type = model_info.data_type
        
        # Store result
        result = {
            "model_id": model_id,
            "description": description,
            "data_type": data_type,
            "requires_iq": requires_iq,
            "requires_spectrogram": requires_spectrogram,
            "dataloader": dataloader_type,
        }
        results.append(result)
        
        # Print result
        print(f"Model: {model_id}")
        print(f"  Description: {description}")
        print(f"  Registry data_type: {data_type}")
        print(f"  Requires IQ raw: {requires_iq}")
        print(f"  Requires spectrogram: {requires_spectrogram}")
        print(f"  → Dataloader: {dataloader_type}")
        print()
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    # Expected behavior validation
    validation_passed = True
    
    for result in results:
        model_id = result["model_id"]
        data_type = result["data_type"]
        dataloader = result["dataloader"]
        
        # Validation rules
        expected_iq = data_type in ["iq_raw", "spectrogram", "hybrid", "multi_modal"]
        actual_iq = "IQ/Spectrogram" in dataloader
        
        if expected_iq != actual_iq:
            print(f"❌ FAIL: {model_id}")
            print(f"   Expected: {'IQ dataloader' if expected_iq else 'Feature dataloader'}")
            print(f"   Actual: {dataloader}")
            validation_passed = False
        else:
            print(f"✅ PASS: {model_id} → {dataloader}")
    
    print()
    print("=" * 80)
    
    if validation_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("The dataloader selection logic correctly routes models to appropriate dataloaders.")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("Some models are routed to incorrect dataloaders.")
        return 1

if __name__ == "__main__":
    sys.exit(test_dataloader_selection())
