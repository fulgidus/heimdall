#!/usr/bin/env python3
"""
Test ONNX export for both LocalizationNet (spectrogram) and HeimdallNet (multi-modal).

Verifies:
1. LocalizationNet exports successfully (regression test)
2. HeimdallNet exports successfully (new functionality)
3. HeimdallNetPro exports successfully (new functionality)
4. Model type auto-detection works
5. ONNX files are valid and loadable
"""

import sys
from pathlib import Path

# Add services/training/src to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "training" / "src"))

import torch
import onnx
import structlog

from models.localization_net import LocalizationNet
from models.heimdall_net import HeimdallNet, HeimdallNetPro
from onnx_export import ONNXExporter, _detect_model_type

logger = structlog.get_logger(__name__)


def test_localization_net_export():
    """Test LocalizationNet (spectrogram-based) ONNX export."""
    print("\n" + "=" * 80)
    print("TEST 1: LocalizationNet (Spectrogram) Export")
    print("=" * 80)
    
    try:
        # Create model and move to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LocalizationNet(pretrained=False)
        model = model.to(device)
        model.eval()
        
        # Detect model type
        model_type = _detect_model_type(model)
        print(f"✅ Model type detected: {model_type}")
        print(f"✅ Model on device: {device}")
        assert model_type == "spectrogram", f"Expected 'spectrogram', got '{model_type}'"
        
        # Export to ONNX
        exporter = ONNXExporter(None, None)
        output_path = Path("/tmp/test_localization_net.onnx")
        exporter.export_to_onnx(model, output_path, model_type=model_type)
        
        # Validate ONNX
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Check input/output shapes
        graph = onnx_model.graph
        assert len(graph.input) == 1, f"Expected 1 input, got {len(graph.input)}"
        assert len(graph.output) == 2, f"Expected 2 outputs, got {len(graph.output)}"
        
        input_name = graph.input[0].name
        output_names = [out.name for out in graph.output]
        
        print(f"✅ ONNX model valid")
        print(f"   Input: {input_name}")
        print(f"   Outputs: {output_names}")
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        print("✅ LocalizationNet export: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ LocalizationNet export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heimdall_net_export():
    """Test HeimdallNet (multi-modal) ONNX export."""
    print("\n" + "=" * 80)
    print("TEST 2: HeimdallNet (Multi-Modal) Export")
    print("=" * 80)
    
    try:
        # Create model and move to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeimdallNet(max_receivers=10)
        model = model.to(device)
        model.eval()
        
        # Detect model type
        model_type = _detect_model_type(model)
        print(f"✅ Model type detected: {model_type}")
        print(f"✅ Model on device: {device}")
        assert model_type == "multi_modal", f"Expected 'multi_modal', got '{model_type}'"
        
        # Export to ONNX
        exporter = ONNXExporter(None, None)
        output_path = Path("/tmp/test_heimdall_net.onnx")
        exporter.export_to_onnx(model, output_path, model_type=model_type)
        
        # Validate ONNX
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Check input/output shapes
        graph = onnx_model.graph
        assert len(graph.input) == 5, f"Expected 5 inputs, got {len(graph.input)}"
        assert len(graph.output) == 2, f"Expected 2 outputs, got {len(graph.output)}"
        
        input_names = [inp.name for inp in graph.input]
        output_names = [out.name for out in graph.output]
        
        expected_inputs = ["iq_data", "features", "positions", "receiver_ids", "mask"]
        for expected in expected_inputs:
            assert expected in input_names, f"Missing input: {expected}"
        
        print(f"✅ ONNX model valid")
        print(f"   Inputs: {input_names}")
        print(f"   Outputs: {output_names}")
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        print("✅ HeimdallNet export: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ HeimdallNet export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heimdall_net_pro_export():
    """Test HeimdallNetPro (multi-modal with Performer attention) ONNX export."""
    print("\n" + "=" * 80)
    print("TEST 3: HeimdallNetPro (Experimental) Export")
    print("=" * 80)
    
    try:
        # Create model and move to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeimdallNetPro(max_receivers=10)
        model = model.to(device)
        model.eval()
        
        # Detect model type
        model_type = _detect_model_type(model)
        print(f"✅ Model type detected: {model_type}")
        print(f"✅ Model on device: {device}")
        assert model_type == "multi_modal", f"Expected 'multi_modal', got '{model_type}'"
        
        # Export to ONNX
        exporter = ONNXExporter(None, None)
        output_path = Path("/tmp/test_heimdall_net_pro.onnx")
        exporter.export_to_onnx(model, output_path, model_type=model_type)
        
        # Validate ONNX
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Check input/output shapes
        graph = onnx_model.graph
        assert len(graph.input) == 5, f"Expected 5 inputs, got {len(graph.input)}"
        assert len(graph.output) == 2, f"Expected 2 outputs, got {len(graph.output)}"
        
        input_names = [inp.name for inp in graph.input]
        output_names = [out.name for out in graph.output]
        
        print(f"✅ ONNX model valid")
        print(f"   Inputs: {input_names}")
        print(f"   Outputs: {output_names}")
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        print("✅ HeimdallNetPro export: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ HeimdallNetPro export: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all ONNX export tests."""
    print("\n" + "=" * 80)
    print("ONNX Export Test Suite for Heimdall Models")
    print("=" * 80)
    
    results = {
        "LocalizationNet": test_localization_net_export(),
        "HeimdallNet": test_heimdall_net_export(),
        "HeimdallNetPro": test_heimdall_net_pro_export(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{model_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
