#!/usr/bin/env python3
"""
Quick test to verify import structure is correct.
This is a temporary file to validate fixes - can be deleted after verification.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("Testing imports...")

# Test 1: Import settings from src.config
print("\n1. Testing: from src.config import settings")
try:
    from src.config import settings
    print(f"   ✓ SUCCESS: settings.service_name = '{settings.service_name}'")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Import Settings class
print("\n2. Testing: from src.config import Settings")
try:
    from src.config import Settings
    print(f"   ✓ SUCCESS: Settings class imported")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Import ModelConfig
print("\n3. Testing: from src.config import ModelConfig, BackboneArchitecture")
try:
    from src.config import ModelConfig, BackboneArchitecture
    print(f"   ✓ SUCCESS: ModelConfig and BackboneArchitecture imported")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Test backward compatibility with old config.py
print("\n4. Testing: from config import settings (backward compat)")
try:
    import os
    os.chdir(src_path)
    from config import settings as settings2
    print(f"   ✓ SUCCESS: settings.service_name = '{settings2.service_name}'")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All import tests passed! ✓")
print("="*60)
