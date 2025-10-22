"""
pytest configuration for api-gateway service tests
Fixes Python import paths
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Add src/ to Python path before tests run"""
    
    # Get the root of the api-gateway service
    service_root = Path(__file__).parent.parent
    src_path = service_root / "src"
    
    # Add src/ to path if not already there
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Also add service root
    if str(service_root) not in sys.path:
        sys.path.insert(0, str(service_root))
