"""
Global pytest configuration for all services
Fixes Python import paths and provides common fixtures
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Pytest hook to configure paths before tests run"""
    
    # Get project root (where this conftest.py is located)
    project_root = Path(__file__).parent
    
    # Per ogni servizio, aggiungi src/ al path
    service_dirs = list(project_root.glob("services/*/"))
    for service_dir in service_dirs:
        src_path = service_dir / "src"
        if src_path.exists() and str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Aggiungi anche la root del servizio
        if str(service_dir) not in sys.path:
            sys.path.insert(0, str(service_dir))


def pytest_collection_modifyitems(config, items):
    """Skip E2E tests that require server running"""
    import pytest
    
    skip_markers = [
        'e2e',
        'health_endpoint',
        'api_contracts',
        'integration_api',  
        'requires_server',
    ]
    
    for item in items:
        # Skip tests in e2e directories or with specific markers
        if "e2e" in str(item.fspath):
            # Don't skip our new e2e tests in services/tests/
            if "services/tests/test_e2e" not in str(item.fspath):
                item.add_marker(pytest.mark.skip(reason="E2E test requires running server"))
