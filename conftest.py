"""
Global pytest configuration for all services
Fixes Python import paths and provides common fixtures
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Pytest hook to configure paths before tests run"""
    
    # Aggiungi le root path comuni
    project_root = Path(__file__).parent.parent
    
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
    """Skip E2E tests che richiedono server running"""
    skip_markers = [
        'e2e',
        'health_endpoint',
        'api_contracts',
        'integration_api',  
        'requires_server',
    ]
    
    for item in items:
        # Se il test è in una directory e2e o ha un marker da skippare, marcalo
        if "e2e" in str(item.fspath):
            item.add_marker("skip")
            item.add_marker("SKIPPED_E2E_REQUIRES_SERVER")
