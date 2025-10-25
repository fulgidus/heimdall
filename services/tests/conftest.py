"""
Pytest configuration for cross-service tests.

Imports shared fixtures from services/common/test_fixtures.py
"""

import sys
from pathlib import Path

# Add services/common to path so we can import fixtures
common_path = Path(__file__).parent.parent / "common"
if str(common_path) not in sys.path:
    sys.path.insert(0, str(common_path))

# Import all shared fixtures
from test_fixtures import *  # noqa: F401, F403
