"""
DEPRECATED: This module has been moved to src/config/settings.py

For backward compatibility, we re-export from the new location.
Please update imports to: from src.config import settings
"""

# Re-export from new location for backward compatibility
from .config.settings import Settings, settings

__all__ = ["Settings", "settings"]
