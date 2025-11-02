"""
Terrain elevation lookup and SRTM data management.

Shared module for terrain operations across backend and training services.
"""

from .terrain import (
    TerrainLookup,
    SRTMDownloader,
    validate_elevation_data,
    ITALIAN_CHECKPOINTS
)

__all__ = [
    "TerrainLookup",
    "SRTMDownloader",
    "validate_elevation_data",
    "ITALIAN_CHECKPOINTS"
]
