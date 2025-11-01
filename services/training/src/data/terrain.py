"""
Terrain elevation lookup and SRTM data management.

Provides elevation data for RF propagation calculations.
Phase 5: Simplified implementation with flat terrain assumption + random variation.
Future: Full SRTM tile download and interpolation.
"""

import structlog
from functools import lru_cache
from typing import Tuple
import numpy as np

logger = structlog.get_logger(__name__)


class TerrainLookup:
    """Terrain elevation lookup with LRU cache."""
    
    def __init__(self, use_srtm: bool = False):
        """
        Initialize terrain lookup.
        
        Args:
            use_srtm: If True, use SRTM data (not implemented yet). If False, use simplified model.
        """
        self.use_srtm = use_srtm
        
        if use_srtm:
            logger.warning("SRTM data not implemented yet, using simplified terrain model")
            self.use_srtm = False
    
    @lru_cache(maxsize=10000)
    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get terrain elevation at given coordinates.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
        
        Returns:
            Elevation in meters ASL
        """
        if self.use_srtm:
            return self._get_elevation_srtm(lat, lon)
        else:
            return self._get_elevation_simplified(lat, lon)
    
    def _get_elevation_simplified(self, lat: float, lon: float) -> float:
        """
        Simplified terrain model for Italian northwest region.
        
        Real terrain:
        - Alps in north/west: 1000-3000m
        - Po Valley: 50-200m
        - Apennines: 500-1500m
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Estimated elevation in meters
        """
        # Base elevation from latitude (Alps in north)
        base_elevation = 200 + (lat - 44.0) * 300  # 200m at 44°N, 500m at 45°N
        
        # Add some noise for realistic variation (deterministic per coordinate)
        seed = hash((round(lat, 6), round(lon, 6)))  # rounding to avoid floating point artifacts
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-50, 50)
        
        elevation = max(50.0, base_elevation + noise)  # Minimum 50m
        return elevation
    
    def _get_elevation_srtm(self, lat: float, lon: float) -> float:
        """
        Get elevation from SRTM data (not implemented yet).
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Elevation in meters
        """
        raise NotImplementedError("SRTM data download and interpolation not implemented yet")


class SRTMDownloader:
    """
    SRTM tile downloader and manager.
    
    Phase 5: Placeholder implementation.
    Future: Download from USGS, NASA, or OpenTopography.
    """
    
    def __init__(self, minio_client=None, db_session=None):
        """
        Initialize SRTM downloader.
        
        Args:
            minio_client: MinIO client for storage
            db_session: Database session for cache tracking
        """
        self.minio_client = minio_client
        self.db_session = db_session
        self.base_url = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
    
    async def download_tile(self, lat: int, lon: int) -> bool:
        """
        Download SRTM tile for given coordinates.
        
        Args:
            lat: Latitude of tile SW corner
            lon: Longitude of tile SW corner
        
        Returns:
            True if successful, False otherwise
        """
        tile_name = self._get_tile_name(lat, lon)
        logger.info(f"Downloading SRTM tile: {tile_name}")
        
        # Phase 5: Not implemented, just log
        logger.warning(f"SRTM download not implemented, tile {tile_name} will use simplified terrain")
        return False
    
    async def download_tiles(self, tiles: list[Tuple[int, int]]) -> dict:
        """
        Download multiple SRTM tiles.
        
        Args:
            tiles: List of (lat, lon) tuples for tile SW corners
        
        Returns:
            Dict with download status for each tile
        """
        results = {}
        
        for lat, lon in tiles:
            tile_name = self._get_tile_name(lat, lon)
            try:
                success = await self.download_tile(lat, lon)
                results[tile_name] = {"success": success, "error": None}
            except Exception as e:
                logger.error(f"Failed to download tile {tile_name}: {e}")
                results[tile_name] = {"success": False, "error": str(e)}
        
        return results
    
    def _get_tile_name(self, lat: int, lon: int) -> str:
        """
        Get SRTM tile name from coordinates.
        
        Args:
            lat: Latitude of tile SW corner
            lon: Longitude of tile SW corner
        
        Returns:
            Tile name (e.g., 'N44E007', 'S01W045')
        """
        lat_prefix = "N" if lat >= 0 else "S"
        lon_prefix = "E" if lon >= 0 else "W"
        
        return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"
    
    async def check_tile_exists(self, lat: int, lon: int) -> bool:
        """
        Check if SRTM tile already exists in cache.
        
        Args:
            lat: Latitude of tile SW corner
            lon: Longitude of tile SW corner
        
        Returns:
            True if tile exists, False otherwise
        """
        if not self.db_session:
            return False
        
        from sqlalchemy import text
        
        tile_name = self._get_tile_name(lat, lon)
        
        query = text("""
            SELECT status FROM heimdall.terrain_tiles
            WHERE tile_name = :tile_name AND status = 'ready'
        """)
        
        result = self.db_session.execute(query, {"tile_name": tile_name}).fetchone()
        return result is not None


def validate_elevation_data(terrain_lookup: TerrainLookup, checkpoints: list[Tuple[float, float, float]]) -> dict:
    """
    Validate terrain elevation data against known checkpoints.
    
    Args:
        terrain_lookup: TerrainLookup instance
        checkpoints: List of (lat, lon, expected_elevation) tuples
    
    Returns:
        Dict with validation results
    """
    results = {
        "total": len(checkpoints),
        "errors": [],
        "max_error": 0.0,
        "mean_error": 0.0
    }
    
    errors = []
    
    for lat, lon, expected in checkpoints:
        actual = terrain_lookup.get_elevation(lat, lon)
        error = abs(actual - expected)
        errors.append(error)
        
        if error > 100:  # More than 100m error
            results["errors"].append({
                "lat": lat,
                "lon": lon,
                "expected": expected,
                "actual": actual,
                "error": error
            })
    
    results["max_error"] = max(errors) if errors else 0.0
    results["mean_error"] = sum(errors) / len(errors) if errors else 0.0
    
    return results


# Terrain checkpoints for Italian northwest region (for validation)
ITALIAN_CHECKPOINTS = [
    (45.03, 7.27, 600),     # Aquila di Giaveno
    (45.234, 7.857, 300),   # Montanaro
    (45.044, 7.672, 250),   # Torino
    (44.561, 8.956, 480),   # Passo del Giovi
    (44.395, 8.956, 100),   # Genova (coast)
    (45.478, 9.123, 120),   # Milano - Baggio
    (44.5, 7.5, 800),       # Alps
    (45.0, 8.0, 300),       # Po Valley
    (44.7, 9.0, 600),       # Apennines
    (45.3, 8.5, 200),       # Po Valley
]
