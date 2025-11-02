"""
Terrain elevation lookup and SRTM data management.

Provides elevation data for RF propagation calculations.
"""

import hashlib
import io
import os
import structlog
import tempfile
from functools import lru_cache
from typing import Tuple, Optional
import numpy as np

logger = structlog.get_logger(__name__)


class TerrainLookup:
    """Terrain elevation lookup with LRU cache."""
    
    def __init__(self, use_srtm: bool = False, minio_client=None):
        """
        Initialize terrain lookup.
        
        Args:
            use_srtm: If True, use SRTM data. If False, use simplified model.
            minio_client: MinIO client for downloading SRTM tiles (required if use_srtm=True)
        """
        self.use_srtm = use_srtm
        self.minio_client = minio_client
        self._tile_cache = {}  # Cache for loaded rasterio datasets
        self._temp_files = []  # Track temporary files for cleanup
        
        if use_srtm and not minio_client:
            logger.warning("SRTM enabled but no MinIO client provided, using simplified terrain model")
            self.use_srtm = False
        elif use_srtm:
            logger.info("TerrainLookup initialized with SRTM data support")
    
    def __del__(self):
        """Cleanup temporary files when instance is destroyed."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.debug(f"Failed to cleanup temp file {temp_file}: {e}")
    
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
        seed = abs(hash((round(lat, 6), round(lon, 6))))  # abs() ensures non-negative seed
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-50, 50)
        
        elevation = max(50.0, base_elevation + noise)  # Minimum 50m
        return elevation
    
    def _get_elevation_srtm(self, lat: float, lon: float) -> float:
        """
        Get elevation from SRTM data.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Elevation in meters, or simplified model if tile not available
        """
        try:
            import rasterio
            from rasterio.errors import RasterioIOError
        except ImportError:
            logger.error("rasterio not installed, falling back to simplified model")
            return self._get_elevation_simplified(lat, lon)
        
        # Determine which tile we need
        tile_lat = int(np.floor(lat))
        tile_lon = int(np.floor(lon))
        tile_name = self._get_tile_name(tile_lat, tile_lon)
        
        # Check if tile is in cache
        if tile_name not in self._tile_cache:
            # Try to load tile from MinIO
            try:
                tile_dataset = self._load_tile_from_minio(tile_name)
                if tile_dataset is None:
                    # Tile not available, fallback to simplified model
                    logger.debug(f"SRTM tile {tile_name} not available, using simplified model")
                    return self._get_elevation_simplified(lat, lon)
                self._tile_cache[tile_name] = tile_dataset
            except Exception as e:
                logger.error(f"Failed to load SRTM tile {tile_name}: {e}")
                return self._get_elevation_simplified(lat, lon)
        
        # Get dataset from cache
        dataset = self._tile_cache[tile_name]
        
        # Convert lat/lon to pixel coordinates
        try:
            row, col = dataset.index(lon, lat)
            
            # Read elevation value
            window = rasterio.windows.Window(col, row, 1, 1)
            elevation_data = dataset.read(1, window=window)
            
            if elevation_data.size > 0:
                elevation = float(elevation_data[0, 0])
                # Handle SRTM no-data values
                if elevation < -1000 or elevation > 9000:
                    logger.debug(f"Invalid SRTM elevation {elevation} at ({lat}, {lon}), using simplified model")
                    return self._get_elevation_simplified(lat, lon)
                return elevation
            else:
                return self._get_elevation_simplified(lat, lon)
        except Exception as e:
            logger.debug(f"Error reading SRTM elevation at ({lat}, {lon}): {e}")
            return self._get_elevation_simplified(lat, lon)
    
    def _load_tile_from_minio(self, tile_name: str):
        """
        Load SRTM tile from MinIO bucket.
        
        Args:
            tile_name: Tile name (e.g., 'N44E007')
        
        Returns:
            rasterio dataset or None if not available
        """
        try:
            import rasterio
            from rasterio.io import MemoryFile
        except ImportError:
            return None
        
        bucket_name = "heimdall-terrain"
        object_name = f"tiles/{tile_name}.tif"
        
        try:
            # Download tile from MinIO
            response = self.minio_client.s3_client.get_object(
                Bucket=bucket_name,
                Key=object_name
            )
            tile_data = response['Body'].read()
            
            # Load into rasterio using MemoryFile
            with MemoryFile(tile_data) as memfile:
                # Open dataset and keep it open (we'll cache it)
                # Note: We need to copy data to a persistent location since MemoryFile is temporary
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    tmp.write(tile_data)
                    tmp_path = tmp.name
                
                # Track temp file for cleanup
                self._temp_files.append(tmp_path)
                
                dataset = rasterio.open(tmp_path)
                return dataset
        except Exception as e:
            logger.debug(f"Tile {tile_name} not found in MinIO: {e}")
            return None
    
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


class SRTMDownloader:
    """
    SRTM tile downloader and manager using OpenTopography API.
    """
    
    def __init__(self, minio_client=None, db_pool=None, api_key: Optional[str] = None):
        """
        Initialize SRTM downloader.
        
        Args:
            minio_client: MinIO client for storage
            db_pool: Database connection pool for cache tracking
            api_key: OpenTopography API key (or from OPENTOPOGRAPHY_API_KEY env var)
        """
        self.minio_client = minio_client
        self.db_pool = db_pool
        self.base_url = "https://portal.opentopography.org/API/globaldem"
        self.api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")
        self.bucket_name = "heimdall-terrain"
        
        if not self.api_key:
            logger.warning("No OpenTopography API key provided, downloads will fail")
        
        # Ensure bucket exists
        if self.minio_client:
            self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure MinIO bucket exists."""
        try:
            self.minio_client.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"Bucket {self.bucket_name} exists")
        except Exception:
            try:
                self.minio_client.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created bucket {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to create bucket {self.bucket_name}: {e}")
    
    async def download_tile(self, lat: int, lon: int) -> dict:
        """
        Download SRTM tile for given coordinates from OpenTopography.
        
        Args:
            lat: Latitude of tile SW corner (e.g., 44 for N44)
            lon: Longitude of tile SW corner (e.g., 7 for E007)
        
        Returns:
            Dict with status, error message, and tile info
        """
        import aiohttp
        
        tile_name = self._get_tile_name(lat, lon)
        logger.info(f"Downloading SRTM tile: {tile_name}")
        
        if not self.api_key:
            return {"success": False, "error": "No API key configured", "tile_name": tile_name}
        
        if not self.minio_client:
            return {"success": False, "error": "No MinIO client configured", "tile_name": tile_name}
        
        # Update DB status to downloading
        if self.db_pool:
            await self._update_tile_status(tile_name, lat, lon, "downloading")
        
        # Build OpenTopography API request
        # Using SRTMGL1 (1 arc-second ~ 30m resolution)
        params = {
            'demtype': 'SRTMGL1',
            'south': lat,
            'north': lat + 1,
            'west': lon,
            'east': lon + 1,
            'outputFormat': 'GTiff',
            'API_Key': self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        logger.error(f"Failed to download tile {tile_name}: {error_msg}")
                        if self.db_pool:
                            await self._update_tile_status(tile_name, lat, lon, "failed", error_msg)
                        return {"success": False, "error": error_msg, "tile_name": tile_name}
                    
                    # Read tile data
                    tile_data = await response.read()
                    file_size = len(tile_data)
                    
                    # Calculate SHA256 checksum
                    checksum = hashlib.sha256(tile_data).hexdigest()
                    
                    # Save to MinIO
                    object_name = f"tiles/{tile_name}.tif"
                    try:
                        self.minio_client.s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=object_name,
                            Body=io.BytesIO(tile_data),
                            ContentType='image/tiff'
                        )
                        logger.info(f"Saved tile {tile_name} to MinIO: {file_size} bytes")
                    except Exception as e:
                        error_msg = f"Failed to save to MinIO: {e}"
                        logger.error(error_msg)
                        if self.db_pool:
                            await self._update_tile_status(tile_name, lat, lon, "failed", error_msg)
                        return {"success": False, "error": error_msg, "tile_name": tile_name}
                    
                    # Update DB status to ready
                    source_url = f"{self.base_url}?{self._encode_params(params)}"
                    if self.db_pool:
                        await self._update_tile_status(
                            tile_name, lat, lon, "ready",
                            error_message=None,
                            file_size=file_size,
                            checksum=checksum,
                            source_url=source_url
                        )
                    
                    return {
                        "success": True,
                        "tile_name": tile_name,
                        "file_size": file_size,
                        "checksum": checksum
                    }
        
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {e}"
            logger.error(f"Failed to download tile {tile_name}: {error_msg}")
            if self.db_pool:
                await self._update_tile_status(tile_name, lat, lon, "failed", error_msg)
            return {"success": False, "error": error_msg, "tile_name": tile_name}
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Failed to download tile {tile_name}: {error_msg}")
            if self.db_pool:
                await self._update_tile_status(tile_name, lat, lon, "failed", error_msg)
            return {"success": False, "error": error_msg, "tile_name": tile_name}
    
    def _encode_params(self, params: dict) -> str:
        """URL encode parameters."""
        from urllib.parse import urlencode
        return urlencode(params)
    
    async def download_tiles(self, tiles: list[Tuple[int, int]]) -> dict:
        """
        Download multiple SRTM tiles.
        
        Args:
            tiles: List of (lat, lon) tuples for tile SW corners
        
        Returns:
            Dict with download summary and per-tile results
        """
        results = []
        successful = 0
        failed = 0
        
        for lat, lon in tiles:
            result = await self.download_tile(lat, lon)
            results.append(result)
            if result["success"]:
                successful += 1
            else:
                failed += 1
        
        return {
            "successful": successful,
            "failed": failed,
            "total": len(tiles),
            "tiles": results
        }
    
    async def _update_tile_status(
        self,
        tile_name: str,
        lat: int,
        lon: int,
        status: str,
        error_message: Optional[str] = None,
        file_size: Optional[int] = None,
        checksum: Optional[str] = None,
        source_url: Optional[str] = None
    ):
        """
        Update tile status in database.
        
        Args:
            tile_name: Tile name
            lat: Latitude
            lon: Longitude
            status: Status (pending/downloading/ready/failed)
            error_message: Optional error message
            file_size: Optional file size in bytes
            checksum: Optional SHA256 checksum
            source_url: Optional source URL
        """
        if not self.db_pool:
            return
        
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        
        async with self.db_pool.acquire() as conn:
            # Check if tile exists
            check_query = "SELECT id FROM heimdall.terrain_tiles WHERE tile_name = $1"
            existing = await conn.fetchrow(check_query, tile_name)
            
            if existing:
                # Update existing tile
                update_query = """
                    UPDATE heimdall.terrain_tiles
                    SET status = $1,
                        error_message = $2,
                        file_size_bytes = COALESCE($3, file_size_bytes),
                        checksum_sha256 = COALESCE($4, checksum_sha256),
                        source_url = COALESCE($5, source_url),
                        downloaded_at = CASE WHEN $1 = 'ready' THEN $6 ELSE downloaded_at END,
                        updated_at = $6
                    WHERE tile_name = $7
                """
                await conn.execute(
                    update_query,
                    status,
                    error_message,
                    file_size,
                    checksum,
                    source_url,
                    now,
                    tile_name
                )
            else:
                # Insert new tile
                insert_query = """
                    INSERT INTO heimdall.terrain_tiles (
                        tile_name, lat_min, lat_max, lon_min, lon_max,
                        minio_bucket, minio_path, status, error_message,
                        file_size_bytes, checksum_sha256, source_url,
                        downloaded_at, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8, $9,
                        $10, $11, $12,
                        CASE WHEN $8 = 'ready' THEN $13 ELSE NULL END,
                        $13, $13
                    )
                """
                await conn.execute(
                    insert_query,
                    tile_name,
                    lat,
                    lat + 1,
                    lon,
                    lon + 1,
                    self.bucket_name,
                    f"tiles/{tile_name}.tif",
                    status,
                    error_message,
                    file_size,
                    checksum,
                    source_url,
                    now
                )
    
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
        if not self.db_pool:
            return False
        
        tile_name = self._get_tile_name(lat, lon)
        
        query = """
            SELECT status FROM heimdall.terrain_tiles
            WHERE tile_name = $1 AND status = 'ready'
        """
        
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow(query, tile_name)
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
