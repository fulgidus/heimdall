"""
Terrain/SRTM data management endpoints.

Provides API for downloading, managing, and querying SRTM terrain tiles.
"""

import math
import os
import sys

import structlog
from fastapi import APIRouter, HTTPException, Depends, Query

# Import terrain module from common
from common.terrain import SRTMDownloader, TerrainLookup

from ..db import get_pool
from ..models.terrain import (
    TerrainTile,
    TerrainTilesList,
    DownloadRequest,
    DownloadResponse,
    CoverageStatus,
    CoverageRegion,
    ElevationResponse,
    TileDownloadResult
)
from ..storage.minio_client import MinIOClient
from ..config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/terrain",
    tags=["terrain"],
    responses={404: {"description": "Not found"}},
)


# Module-level singleton for terrain lookup to enable cache reuse
_terrain_lookup_cache = None

def get_minio_client() -> MinIOClient:
    """Get MinIO client for terrain storage."""
    return MinIOClient(
        endpoint_url=settings.minio_url,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name="heimdall-terrain"
    )


def get_terrain_lookup(minio_client: MinIOClient = Depends(get_minio_client)) -> TerrainLookup:
    """Get or create singleton TerrainLookup instance."""
    global _terrain_lookup_cache
    if _terrain_lookup_cache is None:
        _terrain_lookup_cache = TerrainLookup(use_srtm=True, minio_client=minio_client)
    return _terrain_lookup_cache


@router.get("/tiles", response_model=TerrainTilesList)
async def list_tiles():
    """
    List all terrain tiles in the database.
    
    Returns tile information including status, size, and download metadata.
    """
    query = """
        SELECT 
            id, tile_name, lat_min, lat_max, lon_min, lon_max,
            minio_bucket, minio_path, file_size_bytes,
            status, error_message, checksum_sha256, source_url,
            downloaded_at, created_at, updated_at
        FROM heimdall.terrain_tiles
        ORDER BY tile_name
    """
    
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        
        tiles = []
        status_counts = {"ready": 0, "downloading": 0, "failed": 0, "pending": 0}
        
        for row in rows:
            tile = TerrainTile(
                id=str(row['id']),
                tile_name=row['tile_name'],
                lat_min=row['lat_min'],
                lat_max=row['lat_max'],
                lon_min=row['lon_min'],
                lon_max=row['lon_max'],
                minio_bucket=row['minio_bucket'],
                minio_path=row['minio_path'],
                file_size_bytes=row['file_size_bytes'],
                status=row['status'],
                error_message=row['error_message'],
                checksum_sha256=row['checksum_sha256'],
                source_url=row['source_url'],
                downloaded_at=row['downloaded_at'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            tiles.append(tile)
            status_counts[row['status']] = status_counts.get(row['status'], 0) + 1
        
        return TerrainTilesList(
            tiles=tiles,
            total=len(tiles),
            ready=status_counts["ready"],
            downloading=status_counts["downloading"],
            failed=status_counts["failed"],
            pending=status_counts["pending"]
        )


@router.post("/download", response_model=DownloadResponse)
async def download_tiles(
    request: DownloadRequest,
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """
    Download SRTM tiles for specified bounds or auto-detect from WebSDR stations.
    
    If bounds not provided, automatically determines required tiles based on
    WebSDR station locations.
    """
    api_key = os.getenv("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENTOPOGRAPHY_API_KEY not configured")
    
    pool = get_pool()
    
    # Determine bounds
    if request.bounds:
        lat_min = request.bounds.lat_min
        lat_max = request.bounds.lat_max
        lon_min = request.bounds.lon_min
        lon_max = request.bounds.lon_max
    else:
        # Auto-detect from WebSDR stations
        query = """
            SELECT MIN(latitude) as lat_min, MAX(latitude) as lat_max,
                   MIN(longitude) as lon_min, MAX(longitude) as lon_max
            FROM heimdall.websdr_stations
            WHERE is_active = true
        """
        
        async with pool.acquire() as conn:
            bounds_row = await conn.fetchrow(query)
        
        if not bounds_row or bounds_row['lat_min'] is None:
            raise HTTPException(status_code=400, detail="No active WebSDR stations found")
        
        lat_min = bounds_row['lat_min']
        lat_max = bounds_row['lat_max']
        lon_min = bounds_row['lon_min']
        lon_max = bounds_row['lon_max']
    
    # Calculate required tiles (1°×1° tiles)
    tile_coords = []
    lat_start = int(math.floor(lat_min))
    lat_end = int(math.floor(lat_max))
    lon_start = int(math.floor(lon_min))
    lon_end = int(math.floor(lon_max))
    
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tile_coords.append((lat, lon))
    
    logger.info(
        f"Downloading {len(tile_coords)} SRTM tiles",
        bounds={"lat": (lat_min, lat_max), "lon": (lon_min, lon_max)},
        tiles=len(tile_coords)
    )
    
    # Initialize downloader
    downloader = SRTMDownloader(
        minio_client=minio_client,
        db_pool=pool,
        api_key=api_key
    )
    
    # Download tiles
    result = await downloader.download_tiles(tile_coords)
    
    # Convert to response format
    tiles_results = [
        TileDownloadResult(
            tile_name=tile["tile_name"],
            success=tile["success"],
            error=tile.get("error"),
            file_size=tile.get("file_size"),
            checksum=tile.get("checksum")
        )
        for tile in result["tiles"]
    ]
    
    return DownloadResponse(
        successful=result["successful"],
        failed=result["failed"],
        total=result["total"],
        tiles=tiles_results
    )


@router.get("/coverage", response_model=CoverageStatus)
async def get_coverage_status():
    """
    Get terrain coverage status for WebSDR region.
    
    Shows which tiles are needed, ready, downloading, failed, or missing.
    """
    pool = get_pool()
    
    # Get WebSDR bounds
    bounds_query = """
        SELECT MIN(latitude) as lat_min, MAX(latitude) as lat_max,
               MIN(longitude) as lon_min, MAX(longitude) as lon_max
        FROM heimdall.websdr_stations
        WHERE is_active = true
    """
    
    async with pool.acquire() as conn:
        bounds_row = await conn.fetchrow(bounds_query)
    
    if not bounds_row or bounds_row['lat_min'] is None:
        raise HTTPException(status_code=400, detail="No active WebSDR stations found")
    
    lat_min = float(bounds_row['lat_min'])
    lat_max = float(bounds_row['lat_max'])
    lon_min = float(bounds_row['lon_min'])
    lon_max = float(bounds_row['lon_max'])
    
    # Calculate needed tiles
    needed_tiles = []
    lat_start = int(math.floor(lat_min))
    lat_end = int(math.floor(lat_max))
    lon_start = int(math.floor(lon_min))
    lon_end = int(math.floor(lon_max))
    
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"
            tile_name = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"
            needed_tiles.append(tile_name)
    
    # Get tile status from DB
    tiles_query = """
        SELECT tile_name, status
        FROM heimdall.terrain_tiles
        WHERE tile_name = ANY($1::text[])
    """
    
    async with pool.acquire() as conn:
        tiles_rows = await conn.fetch(tiles_query, needed_tiles)
    
    tile_status_map = {row['tile_name']: row['status'] for row in tiles_rows}
    
    # Categorize tiles
    tiles_ready = []
    tiles_downloading = []
    tiles_failed = []
    tiles_missing = []
    
    for tile_name in needed_tiles:
        status = tile_status_map.get(tile_name)
        if status == "ready":
            tiles_ready.append(tile_name)
        elif status == "downloading":
            tiles_downloading.append(tile_name)
        elif status == "failed":
            tiles_failed.append(tile_name)
        else:
            tiles_missing.append(tile_name)
    
    total_tiles = len(needed_tiles)
    ready_count = len(tiles_ready)
    coverage_percent = (ready_count / total_tiles * 100) if total_tiles > 0 else 0.0
    
    region = CoverageRegion(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        tiles_needed=needed_tiles,
        tiles_ready=tiles_ready,
        tiles_downloading=tiles_downloading,
        tiles_failed=tiles_failed,
        tiles_missing=tiles_missing
    )
    
    return CoverageStatus(
        region=region,
        total_tiles=total_tiles,
        ready_count=ready_count,
        downloading_count=len(tiles_downloading),
        failed_count=len(tiles_failed),
        missing_count=len(tiles_missing),
        coverage_percent=coverage_percent
    )


@router.get("/elevation", response_model=ElevationResponse)
async def query_elevation(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in degrees"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude in degrees"),
    terrain_lookup: TerrainLookup = Depends(get_terrain_lookup)
):
    """
    Query elevation at specific coordinates.
    
    Returns elevation in meters ASL. Uses SRTM data if available,
    otherwise falls back to simplified terrain model.
    """
    
    # Get elevation
    elevation = terrain_lookup.get_elevation(lat, lon)
    
    # Determine tile name
    tile_lat = int(math.floor(lat))
    tile_lon = int(math.floor(lon))
    lat_prefix = "N" if tile_lat >= 0 else "S"
    lon_prefix = "E" if tile_lon >= 0 else "W"
    tile_name = f"{lat_prefix}{abs(tile_lat):02d}{lon_prefix}{abs(tile_lon):03d}"
    
    # Check if we used SRTM or simplified model
    source = "srtm" if terrain_lookup.use_srtm and tile_name in terrain_lookup._tile_cache else "simplified"
    
    return ElevationResponse(
        lat=lat,
        lon=lon,
        elevation_m=elevation,
        tile_name=tile_name,
        source=source
    )


@router.delete("/tiles/{tile_name}")
async def delete_tile(
    tile_name: str,
    minio_client: MinIOClient = Depends(get_minio_client)
):
    """
    Delete terrain tile from database and MinIO storage.
    """
    pool = get_pool()
    
    # Check if tile exists
    check_query = """
        SELECT id, minio_path
        FROM heimdall.terrain_tiles
        WHERE tile_name = $1
    """
    
    async with pool.acquire() as conn:
        tile_row = await conn.fetchrow(check_query, tile_name)
        
        if not tile_row:
            raise HTTPException(status_code=404, detail=f"Tile {tile_name} not found")
        
        # Delete from MinIO
        try:
            object_name = tile_row['minio_path']
            minio_client.s3_client.delete_object(
                Bucket="heimdall-terrain",
                Key=object_name
            )
            logger.info(f"Deleted tile {tile_name} from MinIO")
        except Exception as e:
            logger.error(f"Failed to delete tile {tile_name} from MinIO: {e}")
            # Continue with DB deletion even if MinIO deletion fails
        
        # Delete from database
        delete_query = """
            DELETE FROM heimdall.terrain_tiles
            WHERE tile_name = $1
        """
        await conn.execute(delete_query, tile_name)
        
        logger.info(f"Deleted tile {tile_name} from database")
        
        return {"success": True, "message": f"Tile {tile_name} deleted"}
