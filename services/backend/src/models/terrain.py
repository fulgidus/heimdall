"""
Pydantic models for terrain/SRTM data management.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class TerrainTile(BaseModel):
    """Single SRTM terrain tile information."""
    
    id: Optional[str] = None
    tile_name: str = Field(..., description="Tile name (e.g., 'N44E007')")
    lat_min: int = Field(..., description="Minimum latitude (SW corner)")
    lat_max: int = Field(..., description="Maximum latitude (NE corner)")
    lon_min: int = Field(..., description="Minimum longitude (SW corner)")
    lon_max: int = Field(..., description="Maximum longitude (NE corner)")
    
    minio_bucket: str = Field(default="heimdall-terrain")
    minio_path: str = Field(..., description="Path in MinIO bucket")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    
    status: str = Field(..., description="Status: pending/downloading/ready/failed")
    error_message: Optional[str] = None
    
    checksum_sha256: Optional[str] = None
    source_url: Optional[str] = None
    
    downloaded_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TerrainTilesList(BaseModel):
    """List of terrain tiles."""
    
    tiles: List[TerrainTile]
    total: int
    ready: int
    downloading: int
    failed: int
    pending: int


class BoundsRequest(BaseModel):
    """Geographic bounds for tile download."""
    
    lat_min: float = Field(..., ge=-90, le=90)
    lat_max: float = Field(..., ge=-90, le=90)
    lon_min: float = Field(..., ge=-180, le=180)
    lon_max: float = Field(..., ge=-180, le=180)


class DownloadRequest(BaseModel):
    """Request to download terrain tiles."""
    
    bounds: Optional[BoundsRequest] = Field(
        None,
        description="Optional geographic bounds. If not provided, auto-detect from WebSDR stations"
    )


class TileDownloadResult(BaseModel):
    """Result of a single tile download."""
    
    tile_name: str
    success: bool
    error: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None


class DownloadResponse(BaseModel):
    """Response from tile download operation."""
    
    successful: int
    failed: int
    total: int
    tiles: List[TileDownloadResult]


class CoverageRegion(BaseModel):
    """Coverage region information."""
    
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    tiles_needed: List[str]
    tiles_ready: List[str]
    tiles_downloading: List[str]
    tiles_failed: List[str]
    tiles_missing: List[str]


class CoverageStatus(BaseModel):
    """Terrain coverage status."""
    
    region: CoverageRegion
    total_tiles: int
    ready_count: int
    downloading_count: int
    failed_count: int
    missing_count: int
    coverage_percent: float = Field(..., ge=0, le=100)


class ElevationQuery(BaseModel):
    """Query for elevation at specific coordinates."""
    
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class ElevationResponse(BaseModel):
    """Elevation query response."""
    
    lat: float
    lon: float
    elevation_m: float
    tile_name: str
    source: str = Field(..., description="'srtm' or 'simplified'")
