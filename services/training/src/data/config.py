"""
Training configuration and bounding box calculation.

Extracts WebSDR receiver locations from database and calculates:
- Receiver network bounding box
- Training area with margin
- SRTM tiles needed for terrain data
"""

import structlog
from dataclasses import dataclass
from typing import List, Tuple

logger = structlog.get_logger(__name__)


@dataclass
class ReceiverLocation:
    """WebSDR receiver location."""
    
    name: str
    latitude: float
    longitude: float
    altitude: float  # meters ASL


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    
    def center(self) -> Tuple[float, float]:
        """Return center point (lat, lon)."""
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2
        )
    
    def width_km(self) -> float:
        """Approximate width in km (at center latitude)."""
        import math
        center_lat, _ = self.center()
        lat_rad = math.radians(center_lat)
        km_per_deg = 111.32 * math.cos(lat_rad)  # km per degree longitude
        return (self.lon_max - self.lon_min) * km_per_deg
    
    def height_km(self) -> float:
        """Approximate height in km."""
        return (self.lat_max - self.lat_min) * 111.32  # km per degree latitude


@dataclass
class TrainingConfig:
    """Training area configuration."""
    
    receivers: List[ReceiverLocation]
    receiver_bbox: BoundingBox
    training_bbox: BoundingBox  # with margin
    srtm_tiles: List[Tuple[int, int]]  # (lat, lon) for SRTM tiles
    
    @classmethod
    def from_receivers(cls, receivers: List[ReceiverLocation], margin_degrees: float = 0.5):
        """
        Create training configuration from receiver locations.
        
        Args:
            receivers: List of receiver locations
            margin_degrees: Margin to add around receiver network (default: 0.5° ~50km)
        
        Returns:
            TrainingConfig with calculated bounding boxes and SRTM tiles
        """
        if not receivers:
            raise ValueError("At least one receiver required")
        
        # Calculate receiver network bounding box
        lats = [r.latitude for r in receivers]
        lons = [r.longitude for r in receivers]
        
        receiver_bbox = BoundingBox(
            lat_min=min(lats),
            lat_max=max(lats),
            lon_min=min(lons),
            lon_max=max(lons)
        )
        
        logger.info(
            "Receiver network bounding box",
            bbox={
                "lat_range": f"{receiver_bbox.lat_min:.3f}°N - {receiver_bbox.lat_max:.3f}°N",
                "lon_range": f"{receiver_bbox.lon_min:.3f}°E - {receiver_bbox.lon_max:.3f}°E",
                "width_km": f"{receiver_bbox.width_km():.1f}",
                "height_km": f"{receiver_bbox.height_km():.1f}"
            }
        )
        
        # Calculate training area with margin
        training_bbox = BoundingBox(
            lat_min=receiver_bbox.lat_min - margin_degrees,
            lat_max=receiver_bbox.lat_max + margin_degrees,
            lon_min=receiver_bbox.lon_min - margin_degrees,
            lon_max=receiver_bbox.lon_max + margin_degrees
        )
        
        logger.info(
            "Training area with margin",
            margin_degrees=margin_degrees,
            bbox={
                "lat_range": f"{training_bbox.lat_min:.3f}°N - {training_bbox.lat_max:.3f}°N",
                "lon_range": f"{training_bbox.lon_min:.3f}°E - {training_bbox.lon_max:.3f}°E",
                "width_km": f"{training_bbox.width_km():.1f}",
                "height_km": f"{training_bbox.height_km():.1f}"
            }
        )
        
        # Calculate required SRTM tiles (1° x 1° tiles)
        srtm_tiles = cls._calculate_srtm_tiles(training_bbox)
        
        logger.info(
            "SRTM tiles required",
            num_tiles=len(srtm_tiles),
            tiles=[f"N{lat:02d}E{lon:03d}" if lon >= 0 else f"N{lat:02d}W{abs(lon):03d}" 
                   for lat, lon in srtm_tiles]
        )
        
        return cls(
            receivers=receivers,
            receiver_bbox=receiver_bbox,
            training_bbox=training_bbox,
            srtm_tiles=srtm_tiles
        )
    
    @staticmethod
    def _calculate_srtm_tiles(bbox: BoundingBox) -> List[Tuple[int, int]]:
        """
        Calculate SRTM tile coordinates needed to cover bounding box.
        
        SRTM tiles are 1° x 1° and named by SW corner (e.g., N44E007 covers 44-45°N, 7-8°E).
        
        Args:
            bbox: Bounding box to cover
        
        Returns:
            List of (lat, lon) tuples for SRTM tile SW corners
        """
        import math
        
        # SRTM tiles are named by SW corner, so we need to floor the min values
        lat_start = math.floor(bbox.lat_min)
        lat_end = math.floor(bbox.lat_max)
        lon_start = math.floor(bbox.lon_min)
        lon_end = math.floor(bbox.lon_max)
        
        tiles = []
        for lat in range(lat_start, lat_end + 1):
            for lon in range(lon_start, lon_end + 1):
                tiles.append((lat, lon))
        
        return tiles


async def get_active_receivers_from_db(db_session) -> List[ReceiverLocation]:
    """
    Fetch active WebSDR receivers from database.
    
    Args:
        db_session: Database session
    
    Returns:
        List of active receiver locations
    """
    from sqlalchemy import text
    
    query = text("""
        SELECT name, latitude, longitude, altitude_asl
        FROM heimdall.websdr_stations
        WHERE is_active = TRUE
        ORDER BY name
    """)
    
    result = db_session.execute(query)
    receivers = []
    
    for row in result:
        receivers.append(ReceiverLocation(
            name=row[0],
            latitude=row[1],
            longitude=row[2],
            altitude=row[3] or 0  # default to 0 if NULL
        ))
    
    logger.info(f"Loaded {len(receivers)} active receivers from database")
    return receivers


def get_italian_receivers() -> List[ReceiverLocation]:
    """
    Get Italian WebSDR receivers (hardcoded for Phase 5).
    
    Returns:
        List of 7 Italian receiver locations
    """
    return [
        ReceiverLocation("Aquila di Giaveno", 45.03, 7.27, 600),
        ReceiverLocation("Montanaro", 45.234, 7.857, 300),
        ReceiverLocation("Torino", 45.044, 7.672, 250),
        ReceiverLocation("Coazze", 45.03, 7.27, 700),
        ReceiverLocation("Passo del Giovi", 44.561, 8.956, 480),
        ReceiverLocation("Genova", 44.395, 8.956, 100),
        ReceiverLocation("Milano - Baggio", 45.478, 9.123, 120),
    ]
