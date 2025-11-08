"""
Training configuration and bounding box calculation.

Extracts WebSDR receiver locations from database and calculates:
- Receiver network bounding box
- Training area with margin
- SRTM tiles needed for terrain data
"""

import structlog
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

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
        ReceiverLocation("Coazze", 45.048, 7.238, 700),
        ReceiverLocation("Passo del Giovi", 44.561, 8.956, 480),
        ReceiverLocation("Genova", 44.395, 8.956, 100),
        ReceiverLocation("Milano - Baggio", 45.478, 9.123, 120),
    ]


def generate_random_receivers(
    bbox: BoundingBox,
    num_receivers: int,
    terrain_lookup=None,
    seed: Optional[int] = None
) -> List[ReceiverLocation]:
    """
    Generate random receiver locations within a bounding box.
    
    Args:
        bbox: Geographic bounding box for receiver placement
        num_receivers: Number of receivers to generate
        terrain_lookup: TerrainLookup instance for altitude lookup (optional)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        List of randomly generated receiver locations
    """
    import numpy as np

    if num_receivers < 1:
        raise ValueError("Must generate at least 1 receiver")

    rng = np.random.default_rng(seed)
    receivers = []

    for i in range(num_receivers):
        lat = rng.uniform(bbox.lat_min, bbox.lat_max)
        lon = rng.uniform(bbox.lon_min, bbox.lon_max)

        # Get altitude from terrain data if available, otherwise random
        if terrain_lookup:
            try:
                altitude = terrain_lookup.get_elevation(lat, lon)
            except Exception as e:
                logger.warning(f"Failed to get terrain elevation for receiver {i+1}, using fallback", error=str(e))
                altitude = rng.uniform(50, 800)
        else:
            altitude = rng.uniform(50, 800)

        receivers.append(ReceiverLocation(
            name=f"RX_{i+1:02d}",
            latitude=lat,
            longitude=lon,
            altitude=altitude
        ))

    logger.info(
        "Generated random receivers",
        num_receivers=num_receivers,
        bbox={
            "lat_range": f"{bbox.lat_min:.3f}°N - {bbox.lat_max:.3f}°N",
            "lon_range": f"{bbox.lon_min:.3f}°E - {bbox.lon_max:.3f}°E"
        },
        seed=seed
    )

    return receivers


@dataclass
class TxAntennaDistribution:
    """TX antenna type probability distribution (mobile/portable transmitters)."""
    whip: float = 0.90           # Mobile vehicle antennas (0-2 dBi)
    rubber_duck: float = 0.08     # Handheld radios (-3 to 0 dBi)
    portable_directional: float = 0.02  # Portable beam antennas (3-6 dBi)
    
    def validate(self):
        """Ensure probabilities sum to 1.0."""
        total = self.whip + self.rubber_duck + self.portable_directional
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"TX antenna probabilities must sum to 1.0, got {total:.3f}")
    
    def to_arrays(self) -> Tuple[List[str], List[float]]:
        """Return (antenna_types, probabilities) for np.random.choice()."""
        return (
            ["WHIP", "RUBBER_DUCK", "PORTABLE_DIRECTIONAL"],
            [self.whip, self.rubber_duck, self.portable_directional]
        )


@dataclass
class RxAntennaDistribution:
    """RX antenna type probability distribution (WebSDR fixed stations)."""
    omni_vertical: float = 0.70   # Standard WebSDR omni (0-3 dBi)
    yagi: float = 0.15            # Directional stations (10-15 dBi)
    log_periodic: float = 0.10    # Wideband directional (8-12 dBi)
    collinear: float = 0.05       # High-gain omni (6-9 dBi)
    
    def validate(self):
        """Ensure probabilities sum to 1.0."""
        total = self.omni_vertical + self.yagi + self.log_periodic + self.collinear
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"RX antenna probabilities must sum to 1.0, got {total:.3f}")
    
    def to_arrays(self) -> Tuple[List[str], List[float]]:
        """Return (antenna_types, probabilities) for np.random.choice()."""
        return (
            ["OMNI_VERTICAL", "YAGI", "LOG_PERIODIC", "COLLINEAR"],
            [self.omni_vertical, self.yagi, self.log_periodic, self.collinear]
        )


@dataclass
class MeteorologicalParameters:
    """Meteorological conditions affecting RF propagation."""
    
    # Temperature (Celsius)
    ground_temperature: float = 20.0  # Surface temperature
    temperature_gradient: float = -6.5  # °C per km (standard atmosphere)
    
    # Humidity (relative humidity %)
    relative_humidity: float = 50.0  # 0-100%
    
    # Atmospheric pressure (hPa)
    pressure_hpa: float = 1013.25  # Sea level standard
    
    # Time of day (0-24 hours, affects temperature/humidity)
    time_of_day: float = 12.0  # Noon by default
    
    # Season (0-3: spring, summer, autumn, winter)
    season: int = 1  # Summer by default
    
    # Tropospheric ducting probability (0.0-1.0)
    ducting_probability: float = 0.05  # 5% chance
    
    # Sporadic-E parameters
    solar_flux: float = 70.0  # Solar flux index (70-200, affects ionization)
    season_factor: float = 0.5  # 0=winter, 0.5=spring/fall, 1.0=summer
    
    @classmethod
    def random(cls, seed: Optional[int] = None):
        """
        Generate random meteorological parameters with realistic correlations.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            MeteorologicalParameters with randomized values
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        
        # Random time of day (0-24 hours)
        time_of_day = rng.uniform(0, 24)
        
        # Random season (0=spring, 1=summer, 2=autumn, 3=winter)
        season = rng.integers(0, 4)
        
        # Base temperature depends on season and time of day
        season_base_temps = [15.0, 25.0, 12.0, 5.0]  # Spring, Summer, Autumn, Winter
        base_temp = season_base_temps[season]
        
        # Diurnal temperature variation (peak at ~14:00, minimum at ~04:00)
        hour_offset = (time_of_day - 14.0) / 24.0 * 2 * np.pi
        diurnal_variation = -8.0 * np.cos(hour_offset)  # ±8°C swing
        
        ground_temperature = base_temp + diurnal_variation + rng.uniform(-3, 3)
        
        # Humidity inversely correlated with temperature
        # Higher in winter/night, lower in summer/day
        base_humidity = 70.0 - (ground_temperature - 10.0) * 1.5
        relative_humidity = np.clip(base_humidity + rng.uniform(-15, 15), 20, 95)
        
        # Pressure variation (typical range: 980-1040 hPa)
        pressure_hpa = 1013.25 + rng.uniform(-20, 20)
        
        # Ducting more likely in summer with high temperature gradients
        if season == 1:  # Summer
            ducting_probability = rng.uniform(0.05, 0.15)  # 5-15%
        else:
            ducting_probability = rng.uniform(0.01, 0.05)  # 1-5%
        
        # Solar flux (70-200 typical, higher = more ionization)
        solar_flux = rng.uniform(70.0, 150.0)
        
        # Season factor for sporadic-E (peak in summer)
        season_factors = [0.5, 1.0, 0.5, 0.2]  # Spring, Summer, Autumn, Winter
        season_factor = season_factors[season] * rng.uniform(0.8, 1.2)  # Add variation
        season_factor = np.clip(season_factor, 0.0, 1.0)
        
        return cls(
            ground_temperature=ground_temperature,
            temperature_gradient=-6.5 + rng.uniform(-2, 2),  # Vary around standard
            relative_humidity=relative_humidity,
            pressure_hpa=pressure_hpa,
            time_of_day=time_of_day,
            season=season,
            ducting_probability=ducting_probability,
            solar_flux=solar_flux,
            season_factor=season_factor
        )


@dataclass
class SyntheticGenerationConfig:
    """Configuration for synthetic training data generation."""
    
    # Terrain settings
    use_srtm_terrain: bool = False
    validate_tiles_before_generation: bool = True
    fallback_to_simplified_terrain: bool = True
    
    # Audio library settings (realistic voice/music samples)
    use_audio_library: bool = True  # Enable/disable audio library feature
    audio_library_fallback_to_formant: bool = True  # Fallback to formant synthesis if library empty
    audio_library_categories: List[str] = field(default_factory=lambda: [
        "voice", "music", "documentary", "conference", "custom"
    ])
    
    # Antenna pattern distributions (must sum to 1.0 each)
    tx_antenna_dist: TxAntennaDistribution = field(default_factory=TxAntennaDistribution)
    rx_antenna_dist: RxAntennaDistribution = field(default_factory=RxAntennaDistribution)
    
    # Frequency stability
    frequency_drift_ratio: float = 0.10  # 10% of samples have oscillator drift
    frequency_drift_ppm_range: Tuple[float, float] = (-2.0, 2.0)  # ±2 ppm drift
    gpsdo_ratio: float = 0.90  # 90% of transmitters use GPSDO (no drift)
    
    # Transmitter dynamics
    enable_tx_power_envelope: bool = True  # CW/FM/SSB power variations
    enable_keying_patterns: bool = True  # Morse code, packet bursts
    enable_doppler_shift: bool = False  # Mobile transmitters (disabled by default)
    
    # RF propagation
    enable_multipath: bool = True
    enable_atmospheric_effects: bool = True
    tropospheric_ducting_probability: float = 0.05  # 5% chance of ducting
    
    # Signal quality
    min_snr_db: float = 0.0  # Minimum SNR for signal detection (realistic weak signals)
    noise_floor_dbm: float = -115.0  # Visible noise floor for waterfall
    
    # Geometry constraints
    min_receivers_with_signal: int = 3  # Minimum for localization
    max_gdop: float = 10.0  # Maximum geometric dilution of precision
    
    # Dataset splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Check antenna distributions sum to 1.0
        self.tx_antenna_dist.validate()
        self.rx_antenna_dist.validate()
        
        # Check dataset splits sum to 1.0
        total_split = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_split <= 1.01):
            raise ValueError(f"Dataset splits must sum to 1.0, got {total_split}")
        
        # Validate ranges
        if not (0.0 <= self.frequency_drift_ratio <= 1.0):
            raise ValueError("frequency_drift_ratio must be between 0.0 and 1.0")
        
        if not (0.0 <= self.gpsdo_ratio <= 1.0):
            raise ValueError("gpsdo_ratio must be between 0.0 and 1.0")
        
        logger.info(
            "SyntheticGenerationConfig initialized",
            use_srtm_terrain=self.use_srtm_terrain,
            use_audio_library=self.use_audio_library,
            audio_library_fallback=self.audio_library_fallback_to_formant,
            tx_antenna_dist=self.tx_antenna_dist.to_arrays(),
            rx_antenna_dist=self.rx_antenna_dist.to_arrays(),
            frequency_drift_ratio=self.frequency_drift_ratio,
            min_receivers=self.min_receivers_with_signal
        )
