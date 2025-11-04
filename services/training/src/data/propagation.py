"""
RF propagation simulator for synthetic data generation.

Implements physics-based propagation model:
- Free Space Path Loss (FSPL)
- Terrain blockage (LOS check)
- Environment loss (statistical)
- Multipath fading (Rayleigh)
- Antenna patterns (omnidirectional, directional)
"""

import math
import numpy as np
import structlog
from typing import Tuple, Optional
from enum import Enum

logger = structlog.get_logger(__name__)


class AntennaType(Enum):
    """Antenna types for VHF/UHF amateur radio."""
    # RX antennas (WebSDR - typically fixed stations)
    OMNI_VERTICAL = "omni_vertical"      # 0-3 dBi, circular pattern (80% of RX)
    YAGI = "yagi"                        # 10-15 dBi, directional 30-60° (15% of RX)
    COLLINEAR = "collinear"              # 6-9 dBi, omnidirectional (5% of RX)
    
    # TX antennas (mobile/portable)
    WHIP = "whip"                        # 0-2 dBi, omnidirectional (90% of TX)
    RUBBER_DUCK = "rubber_duck"          # -3 to 0 dBi, inefficient handheld (8% of TX)
    PORTABLE_DIRECTIONAL = "portable_directional"  # 3-6 dBi, portable yagi (2% of TX)


class AntennaPattern:
    """
    Antenna radiation pattern for VHF/UHF frequencies.
    
    Provides gain based on antenna type and direction (azimuth/elevation).
    """
    
    def __init__(self, antenna_type: AntennaType, pointing_azimuth: float = 0.0):
        """
        Initialize antenna pattern.
        
        Args:
            antenna_type: Type of antenna (enum)
            pointing_azimuth: Direction antenna is pointing (degrees, 0=North, for directional antennas)
        """
        self.antenna_type = antenna_type
        self.pointing_azimuth = pointing_azimuth
        
        # Define antenna parameters based on type
        self._setup_antenna_parameters()
    
    def _setup_antenna_parameters(self):
        """Setup antenna parameters (gain, beamwidth) based on type."""
        if self.antenna_type == AntennaType.OMNI_VERTICAL:
            self.max_gain_dbi = np.random.uniform(0.0, 3.0)
            self.azimuth_beamwidth = 360.0  # Omnidirectional
            self.elevation_beamwidth = 60.0
            self.front_to_back_ratio = 0.0
            
        elif self.antenna_type == AntennaType.YAGI:
            self.max_gain_dbi = np.random.uniform(10.0, 15.0)
            self.azimuth_beamwidth = np.random.uniform(30.0, 60.0)
            self.elevation_beamwidth = np.random.uniform(40.0, 70.0)
            self.front_to_back_ratio = np.random.uniform(15.0, 25.0)  # dB
            
        elif self.antenna_type == AntennaType.COLLINEAR:
            self.max_gain_dbi = np.random.uniform(6.0, 9.0)
            self.azimuth_beamwidth = 360.0  # Omnidirectional
            self.elevation_beamwidth = 30.0  # Narrower than vertical
            self.front_to_back_ratio = 0.0
            
        elif self.antenna_type == AntennaType.WHIP:
            self.max_gain_dbi = np.random.uniform(0.0, 2.0)
            self.azimuth_beamwidth = 360.0
            self.elevation_beamwidth = 80.0
            self.front_to_back_ratio = 0.0
            
        elif self.antenna_type == AntennaType.RUBBER_DUCK:
            self.max_gain_dbi = np.random.uniform(-3.0, 0.0)
            self.azimuth_beamwidth = 360.0
            self.elevation_beamwidth = 90.0
            self.front_to_back_ratio = 0.0
            
        elif self.antenna_type == AntennaType.PORTABLE_DIRECTIONAL:
            self.max_gain_dbi = np.random.uniform(3.0, 6.0)
            self.azimuth_beamwidth = np.random.uniform(60.0, 90.0)
            self.elevation_beamwidth = np.random.uniform(60.0, 90.0)
            self.front_to_back_ratio = np.random.uniform(10.0, 15.0)
    
    def get_gain(self, azimuth_to_target: float, elevation_to_target: float = 0.0) -> float:
        """
        Calculate antenna gain in direction of target.
        
        Args:
            azimuth_to_target: Azimuth angle to target (degrees, 0=North, clockwise)
            elevation_to_target: Elevation angle to target (degrees, 0=horizon, positive=up)
        
        Returns:
            Antenna gain in dBi (can be negative for poor directions)
        """
        # Azimuth pattern
        azimuth_offset = abs(self._angle_difference(azimuth_to_target, self.pointing_azimuth))
        
        if self.azimuth_beamwidth >= 360.0:
            # Omnidirectional in azimuth
            azimuth_loss = 0.0
        else:
            # Directional pattern (approximate as cosine squared)
            # Half-power beamwidth corresponds to -3 dB point
            if azimuth_offset <= self.azimuth_beamwidth / 2:
                # Within main lobe
                azimuth_loss = -3.0 * (azimuth_offset / (self.azimuth_beamwidth / 2)) ** 2
            elif azimuth_offset >= 150.0:
                # Back lobe
                azimuth_loss = -self.front_to_back_ratio
            else:
                # Side lobes (transition region)
                # Linear interpolation between main lobe and back lobe
                transition_angle = (150.0 - self.azimuth_beamwidth / 2)
                progress = (azimuth_offset - self.azimuth_beamwidth / 2) / transition_angle
                azimuth_loss = -3.0 - progress * (self.front_to_back_ratio - 3.0)
        
        # Elevation pattern (simplified - assumes horizontal main lobe)
        elevation_offset = abs(elevation_to_target)
        if elevation_offset <= self.elevation_beamwidth / 2:
            elevation_loss = -3.0 * (elevation_offset / (self.elevation_beamwidth / 2)) ** 2
        else:
            # Beyond main lobe - steep drop-off
            elevation_loss = -3.0 - (elevation_offset - self.elevation_beamwidth / 2) * 0.5
            elevation_loss = max(elevation_loss, -20.0)  # Clamp to -20 dB
        
        # Total gain
        total_gain = self.max_gain_dbi + azimuth_loss + elevation_loss
        
        return total_gain
    
    @staticmethod
    def _angle_difference(angle1: float, angle2: float) -> float:
        """
        Calculate smallest angular difference between two angles.
        
        Args:
            angle1, angle2: Angles in degrees
        
        Returns:
            Difference in degrees (-180 to +180)
        """
        diff = angle1 - angle2
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        return diff


class RFPropagationModel:
    """RF propagation model for VHF/UHF frequencies."""
    
    def __init__(
        self,
        env_loss_min_db: float = 5.0,
        env_loss_max_db: float = 15.0,
        fading_scale_db: float = 5.57,
        noise_floor_dbm: float = -120.0
    ):
        """
        Initialize propagation model.
        
        Args:
            env_loss_min_db: Minimum environment loss (urban/suburban)
            env_loss_max_db: Maximum environment loss
            fading_scale_db: Rayleigh fading scale parameter
            noise_floor_dbm: Receiver noise floor
        """
        self.env_loss_min_db = env_loss_min_db
        self.env_loss_max_db = env_loss_max_db
        self.fading_scale_db = fading_scale_db
        self.noise_floor_dbm = noise_floor_dbm
    
    def calculate_fspl(self, distance_km: float, frequency_mhz: float) -> float:
        """
        Calculate Free Space Path Loss.
        
        Formula: FSPL = 20*log10(d_km) + 20*log10(f_mhz) + 32.44
        
        Args:
            distance_km: Distance in kilometers
            frequency_mhz: Frequency in MHz
        
        Returns:
            Path loss in dB
        """
        if distance_km <= 0:
            return 0.0
        
        fspl = 20 * math.log10(distance_km) + 20 * math.log10(frequency_mhz) + 32.44
        return fspl
    
    def calculate_terrain_loss(
        self,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float,
        rx_lat: float,
        rx_lon: float,
        rx_alt: float,
        terrain_lookup=None,
        frequency_mhz: float = 145.0
    ) -> float:
        """
        Calculate terrain blockage loss using line-of-sight check with Fresnel zone.
        
        Args:
            tx_lat: Transmitter latitude
            tx_lon: Transmitter longitude
            tx_alt: Transmitter altitude (meters ASL)
            rx_lat: Receiver latitude
            rx_lon: Receiver longitude
            rx_alt: Receiver altitude (meters ASL)
            terrain_lookup: Optional terrain elevation lookup (TerrainLookup instance)
            frequency_mhz: Frequency in MHz (for Fresnel zone calculation)
        
        Returns:
            Terrain loss in dB (0 if LOS, 10-40 if blocked)
        """
        distance_km = self._haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        
        # If no terrain lookup provided, use simplified model
        if terrain_lookup is None:
            alt_diff = abs(tx_alt - rx_alt)
            los_score = (alt_diff / 100.0) - (distance_km / 50.0)
            
            if los_score > 0:
                return 0.0
            elif los_score > -2:
                return np.random.uniform(10.0, 25.0)
            else:
                return np.random.uniform(25.0, 40.0)
        
        # Real terrain-based LOS check
        # Sample 20 points along the path TX→RX
        num_samples = 20
        earth_radius_km = 6371.0
        wavelength_m = 299.792458 / frequency_mhz  # c/f in meters
        
        max_intrusion_ratio = 0.0  # Track maximum Fresnel zone intrusion
        
        for i in range(1, num_samples):  # Skip endpoints
            fraction = i / num_samples
            
            # Interpolate position along great circle path
            sample_lat = tx_lat + fraction * (rx_lat - tx_lat)
            sample_lon = tx_lon + fraction * (rx_lon - tx_lon)
            
            # Get terrain elevation at sample point
            terrain_elevation = terrain_lookup.get_elevation(sample_lat, sample_lon)
            
            # Calculate LOS altitude at this point (with Earth curvature correction)
            # LOS altitude = linear interpolation + curvature correction
            linear_alt = tx_alt + fraction * (rx_alt - tx_alt)
            
            # Earth curvature correction
            # Height correction = d1 * d2 / (2 * R) where d1, d2 are distances from endpoints
            d1_km = distance_km * fraction
            d2_km = distance_km * (1 - fraction)
            curvature_correction = (d1_km * d2_km) / (2 * earth_radius_km)
            
            los_altitude = linear_alt + curvature_correction * 1000  # Convert to meters
            
            # Calculate first Fresnel zone radius at this point
            # Radius = sqrt(wavelength * d1 * d2 / distance_total)
            fresnel_radius = math.sqrt(wavelength_m * d1_km * d2_km / distance_km)
            
            # Check if terrain intrudes into Fresnel zone
            clearance = los_altitude - terrain_elevation
            if clearance < fresnel_radius:
                intrusion_ratio = 1.0 - (clearance / fresnel_radius)
                intrusion_ratio = max(0.0, min(1.0, intrusion_ratio))  # Clamp to [0, 1]
                max_intrusion_ratio = max(max_intrusion_ratio, intrusion_ratio)
        
        # Calculate loss based on maximum Fresnel zone intrusion
        if max_intrusion_ratio < 0.2:
            # Clear LOS (< 20% intrusion)
            return 0.0
        elif max_intrusion_ratio < 0.6:
            # Partial blockage (20-60% intrusion)
            # Linear interpolation: 0-15 dB loss
            return max_intrusion_ratio * 25.0
        else:
            # Heavy blockage (> 60% intrusion)
            # 25-40 dB loss
            return 25.0 + (max_intrusion_ratio - 0.6) * 37.5
    
    def calculate_environment_loss(self) -> float:
        """
        Calculate statistical environment loss (urban/suburban clutter).
        
        Returns:
            Environment loss in dB
        """
        return np.random.uniform(self.env_loss_min_db, self.env_loss_max_db)
    
    def calculate_fading(self) -> float:
        """
        Calculate multipath fading using Rayleigh distribution.
        
        Returns:
            Fading loss in dB
        """
        # Rayleigh fading in linear scale
        fading_linear = np.random.rayleigh(scale=self.fading_scale_db / (math.sqrt(2)))
        # Clamp to minimum positive value to avoid log10(0) or log10(negative)
        if fading_linear <= 0:
            fading_linear = 1e-12
        # Convert to dB (relative to mean)
        fading_db = 20 * math.log10(fading_linear / (self.fading_scale_db / math.sqrt(2)))
        return fading_db
    
    def calculate_received_power(
        self,
        tx_power_dbm: float,
        tx_lat: float,
        tx_lon: float,
        tx_alt: float,
        rx_lat: float,
        rx_lon: float,
        rx_alt: float,
        frequency_mhz: float,
        terrain_lookup=None,
        tx_antenna: Optional[AntennaPattern] = None,
        rx_antenna: Optional[AntennaPattern] = None
    ) -> Tuple[float, float, dict]:
        """
        Calculate received power at receiver.
        
        Args:
            tx_power_dbm: Transmitter power in dBm
            tx_lat: Transmitter latitude
            tx_lon: Transmitter longitude
            tx_alt: Transmitter altitude (meters ASL)
            rx_lat: Receiver latitude
            rx_lon: Receiver longitude
            rx_alt: Receiver altitude (meters ASL)
            frequency_mhz: Frequency in MHz
            terrain_lookup: Optional terrain elevation lookup
            tx_antenna: Optional transmitter antenna pattern
            rx_antenna: Optional receiver antenna pattern
        
        Returns:
            Tuple of (rx_power_dbm, snr_db, details_dict)
        """
        # Calculate distance
        distance_km = self._haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        
        # Calculate azimuth and elevation angles for antenna patterns
        tx_azimuth_to_rx, tx_elevation_to_rx = self._calculate_bearing_and_elevation(
            tx_lat, tx_lon, tx_alt, rx_lat, rx_lon, rx_alt, distance_km
        )
        rx_azimuth_to_tx, rx_elevation_to_tx = self._calculate_bearing_and_elevation(
            rx_lat, rx_lon, rx_alt, tx_lat, tx_lon, tx_alt, distance_km
        )
        
        # Calculate antenna gains
        tx_antenna_gain_db = 0.0
        rx_antenna_gain_db = 0.0
        if tx_antenna is not None:
            tx_antenna_gain_db = tx_antenna.get_gain(tx_azimuth_to_rx, tx_elevation_to_rx)
        if rx_antenna is not None:
            rx_antenna_gain_db = rx_antenna.get_gain(rx_azimuth_to_tx, rx_elevation_to_tx)
        
        # Calculate losses
        fspl = self.calculate_fspl(distance_km, frequency_mhz)
        terrain_loss = self.calculate_terrain_loss(
            tx_lat, tx_lon, tx_alt, rx_lat, rx_lon, rx_alt, terrain_lookup, frequency_mhz
        )
        env_loss = self.calculate_environment_loss()
        fading = self.calculate_fading()
        
        # Total received power (including antenna gains)
        rx_power_dbm = (
            tx_power_dbm 
            + tx_antenna_gain_db 
            + rx_antenna_gain_db
            - fspl 
            - terrain_loss 
            - env_loss 
            + fading
        )
        
        # SNR
        snr_db = rx_power_dbm - self.noise_floor_dbm
        
        details = {
            "distance_km": distance_km,
            "fspl_db": fspl,
            "terrain_loss_db": terrain_loss,
            "env_loss_db": env_loss,
            "fading_db": fading,
            "tx_antenna_gain_db": tx_antenna_gain_db,
            "rx_antenna_gain_db": rx_antenna_gain_db,
            "rx_power_dbm": rx_power_dbm,
            "snr_db": snr_db
        }
        
        return rx_power_dbm, snr_db, details
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point (degrees)
            lat2, lon2: Second point (degrees)
        
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def _calculate_bearing_and_elevation(
        self,
        from_lat: float,
        from_lon: float,
        from_alt: float,
        to_lat: float,
        to_lon: float,
        to_alt: float,
        distance_km: float
    ) -> Tuple[float, float]:
        """
        Calculate bearing (azimuth) and elevation angle from one point to another.
        
        Args:
            from_lat, from_lon: Starting point coordinates (degrees)
            from_alt: Starting altitude (meters ASL)
            to_lat, to_lon: Target point coordinates (degrees)
            to_alt: Target altitude (meters ASL)
            distance_km: Horizontal distance in kilometers (from haversine)
        
        Returns:
            Tuple of (azimuth_degrees, elevation_degrees)
            - azimuth: 0=North, 90=East, 180=South, 270=West
            - elevation: 0=horizon, positive=up, negative=down
        """
        # Calculate bearing (azimuth)
        from_lat_rad = math.radians(from_lat)
        to_lat_rad = math.radians(to_lat)
        delta_lon_rad = math.radians(to_lon - from_lon)
        
        x = math.sin(delta_lon_rad) * math.cos(to_lat_rad)
        y = math.cos(from_lat_rad) * math.sin(to_lat_rad) - math.sin(from_lat_rad) * math.cos(to_lat_rad) * math.cos(delta_lon_rad)
        
        azimuth_rad = math.atan2(x, y)
        azimuth_deg = math.degrees(azimuth_rad)
        
        # Normalize to 0-360
        if azimuth_deg < 0:
            azimuth_deg += 360.0
        
        # Calculate elevation angle
        altitude_diff_m = to_alt - from_alt
        distance_m = distance_km * 1000.0
        
        if distance_m > 0:
            elevation_rad = math.atan2(altitude_diff_m, distance_m)
            elevation_deg = math.degrees(elevation_rad)
        else:
            elevation_deg = 0.0
        
        return azimuth_deg, elevation_deg


def calculate_psd(snr_db: float, noise_floor_dbm: float = -120.0) -> float:
    """
    Calculate Power Spectral Density from SNR.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        noise_floor_dbm: Noise floor in dBm
    
    Returns:
        PSD in dBm/Hz
    """
    # PSD = signal_power / bandwidth
    # For WebSDR with 12.5 kHz bandwidth
    bandwidth_hz = 12500
    signal_power_dbm = noise_floor_dbm + snr_db
    psd_dbm_hz = signal_power_dbm - 10 * math.log10(bandwidth_hz)
    return psd_dbm_hz


def calculate_frequency_offset(distance_km: float, max_offset_hz: int = 50) -> int:
    """
    Calculate realistic frequency offset based on distance.
    
    Simulates Doppler effect and oscillator drift.
    
    Args:
        distance_km: Distance in kilometers
        max_offset_hz: Maximum offset
    
    Returns:
        Frequency offset in Hz
    """
    # Longer distance = more phase noise and Doppler uncertainty
    offset_range = min(max_offset_hz, int(distance_km * 0.5))
    return np.random.randint(-offset_range, offset_range + 1)


def calculate_gdop(receiver_positions: list, tx_position: Tuple[float, float]) -> float:
    """
    Calculate Geometric Dilution of Precision.
    
    Lower GDOP = better geometry for triangulation.
    
    Improved algorithm that considers:
    - Angular spread (minimum gap between receivers)
    - Number of receivers (redundancy)
    - Angular uniformity (variance of gaps)
    
    Args:
        receiver_positions: List of (lat, lon) tuples for receivers
        tx_position: (lat, lon) tuple for transmitter
    
    Returns:
        GDOP value (lower is better, <10 is excellent, <30 is good, <100 is acceptable)
    """
    if len(receiver_positions) < 3:
        return 999.0  # Invalid
    
    tx_lat, tx_lon = tx_position
    
    # Calculate angles from TX to each receiver
    angles = []
    for rx_lat, rx_lon in receiver_positions:
        angle = math.atan2(rx_lon - tx_lon, rx_lat - tx_lat)
        angles.append(angle)
    
    # Sort angles
    angles = sorted(angles)
    
    # Calculate angular gaps
    gaps = []
    for i in range(len(angles)):
        next_i = (i + 1) % len(angles)
        gap = (angles[next_i] - angles[i]) % (2 * math.pi)
        gaps.append(gap)
    
    # Metrics for geometry quality
    min_gap = min(gaps)
    max_gap = max(gaps)
    mean_gap = sum(gaps) / len(gaps)
    
    # Calculate gap variance (uniformity measure)
    gap_variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    gap_std = math.sqrt(gap_variance)
    
    # Base GDOP from minimum gap (primary factor)
    # More gradual penalty function
    if min_gap < 0.05:  # <3 degrees - very poor geometry
        base_gdop = 100.0
    elif min_gap < 0.15:  # <9 degrees - poor geometry
        base_gdop = 60.0 + (0.15 - min_gap) * 400.0  # 60-100 range
    elif min_gap < 0.5:  # <29 degrees - fair geometry
        base_gdop = 20.0 + (0.5 - min_gap) * 114.3  # 20-60 range
    else:  # >=29 degrees - good geometry
        base_gdop = 10.0 / math.sqrt(len(receiver_positions))
    
    # Penalty for non-uniform distribution (clustered receivers)
    # Ideal distribution has uniform gaps (low std deviation)
    uniformity_penalty = 1.0 + (gap_std / mean_gap) * 0.3
    
    # Bonus for more receivers (redundancy helps)
    receiver_bonus = 1.0 / math.sqrt(len(receiver_positions) / 4.0)
    
    gdop = base_gdop * uniformity_penalty * receiver_bonus
    
    # Cap at reasonable maximum
    return min(gdop, 999.0)
