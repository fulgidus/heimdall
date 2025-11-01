"""
RF propagation simulator for synthetic data generation.

Implements physics-based propagation model:
- Free Space Path Loss (FSPL)
- Terrain blockage (LOS check)
- Environment loss (statistical)
- Multipath fading (Rayleigh)
"""

import math
import numpy as np
import structlog
from typing import Tuple

logger = structlog.get_logger(__name__)


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
        terrain_lookup=None
    ) -> float:
        """
        Calculate terrain blockage loss using line-of-sight check.
        
        Args:
            tx_lat: Transmitter latitude
            tx_lon: Transmitter longitude
            tx_alt: Transmitter altitude (meters ASL)
            rx_lat: Receiver latitude
            rx_lon: Receiver longitude
            rx_alt: Receiver altitude (meters ASL)
            terrain_lookup: Optional terrain elevation lookup function
        
        Returns:
            Terrain loss in dB (0 if LOS, 20-40 if blocked)
        """
        # Simplified model: check if terrain blocks first Fresnel zone
        # For Phase 5, we use a probabilistic model based on distance and altitude difference
        
        distance_km = self._haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        
        # Altitude difference
        alt_diff = abs(tx_alt - rx_alt)
        
        # Simplified LOS probability based on distance and altitude
        # Higher altitude difference and shorter distance = better LOS
        los_score = (alt_diff / 100.0) - (distance_km / 50.0)
        
        if los_score > 0:
            # Good LOS
            return 0.0
        elif los_score > -2:
            # Partial blockage
            return np.random.uniform(10.0, 25.0)
        else:
            # Heavy blockage
            return np.random.uniform(25.0, 40.0)
    
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
        terrain_lookup=None
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
        
        Returns:
            Tuple of (rx_power_dbm, snr_db, details_dict)
        """
        # Calculate distance
        distance_km = self._haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        
        # Calculate losses
        fspl = self.calculate_fspl(distance_km, frequency_mhz)
        terrain_loss = self.calculate_terrain_loss(
            tx_lat, tx_lon, tx_alt, rx_lat, rx_lon, rx_alt, terrain_lookup
        )
        env_loss = self.calculate_environment_loss()
        fading = self.calculate_fading()
        
        # Total received power
        rx_power_dbm = tx_power_dbm - fspl - terrain_loss - env_loss + fading
        
        # SNR
        snr_db = rx_power_dbm - self.noise_floor_dbm
        
        details = {
            "distance_km": distance_km,
            "fspl_db": fspl,
            "terrain_loss_db": terrain_loss,
            "env_loss_db": env_loss,
            "fading_db": fading,
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
    
    Args:
        receiver_positions: List of (lat, lon) tuples for receivers
        tx_position: (lat, lon) tuple for transmitter
    
    Returns:
        GDOP value (lower is better, <10 is good)
    """
    if len(receiver_positions) < 3:
        return 999.0  # Invalid
    
    # Simplified GDOP calculation
    # Real GDOP requires full Jacobian matrix, but we use angular spread as proxy
    
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
    
    # GDOP inversely proportional to minimum gap
    # Smaller gaps = worse geometry = higher GDOP
    min_gap = min(gaps)
    
    if min_gap < 0.1:  # Very small gap
        gdop = 50.0
    elif min_gap < 0.5:
        gdop = 20.0 / min_gap
    else:
        gdop = 10.0 / math.sqrt(len(receiver_positions))
    
    return gdop
