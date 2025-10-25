"""Uncertainty ellipse calculations for Phase 6 Inference Service."""
import numpy as np
from typing import Dict
import math
import logging

logger = logging.getLogger(__name__)


def compute_uncertainty_ellipse(
    sigma_x: float,
    sigma_y: float,
    covariance_xy: float = 0.0,
    confidence_level: float = 0.68,  # 1-sigma
) -> Dict:
    """
    Convert (sigma_x, sigma_y) to uncertainty ellipse parameters.
    
    Computes semi-major and semi-minor axes and rotation angle
    from the covariance matrix eigenvalues/eigenvectors.
    
    Args:
        sigma_x: Standard deviation in X direction (meters)
        sigma_y: Standard deviation in Y direction (meters)
        covariance_xy: Covariance between X and Y (default 0 = uncorrelated)
        confidence_level: Confidence interval
                         0.68 = 1-sigma (68%)
                         0.95 = 2-sigma (95%)
                         0.997 = 3-sigma (99.7%)
    
    Returns:
        Dict with ellipse parameters:
        {
            "semi_major_axis": float,    # Meters
            "semi_minor_axis": float,    # Meters
            "rotation_angle": float,     # Degrees (-180 to 180)
            "confidence_interval": float,
            "area_m2": float,            # Area of ellipse
            "semi_major_scaled": float,  # Scaled for confidence level
            "semi_minor_scaled": float,  # Scaled for confidence level
        }
    """
    try:
        # Create covariance matrix
        cov_matrix = np.array([
            [sigma_x**2, covariance_xy],
            [covariance_xy, sigma_y**2],
        ])
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Semi-major and semi-minor axes (1-sigma)
        semi_major = math.sqrt(max(eigenvalues[0], 0.0))  # Avoid negative due to numerical errors
        semi_minor = math.sqrt(max(eigenvalues[1], 0.0))
        
        # Rotation angle (eigenvector orientation)
        rotation_rad = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
        rotation_deg = math.degrees(rotation_rad)
        
        # Normalize rotation to [-180, 180]
        while rotation_deg > 180:
            rotation_deg -= 360
        while rotation_deg < -180:
            rotation_deg += 360
        
        # Scale for confidence level
        # For 1-sigma: scale = 1.0
        # For 2-sigma: scale = 2.0
        # For 95%: scale ≈ 2.45 (chi-square distribution for 2D)
        if confidence_level >= 0.95:
            scale_factor = 2.45  # 2-sigma ≈ 95% for Gaussian
        elif confidence_level >= 0.90:
            scale_factor = 2.15
        else:
            scale_factor = 1.0
        
        result = {
            "semi_major_axis": float(semi_major),
            "semi_minor_axis": float(semi_minor),
            "rotation_angle": float(rotation_deg),
            "confidence_interval": float(confidence_level),
            "area_m2": float(math.pi * semi_major * semi_minor),
            "semi_major_scaled": float(semi_major * scale_factor),
            "semi_minor_scaled": float(semi_minor * scale_factor),
        }
        
        logger.debug(f"Uncertainty ellipse computed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Failed to compute uncertainty ellipse: {e}", exc_info=True)
        raise


def ellipse_to_geojson(
    center_lat: float,
    center_lon: float,
    semi_major_m: float,
    semi_minor_m: float,
    rotation_deg: float,
    num_points: int = 64,
) -> Dict:
    """
    Convert uncertainty ellipse to GeoJSON Feature for Mapbox visualization.
    
    Generates a polygon approximating the ellipse for rendering on map.
    Handles the conversion from meters to geographic coordinates (lat/lon).
    
    Args:
        center_lat: Center latitude in decimal degrees (-90 to 90)
        center_lon: Center longitude in decimal degrees (-180 to 180)
        semi_major_m: Semi-major axis in meters
        semi_minor_m: Semi-minor axis in meters
        rotation_deg: Rotation angle in degrees
        num_points: Number of points to approximate ellipse (default 64 for smooth curve)
    
    Returns:
        GeoJSON Feature dict with Polygon geometry:
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon, lat], [lon, lat], ...]]
            },
            "properties": {
                "name": "Uncertainty Ellipse",
                "semi_major_m": float,
                "semi_minor_m": float,
                "rotation_deg": float,
            }
        }
    """
    try:
        # Earth radius in meters (WGS84)
        EARTH_RADIUS_M = 6371000.0
        
        # Convert meter-based ellipse to degrees
        # At equator: 1 degree ≈ 111km
        lat_scale = semi_major_m / EARTH_RADIUS_M
        lon_scale = semi_minor_m / (EARTH_RADIUS_M * math.cos(math.radians(center_lat)))
        
        # Generate ellipse points
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        rotation_rad = math.radians(rotation_deg)
        
        coordinates = []
        for angle in angles:
            # Ellipse parametric equations
            x = semi_major_m * math.cos(angle)
            y = semi_minor_m * math.sin(angle)
            
            # Apply rotation
            x_rot = x * math.cos(rotation_rad) - y * math.sin(rotation_rad)
            y_rot = x * math.sin(rotation_rad) + y * math.cos(rotation_rad)
            
            # Convert to lat/lon
            lat = center_lat + y_rot / EARTH_RADIUS_M
            lon = center_lon + x_rot / (EARTH_RADIUS_M * math.cos(math.radians(center_lat)))
            
            coordinates.append([lon, lat])
        
        # Close polygon (first point == last point)
        coordinates.append(coordinates[0])
        
        result = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates],
            },
            "properties": {
                "name": "Uncertainty Ellipse",
                "semi_major_m": float(semi_major_m),
                "semi_minor_m": float(semi_minor_m),
                "rotation_deg": float(rotation_deg),
                "center_lat": float(center_lat),
                "center_lon": float(center_lon),
            },
        }
        
        logger.debug(f"GeoJSON ellipse generated with {len(coordinates)} points")
        return result
    
    except Exception as e:
        logger.error(f"Failed to convert ellipse to GeoJSON: {e}", exc_info=True)
        raise


def create_uncertainty_circle(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    num_points: int = 32,
) -> Dict:
    """
    Create a circle (special case of ellipse) as GeoJSON.
    
    Useful for simpler visualization when sigma_x ≈ sigma_y.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        num_points: Number of points (default 32)
    
    Returns:
        GeoJSON Feature with circular polygon
    """
    return ellipse_to_geojson(
        center_lat, center_lon, radius_m, radius_m, 0.0, num_points
    )
