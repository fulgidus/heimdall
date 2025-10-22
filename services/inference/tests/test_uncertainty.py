"""Tests for T6.3: Uncertainty Ellipse Calculations."""
import pytest
import numpy as np
import math
from src.utils.uncertainty import (
    compute_uncertainty_ellipse,
    ellipse_to_geojson,
    create_uncertainty_circle,
)


class TestComputeUncertaintyEllipse:
    """Test uncertainty ellipse computation."""
    
    def test_circular_uncertainty(self):
        """Test with equal sigma_x and sigma_y (circular uncertainty)."""
        result = compute_uncertainty_ellipse(
            sigma_x=50.0,
            sigma_y=50.0,
            covariance_xy=0.0,
        )
        
        # Should be approximately circular
        assert result["semi_major_axis"] == pytest.approx(50.0, rel=0.01)
        assert result["semi_minor_axis"] == pytest.approx(50.0, rel=0.01)
        assert result["rotation_angle"] == pytest.approx(0.0, abs=1.0)
    
    def test_elliptical_uncertainty(self):
        """Test with different sigma_x and sigma_y."""
        result = compute_uncertainty_ellipse(
            sigma_x=100.0,
            sigma_y=50.0,
            covariance_xy=0.0,
        )
        
        # Major axis should be larger
        assert result["semi_major_axis"] > result["semi_minor_axis"]
        assert result["semi_major_axis"] == pytest.approx(100.0, rel=0.01)
        assert result["semi_minor_axis"] == pytest.approx(50.0, rel=0.01)
    
    def test_area_calculation(self):
        """Test that ellipse area is computed correctly."""
        sigma_x = 50.0
        sigma_y = 40.0
        
        result = compute_uncertainty_ellipse(sigma_x, sigma_y)
        
        expected_area = math.pi * sigma_x * sigma_y
        assert result["area_m2"] == pytest.approx(expected_area, rel=0.01)
    
    def test_confidence_levels(self):
        """Test confidence interval scaling."""
        result_1sigma = compute_uncertainty_ellipse(50.0, 50.0, confidence_level=0.68)
        result_2sigma = compute_uncertainty_ellipse(50.0, 50.0, confidence_level=0.95)
        
        # 2-sigma should be scaled larger
        assert result_2sigma["semi_major_scaled"] > result_1sigma["semi_major_scaled"]
        assert result_1sigma["semi_major_scaled"] == pytest.approx(50.0, rel=0.01)
    
    def test_rotation_angle_range(self):
        """Test that rotation angle is in valid range."""
        result = compute_uncertainty_ellipse(
            sigma_x=100.0,
            sigma_y=50.0,
            covariance_xy=30.0,
        )
        
        # Rotation angle should be in [-180, 180]
        assert -180 <= result["rotation_angle"] <= 180
    
    def test_correlated_uncertainty(self):
        """Test with non-zero covariance."""
        result = compute_uncertainty_ellipse(
            sigma_x=50.0,
            sigma_y=50.0,
            covariance_xy=20.0,  # Positive correlation
        )
        
        # Rotation angle should be non-zero
        assert abs(result["rotation_angle"]) > 0.1
    
    def test_zero_uncertainty(self):
        """Test with zero uncertainty (edge case)."""
        result = compute_uncertainty_ellipse(0.0, 0.0, 0.0)
        
        assert result["semi_major_axis"] == 0.0
        assert result["semi_minor_axis"] == 0.0
        assert result["area_m2"] == 0.0
    
    def test_small_uncertainty(self):
        """Test with very small uncertainty values."""
        result = compute_uncertainty_ellipse(1.0, 0.5, 0.0)
        
        assert result["semi_major_axis"] > result["semi_minor_axis"]
        assert result["semi_major_axis"] == pytest.approx(1.0, rel=0.01)


class TestEllipseToGeoJSON:
    """Test GeoJSON ellipse conversion."""
    
    def test_geojson_structure(self):
        """Test that GeoJSON has correct structure."""
        result = ellipse_to_geojson(
            center_lat=45.0,
            center_lon=7.0,
            semi_major_m=1000.0,
            semi_minor_m=800.0,
            rotation_deg=0.0,
        )
        
        assert result["type"] == "Feature"
        assert "geometry" in result
        assert "properties" in result
        assert result["geometry"]["type"] == "Polygon"
    
    def test_geojson_polygon_closed(self):
        """Test that polygon is closed (first point == last point)."""
        result = ellipse_to_geojson(45.0, 7.0, 1000.0, 800.0, 0.0)
        
        coordinates = result["geometry"]["coordinates"][0]
        assert coordinates[0] == coordinates[-1]  # Polygon must be closed
    
    def test_geojson_polygon_points(self):
        """Test that polygon has expected number of points."""
        result = ellipse_to_geojson(45.0, 7.0, 1000.0, 800.0, 0.0, num_points=32)
        
        coordinates = result["geometry"]["coordinates"][0]
        # Should have num_points + 1 (closing point)
        assert len(coordinates) == 33
    
    def test_geojson_properties(self):
        """Test that GeoJSON properties are correct."""
        semi_major = 1000.0
        semi_minor = 800.0
        rotation = 25.0
        
        result = ellipse_to_geojson(45.0, 7.0, semi_major, semi_minor, rotation)
        
        props = result["properties"]
        assert props["semi_major_m"] == pytest.approx(semi_major)
        assert props["semi_minor_m"] == pytest.approx(semi_minor)
        assert props["rotation_deg"] == pytest.approx(rotation)
    
    def test_geojson_coordinates_are_valid(self):
        """Test that all coordinates are valid lat/lon."""
        result = ellipse_to_geojson(
            center_lat=45.0,
            center_lon=7.0,
            semi_major_m=1000.0,
            semi_minor_m=800.0,
            rotation_deg=0.0,
        )
        
        coordinates = result["geometry"]["coordinates"][0]
        for lon, lat in coordinates:
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90
    
    def test_geojson_circle_approximation(self):
        """Test circle as special case of ellipse."""
        result = ellipse_to_geojson(
            center_lat=45.0,
            center_lon=7.0,
            semi_major_m=1000.0,
            semi_minor_m=1000.0,  # Equal to semi_major
            rotation_deg=0.0,
            num_points=64,
        )
        
        # Should have 65 points (64 + closing)
        assert len(result["geometry"]["coordinates"][0]) == 65
    
    def test_geojson_with_rotation(self):
        """Test ellipse with rotation angle."""
        result_0deg = ellipse_to_geojson(45.0, 7.0, 1000.0, 800.0, 0.0)
        result_45deg = ellipse_to_geojson(45.0, 7.0, 1000.0, 800.0, 45.0)
        
        # Different rotations should produce different coordinates
        coords_0 = result_0deg["geometry"]["coordinates"][0]
        coords_45 = result_45deg["geometry"]["coordinates"][0]
        
        # Points should be different (except possibly center-ish points)
        assert coords_0 != coords_45
    
    def test_geojson_at_equator(self):
        """Test GeoJSON generation at equator."""
        result = ellipse_to_geojson(0.0, 0.0, 1000.0, 1000.0, 0.0)
        
        # Should work without errors
        assert result["type"] == "Feature"
        assert len(result["geometry"]["coordinates"][0]) > 2
    
    def test_geojson_at_poles(self):
        """Test GeoJSON generation near poles."""
        result_north = ellipse_to_geojson(80.0, 0.0, 1000.0, 1000.0, 0.0)
        result_south = ellipse_to_geojson(-80.0, 0.0, 1000.0, 1000.0, 0.0)
        
        assert result_north["type"] == "Feature"
        assert result_south["type"] == "Feature"
    
    def test_geojson_international_dateline(self):
        """Test GeoJSON near international dateline."""
        result_east = ellipse_to_geojson(45.0, 179.0, 1000.0, 1000.0, 0.0)
        result_west = ellipse_to_geojson(45.0, -179.0, 1000.0, 1000.0, 0.0)
        
        assert result_east["type"] == "Feature"
        assert result_west["type"] == "Feature"


class TestCreateUncertaintyCircle:
    """Test circle creation helper."""
    
    def test_circle_creation(self):
        """Test that circle is created as ellipse with equal axes."""
        result = create_uncertainty_circle(45.0, 7.0, 500.0)
        
        # Check structure
        assert result["type"] == "Feature"
        assert result["geometry"]["type"] == "Polygon"
        
        # Properties should show equal axes
        assert result["properties"]["semi_major_m"] == pytest.approx(500.0)
        assert result["properties"]["semi_minor_m"] == pytest.approx(500.0)
    
    def test_circle_polygon_closed(self):
        """Test that circle polygon is closed."""
        result = create_uncertainty_circle(45.0, 7.0, 500.0)
        
        coordinates = result["geometry"]["coordinates"][0]
        assert coordinates[0] == coordinates[-1]
    
    def test_circle_with_custom_points(self):
        """Test circle with custom number of points."""
        result = create_uncertainty_circle(45.0, 7.0, 500.0, num_points=16)
        
        coordinates = result["geometry"]["coordinates"][0]
        # Should have num_points + 1 (closing point)
        assert len(coordinates) == 17


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_large_uncertainty(self):
        """Test with very large uncertainty values."""
        result = compute_uncertainty_ellipse(10000.0, 8000.0)
        
        assert result["semi_major_axis"] == pytest.approx(10000.0, rel=0.01)
        assert result["area_m2"] > 0
    
    def test_negative_covariance(self):
        """Test with negative covariance."""
        result = compute_uncertainty_ellipse(50.0, 50.0, covariance_xy=-20.0)
        
        # Should produce valid result
        assert result["semi_major_axis"] > 0
        assert result["semi_minor_axis"] > 0
    
    def test_ellipse_very_small_area(self):
        """Test ellipse with very small area."""
        result = ellipse_to_geojson(45.0, 7.0, 0.1, 0.05, 0.0, num_points=4)
        
        # Should still produce valid GeoJSON
        assert result["type"] == "Feature"
        assert len(result["geometry"]["coordinates"][0]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
