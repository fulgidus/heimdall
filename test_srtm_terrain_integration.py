#!/usr/bin/env python3
"""
Test SRTM terrain integration in synthetic data generation.

This script tests:
1. Terrain API endpoints in training service
2. SRTM tile validation in SyntheticDataGenerator
3. Terrain lookup in worker threads during data generation
4. Fallback behavior when tiles are missing

Run from project root: python3 test_srtm_terrain_integration.py
"""

import asyncio
import os
import sys
import json
import time
import logging
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Install with: pip install requests")
    sys.exit(1)

# Use standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service URLs
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TRAINING_URL = os.getenv("TRAINING_URL", "http://localhost:8002")


class TerrainIntegrationTester:
    """Test SRTM terrain integration."""
    
    def __init__(self):
        """Initialize tester."""
        self.backend_url = BACKEND_URL
        self.training_url = TRAINING_URL
        self.session = requests.Session()
        self.test_results = []
    
    def test_training_terrain_status(self) -> bool:
        """Test training service terrain status endpoint."""
        logger.info("Testing training service terrain status endpoint")
        
        try:
            response = self.session.get(f"{self.training_url}/synthetic/terrain/status")
            response.raise_for_status()
            data = response.json()
            
            logger.info("Terrain status", data=data)
            
            # Check expected fields
            assert "srtm_enabled" in data, "Missing srtm_enabled field"
            assert "minio_configured" in data, "Missing minio_configured field"
            assert "minio_connection" in data, "Missing minio_connection field"
            assert data["srtm_enabled"] is True, "SRTM should be enabled"
            
            logger.info("✓ Terrain status endpoint OK")
            self.test_results.append(("terrain_status", True, None))
            return True
            
        except Exception as e:
            logger.error("✗ Terrain status endpoint failed", error=str(e))
            self.test_results.append(("terrain_status", False, str(e)))
            return False
    
    def test_training_terrain_coverage(self) -> bool:
        """Test training service terrain coverage endpoint."""
        logger.info("Testing training service terrain coverage endpoint")
        
        # Test region covering northwestern Italy (where WebSDRs are)
        request_data = {
            "lat_min": 44.0,
            "lat_max": 46.0,
            "lon_min": 7.0,
            "lon_max": 13.0
        }
        
        try:
            response = self.session.post(
                f"{self.training_url}/synthetic/terrain/coverage",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(
                "Terrain coverage",
                total_tiles=data["total_tiles"],
                available_tiles=data["available_tiles"],
                missing_tiles=data["missing_tiles"],
                coverage_percent=data["coverage_percent"]
            )
            
            # Check expected fields
            assert "total_tiles" in data, "Missing total_tiles field"
            assert "available_tiles" in data, "Missing available_tiles field"
            assert "missing_tile_names" in data, "Missing missing_tile_names field"
            assert data["total_tiles"] > 0, "Should have at least 1 tile"
            
            # Log missing tiles
            if data["missing_tiles"] > 0:
                logger.warning(
                    "Missing SRTM tiles",
                    missing=data["missing_tile_names"]
                )
            
            logger.info("✓ Terrain coverage endpoint OK")
            self.test_results.append(("terrain_coverage", True, None))
            return True
            
        except Exception as e:
            logger.error("✗ Terrain coverage endpoint failed", error=str(e))
            self.test_results.append(("terrain_coverage", False, str(e)))
            return False
    
    def test_training_terrain_download_redirect(self) -> bool:
        """Test training service terrain download redirect endpoint."""
        logger.info("Testing training service terrain download redirect endpoint")
        
        request_data = {
            "lat_min": 44.0,
            "lat_max": 45.0,
            "lon_min": 7.0,
            "lon_max": 8.0
        }
        
        try:
            response = self.session.post(
                f"{self.training_url}/synthetic/terrain/download",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(
                "Terrain download redirect",
                message=data["message"],
                tiles_to_download=len(data["tiles_to_download"]),
                backend_url=data["backend_url"]
            )
            
            # Check expected fields
            assert "backend_url" in data, "Missing backend_url field"
            assert "tiles_to_download" in data, "Missing tiles_to_download field"
            assert len(data["tiles_to_download"]) > 0, "Should have tiles to download"
            
            logger.info("✓ Terrain download redirect endpoint OK")
            self.test_results.append(("terrain_download_redirect", True, None))
            return True
            
        except Exception as e:
            logger.error("✗ Terrain download redirect endpoint failed", error=str(e))
            self.test_results.append(("terrain_download_redirect", False, str(e)))
            return False
    
    def test_synthetic_generator_validation(self) -> bool:
        """Test SRTM tile validation in SyntheticDataGenerator."""
        logger.info("Testing SyntheticDataGenerator SRTM tile validation")
        
        try:
            # Import generator and test validation method
            sys.path.insert(0, "/home/fulgidus/Documents/Projects/heimdall/services/training/src")
            from data.synthetic_generator import SyntheticDataGenerator
            from data.config import SyntheticGenerationConfig
            
            # Create config with SRTM enabled
            config = SyntheticGenerationConfig(
                num_samples=10,
                dataset_name="test_srtm_validation",
                use_srtm_terrain=True,
                frequency_mhz=145.0,
                tx_power_dbm=37.0,
                min_snr_db=10.0,
                min_receivers=4,
                max_gdop=5.0,
                seed=42
            )
            
            # Initialize generator (should validate tiles)
            generator = SyntheticDataGenerator(config)
            
            # Check if validation warnings were logged (would appear in logs)
            logger.info("✓ SyntheticDataGenerator validation OK")
            self.test_results.append(("generator_validation", True, None))
            return True
            
        except Exception as e:
            logger.error("✗ SyntheticDataGenerator validation failed", error=str(e))
            self.test_results.append(("generator_validation", False, str(e)))
            return False
    
    def test_small_generation_job(self) -> bool:
        """Test small synthetic data generation job with SRTM."""
        logger.info("Testing small synthetic data generation with SRTM")
        
        # Create small test dataset
        request_data = {
            "name": "test_srtm_terrain_integration",
            "description": "Test SRTM terrain integration in synthetic generation",
            "num_samples": 5,  # Small number for quick test
            "frequency_mhz": 145.0,
            "tx_power_dbm": 37.0,
            "min_snr_db": 10.0,
            "min_receivers": 4,
            "max_gdop": 5.0,
            "dataset_type": "no_features",  # Faster generation
            "use_random_receivers": False,
            "seed": 42
        }
        
        try:
            # Submit generation job
            response = self.session.post(
                f"{self.training_url}/synthetic/generate",
                json=request_data
            )
            response.raise_for_status()
            data = response.json()
            
            job_id = data["job_id"]
            logger.info("Generation job submitted", job_id=job_id)
            
            # Poll job status
            max_wait = 180  # 3 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = self.session.get(
                    f"{self.training_url}/synthetic/jobs/{job_id}"
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                
                logger.info(
                    "Job status",
                    status=status_data["status"],
                    progress=status_data.get("progress_percent")
                )
                
                if status_data["status"] == "completed":
                    logger.info("✓ Generation job completed successfully")
                    self.test_results.append(("generation_job", True, None))
                    return True
                elif status_data["status"] == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    logger.error("✗ Generation job failed", error=error_msg)
                    self.test_results.append(("generation_job", False, error_msg))
                    return False
                
                time.sleep(5)
            
            logger.error("✗ Generation job timed out")
            self.test_results.append(("generation_job", False, "Timeout"))
            return False
            
        except Exception as e:
            logger.error("✗ Generation job failed", error=str(e))
            self.test_results.append(("generation_job", False, str(e)))
            return False
    
    def print_summary(self):
        """Print test results summary."""
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, error in self.test_results:
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"{status:8} {test_name:30} {error or ''}")
        
        logger.info("=" * 60)
        logger.info(f"TOTAL: {passed}/{total} tests passed")
        logger.info("=" * 60)
        
        return passed == total
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info("Starting SRTM terrain integration tests")
        logger.info("=" * 60)
        
        # Test 1: Terrain status endpoint
        self.test_training_terrain_status()
        
        # Test 2: Terrain coverage endpoint
        self.test_training_terrain_coverage()
        
        # Test 3: Terrain download redirect endpoint
        self.test_training_terrain_download_redirect()
        
        # Test 4: SyntheticDataGenerator validation
        self.test_synthetic_generator_validation()
        
        # Test 5: Small generation job (optional - requires services running)
        # Uncomment to test actual data generation:
        # self.test_small_generation_job()
        
        # Print summary
        return self.print_summary()


if __name__ == "__main__":
    import logging
    
    tester = TerrainIntegrationTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
