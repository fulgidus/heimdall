#!/usr/bin/env python3
"""
Seed WebSDR stations from health-check endpoints.

This script fetches health-check JSON from each WebSDR receiver,
extracts GPS coordinates, frequencies, and capabilities, then
populates the database with accurate, up-to-date information.

Usage:
    python scripts/seed_websdrs.py
"""

import sys
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import os

# Set working directory to services/backend/src
src_dir = Path(__file__).parent.parent / "services" / "backend" / "src"
if src_dir.exists():
    os.chdir(src_dir)
    sys.path.insert(0, str(src_dir))
else:
    # Fallback for Docker container
    sys.path.insert(0, "/app/src")

from storage.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Italian WebSDR receivers from DEFAULT_WEBSDRS in acquisition.py
WEBSDRS = [
    {
        "name": "Aquila di Giaveno",
        "url": "http://sdr1.ik1jns.it:8076",
    },
    {
        "name": "Montanaro",
        "url": "http://cbfenis.ddns.net:43510",
    },
    {
        "name": "Torino",
        "url": "http://vst-aero.it:8073",
    },
    {
        "name": "Coazze",
        "url": "http://94.247.189.130:8076",
    },
    {
        "name": "Passo del Giovi",
        "url": "http://iz1mlt.ddns.net:8074",
    },
    {
        "name": "Genova",
        "url": "http://iq1zw.ddns.net:42154",
    },
    {
        "name": "Milano - Baggio",
        "url": "http://iu2mch.duckdns.org:8073",
    },
]


def fetch_health_check(base_url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch health-check JSON from WebSDR.
    
    Args:
        base_url: Base URL of the WebSDR (e.g., "http://sdr1.ik1jns.it:8076")
        timeout: Request timeout in seconds
    
    Returns:
        Parsed JSON dict if successful, None otherwise
    """
    # Ensure base_url ends without trailing slash
    base_url = base_url.rstrip('/')
    # OpenWebRX uses /status.json endpoint
    status_url = f"{base_url}/status.json"
    
    try:
        logger.info(f"Fetching health-check from {status_url}")
        response = requests.get(status_url, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Successfully fetched health-check from {base_url}")
        return data
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch health-check from {base_url}: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid JSON from {base_url}: {e}")
        return None


def seed_websdrs():
    """Seed database with WebSDR stations from health-check endpoints."""
    db_manager = DatabaseManager()
    
    # Check database connection
    if not db_manager.check_connection():
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)
    
    logger.info("Database connection successful")
    logger.info(f"Seeding {len(WEBSDRS)} WebSDR stations...")
    
    success_count = 0
    failed_count = 0
    
    for websdr_config in WEBSDRS:
        name = websdr_config["name"]
        base_url = websdr_config["url"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {name}")
        logger.info(f"URL: {base_url}")
        logger.info(f"{'='*60}")
        
        # Fetch health-check
        health_data = fetch_health_check(base_url)
        
        if health_data is None:
            logger.warning(f"Skipping {name} - health-check failed")
            failed_count += 1
            continue
        
        # Log receiver info
        receiver = health_data.get("receiver", {})
        gps = receiver.get("gps", {})
        logger.info(f"  Name: {receiver.get('name', 'N/A')}")
        logger.info(f"  Location: {receiver.get('location', 'N/A')}")
        logger.info(f"  GPS: {gps.get('lat', 'N/A')}, {gps.get('lon', 'N/A')}")
        logger.info(f"  Altitude: {receiver.get('asl', 'N/A')} m ASL")
        logger.info(f"  Admin: {receiver.get('admin', 'N/A')}")
        
        # Log SDR profiles
        sdrs = health_data.get("sdrs", [])
        logger.info(f"  SDRs: {len(sdrs)} receivers")
        for sdr in sdrs:
            profiles = sdr.get("profiles", [])
            logger.info(f"    - {sdr.get('name', 'N/A')} ({sdr.get('type', 'N/A')}): {len(profiles)} profiles")
        
        # Upsert to database
        station_id = db_manager.upsert_websdr_from_health_check(
            name=name,
            url=base_url,
            health_data=health_data
        )
        
        if station_id:
            logger.info(f"✅ Successfully upserted {name} (ID: {station_id})")
            success_count += 1
        else:
            logger.error(f"❌ Failed to upsert {name}")
            failed_count += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SEEDING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Success: {success_count}")
    logger.info(f"❌ Failed: {failed_count}")
    logger.info(f"Total: {success_count + failed_count}")
    
    # Verify database state
    all_stations = db_manager.get_all_websdrs()
    active_stations = db_manager.get_active_websdrs()
    
    logger.info(f"\nDatabase contains:")
    logger.info(f"  Total stations: {len(all_stations)}")
    logger.info(f"  Active stations: {len(active_stations)}")
    
    if active_stations:
        logger.info(f"\nActive WebSDR stations:")
        for station in active_stations:
            logger.info(f"  - {station.name} ({station.latitude}, {station.longitude})")
    
    logger.info("\n✅ Seeding complete!")


if __name__ == "__main__":
    seed_websdrs()
