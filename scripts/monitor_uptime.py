#!/usr/bin/env python
"""Monitor WebSDR uptime data being collected in the database."""

import logging
import time
from datetime import datetime, timedelta
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

HEALTH_ENDPOINT = "http://localhost:8000/api/v1/acquisition/websdrs/health"
DB_QUERY_ENDPOINT = "http://localhost:5432"  # Direct PostgreSQL

def monitor_uptime():
    """Monitor uptime collection and display progress."""
    logger.info("🚀 Starting WebSDR uptime monitoring (every 60s will add records)")
    logger.info("⏰ Waiting for Celery Beat to collect data...")
    
    for i in range(5):  # Monitor for ~5 minutes
        logger.info(f"\n{'='*60}")
        logger.info(f"Check {i+1}/5 - Time: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                logger.info(f"✅ Got health data for {len(data)} WebSDRs")
                
                # Show first 3 SDRs
                for i, (ws_id, health) in enumerate(list(data.items())[:3]):
                    logger.info(
                        f"  SDR #{ws_id} ({health['name']}): "
                        f"status={health['status']}, "
                        f"uptime={health.get('uptime', 'N/A')}%, "
                        f"avg_snr={health.get('avg_snr', 'N/A')}"
                    )
            else:
                logger.error(f"❌ HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        
        if i < 4:
            logger.info("⏳ Waiting 60 seconds... (Celery Beat runs every 60s)")
            time.sleep(60)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Monitoring complete!")
    logger.info("📊 After Celery Beat collects ~5 records per SDR,")
    logger.info("   the uptime% will be calculated from the database history")
    logger.info("="*60)

if __name__ == "__main__":
    monitor_uptime()
