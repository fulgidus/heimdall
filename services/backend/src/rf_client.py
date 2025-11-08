"""
Client for RF acquisition service
"""

import logging
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)


class RFAcquisitionClient:
    """Client for interacting with RF acquisition service"""

    def __init__(self, base_url: str = "http://backend:8001"):
        self.base_url = base_url
        self.timeout = httpx.Timeout(300.0, connect=10.0)  # 5 min for acquisition

    async def trigger_acquisition(
        self,
        frequency_hz: int,
        duration_seconds: float,
        start_time: datetime | None = None,
    ) -> dict:
        """
        Trigger RF acquisition from WebSDR receivers

        Args:
            frequency_hz: Frequency in Hz
            duration_seconds: Duration in seconds
            start_time: Start time (defaults to now)

        Returns:
            dict with task_id and status
        """
        if start_time is None:
            start_time = datetime.now(UTC)

        frequency_mhz = frequency_hz / 1_000_000.0

        payload = {
            "frequency_mhz": frequency_mhz,
            "duration_seconds": duration_seconds,
            "start_time": start_time.isoformat() + "Z",
        }

        logger.info(f"Triggering RF acquisition: {frequency_mhz} MHz for {duration_seconds}s")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/acquisition/acquire",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                logger.info(f"RF acquisition triggered: task_id={result.get('task_id')}")
                return result

        except httpx.HTTPError as e:
            logger.error(f"Failed to trigger RF acquisition: {e}")
            raise

    async def get_task_status(self, task_id: str) -> dict:
        """Get status of an acquisition task"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/v1/acquisition/status/{task_id}")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get task status: {e}")
            raise
