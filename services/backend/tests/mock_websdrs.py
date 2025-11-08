"""
Mock WebSDR Receiver for Testing

This module provides mock WebSDR receivers to eliminate network dependencies
and flakiness in tests.
"""

from datetime import datetime
from typing import Any

import numpy as np


class MockWebSDRReceiver:
    """Mock WebSDR receiver for testing."""

    def __init__(self, receiver_id: str, frequency_mhz: float = 145.0):
        self.receiver_id = receiver_id
        self.frequency_mhz = frequency_mhz
        self.is_connected = True

    async def fetch_iq_data(
        self, frequency_mhz: float, duration_seconds: float, sample_rate: int = 192000
    ) -> dict[str, Any]:
        """Generate synthetic IQ data."""
        if not self.is_connected:
            raise ConnectionError(f"Receiver {self.receiver_id} not connected")

        num_samples = int(sample_rate * duration_seconds)

        # Generate synthetic IQ data with signal
        t = np.arange(num_samples) / sample_rate
        # Carrier + modulation + noise
        signal = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(num_samples)

        iq_data = signal.astype(np.complex64)

        return {
            "iq_data": iq_data,
            "sample_rate": sample_rate,
            "timestamp": (
                datetime.now(datetime.UTC).isoformat()
                if hasattr(datetime, "UTC")
                else datetime.utcnow().isoformat()
            ),
            "receiver_id": self.receiver_id,
            "frequency_mhz": frequency_mhz,
            "duration_seconds": duration_seconds,
            "signal_strength": -50 + 10 * np.random.randn(),  # dBm
        }

    def disconnect(self):
        """Simulate disconnect."""
        self.is_connected = False

    def reconnect(self):
        """Simulate reconnect."""
        self.is_connected = True


def create_mock_websdrs(count: int = 7) -> list[MockWebSDRReceiver]:
    """Create multiple mock WebSDR receivers."""
    return [MockWebSDRReceiver(f"websdr-{i}", 145.0 + i * 0.5) for i in range(count)]


async def mock_concurrent_fetch(receivers, frequency_mhz, duration):
    """Mock concurrent fetch from multiple receivers."""
    tasks = [r.fetch_iq_data(frequency_mhz, duration) for r in receivers]
    import asyncio

    return await asyncio.gather(*tasks, return_exceptions=True)


class MockWebSDRFetcher:
    """Mock WebSDR fetcher that mimics the real fetcher interface."""

    def __init__(self, receivers: list[MockWebSDRReceiver]):
        self.receivers = receivers
        self.fetch_count = 0

    async def fetch_iq_simultaneous(
        self, frequency_mhz: float, duration_seconds: float, sample_rate: int = 192000
    ) -> dict[str, tuple[np.ndarray | None, str | None]]:
        """
        Fetch IQ data from all receivers simultaneously.

        Returns:
            Dict mapping receiver_id to (iq_data, error) tuple
        """
        self.fetch_count += 1
        results = {}

        for receiver in self.receivers:
            try:
                data = await receiver.fetch_iq_data(frequency_mhz, duration_seconds, sample_rate)
                results[receiver.receiver_id] = (data["iq_data"], None)
            except Exception as e:
                results[receiver.receiver_id] = (None, str(e))

        return results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_mock_fetcher(
    receiver_count: int = 7, failure_indices: list[int] | None = None
) -> MockWebSDRFetcher:
    """
    Create a mock WebSDR fetcher with optional failures.

    Args:
        receiver_count: Number of receivers to create
        failure_indices: List of receiver indices that should fail

    Returns:
        MockWebSDRFetcher instance
    """
    receivers = create_mock_websdrs(receiver_count)

    if failure_indices:
        for idx in failure_indices:
            if 0 <= idx < len(receivers):
                receivers[idx].disconnect()

    return MockWebSDRFetcher(receivers)
