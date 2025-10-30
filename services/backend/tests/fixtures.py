"""Test fixtures for backend service."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
import aiohttp

from src.models.websdrs import (
    WebSDRConfig,
    SignalMetrics,
    AcquisitionRequest,
)
from src.fetchers.websdr_fetcher import WebSDRFetcher


@pytest.fixture
def sample_websdrs() -> list[WebSDRConfig]:
    """Create sample WebSDR configurations."""
    return [
        WebSDRConfig(
            id=1,
            name="F5LEN Toulouse",
            url="http://websdr.f5len.net:8901",
            location_name="Toulouse, France",
            latitude=43.5,
            longitude=1.4,
            is_active=True,
            timeout_seconds=30,
            retry_count=3
        ),
        WebSDRConfig(
            id=2,
            name="PH0M Pachmarke",
            url="http://websdr.pa3weg.nl:8901",
            location_name="Pachmarke, Netherlands",
            latitude=52.5,
            longitude=4.8,
            is_active=True,
            timeout_seconds=30,
            retry_count=3
        ),
    ]


@pytest.fixture
def sample_iq_data() -> np.ndarray:
    """Create sample IQ data."""
    # Generate random complex data
    n_samples = 125000  # 10 seconds at 12.5 kHz
    iq_data = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    # Add some signal component (simulated narrowband signal)
    signal_freq = 1000  # 1 kHz offset
    t = np.arange(n_samples) / 12500
    signal = 0.1 * np.exp(2j * np.pi * signal_freq * t)
    return (iq_data + signal).astype(np.complex64)


@pytest.fixture
def sample_signal_metrics() -> SignalMetrics:
    """Create sample signal metrics."""
    return SignalMetrics(
        snr_db=15.5,
        psd_dbm=-80.2,
        frequency_offset_hz=50.0,
        signal_power_dbm=-50.0,
        noise_power_dbm=-65.5
    )


@pytest.fixture
def sample_acquisition_request() -> AcquisitionRequest:
    """Create sample acquisition request."""
    return AcquisitionRequest(
        frequency_mhz=145.5,
        duration_seconds=10,
        start_time=datetime.utcnow(),
        websdrs=None
    )


@pytest.fixture
async def mock_aiohttp_session():
    """Create mock aiohttp session for testing."""
    session = AsyncMock()
    
    # Mock successful response
    response = AsyncMock()
    response.status = 200
    
    # Generate mock IQ binary data
    n_samples = 125000
    iq_data = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)
    
    # Convert to int16 interleaved format (as WebSDR sends it)
    iq_int16 = (iq_data * 32767).astype(np.int16)
    binary_data = b''
    for i in range(0, len(iq_int16), 2):
        if i + 1 < len(iq_int16):
            binary_data += iq_int16[i].tobytes() + iq_int16[i + 1].tobytes()
    
    response.read = AsyncMock(return_value=binary_data)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    
    session.get = MagicMock(return_value=response)
    
    return session


@pytest.fixture
def mock_websdr_fetcher_success(sample_websdrs, sample_iq_data):
    """Create mock WebSDR fetcher with successful response."""
    fetcher = MagicMock(spec=WebSDRFetcher)
    
    # Mock successful IQ fetch
    async def mock_fetch(*args, **kwargs):
        return {
            ws.id: (sample_iq_data, None)
            for ws in sample_websdrs
        }
    
    fetcher.fetch_iq_simultaneous = mock_fetch
    fetcher.__aenter__ = AsyncMock(return_value=fetcher)
    fetcher.__aexit__ = AsyncMock(return_value=None)
    
    return fetcher


@pytest.fixture
def mock_websdr_fetcher_partial_failure(sample_websdrs, sample_iq_data):
    """Create mock WebSDR fetcher with partial failure."""
    fetcher = MagicMock(spec=WebSDRFetcher)
    
    # Mock partial IQ fetch (first succeeds, second fails)
    async def mock_fetch(*args, **kwargs):
        results = {}
        for i, ws in enumerate(sample_websdrs):
            if i == 0:
                results[ws.id] = (sample_iq_data, None)
            else:
                results[ws.id] = (None, "Connection timeout")
        return results
    
    fetcher.fetch_iq_simultaneous = mock_fetch
    fetcher.__aenter__ = AsyncMock(return_value=fetcher)
    fetcher.__aexit__ = AsyncMock(return_value=None)
    
    return fetcher
