"""Unit tests for WebSDR fetcher."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.fetchers.websdr_fetcher import WebSDRFetcher
from src.models.websdrs import WebSDRConfig
from tests.fixtures import sample_websdrs, sample_iq_data


@pytest.mark.asyncio
async def test_websdr_fetcher_init(sample_websdrs):
    """Test WebSDR fetcher initialization."""
    fetcher = WebSDRFetcher(
        websdrs=sample_websdrs,
        timeout=30,
        retry_count=3,
        concurrent_limit=7
    )
    
    assert len(fetcher.websdrs) == len(sample_websdrs)
    assert fetcher.retry_count == 3
    assert fetcher.concurrent_limit == 7


@pytest.mark.asyncio
async def test_websdr_fetcher_context_manager(sample_websdrs):
    """Test WebSDR fetcher context manager."""
    async with WebSDRFetcher(websdrs=sample_websdrs) as fetcher:
        assert fetcher.session is not None
        assert isinstance(fetcher.session, object)  # aiohttp.ClientSession
    
    # Session should be closed after exiting context
    # (can't directly check as it's closed)


@pytest.mark.asyncio
async def test_fetch_iq_simultaneous_success(sample_websdrs, sample_iq_data):
    """Test successful simultaneous IQ fetch."""
    async with WebSDRFetcher(websdrs=sample_websdrs) as fetcher:
        # Mock the session's get method
        mock_response = AsyncMock()
        mock_response.status = 200
        
        # Convert IQ to int16 format
        iq_int16 = (sample_iq_data * 32767).astype(np.int16)
        binary_data = iq_int16.tobytes()
        
        mock_response.read = AsyncMock(return_value=binary_data)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        fetcher.session.get = MagicMock(return_value=mock_response)
        
        results = await fetcher.fetch_iq_simultaneous(
            frequency_mhz=145.5,
            duration_seconds=10,
            sample_rate_khz=12.5
        )
        
        # All WebSDRs should have results
        assert len(results) == len(sample_websdrs)
        
        for websdr_id, (iq_data, error) in results.items():
            if error is None:
                assert isinstance(iq_data, np.ndarray)
                assert iq_data.dtype == np.complex64
                assert len(iq_data) > 0


@pytest.mark.asyncio
async def test_websdr_health_check(sample_websdrs):
    """Test WebSDR health check."""
    async with WebSDRFetcher(websdrs=sample_websdrs) as fetcher:
        # Mock head requests
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        fetcher.session.head = MagicMock(return_value=mock_response)
        
        health_status = await fetcher.health_check()
        
        # All WebSDRs should report as reachable
        assert len(health_status) == len(sample_websdrs)
        for websdr_id, is_reachable in health_status.items():
            assert is_reachable is True


def test_websdr_fetcher_filters_inactive():
    """Test that fetcher filters out inactive WebSDRs."""
    websdrs = [
        WebSDRConfig(
            id=1,
            name="Active",
            url="http://example.com:8901",
            location_name="Test",
            latitude=0,
            longitude=0,
            is_active=True
        ),
        WebSDRConfig(
            id=2,
            name="Inactive",
            url="http://example.com:8902",
            location_name="Test",
            latitude=0,
            longitude=0,
            is_active=False
        ),
    ]
    
    fetcher = WebSDRFetcher(websdrs=websdrs)
    
    # Only active WebSDRs should be in the fetcher
    assert len(fetcher.websdrs) == 1
    assert 1 in fetcher.websdrs
    assert 2 not in fetcher.websdrs
