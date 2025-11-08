"""Unit tests for services health monitor task."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tasks.services_health_monitor import check_service_health, monitor_services_health


@pytest.mark.asyncio
async def test_check_service_health_success():
    """Test successful health check."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"version": "1.0.0"})
        
        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.__aexit__.return_value = None
        mock_session_instance.get = AsyncMock()
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        mock_session_instance.get.return_value.__aexit__.return_value = None
        
        mock_session.return_value = mock_session_instance
        
        result = await check_service_health("backend", "http://backend:8001")
        
        assert result["service"] == "backend"
        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert result["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_check_service_health_unhealthy():
    """Test health check with unhealthy service."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 503
        
        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.__aexit__.return_value = None
        mock_session_instance.get = AsyncMock()
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        mock_session_instance.get.return_value.__aexit__.return_value = None
        
        mock_session.return_value = mock_session_instance
        
        result = await check_service_health("backend", "http://backend:8001")
        
        assert result["service"] == "backend"
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "503" in result["error"]


@pytest.mark.asyncio
async def test_check_service_health_timeout():
    """Test health check with timeout."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.__aexit__.return_value = None
        mock_session_instance.get = AsyncMock(side_effect=asyncio.TimeoutError())
        
        mock_session.return_value = mock_session_instance
        
        result = await check_service_health("backend", "http://backend:8001", timeout=1)
        
        assert result["service"] == "backend"
        assert result["status"] == "unhealthy"
        assert result["error"] == "Timeout"


@pytest.mark.asyncio
async def test_check_service_health_exception():
    """Test health check with exception."""
    with patch("aiohttp.ClientSession") as mock_session:
        mock_session_instance = AsyncMock()
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session_instance.__aexit__.return_value = None
        mock_session_instance.get = AsyncMock(side_effect=Exception("Connection refused"))
        
        mock_session.return_value = mock_session_instance
        
        result = await check_service_health("backend", "http://backend:8001")
        
        assert result["service"] == "backend"
        assert result["status"] == "unhealthy"
        assert "Connection refused" in result["error"]


def test_monitor_services_health_task():
    """Test monitor_services_health Celery task."""
    with patch("asyncio.get_event_loop") as mock_loop, \
         patch("src.tasks.services_health_monitor.check_service_health") as mock_check:
        
        # Mock event loop
        mock_loop_instance = MagicMock()
        mock_loop.return_value = mock_loop_instance
        
        # Mock health check results
        mock_health_results = {
            "backend": {"service": "backend", "status": "healthy"},
            "training": {"service": "training", "status": "healthy"},
            "inference": {"service": "inference", "status": "healthy"},
        }
        
        async def mock_check_all():
            return mock_health_results
        
        mock_loop_instance.run_until_complete.return_value = mock_health_results
        
        # Mock WebSocket manager
        with patch("src.tasks.services_health_monitor.ws_manager") as mock_ws_manager:
            mock_ws_manager.active_connections = [MagicMock()]
            
            # Mock Celery task self parameter
            mock_self = MagicMock()
            
            result = monitor_services_health(mock_self)
            
            assert result["status"] == "success"
            assert result["services_checked"] == 3


def test_monitor_services_health_no_websocket():
    """Test monitor_services_health when WebSocket is not available."""
    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop_instance = MagicMock()
        mock_loop.return_value = mock_loop_instance
        
        mock_health_results = {
            "backend": {"service": "backend", "status": "healthy"},
        }
        
        mock_loop_instance.run_until_complete.return_value = mock_health_results
        
        # WebSocket import fails
        with patch("src.tasks.services_health_monitor.ws_manager", side_effect=ImportError()):
            mock_self = MagicMock()
            
            # Should still succeed even if WebSocket broadcast fails
            result = monitor_services_health(mock_self)
            
            assert result["status"] == "success"
