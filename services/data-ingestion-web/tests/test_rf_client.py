"""
Tests for RF acquisition client
"""
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import httpx
from src.rf_client import RFAcquisitionClient


@pytest.mark.asyncio
class TestRFAcquisitionClient:
    """Test RF acquisition client"""
    
    async def test_trigger_acquisition_success(self):
        """Test successful acquisition trigger"""
        client = RFAcquisitionClient(base_url="http://test-rf:8001")
        
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.json.return_value = {
            "task_id": "test-task-123",
            "status": "pending",
        }
        mock_response.raise_for_status = lambda: None
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await client.trigger_acquisition(
                frequency_hz=145000000,
                duration_seconds=10.0,
            )
            
            assert result["task_id"] == "test-task-123"
            assert result["status"] == "pending"
            
            # Verify the correct endpoint was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://test-rf:8001/api/v1/acquisition/acquire"
            
            # Verify payload
            payload = call_args[1]["json"]
            assert payload["frequency_mhz"] == 145.0
            assert payload["duration_seconds"] == 10.0
    
    async def test_trigger_acquisition_with_start_time(self):
        """Test acquisition trigger with specific start time"""
        client = RFAcquisitionClient()
        
        start_time = datetime(2024, 1, 15, 12, 30, 0)
        
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.json.return_value = {"task_id": "test-task-123"}
        mock_response.raise_for_status = lambda: None
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            await client.trigger_acquisition(
                frequency_hz=145000000,
                duration_seconds=10.0,
                start_time=start_time,
            )
            
            payload = mock_client.post.call_args[1]["json"]
            assert "2024-01-15T12:30:00Z" in payload["start_time"]
    
    async def test_trigger_acquisition_http_error(self):
        """Test acquisition trigger with HTTP error"""
        client = RFAcquisitionClient()
        
        def raise_error():
            raise httpx.HTTPError("Connection failed")
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = raise_error
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            with pytest.raises(httpx.HTTPError):
                await client.trigger_acquisition(
                    frequency_hz=145000000,
                    duration_seconds=10.0,
                )
    
    async def test_get_task_status_success(self):
        """Test getting task status"""
        client = RFAcquisitionClient()
        
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.json.return_value = {
            "task_id": "test-task-123",
            "status": "completed",
            "result": {"measurements": 100},
        }
        mock_response.raise_for_status = lambda: None
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            result = await client.get_task_status("test-task-123")
            
            assert result["status"] == "completed"
            assert result["result"]["measurements"] == 100
    
    async def test_get_task_status_not_found(self):
        """Test getting status of non-existent task"""
        client = RFAcquisitionClient()
        
        def raise_error():
            raise httpx.HTTPStatusError(
                "404 Not Found",
                request=AsyncMock(),
                response=AsyncMock(),
            )
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = raise_error
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_task_status("non-existent-task")
