"""
Tests for WebSocket functionality
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import app


def test_websocket_connection():
    """Test WebSocket connection establishment."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/updates") as websocket:
        # Connection should be established
        assert websocket is not None
        
        # Send a ping
        websocket.send_json({
            "event": "ping",
            "data": {},
            "timestamp": "2024-01-01T00:00:00"
        })
        
        # Receive pong
        data = websocket.receive_json()
        assert data["event"] == "pong"


def test_websocket_ping_pong():
    """Test WebSocket ping/pong heartbeat."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/updates") as websocket:
        # Send ping
        websocket.send_json({
            "event": "ping",
            "data": {}
        })
        
        # Expect pong response
        response = websocket.receive_json()
        assert response["event"] == "pong"
        assert "timestamp" in response


def test_websocket_disconnect():
    """Test WebSocket disconnection."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/updates") as websocket:
        # Connection established
        assert websocket is not None
        
    # Connection should be closed after context manager exit
    # (No exception should be raised)


def test_websocket_subscribe():
    """Test WebSocket subscription to events."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/updates") as websocket:
        # Subscribe to an event
        websocket.send_json({
            "event": "subscribe",
            "data": {
                "event_name": "services:health"
            }
        })
        
        # Should not raise an error
        # (Server handles subscription internally)


def test_websocket_unsubscribe():
    """Test WebSocket unsubscription from events."""
    client = TestClient(app)
    
    with client.websocket_connect("/ws/updates") as websocket:
        # Subscribe first
        websocket.send_json({
            "event": "subscribe",
            "data": {
                "event_name": "services:health"
            }
        })
        
        # Then unsubscribe
        websocket.send_json({
            "event": "unsubscribe",
            "data": {
                "event_name": "services:health"
            }
        })
        
        # Should not raise an error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
