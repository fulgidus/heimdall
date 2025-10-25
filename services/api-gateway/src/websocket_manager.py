"""
WebSocket connection manager for broadcasting real-time updates to connected clients.
"""

from typing import Set, Dict, Any
from fastapi import WebSocket
from datetime import datetime
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages."""
    
    def __init__(self):
        # Active WebSocket connections
        self.active_connections: Set[WebSocket] = set()
        # Subscription tracking (event -> set of websockets)
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"✅ WebSocket client connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        # Remove from all subscriptions
        for subscribers in self.subscriptions.values():
            subscribers.discard(websocket)
        logger.info(f"❌ WebSocket client disconnected. Total: {len(self.active_connections)}")
        
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")
            self.disconnect(websocket)
            
    async def broadcast(self, event: str, data: Any):
        """Broadcast a message to all connected clients."""
        message = {
            "event": event,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all active connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
            
        logger.debug(f"📡 Broadcasted {event} to {len(self.active_connections)} clients")
        
    async def broadcast_to_subscribers(self, event: str, data: Any):
        """Broadcast a message only to clients subscribed to a specific event."""
        if event not in self.subscriptions:
            return
            
        message = {
            "event": event,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        subscribers = self.subscriptions[event].copy()
        disconnected = set()
        
        for connection in subscribers:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to subscriber: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
            
        logger.debug(f"📡 Sent {event} to {len(subscribers)} subscribers")
        
    def subscribe(self, websocket: WebSocket, event: str):
        """Subscribe a WebSocket to a specific event."""
        if event not in self.subscriptions:
            self.subscriptions[event] = set()
        self.subscriptions[event].add(websocket)
        logger.debug(f"📝 Client subscribed to {event}")
        
    def unsubscribe(self, websocket: WebSocket, event: str):
        """Unsubscribe a WebSocket from a specific event."""
        if event in self.subscriptions:
            self.subscriptions[event].discard(websocket)
        logger.debug(f"📝 Client unsubscribed from {event}")


# Global connection manager instance
manager = ConnectionManager()


async def heartbeat_task(websocket: WebSocket):
    """Send periodic heartbeat to keep connection alive."""
    try:
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            await websocket.send_json({
                "event": "pong",
                "data": {},
                "timestamp": datetime.utcnow().isoformat()
            })
    except Exception as e:
        logger.debug(f"Heartbeat task ended: {e}")
