"""
WebSocket router for real-time dashboard updates
Provides live updates for:
- WebSDR status changes
- Signal detections
- Localization results
- System health metrics
"""

import json
import asyncio
import logging
from typing import Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

# Track active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Unregister and close WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)

# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates
    
    Accepts connections and maintains them for broadcasting updates.
    Client will receive JSON messages with event types:
    - websdrs_update: WebSDR status changes
    - signal_detection: New signal detection
    - localization_result: Localization solution
    - health_status: System health metrics
    - heartbeat: Keep-alive ping
    """
    await manager.connect(websocket)
    
    heartbeat_task = None
    try:
        # Start heartbeat to keep connection alive
        async def send_heartbeat():
            """Send periodic heartbeats to detect stale connections"""
            while True:
                try:
                    await asyncio.sleep(30)  # Heartbeat every 30 seconds
                    await manager.send_personal(websocket, {
                        "event": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": "Connection alive"
                    })
                except Exception as e:
                    logger.debug(f"Heartbeat failed: {e}")
                    break
        
        heartbeat_task = asyncio.create_task(send_heartbeat())
        
        # NOTE: No welcome message here - wait for client to send first message
        # This avoids timing issues where the browser isn't ready to receive messages yet
        # The client will request data via the 'get_data' event
        
        # Listen for client messages
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                event = message.get("event", "unknown")
                
                if event == "ping":
                    # Client ping - respond with pong
                    await manager.send_personal(websocket, {
                        "event": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif event == "subscribe":
                    # Client subscription to specific event types
                    channels = message.get("channels", [])
                    logger.info(f"Client subscribed to channels: {channels}")
                    await manager.send_personal(websocket, {
                        "event": "subscribed",
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif event == "get_data":
                    # Client requesting initial data
                    data_type = message.get("data_type", "")
                    
                    if data_type == "websdrs":
                        # Send all WebSDRs configuration
                        try:
                            from ..storage.db_manager import get_db_manager
                            db_manager = get_db_manager()
                            websdrs = db_manager.get_all_websdrs()
                            
                            # Convert to JSON-serializable format
                            websdrs_data = []
                            for websdr in websdrs:
                                created_at = getattr(websdr, 'created_at', None)
                                updated_at = getattr(websdr, 'updated_at', None)
                                
                                websdrs_data.append({
                                    "id": str(getattr(websdr, 'id', '')),
                                    "name": getattr(websdr, 'name', ''),
                                    "url": getattr(websdr, 'url', ''),
                                    "latitude": float(getattr(websdr, 'latitude', 0)),
                                    "longitude": float(getattr(websdr, 'longitude', 0)),
                                    "location_description": getattr(websdr, 'location_description', None),
                                    "country": getattr(websdr, 'country', None),
                                    "admin_email": getattr(websdr, 'admin_email', None),
                                    "altitude_asl": getattr(websdr, 'altitude_asl', None),
                                    "timeout_seconds": getattr(websdr, 'timeout_seconds', 30),
                                    "retry_count": getattr(websdr, 'retry_count', 3),
                                    "is_active": getattr(websdr, 'is_active', True),
                                    "created_at": created_at.isoformat() if created_at else None,
                                    "updated_at": updated_at.isoformat() if updated_at else None,
                                })
                            
                            await manager.send_personal(websocket, {
                                "event": "websdrs_data",
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": websdrs_data,
                                "count": len(websdrs_data)
                            })
                            logger.info(f"Sent {len(websdrs_data)} WebSDRs via WebSocket")
                        except Exception as e:
                            logger.error(f"Error fetching WebSDRs for WebSocket: {e}")
                            await manager.send_personal(websocket, {
                                "event": "error",
                                "message": f"Failed to fetch WebSDRs: {str(e)}",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    
                    else:
                        await manager.send_personal(websocket, {
                            "event": "error",
                            "message": f"Unknown data type: {data_type}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                else:
                    logger.debug(f"Received unknown event: {event}")
            
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client: {data}")
                await manager.send_personal(websocket, {
                    "event": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info("WebSocket disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await manager.disconnect(websocket)
    
    finally:
        # Cancel heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


@router.post("/ws/broadcast")
async def broadcast_update(event_type: str, data: dict):
    """
    Broadcast an update to all connected WebSocket clients
    
    Internal endpoint for other services to push updates to dashboard.
    
    Args:
        event_type: Type of event (e.g., 'websdrs_update', 'signal_detection')
        data: Event payload
    
    Returns:
        Number of clients that received the message
    """
    message = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    
    await manager.broadcast(message)
    
    return {
        "status": "broadcasted",
        "event": event_type,
        "clients_connected": len(manager.active_connections)
    }


def get_connection_count() -> int:
    """Get current number of active WebSocket connections"""
    return len(manager.active_connections)
