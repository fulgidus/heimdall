#!/usr/bin/env python3
"""
WebSocket Real-Time Updates Test

Tests the complete flow:
1. Connect to WebSocket at ws://localhost:8001/ws
2. Monitor for training events
3. Report what events are received
"""
import asyncio
import json
import sys
from datetime import datetime

import websockets


async def test_websocket():
    """Connect to WebSocket and monitor for events."""
    uri = "ws://localhost:8001/ws"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri, ping_interval=20) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected successfully!")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening for events... (Press Ctrl+C to stop)\n")
            
            event_count = 0
            async for message in websocket:
                event_count += 1
                try:
                    data = json.loads(message)
                    event_type = data.get('type', 'unknown')
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    print(f"\n[{timestamp}] 📡 Event #{event_count}: {event_type}")
                    print(f"   Raw: {json.dumps(data, indent=2)[:300]}...")
                    
                    # Parse based on event type
                    if event_type.startswith('training:'):
                        job_id = data.get('job_id', 'unknown')
                        status = data.get('status', 'unknown')
                        print(f"   Job ID: {job_id}")
                        print(f"   Status: {status}")
                        
                        if event_type == 'training:progress':
                            progress = data.get('progress', {})
                            print(f"   Epoch: {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 0)}")
                            print(f"   Progress: {progress.get('progress_percent', 0):.1f}%")
                    
                except json.JSONDecodeError as e:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] ⚠️  Non-JSON message: {message[:100]}")
                    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ Connection closed: {e}")
        return False
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket Real-Time Updates Test")
    print("=" * 60)
    
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Test interrupted by user")
        sys.exit(0)
