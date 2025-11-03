#!/usr/bin/env python3
"""
Test WebSocket message format from backend
Verifies that all broadcast messages include the required 'timestamp' field
"""
import asyncio
import websockets
import json
import sys
from datetime import datetime

async def test_websocket_format():
    uri = "ws://localhost:8000/ws"
    print(f"[{datetime.now().isoformat()}] Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[{datetime.now().isoformat()}] ✓ Connected successfully")
            print(f"[{datetime.now().isoformat()}] Listening for messages (20 seconds)...")
            
            message_count = 0
            valid_count = 0
            invalid_count = 0
            
            # Listen for 20 seconds
            try:
                async for message in asyncio.wait_for(websocket.__aiter__().__anext__(), timeout=20):
                    message_count += 1
                    timestamp = datetime.now().isoformat()
                    
                    try:
                        data = json.loads(message)
                        
                        # Check required fields
                        has_event = 'event' in data
                        has_data = 'data' in data
                        has_timestamp = 'timestamp' in data
                        
                        if has_event and has_data and has_timestamp:
                            valid_count += 1
                            print(f"[{timestamp}] ✓ Valid: event='{data['event']}', timestamp='{data['timestamp']}'")
                        else:
                            invalid_count += 1
                            missing = []
                            if not has_event: missing.append('event')
                            if not has_data: missing.append('data')
                            if not has_timestamp: missing.append('timestamp')
                            print(f"[{timestamp}] ✗ INVALID: Missing {missing}")
                            print(f"    Raw: {data}")
                    
                    except json.JSONDecodeError as e:
                        invalid_count += 1
                        print(f"[{timestamp}] ✗ INVALID JSON: {e}")
                        
            except asyncio.TimeoutError:
                print(f"\n[{datetime.now().isoformat()}] Timeout reached (20s)")
            
            print(f"\n{'='*60}")
            print(f"Test Summary:")
            print(f"  Total messages: {message_count}")
            print(f"  Valid messages: {valid_count}")
            print(f"  Invalid messages: {invalid_count}")
            if message_count > 0:
                print(f"  Success rate: {valid_count/message_count*100:.1f}%")
            else:
                print(f"  No messages received (this is OK if no activity)")
            print(f"{'='*60}")
            
            return invalid_count == 0
            
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] ✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_websocket_format())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(2)
