#!/usr/bin/env python3
"""Debug script to test OpenWebRX WebSocket protocol."""

import asyncio
import websockets
import json

async def test_openwebrx():
    """Connect to OpenWebRX and log all messages."""
    url = "ws://sdr1.ik1jns.it:8076/ws/"
    
    print(f"Connecting to {url}...")
    
    try:
        async with websockets.connect(url, max_size=10_000_000) as ws:
            print("✓ Connected!")
            
            # Collect initial messages
            messages = []
            for i in range(20):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    
                    if isinstance(msg, str):
                        print(f"\n[{i}] TEXT ({len(msg)} bytes):")
                        # Try to pretty-print JSON
                        if msg.strip().startswith('{'):
                            try:
                                data = json.loads(msg)
                                print(json.dumps(data, indent=2))
                            except:
                                print(msg[:500])
                        else:
                            print(msg[:500])
                    else:
                        print(f"\n[{i}] BINARY ({len(msg)} bytes)")
                        print(f"  First 32 bytes (hex): {msg[:32].hex()}")
                    
                    messages.append(msg)
                    
                except asyncio.TimeoutError:
                    print(f"\n[{i}] Timeout - no more messages")
                    break
            
            print(f"\n✓ Received {len(messages)} initial messages")
            
            # Now try sending commands
            print("\n--- Testing commands ---")
            
            # Set frequency to 145 MHz
            cmd1 = "SET mod=iq freq=145000000\n"
            print(f"\nSending: {cmd1.strip()}")
            await ws.send(cmd1)
            await asyncio.sleep(0.5)
            
            # Start stream
            cmd2 = "START\n"
            print(f"Sending: {cmd2.strip()}")
            await ws.send(cmd2)
            
            # Wait for data
            print("\nWaiting for data after START command...")
            for i in range(10):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    
                    if isinstance(msg, str):
                        print(f"\n[DATA {i}] TEXT ({len(msg)} bytes): {msg[:200]}")
                    else:
                        print(f"\n[DATA {i}] BINARY ({len(msg)} bytes)")
                        print(f"  First 32 bytes (hex): {msg[:32].hex()}")
                        print(f"  Last 32 bytes (hex): {msg[-32:].hex()}")
                        
                except asyncio.TimeoutError:
                    print(f"\n[DATA {i}] Timeout")
                    break
            
            # Stop
            print("\nSending: STOP")
            await ws.send("STOP\n")
            
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_openwebrx())
