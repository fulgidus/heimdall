#!/usr/bin/env python3
"""Test script to understand OpenWebRX WebSocket protocol."""

import asyncio
import json

import websockets


async def test_openwebrx():
    """Connect to OpenWebRX and log all messages."""
    url = "ws://sdr1.ik1jns.it:8076/ws/"

    print(f"Connecting to {url}...")

    async with websockets.connect(url, max_size=10_000_000) as ws:
        print("Connected!")

        # Wait for initial messages
        for i in range(20):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)

                if isinstance(msg, str):
                    print(f"\n[{i}] TEXT MESSAGE ({len(msg)} chars):")
                    if len(msg) < 200:
                        print(msg)
                    else:
                        print(msg[:200] + "...")

                    # Try to parse as JSON
                    if msg.startswith("{"):
                        try:
                            data = json.loads(msg)
                            print(f"    JSON keys: {list(data.keys())}")
                        except:
                            pass

                elif isinstance(msg, bytes):
                    print(f"\n[{i}] BINARY MESSAGE ({len(msg)} bytes)")
                    print(f"    First 20 bytes: {msg[:20]}")

            except TimeoutError:
                print(f"\n[{i}] Timeout")
                break

        # Try sending commands
        print("\n\nSending commands...")

        commands = [
            "SET mod=iq freq=145000000",
            "START",
        ]

        for cmd in commands:
            print(f"Sending: {cmd}")
            await ws.send(cmd + "\n")
            await asyncio.sleep(0.5)

            # Check for responses
            try:
                for _ in range(3):
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(msg, str):
                        print(f"  Response (text): {msg[:100]}")
                    else:
                        print(f"  Response (binary): {len(msg)} bytes")
            except TimeoutError:
                print("  No response")


if __name__ == "__main__":
    asyncio.run(test_openwebrx())
