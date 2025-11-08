#!/usr/bin/env python3
"""
Test OpenWebRX Multiplexed WebSocket Connection.

This script tests the REAL OpenWebRX protocol which uses a SINGLE multiplexed
WebSocket for FFT, Audio, and Control data (not separate /ws/fft and /ws/audio endpoints).

Usage:
    python test_openwebrx_multiplexed.py --url http://sdr1.ik1jns.it:8076
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

import aiohttp
import websockets


class OpenWebRXTester:
    """Test OpenWebRX multiplexed WebSocket connection."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path('data/openwebrx_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.http_session = None
        self.ws = None  # Single multiplexed WebSocket
        
    async def run(self):
        """Run all test steps."""
        
        print()
        print("=" * 60)
        print("🔍 OpenWebRX Multiplexed WebSocket Test")
        print("=" * 60)
        print(f"Target: {self.base_url}")
        print(f"Output: {self.output_dir}")
        print()
        
        try:
            # Create HTTP session
            self.http_session = aiohttp.ClientSession()
            
            # Step 1: Load main page
            await self.step1_load_main_page()
            
            # Step 2: Connect to multiplexed WebSocket
            await self.step2_connect_websocket()
            
            # Step 3: Send control commands
            await self.step3_send_commands()
            
            # Step 4: Capture frames
            await self.step4_capture_frames()
            
            print()
            print("=" * 60)
            print("✅ Test Completed Successfully!")
            print("=" * 60)
            print(f"📁 Output saved to: {self.output_dir}")
            
        except Exception as e:
            print()
            print("=" * 60)
            print(f"❌ Test Failed: {e}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            await self.cleanup()
    
    async def step1_load_main_page(self):
        """Step 1: HTTP GET main page."""
        
        print("📡 STEP 1: Loading main page...")
        print(f"   GET {self.base_url}/")
        
        async with self.http_session.get(self.base_url) as response:
            print(f"   Status: {response.status}")
            print(f"   Content-Type: {response.headers.get('Content-Type')}")
            
            html = await response.text()
            print(f"   HTML size: {len(html)} bytes")
            
            # Save HTML for analysis
            html_file = self.output_dir / "main_page.html"
            with open(html_file, 'w') as f:
                f.write(html)
            print(f"   ✅ Saved to: {html_file}")
        
        print()
    
    async def step2_connect_websocket(self):
        """Step 2: Connect to multiplexed WebSocket."""
        
        print("📡 STEP 2: Connecting to multiplexed WebSocket...")
        print("   NOTE: OpenWebRX uses a SINGLE WebSocket at /ws/ for all data")
        
        # Build WebSocket URL - note the trailing slash!
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/'
        print(f"   WS URL: {ws_url}")
        
        try:
            self.ws = await websockets.connect(
                ws_url,
                open_timeout=10,
                close_timeout=10
            )
            print("   ✅ Connected!")
            
            # Send handshake (REQUIRED!)
            handshake = "SERVER DE CLIENT client=openwebrx.js type=receiver"
            print(f"   Sending handshake: {handshake}")
            await self.ws.send(handshake)
            
            # Receive initial messages
            print("   Waiting for initial server messages...")
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=2.0)
                    if isinstance(msg, str):
                        print(f"   📩 Text: {msg[:150]}")
                    else:
                        print(f"   📦 Binary: {len(msg)} bytes")
                except asyncio.TimeoutError:
                    print(f"   ⏱️  Timeout on message {i+1}")
                    break
        
        except Exception as e:
            print(f"   ❌ Failed to connect: {e}")
            self.ws = None
        
        print()
    
    async def step3_send_commands(self):
        """Step 3: Send control commands to configure receiver."""
        
        print("📡 STEP 3: Sending control commands...")
        
        if not self.ws:
            print("   ⚠️  No WebSocket connection, skipping...")
            print()
            return
        
        # OpenWebRX control commands (reverse engineered from JavaScript)
        commands = [
            # Start demodulator (REQUIRED first!)
            {
                "type": "dspcontrol",
                "action": "start"
            },
            # Set connection properties
            {
                "type": "connectionproperties",
                "params": {
                    "output_rate": 12000,  # Audio sample rate
                    "hd_output_rate": 48000  # HD audio sample rate
                }
            },
            # Set frequency (145.500 MHz = 2m band)
            {
                "type": "dspcontrol",
                "params": {
                    "offset_freq": 0,  # Offset from center
                }
            },
        ]
        
        for cmd in commands:
            try:
                cmd_str = json.dumps(cmd)
                print(f"   Sending: {cmd_str}")
                await self.ws.send(cmd_str)
                print(f"   ✅ Sent")
                
                # Small delay between commands
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"   ❌ Failed to send: {e}")
        
        print()
    
    async def step4_capture_frames(self):
        """Step 4: Capture frames for analysis."""
        
        print("📡 STEP 4: Capturing frames...")
        print("   Duration: 10 seconds")
        print()
        
        if not self.ws:
            print("   ⚠️  No WebSocket connection, skipping...")
            print()
            return
        
        fft_frames = 0
        audio_frames = 0
        text_messages = 0
        small_frames = 0
        frame_types_seen = {}  # Track unique frame signatures
        
        start_time = asyncio.get_event_loop().time()
        duration = 10.0
        
        while asyncio.get_event_loop().time() - start_time < duration:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                
                if isinstance(msg, str):
                    # Text message (JSON control)
                    text_messages += 1
                    if text_messages <= 10 or text_messages % 10 == 0:
                        print(f"   Text message #{text_messages}: {msg[:150]}")
                    
                    # Save first text message
                    if text_messages == 1:
                        text_file = self.output_dir / 'text_message.json'
                        with open(text_file, 'w') as f:
                            f.write(msg)
                        print(f"   💾 Saved to: {text_file}")
                
                elif isinstance(msg, bytes):
                    # Binary message - analyze signature
                    
                    # Get first 4 bytes as signature
                    if len(msg) >= 4:
                        sig = msg[:4].hex()
                        frame_types_seen[sig] = frame_types_seen.get(sig, 0) + 1
                    else:
                        sig = "short"
                    
                    # Categorize by signature and size
                    if msg[:4] == b'\x02SYN':  # Audio codec header
                        audio_frames += 1
                        frame_type = "Audio (SYNC)"
                        
                        # Save first few audio frames
                        if audio_frames <= 3:
                            audio_file = self.output_dir / f'audio_frame_{audio_frames}.bin'
                            with open(audio_file, 'wb') as f:
                                f.write(msg)
                            print(f"   Audio frame #{audio_frames}: {len(msg)} bytes, sig={sig}")
                            print(f"   💾 Saved to: {audio_file}")
                        elif audio_frames % 50 == 0:
                            print(f"   Audio frame #{audio_frames}: {len(msg)} bytes")
                    
                    elif len(msg) > 5000:  # Likely FFT (large binary)
                        fft_frames += 1
                        frame_type = "FFT (large)"
                        
                        # Save first few FFT frames
                        if fft_frames <= 3:
                            fft_file = self.output_dir / f'fft_frame_{fft_frames}.bin'
                            with open(fft_file, 'wb') as f:
                                f.write(msg)
                            print(f"   FFT frame #{fft_frames}: {len(msg)} bytes, sig={sig}")
                            print(f"   💾 Saved to: {fft_file}")
                        elif fft_frames % 10 == 0:
                            print(f"   FFT frame #{fft_frames}: {len(msg)} bytes")
                    
                    else:
                        small_frames += 1
                        if small_frames <= 5:
                            small_file = self.output_dir / f'small_frame_{small_frames}.bin'
                            with open(small_file, 'wb') as f:
                                f.write(msg)
                            print(f"   Small binary #{small_frames}: {len(msg)} bytes, sig={sig}")
                            print(f"   💾 Saved to: {small_file}")
            
            except asyncio.TimeoutError:
                # No data for 1 second, continue
                pass
            except Exception as e:
                print(f"   ❌ Error receiving: {e}")
                break
        
        print()
        print(f"   📊 Captured:")
        print(f"      FFT frames: {fft_frames}")
        print(f"      Audio frames: {audio_frames}")
        print(f"      Text messages: {text_messages}")
        print(f"      Small binary frames: {small_frames}")
        print(f"   🔍 Frame signatures seen:")
        for sig, count in sorted(frame_types_seen.items(), key=lambda x: -x[1]):
            print(f"      {sig}: {count} frames")
        print()
    
    async def cleanup(self):
        """Cleanup resources."""
        
        if self.ws:
            await self.ws.close()
        
        if self.http_session:
            await self.http_session.close()


async def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Test OpenWebRX multiplexed WebSocket')
    parser.add_argument('--url', required=True, help='Base URL of OpenWebRX server')
    args = parser.parse_args()
    
    tester = OpenWebRXTester(args.url)
    await tester.run()


if __name__ == '__main__':
    asyncio.run(main())
