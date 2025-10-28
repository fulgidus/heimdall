#!/usr/bin/env python3
"""
Script di test per validare la sequenza di connessione OpenWebRX.

Questo script:
1. Si connette a una WebSDR
2. Esegue la sequenza HTTP + WebSocket corretta
3. Cattura alcuni frame per analisi
4. Salva i frame raw per reverse engineering

Usage:
    python scripts/test_openwebrx_connection.py --url http://sdr1.ik1jns.it:8076
"""

import asyncio
import aiohttp
import websockets
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


class OpenWebRXTester:
    """Test della sequenza di connessione OpenWebRX."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.http_session = None
        self.ws_fft = None
        self.ws_audio = None
        self.output_dir = Path("data/openwebrx_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_full_sequence(self):
        """Esegue l'intera sequenza di test."""
        
        print("=" * 60)
        print("🧪 OpenWebRX Connection Sequence Test")
        print("=" * 60)
        print(f"Target: {self.base_url}")
        print()
        
        try:
            # Create HTTP session
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Step 1: Load main page
            await self.step1_load_main_page()
            
            # Step 2: Start receiver (if endpoint exists)
            await self.step2_start_receiver()
            
            # Step 3: Configure DSP (if endpoint exists)
            await self.step3_configure_dsp()
            
            # Step 4: Connect FFT WebSocket
            await self.step4_connect_fft()
            
            # Step 5: Connect Audio WebSocket
            await self.step5_connect_audio()
            
            # Step 6: Capture some frames
            await self.step6_capture_frames()
            
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
            
            # Try to extract config
            if 'center_freq' in html:
                print("   ✅ Found 'center_freq' in HTML - config present")
            else:
                print("   ⚠️  No 'center_freq' found - may need different parsing")
        
        print()
    
    async def step2_start_receiver(self):
        """Step 2: HTTP POST to start receiver."""
        
        print("📡 STEP 2: Starting receiver session...")
        
        url = f"{self.base_url}/api/receiver"
        print(f"   POST {url}")
        
        data = aiohttp.FormData()
        data.add_field('action', 'start')
        data.add_field('frequency', '145500000')
        data.add_field('mode', 'usb')
        
        try:
            async with self.http_session.post(url, data=data) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"   Response: {json.dumps(result, indent=2)}")
                    print("   ✅ Receiver started")
                else:
                    text = await response.text()
                    print(f"   Response: {text[:200]}")
                    print("   ⚠️  Endpoint may not exist (this is OK)")
        
        except aiohttp.ClientConnectorError as e:
            print(f"   ⚠️  Connection error: {e}")
            print("   ⚠️  Endpoint may not exist (this is OK)")
        
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            print("   ⚠️  Continuing anyway...")
        
        print()
    
    async def step3_configure_dsp(self):
        """Step 3: HTTP POST to configure DSP."""
        
        print("📡 STEP 3: Configuring DSP...")
        
        url = f"{self.base_url}/api/dspcontrol"
        print(f"   POST {url}")
        
        payload = {
            "type": "dspcontrol",
            "params": {
                "offset_freq": 0,
                "low_cut": 300,
                "high_cut": 2700,
                "mod": "usb",
                "squelch_level": -150,
            }
        }
        
        print(f"   Payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with self.http_session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 200:
                    result = await response.text()
                    print(f"   Response: {result[:200]}")
                    print("   ✅ DSP configured")
                else:
                    text = await response.text()
                    print(f"   Response: {text[:200]}")
                    print("   ⚠️  Endpoint may not exist (this is OK)")
        
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            print("   ⚠️  Continuing anyway...")
        
        print()
    
    async def step4_connect_fft(self):
        """Step 4: Connect to FFT WebSocket."""
        
        print("📡 STEP 4: Connecting to FFT WebSocket...")
        
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/fft'
        print(f"   WS URL: {ws_url}")
        
        try:
            # websockets.connect() doesn't take 'timeout' parameter in newer versions
            # Use open_timeout and close_timeout instead
            self.ws_fft = await websockets.connect(
                ws_url,
                open_timeout=10,
                close_timeout=10
            )
            print("   ✅ Connected!")
            
            # Try sending config
            config_msg = {
                "type": "config",
                "params": {
                    "fft_size": 4096,
                    "fft_fps": 20,
                }
            }
            
            print(f"   Sending config: {json.dumps(config_msg)}")
            await self.ws_fft.send(json.dumps(config_msg))
            print("   ✅ Config sent")
        
        except Exception as e:
            print(f"   ❌ Failed to connect: {e}")
            self.ws_fft = None
        
        print()
    
    async def step5_connect_audio(self):
        """Step 5: Connect to Audio WebSocket."""
        
        print("📡 STEP 5: Connecting to Audio WebSocket...")
        
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/audio'
        print(f"   WS URL: {ws_url}")
        
        try:
            self.ws_audio = await websockets.connect(
                ws_url,
                open_timeout=10,
                close_timeout=10
            )
            print("   ✅ Connected!")
            
            # Try sending config
            config_msg = {
                "type": "audioconfig",
                "params": {
                    "codec": "adpcm",
                    "sample_rate": 12000,
                }
            }
            
            print(f"   Sending config: {json.dumps(config_msg)}")
            await self.ws_audio.send(json.dumps(config_msg))
            print("   ✅ Config sent")
        
        except Exception as e:
            print(f"   ❌ Failed to connect: {e}")
            self.ws_audio = None
        
        print()
    
    async def step6_capture_frames(self):
        """Step 6: Capture some frames for analysis."""
        
        print("📡 STEP 6: Capturing frames for analysis...")
        print("   Duration: 5 seconds")
        print()
        
        fft_frames = []
        audio_frames = []
        
        start_time = asyncio.get_event_loop().time()
        duration = 5.0
        
        while asyncio.get_event_loop().time() - start_time < duration:
            
            # Receive FFT frames
            if self.ws_fft:
                try:
                    frame = await asyncio.wait_for(self.ws_fft.recv(), timeout=0.5)
                    fft_frames.append(frame)
                    
                    if len(fft_frames) <= 3:  # Log first few
                        if isinstance(frame, bytes):
                            print(f"   📦 FFT Frame #{len(fft_frames)}: {len(frame)} bytes (binary)")
                            print(f"      Hex (first 32): {frame[:32].hex()}")
                        else:
                            print(f"   📄 FFT Frame #{len(fft_frames)}: {frame[:100]}")
                
                except asyncio.TimeoutError:
                    pass
            
            # Receive Audio frames
            if self.ws_audio:
                try:
                    frame = await asyncio.wait_for(self.ws_audio.recv(), timeout=0.5)
                    audio_frames.append(frame)
                    
                    if len(audio_frames) <= 3:  # Log first few
                        if isinstance(frame, bytes):
                            print(f"   🔊 Audio Frame #{len(audio_frames)}: {len(frame)} bytes (binary)")
                            print(f"      Hex (first 32): {frame[:32].hex()}")
                        else:
                            print(f"   📄 Audio Frame #{len(audio_frames)}: {frame[:100]}")
                
                except asyncio.TimeoutError:
                    pass
        
        print()
        print(f"   ✅ Captured {len(fft_frames)} FFT frames")
        print(f"   ✅ Captured {len(audio_frames)} Audio frames")
        
        # Save frames for analysis
        if fft_frames:
            fft_file = self.output_dir / "fft_frames_sample.bin"
            with open(fft_file, 'wb') as f:
                for frame in fft_frames[:10]:  # Save first 10
                    if isinstance(frame, bytes):
                        f.write(frame)
                        f.write(b'\n---FRAME_SEPARATOR---\n')
            print(f"   💾 FFT frames saved to: {fft_file}")
        
        if audio_frames:
            audio_file = self.output_dir / "audio_frames_sample.bin"
            with open(audio_file, 'wb') as f:
                for frame in audio_frames[:10]:
                    if isinstance(frame, bytes):
                        f.write(frame)
                        f.write(b'\n---FRAME_SEPARATOR---\n')
            print(f"   💾 Audio frames saved to: {audio_file}")
        
        print()
    
    async def cleanup(self):
        """Clean up connections."""
        
        if self.ws_fft:
            await self.ws_fft.close()
        
        if self.ws_audio:
            await self.ws_audio.close()
        
        if self.http_session:
            await self.http_session.close()


async def main():
    parser = argparse.ArgumentParser(
        description='Test OpenWebRX connection sequence'
    )
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='WebSDR base URL (e.g., http://sdr1.ik1jns.it:8076)'
    )
    
    args = parser.parse_args()
    
    tester = OpenWebRXTester(args.url)
    await tester.run_full_sequence()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
