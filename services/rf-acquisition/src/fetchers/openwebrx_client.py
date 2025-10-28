"""
OpenWebRX WebSocket client for FFT and Audio stream acquisition.

This module implements the REAL OpenWebRX protocol discovered via reverse engineering:
- Single multiplexed WebSocket at ws://host:port/ws/
- Handshake: "SERVER DE CLIENT client=openwebrx.js type=receiver"
- Binary frames with type byte prefix (0x01=FFT, 0x02=Audio)
- JSON text messages for control and telemetry

See docs/WEBSDR_INTEGRATION_GUIDE.md for full protocol specification.
"""

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlparse

import websockets
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FFTFrame:
    """FFT/Waterfall spectrum frame."""
    timestamp: datetime
    websdr_id: str
    center_freq: int  # Hz
    bandwidth: int  # Hz
    bins: bytes  # uint8 array, len=2048+
    
    def to_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert raw bins to (frequencies, power_dbm) arrays.
        
        Returns:
            (frequencies_hz, power_dbm): Both as numpy float32 arrays
        """
        bins_array = np.frombuffer(self.bins, dtype=np.uint8)
        num_bins = len(bins_array)
        
        # Map bins to frequencies
        freq_per_bin = self.bandwidth / num_bins
        frequencies = np.linspace(
            self.center_freq - self.bandwidth / 2,
            self.center_freq + self.bandwidth / 2,
            num_bins,
            dtype=np.float32
        )
        
        # Convert uint8 (0-255) to dBm
        # Typical range: 0x00=-88dBm, 0xFF=-20dBm
        power_dbm = -88.0 + (bins_array.astype(np.float32) / 255.0) * (88.0 - 20.0)
        
        return frequencies, power_dbm


@dataclass
class AudioFrame:
    """ADPCM compressed audio frame."""
    timestamp: datetime
    websdr_id: str
    sample_rate: int  # Hz (typically 12000)
    codec: str  # "adpcm"
    adpcm_data: bytes  # Compressed audio (skip first 5 bytes: type + "SYNC")
    
    def decompress(self) -> np.ndarray:
        """
        Decompress ADPCM to PCM int16 samples.
        
        Returns:
            PCM audio as numpy int16 array
        
        Raises:
            ImportError: If audioop not available (removed in Python 3.13+)
        """
        try:
            import audioop
        except ImportError:
            logger.error("audioop module not available (removed in Python 3.13+)")
            raise ImportError(
                "ADPCM decompression requires audioop module. "
                "Use Python ≤3.12 or implement alternative decoder."
            )
        
        # Decompress ADPCM → PCM
        # Note: audioop.adpcm2lin requires state for continuous decoding
        # For single-frame decode, state=None
        pcm_bytes, _ = audioop.adpcm2lin(self.adpcm_data, 2, None)
        
        # Convert to numpy int16
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        return pcm_array


class OpenWebRXClient:
    """
    WebSocket client for OpenWebRX receiver.
    
    Connects to a single OpenWebRX instance and receives multiplexed streams:
    - FFT/Waterfall frames (type 0x01, ~7 fps, 2054 bytes)
    - Audio frames (type 0x02, ~20 fps, variable size, ADPCM)
    - JSON control messages (text, various types)
    
    Example:
        >>> client = OpenWebRXClient("http://sdr1.ik1jns.it:8076")
        >>> 
        >>> fft_count = 0
        >>> def on_fft(frame: FFTFrame):
        ...     global fft_count
        ...     fft_count += 1
        ...     freqs, power = frame.to_spectrum()
        ...     print(f"FFT: {len(freqs)} bins, peak={power.max():.1f} dBm")
        >>> 
        >>> client.on_fft = on_fft
        >>> 
        >>> await client.connect()
        >>> await client.receive_loop(duration=10)  # Run for 10 seconds
        >>> await client.disconnect()
        >>> print(f"Received {fft_count} FFT frames")
    """
    
    def __init__(self, base_url: str):
        """
        Initialize OpenWebRX client.
        
        Args:
            base_url: Base HTTP URL (e.g., "http://sdr1.ik1jns.it:8076")
        """
        self.base_url = base_url.rstrip('/')
        parsed = urlparse(base_url)
        self.websdr_id = f"{parsed.netloc}"
        
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        
        # Server configuration (received during handshake)
        self.center_freq: Optional[int] = None
        self.bandwidth: Optional[int] = None
        self.fft_size: int = 2048
        self.sample_rate: int = 12000
        self.receiver_gps: Optional[tuple[float, float]] = None  # (lat, lon)
        self.receiver_asl: Optional[int] = None  # Altitude ASL (meters)
        
        # Callbacks (set these before calling connect)
        self.on_fft: Optional[Callable[[FFTFrame], None]] = None
        self.on_audio: Optional[Callable[[AudioFrame], None]] = None
        self.on_control: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Statistics
        self.stats = {
            "fft_frames": 0,
            "audio_frames": 0,
            "text_messages": 0,
            "errors": 0,
        }
    
    async def connect(self, timeout: int = 10):
        """
        Connect to WebSDR and perform handshake.
        
        Args:
            timeout: Connection timeout in seconds
        
        Raises:
            websockets.exceptions.WebSocketException: If connection fails
            asyncio.TimeoutError: If handshake times out
        """
        # Build WebSocket URL
        ws_scheme = 'wss' if self.base_url.startswith('https') else 'ws'
        ws_url = f"{ws_scheme}://{self.base_url.split('://')[1]}/ws/"
        
        logger.info(f"Connecting to {ws_url}...")
        
        # Connect WebSocket
        self.ws = await websockets.connect(
            ws_url,
            open_timeout=timeout,
            close_timeout=timeout,
            ping_interval=30,  # Keep-alive ping every 30s
            ping_timeout=10,
        )
        
        logger.info(f"✅ WebSocket connected to {self.websdr_id}")
        
        # Send handshake (REQUIRED!)
        handshake = "SERVER DE CLIENT client=openwebrx.js type=receiver"
        await self.ws.send(handshake)
        logger.debug(f"Sent handshake: {handshake}")
        
        # Receive initial configuration messages
        logger.debug("Waiting for server configuration...")
        
        config_received = False
        for _ in range(10):
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                
                if isinstance(msg, str):
                    # Parse JSON config
                    if msg.startswith("CLIENT DE SERVER"):
                        # Server acknowledgment
                        logger.debug(f"Server ACK: {msg}")
                    else:
                        data = json.loads(msg)
                        self._handle_config(data)
                        
                        if data.get("type") == "config":
                            config_received = True
                
                # Stop after receiving main config
                if config_received:
                    break
                    
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for config message")
                break
        
        # Start demodulator (REQUIRED to receive data streams!)
        await self.ws.send(json.dumps({
            "type": "dspcontrol",
            "action": "start"
        }))
        logger.debug("Sent: dspcontrol start")
        
        # Configure audio parameters
        await self.ws.send(json.dumps({
            "type": "connectionproperties",
            "params": {
                "output_rate": 12000,
                "hd_output_rate": 48000
            }
        }))
        logger.debug("Sent: connectionproperties")
        
        logger.info(f"✅ Handshake complete - center_freq={self.center_freq} Hz, bw={self.bandwidth} Hz")
        self._running = True
    
    def _handle_config(self, msg: Dict[str, Any]):
        """
        Extract configuration from server messages.
        
        Args:
            msg: Parsed JSON message
        """
        msg_type = msg.get("type")
        value = msg.get("value", {})
        
        if msg_type == "config":
            # Main configuration
            self.center_freq = value.get("center_freq") or value.get("start_freq")
            self.bandwidth = value.get("samp_rate")
            
            logger.debug(
                f"Config: center={self.center_freq} Hz, bw={self.bandwidth} Hz, "
                f"profile={value.get('profile_id')}, mode={value.get('start_mod')}"
            )
        
        elif msg_type == "secondary_config":
            # FFT configuration
            self.fft_size = value.get("secondary_fft_size", 2048)
            logger.debug(f"FFT size: {self.fft_size}")
        
        elif msg_type == "receiver_details":
            # GPS coordinates and altitude
            gps = value.get("receiver_gps", {})
            if "lat" in gps and "lon" in gps:
                self.receiver_gps = (gps["lat"], gps["lon"])
            self.receiver_asl = value.get("receiver_asl")
            
            logger.debug(
                f"Receiver location: {self.receiver_gps}, "
                f"altitude={self.receiver_asl}m ASL"
            )
    
    async def receive_loop(self, duration: Optional[float] = None):
        """
        Receive and dispatch frames.
        
        Args:
            duration: Optional duration in seconds. If None, runs until disconnect.
        
        Raises:
            RuntimeError: If not connected
        """
        if not self.ws or not self._running:
            raise RuntimeError("Not connected. Call connect() first.")
        
        logger.info(f"Starting receive loop (duration={duration}s)")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            while self._running:
                # Check duration timeout
                if duration is not None:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration:
                        logger.info(f"Duration timeout reached ({duration}s)")
                        break
                
                # Receive next message
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    # No data for 5 seconds - check if still alive
                    if not self.ws.open:
                        logger.error("WebSocket closed unexpectedly")
                        break
                    continue
                
                # Dispatch message
                await self._dispatch_message(msg)
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
        
        except Exception as e:
            logger.error(f"Error in receive loop: {e}", exc_info=True)
            self.stats["errors"] += 1
        
        finally:
            logger.info(
                f"Receive loop ended. Stats: "
                f"FFT={self.stats['fft_frames']}, "
                f"Audio={self.stats['audio_frames']}, "
                f"Text={self.stats['text_messages']}, "
                f"Errors={self.stats['errors']}"
            )
    
    async def _dispatch_message(self, msg):
        """
        Parse and dispatch a received message.
        
        Args:
            msg: Raw message (str or bytes)
        """
        if isinstance(msg, str):
            # JSON text message (control/telemetry)
            self.stats["text_messages"] += 1
            
            try:
                data = json.loads(msg)
                
                # Call callback if set
                if self.on_control:
                    try:
                        self.on_control(data)
                    except Exception as e:
                        logger.error(f"Error in on_control callback: {e}", exc_info=True)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON message: {msg[:100]}")
                self.stats["errors"] += 1
        
        elif isinstance(msg, bytes):
            # Binary frame
            if len(msg) < 5:
                logger.warning(f"Frame too short: {len(msg)} bytes")
                self.stats["errors"] += 1
                return
            
            frame_type = msg[0]
            
            if frame_type == 0x01:
                # FFT/Waterfall frame
                self.stats["fft_frames"] += 1
                
                frame = FFTFrame(
                    timestamp=datetime.now(timezone.utc),
                    websdr_id=self.websdr_id,
                    center_freq=self.center_freq or 0,
                    bandwidth=self.bandwidth or 0,
                    bins=msg[5:]  # Skip type (1) + timestamp (4)
                )
                
                # Call callback if set
                if self.on_fft:
                    try:
                        self.on_fft(frame)
                    except Exception as e:
                        logger.error(f"Error in on_fft callback: {e}", exc_info=True)
            
            elif frame_type == 0x02:
                # Audio frame (ADPCM)
                self.stats["audio_frames"] += 1
                
                frame = AudioFrame(
                    timestamp=datetime.now(timezone.utc),
                    websdr_id=self.websdr_id,
                    sample_rate=self.sample_rate,
                    codec="adpcm",
                    adpcm_data=msg[5:]  # Skip type (1) + "SYNC" header (4)
                )
                
                # Call callback if set
                if self.on_audio:
                    try:
                        self.on_audio(frame)
                    except Exception as e:
                        logger.error(f"Error in on_audio callback: {e}", exc_info=True)
            
            else:
                logger.warning(f"Unknown frame type: 0x{frame_type:02x}")
                self.stats["errors"] += 1
    
    async def disconnect(self):
        """Disconnect WebSocket gracefully."""
        self._running = False
        
        if self.ws:
            try:
                await self.ws.close()
                logger.info(f"✅ Disconnected from {self.websdr_id}")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.ws = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        client = OpenWebRXClient("http://sdr1.ik1jns.it:8076")
        
        fft_count = 0
        audio_count = 0
        
        def on_fft(frame: FFTFrame):
            nonlocal fft_count
            fft_count += 1
            if fft_count % 10 == 1:
                freqs, power = frame.to_spectrum()
                print(f"FFT #{fft_count}: {len(freqs)} bins, peak={power.max():.1f} dBm")
        
        def on_audio(frame: AudioFrame):
            nonlocal audio_count
            audio_count += 1
            if audio_count % 50 == 1:
                print(f"Audio #{audio_count}: {len(frame.adpcm_data)} bytes ADPCM")
        
        def on_control(msg: dict):
            msg_type = msg.get("type")
            if msg_type in ["smeter", "temperature", "cpuusage"]:
                # Skip verbose telemetry
                pass
            else:
                print(f"Control: {msg_type} = {str(msg)[:100]}")
        
        client.on_fft = on_fft
        client.on_audio = on_audio
        client.on_control = on_control
        
        async with client:
            await client.receive_loop(duration=10)
        
        print(f"\nFinal: {fft_count} FFT frames, {audio_count} Audio frames")
    
    asyncio.run(main())
