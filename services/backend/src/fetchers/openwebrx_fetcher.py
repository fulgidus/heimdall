"""OpenWebRX WebSocket client for IQ data acquisition.

OpenWebRX uses WebSocket for streaming audio/waterfall/IQ data.
Protocol: Text commands for control, binary frames for data.
"""

import asyncio
import logging
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import numpy as np
import websockets
import json

from ..models.websdrs import WebSDRConfig, IQDataPoint

logger = logging.getLogger(__name__)


class OpenWebRXClient:
    """WebSocket client for OpenWebRX receiver."""
    
    # OpenWebRX WebSocket message types
    MSG_CLIENT_DE = "CLIENT DE SERVER"  # Server handshake
    MSG_RECEIVER_DETAILS = "receiver_details"
    
    def __init__(
        self,
        websdr_cfg: WebSDRConfig,
        timeout: int = 30,
    ):
        """
        Initialize OpenWebRX client.
        
        Args:
            websdr_cfg: WebSDR configuration
            timeout: Connection timeout in seconds
        """
        self.config = websdr_cfg
        self.timeout = timeout
        self.ws: Any = None  # WebSocket connection
        self.sample_rate = 12000  # Default for OpenWebRX audio
        self._receiver_info = {}
        
    async def __aenter__(self):
        """Context manager entry - connect to WebSocket."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close WebSocket."""
        await self.close()
    
    async def connect(self):
        """
        Connect to OpenWebRX WebSocket endpoint.
        
        OpenWebRX WebSocket is at ws://<host>:<port>/ws/
        """
        # Parse base URL and construct WebSocket URL
        parsed = urlparse(str(self.config.url))
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = f"{ws_scheme}://{parsed.netloc}/ws/"
        
        logger.info(f"Connecting to OpenWebRX: {ws_url}")
        
        try:
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    ws_url,
                    max_size=10_000_000,  # 10MB max message size for IQ data
                    ping_interval=20,
                    ping_timeout=10,
                ),
                timeout=self.timeout
            )
            logger.info(f"Connected to {self.config.name}")
            
            # Wait for server handshake
            await self._wait_for_handshake()
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Connection timeout to {self.config.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {self.config.name}: {e}")
    
    async def _wait_for_handshake(self):
        """
        Wait for server handshake and initialization.
        
        OpenWebRX requires client to send handshake first:
        "SERVER DE CLIENT client=<client_name> type=receiver"
        """
        try:
            # CRITICAL: Send client handshake first (this was the missing piece!)
            handshake_msg = "SERVER DE CLIENT client=heimdall type=receiver"
            await self.ws.send(handshake_msg)
            logger.debug(f"Sent handshake: {handshake_msg}")
            
            # Now OpenWebRX will send initial messages
            for _ in range(10):  # Wait for up to 10 messages
                message = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=5.0
                )
                
                if isinstance(message, str):
                    logger.debug(f"Received: {message[:100]}")
                    
                    # Parse JSON messages
                    if message.startswith('{'):
                        try:
                            data = json.loads(message)
                            if 'type' in data and data['type'] == 'receiver_details':
                                self._receiver_info = data
                                logger.info(f"Receiver details: {data.get('name', 'unknown')}")
                        except json.JSONDecodeError:
                            pass
                    
                    # Server ready when we get CLIENT DE SERVER
                    if self.MSG_CLIENT_DE in message:
                        logger.info("Handshake complete")
                        return
                        
        except asyncio.TimeoutError:
            logger.warning("Handshake timeout, proceeding anyway")
    
    async def set_frequency(self, frequency_hz: int):
        """
        Set receiver frequency.
        
        Args:
            frequency_hz: Center frequency in Hz
        """
        # OpenWebRX command format: "SET mod=<mode> sql=<squelch> freq=<freq>"
        command = f"SET mod=iq freq={frequency_hz}\n"
        await self.ws.send(command)
        logger.debug(f"Set frequency to {frequency_hz} Hz")
        
        # Small delay to let receiver tune
        await asyncio.sleep(0.5)
    
    async def start_audio_stream(self):
        """Start audio/IQ stream."""
        # Request IQ data stream
        await self.ws.send("SET mod=iq\n")
        await self.ws.send("START\n")
        logger.debug("Started IQ stream")
    
    async def stop_stream(self):
        """Stop audio/IQ stream."""
        try:
            await self.ws.send("STOP\n")
            logger.debug("Stopped stream")
        except Exception as e:
            logger.warning(f"Error stopping stream: {e}")
    
    async def fetch_iq_data(
        self,
        frequency_hz: int,
        duration_seconds: float,
    ) -> np.ndarray:
        """
        Fetch IQ data from OpenWebRX.
        
        Args:
            frequency_hz: Center frequency in Hz
            duration_seconds: Acquisition duration in seconds
        
        Returns:
            Complex64 IQ data array
        """
        if not self.ws:
            raise RuntimeError("Not connected")
        
        # Set frequency and start stream
        await self.set_frequency(frequency_hz)
        await self.start_audio_stream()
        
        # Collect IQ data
        iq_samples = []
        start_time = asyncio.get_event_loop().time()
        expected_samples = int(self.sample_rate * duration_seconds)
        
        logger.info(f"Collecting {expected_samples} samples over {duration_seconds}s")
        
        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                
                if elapsed >= duration_seconds:
                    logger.info(f"Duration reached: {elapsed:.2f}s")
                    break
                
                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Receive timeout, stopping")
                    break
                
                # Process binary IQ data
                if isinstance(message, bytes):
                    samples = self._parse_iq_data(message)
                    if samples is not None and len(samples) > 0:
                        iq_samples.append(samples)
                        logger.debug(f"Received {len(samples)} samples, total: {sum(len(s) for s in iq_samples)}")
                
                # Stop if we have enough samples
                total_samples = sum(len(s) for s in iq_samples)
                if total_samples >= expected_samples:
                    logger.info(f"Collected enough samples: {total_samples}")
                    break
                    
        finally:
            await self.stop_stream()
        
        # Concatenate all samples
        if iq_samples:
            iq_array = np.concatenate(iq_samples)
            logger.info(f"Total IQ samples collected: {len(iq_array)}")
            return iq_array.astype(np.complex64)
        else:
            raise RuntimeError("No IQ data received")
    
    def _parse_iq_data(self, data: bytes) -> Optional[np.ndarray]:
        """
        Parse IQ data from OpenWebRX binary message.
        
        OpenWebRX sends audio as 16-bit PCM (int16).
        For IQ mode, it's interleaved I/Q samples.
        
        Args:
            data: Binary data from WebSocket
        
        Returns:
            Complex IQ samples or None if invalid
        """
        try:
            # Check if data is long enough
            if len(data) < 4:
                return None
            
            # OpenWebRX audio data format: int16 samples
            # IQ mode: alternating I and Q samples
            num_samples = len(data) // 2
            samples = struct.unpack(f'<{num_samples}h', data[:num_samples*2])
            
            if len(samples) < 2:
                return None
            
            # Convert to numpy array
            samples_array = np.array(samples, dtype=np.int16)
            
            # Normalize to [-1, 1] range
            normalized = samples_array.astype(np.float32) / 32768.0
            
            # Convert to complex (I + jQ)
            # Even indices are I, odd indices are Q
            i_samples = normalized[0::2]
            q_samples = normalized[1::2]
            
            # Make sure I and Q have same length
            min_len = min(len(i_samples), len(q_samples))
            if min_len == 0:
                return None
                
            iq_complex = i_samples[:min_len] + 1j * q_samples[:min_len]
            
            return iq_complex
            
        except Exception as e:
            logger.warning(f"Failed to parse IQ data: {e}")
            return None
    
    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                await self.stop_stream()
                await self.ws.close()
                logger.info(f"Closed connection to {self.config.name}")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.ws = None


class OpenWebRXFetcher:
    """Fetch IQ data simultaneously from multiple OpenWebRX receivers."""
    
    def __init__(
        self,
        websdrs: List[WebSDRConfig],
        timeout: int = 30,
        retry_count: int = 3,
        concurrent_limit: int = 7,
    ):
        """
        Initialize OpenWebRX fetcher.
        
        Args:
            websdrs: List of WebSDRConfig objects
            timeout: Request timeout in seconds
            retry_count: Number of retries per request
            concurrent_limit: Max concurrent connections
        """
        self.websdrs = {ws.id: ws for ws in websdrs if ws.is_active}
        self.timeout = timeout
        self.retry_count = retry_count
        self.concurrent_limit = concurrent_limit
        self.semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def fetch_iq_simultaneous(
        self,
        frequency_mhz: float,
        duration_seconds: float,
        sample_rate_khz: float = 12.0,  # OpenWebRX default
    ) -> Dict[int, Tuple[Optional[np.ndarray], Optional[str]]]:
        """
        Fetch IQ data simultaneously from all active WebSDRs.
        
        Args:
            frequency_mhz: Center frequency in MHz
            duration_seconds: Duration of acquisition
            sample_rate_khz: Sample rate in kHz (ignored, OpenWebRX decides)
        
        Returns:
            Dict mapping WebSDR ID to (IQ data array, error message if any)
        """
        frequency_hz = int(frequency_mhz * 1e6)
        
        logger.info(
            f"Starting simultaneous fetch from {len(self.websdrs)} OpenWebRX receivers",
            extra={
                "frequency_mhz": frequency_mhz,
                "duration_seconds": duration_seconds,
            }
        )
        
        # Create fetch tasks for all receivers
        tasks = [
            self._fetch_from_websdr(
                websdr_id,
                websdr_cfg,
                frequency_hz,
                duration_seconds
            )
            for websdr_id, websdr_cfg in self.websdrs.items()
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Map results back to WebSDR IDs
        iq_data_dict = {}
        for (websdr_id, _), result in zip(
            sorted(self.websdrs.items(), key=lambda x: x[0]),
            results
        ):
            iq_data, error = result
            iq_data_dict[websdr_id] = (iq_data, error)
            
            if error:
                logger.warning(f"Failed to fetch from WebSDR {websdr_id}: {error}")
            else:
                logger.info(
                    f"Successfully fetched {len(iq_data) if iq_data is not None else 0} samples from WebSDR {websdr_id}"
                )
        
        return iq_data_dict
    
    async def _fetch_from_websdr(
        self,
        websdr_id: int,
        websdr_cfg: WebSDRConfig,
        frequency_hz: int,
        duration_seconds: float,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Fetch IQ data from a single WebSDR with retries.
        
        Returns:
            Tuple of (IQ data array or None, error message or None)
        """
        async with self.semaphore:
            for attempt in range(self.retry_count):
                try:
                    async with OpenWebRXClient(websdr_cfg, self.timeout) as client:
                        iq_data = await client.fetch_iq_data(
                            frequency_hz,
                            duration_seconds
                        )
                        return iq_data, None
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.retry_count} failed for WebSDR {websdr_id}: {error_msg}"
                    )
                    
                    if attempt < self.retry_count - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    else:
                        # Last attempt failed, return error
                        return None, f"Failed after {self.retry_count} attempts: {error_msg}"
            
            return None, "Failed without specific error"
    
    async def health_check(self) -> Dict[int, bool]:
        """
        Check connectivity to all OpenWebRX receivers.
        
        Returns:
            Dict mapping WebSDR ID to True if reachable, False otherwise
        """
        async def check_single(websdr_id: int, websdr_cfg: WebSDRConfig) -> Tuple[int, bool]:
            """Check if a single OpenWebRX is online."""
            try:
                logger.debug(f"Health check for WebSDR {websdr_id}: {websdr_cfg.url}")
                
                # Try to connect via WebSocket
                async with OpenWebRXClient(websdr_cfg, timeout=5) as client:
                    logger.info(f"WebSDR {websdr_id} ({websdr_cfg.name}) is ONLINE")
                    return websdr_id, True
                    
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout for WebSDR {websdr_id} - marking OFFLINE")
                return websdr_id, False
            except Exception as e:
                logger.warning(f"Health check error for WebSDR {websdr_id}: {type(e).__name__}: {e} - marking OFFLINE")
                return websdr_id, False
        
        tasks = [
            check_single(wsid, wscfg)
            for wsid, wscfg in self.websdrs.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)
