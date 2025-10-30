"""WebSDR data fetcher for simultaneous IQ acquisition from multiple receivers."""

import asyncio
import logging
import struct
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp
import numpy as np

from ..models.websdrs import WebSDRConfig, IQDataPoint

logger = logging.getLogger(__name__)


class WebSDRFetcher:
    """Fetch IQ data simultaneously from multiple WebSDR receivers."""
    
    # WebSDR API endpoint pattern: http://<host>:<port>/iq
    # Request parameters: tune=<freq_hz>, rate=<sample_rate_hz>
    # Response: binary IQ data (alternating int16 I and Q samples)
    
    def __init__(
        self,
        websdrs: List[WebSDRConfig],
        timeout: int = 30,
        retry_count: int = 3,
        concurrent_limit: int = 7,
    ):
        """
        Initialize WebSDR fetcher.
        
        Args:
            websdrs: List of WebSDRConfig objects
            timeout: Request timeout in seconds
            retry_count: Number of retries per request
            concurrent_limit: Max concurrent requests
        """
        self.websdrs = {ws.id: ws for ws in websdrs if ws.is_active}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_count = retry_count
        self.concurrent_limit = concurrent_limit
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def __aenter__(self):
        """Context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=self.concurrent_limit,
            limit_per_host=2,
            ttl_dns_cache=300
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_iq_simultaneous(
        self,
        frequency_mhz: float,
        duration_seconds: float,
        sample_rate_khz: float = 12.5,
    ) -> Dict[int, Tuple[np.ndarray, Optional[str]]]:
        """
        Fetch IQ data simultaneously from all active WebSDRs.
        
        Args:
            frequency_mhz: Center frequency in MHz
            duration_seconds: Duration of acquisition
            sample_rate_khz: Sample rate in kHz
        
        Returns:
            Dict mapping WebSDR ID to (IQ data array, error message if any)
            IQ data is complex64 array with shape (num_samples,)
        """
        if not self.session:
            raise RuntimeError("Fetcher not in context manager")
        
        # Convert frequencies
        frequency_hz = int(frequency_mhz * 1e6)
        sample_rate_hz = int(sample_rate_khz * 1e3)
        
        # Calculate expected samples
        expected_samples = int(duration_seconds * sample_rate_hz)
        
        logger.info(
            "Starting simultaneous fetch from %d WebSDRs",
            len(self.websdrs),
            extra={
                "frequency_mhz": frequency_mhz,
                "duration_seconds": duration_seconds,
                "sample_rate_hz": sample_rate_hz,
                "expected_samples": expected_samples
            }
        )
        
        # Create fetch tasks for all receivers
        tasks = [
            self._fetch_from_websdr(
                websdr_id,
                websdr_cfg,
                frequency_hz,
                sample_rate_hz,
                expected_samples
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
                logger.warning(
                    "Failed to fetch from WebSDR %d: %s",
                    websdr_id,
                    error
                )
            else:
                logger.info(
                    "Successfully fetched %d samples from WebSDR %d",
                    len(iq_data) if iq_data is not None else 0,
                    websdr_id
                )
        
        return iq_data_dict
    
    async def _fetch_from_websdr(
        self,
        websdr_id: int,
        websdr_cfg: WebSDRConfig,
        frequency_hz: int,
        sample_rate_hz: int,
        expected_samples: int,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Fetch IQ data from a single WebSDR with retries.
        
        Returns:
            Tuple of (IQ data array or None, error message or None)
        """
        async with self.semaphore:
            for attempt in range(self.retry_count):
                try:
                    return await self._fetch_single_attempt(
                        websdr_cfg,
                        frequency_hz,
                        sample_rate_hz,
                        expected_samples
                    ), None
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(
                        "Attempt %d/%d failed for WebSDR %d: %s",
                        attempt + 1,
                        self.retry_count,
                        websdr_id,
                        error_msg
                    )
                    
                    if attempt < self.retry_count - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
        
        return None, f"Failed after {self.retry_count} attempts: {error_msg}"
    
    async def _fetch_single_attempt(
        self,
        websdr_cfg: WebSDRConfig,
        frequency_hz: int,
        sample_rate_hz: int,
        expected_samples: int,
    ) -> np.ndarray:
        """
        Single fetch attempt from WebSDR.
        
        Raises:
            Exception: If fetch fails
        """
        url = f"{str(websdr_cfg.url).rstrip('/')}/iq"
        params = {
            "tune": frequency_hz,
            "rate": sample_rate_hz,
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status} from {websdr_cfg.name}"
                )
            
            # Read binary IQ data
            data = await response.read()
            
            # WebSDR returns interleaved int16 I and Q samples
            # Each sample is 4 bytes (2 bytes I, 2 bytes Q)
            iq_samples = struct.unpack(f'<{len(data)//4}h', data[:len(data)//4*4])
            
            # Reshape to pairs and convert to complex
            iq_array = np.array(iq_samples, dtype=np.int16)
            
            # Normalize to [-1, 1] range and convert to complex
            iq_normalized = iq_array.astype(np.float32) / 32767.0
            iq_complex = iq_normalized[0::2] + 1j * iq_normalized[1::2]
            
            if len(iq_complex) == 0:
                raise RuntimeError(f"No IQ data received from {websdr_cfg.name}")
            
            logger.debug(
                "Received %d samples from %s",
                len(iq_complex),
                websdr_cfg.name
            )
            
            return iq_complex.astype(np.complex64)
    
    async def health_check(self) -> Dict[int, bool]:
        """
        Check connectivity to all WebSDR receivers.
        
        Returns:
            Dict mapping WebSDR ID to True if reachable, False otherwise
        """
        if not self.session:
            raise RuntimeError("Fetcher not in context manager")
        
        async def check_single(websdr_id: int, websdr_cfg: WebSDRConfig) -> Tuple[int, bool]:
            """
            Check if a single WebSDR is online.
            
            Try HEAD first (fast), fall back to GET if HEAD fails with 501.
            A WebSDR is considered ONLINE if:
            - HEAD request returns 200-299 (success)
            - GET request returns 200-299 (success) 
            - GET request returns 301-399 (redirect - still responding)
            - HEAD returns 501 (method not supported - server is responding!)
            
            A WebSDR is OFFLINE if:
            - Connection timeout
            - DNS resolution fails
            - Server returns 4xx/5xx (not 501)
            """
            try:
                # Use SHORT timeout for health checks - 3 seconds max per receiver
                timeout = aiohttp.ClientTimeout(total=3)
                
                # First try HEAD (faster)
                logger.debug(f"Health check: HEAD {websdr_cfg.url} (SDR #{websdr_id}, timeout=3s)")
                async with self.session.head(
                    str(websdr_cfg.url),
                    timeout=timeout,
                    allow_redirects=False
                ) as response:
                    # 501 = "Method Not Supported" - server is alive but doesn't support HEAD
                    if response.status == 501:
                        logger.debug(f"  → 501 (HEAD not supported), trying GET... (SDR #{websdr_id})")
                        # Fall through to try GET
                    # 200-299 = Success
                    elif 200 <= response.status < 300:
                        logger.info(f"  → {response.status} OK (SDR #{websdr_id} ONLINE)")
                        return websdr_id, True
                    # 300-399 = Redirect (server is responding)
                    elif 300 <= response.status < 400:
                        logger.info(f"  → {response.status} Redirect (SDR #{websdr_id} ONLINE)")
                        return websdr_id, True
                    # 4xx/5xx (except 501) = Error
                    else:
                        logger.warning(f"  → {response.status} Error (SDR #{websdr_id} appears OFFLINE)")
                        return websdr_id, False
                
                # If we got here, we got 501, try GET
                logger.debug(f"Health check: GET {websdr_cfg.url} (SDR #{websdr_id}, timeout=3s)")
                async with self.session.get(
                    str(websdr_cfg.url),
                    timeout=timeout,
                    allow_redirects=False
                ) as response:
                    # Accept 200-399 range as "online"
                    is_online = 200 <= response.status < 400
                    status_msg = "ONLINE" if is_online else "OFFLINE"
                    logger.info(f"  → GET {response.status} (SDR #{websdr_id} {status_msg})")
                    return websdr_id, is_online
                    
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout for {websdr_cfg.url} (SDR #{websdr_id}) - marking OFFLINE")
                return websdr_id, False
            except Exception as e:
                logger.warning(f"Health check error for {websdr_cfg.url} (SDR #{websdr_id}): {type(e).__name__}: {e} - marking OFFLINE")
                return websdr_id, False
        
        tasks = [
            check_single(wsid, wscfg)
            for wsid, wscfg in self.websdrs.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)
