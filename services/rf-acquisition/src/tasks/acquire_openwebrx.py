"""
Celery tasks for OpenWebRX data acquisition.

These tasks replace the old IQ-based acquisition (which assumed /iq endpoint)
with the real OpenWebRX WebSocket protocol for FFT and Audio streams.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from celery import shared_task
import numpy as np

from ..fetchers.openwebrx_client import OpenWebRXClient, FFTFrame, AudioFrame
from ..models.websdrs import WebSDRConfig

logger = logging.getLogger(__name__)


# WebSDR URLs from WEBSDRS.md
OPENWEBRX_URLS = [
    "http://sdr1.ik1jns.it:8076",  # Aquila di Giaveno
    "http://sdr3.ik1jns.it:8073",  # Rubiana
    "http://iz1pnv.duckdns.org:8074",  # Caselle T.se
    "http://195.231.18.92:8080",  # Morozzo
    "http://212.237.38.25:8073",  # Claviere
    "http://websdr.iz1kqw.com:8076",  # Borgosesia 2m
    "http://websdr.iz1kqw.com:8080",  # Borgosesia 70cm
]


@shared_task(name="rf_acquisition.acquire_openwebrx_single")
def acquire_openwebrx_single(
    websdr_url: str,
    duration_seconds: int = 60,
    save_fft: bool = True,
    save_audio: bool = False,
) -> Dict:
    """
    Acquire FFT and Audio data from a single OpenWebRX receiver.
    
    Args:
        websdr_url: Base URL (e.g., "http://sdr1.ik1jns.it:8076")
        duration_seconds: Acquisition duration (default 60s)
        save_fft: Save FFT frames to database (default True)
        save_audio: Save audio frames to database (default False, uses lots of storage!)
    
    Returns:
        Dict with acquisition statistics:
        {
            "websdr_url": str,
            "duration": int,
            "fft_frames": int,
            "audio_frames": int,
            "text_messages": int,
            "errors": int,
            "success": bool
        }
    """
    logger.info(
        f"Starting OpenWebRX acquisition: {websdr_url}, duration={duration_seconds}s"
    )
    
    try:
        # Run async acquisition
        result = asyncio.run(
            _acquire_async_single(websdr_url, duration_seconds, save_fft, save_audio)
        )
        
        logger.info(
            f"✅ Acquisition complete: {result['fft_frames']} FFT, "
            f"{result['audio_frames']} Audio from {websdr_url}"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Acquisition failed for {websdr_url}: {e}", exc_info=True)
        return {
            "websdr_url": websdr_url,
            "duration": duration_seconds,
            "fft_frames": 0,
            "audio_frames": 0,
            "text_messages": 0,
            "errors": 1,
            "success": False,
            "error": str(e),
        }


async def _acquire_async_single(
    websdr_url: str,
    duration_seconds: int,
    save_fft: bool,
    save_audio: bool,
) -> Dict:
    """
    Async worker for single WebSDR acquisition.
    
    This function runs in the event loop and handles WebSocket communication.
    """
    client = OpenWebRXClient(websdr_url)
    
    # Storage counters
    fft_saved = 0
    audio_saved = 0
    
    # Callbacks
    def on_fft(frame: FFTFrame):
        nonlocal fft_saved
        
        if save_fft:
            try:
                _save_fft_frame(frame)
                fft_saved += 1
            except Exception as e:
                logger.error(f"Failed to save FFT frame: {e}")
    
    def on_audio(frame: AudioFrame):
        nonlocal audio_saved
        
        if save_audio:
            try:
                _save_audio_frame(frame)
                audio_saved += 1
            except Exception as e:
                logger.error(f"Failed to save audio frame: {e}")
    
    # Set callbacks
    client.on_fft = on_fft
    client.on_audio = on_audio
    
    # Acquire
    async with client:
        await client.receive_loop(duration=duration_seconds)
    
    return {
        "websdr_url": websdr_url,
        "duration": duration_seconds,
        "fft_frames": client.stats["fft_frames"],
        "audio_frames": client.stats["audio_frames"],
        "text_messages": client.stats["text_messages"],
        "errors": client.stats["errors"],
        "fft_saved": fft_saved,
        "audio_saved": audio_saved,
        "success": client.stats["errors"] == 0,
    }


def _save_fft_frame(frame: FFTFrame):
    """
    Save FFT frame to database.
    
    TODO: Implement actual database save using SQLAlchemy models
    For now, just log
    """
    logger.debug(
        f"FFT frame: {frame.websdr_id}, "
        f"freq={frame.center_freq/1e6:.3f} MHz, "
        f"bins={len(frame.bins)}"
    )
    
    # TODO: Insert into fft_captures table
    # from ..models.acquisitions import FFTCapture
    # FFTCapture.create(
    #     websdr_id=frame.websdr_id,
    #     timestamp=frame.timestamp,
    #     center_freq=frame.center_freq,
    #     bandwidth=frame.bandwidth,
    #     num_bins=len(frame.bins),
    #     spectral_data=frame.bins
    # )


def _save_audio_frame(frame: AudioFrame):
    """
    Save audio frame to database.
    
    WARNING: Audio frames are frequent (~20 fps) and large (2KB each).
    This generates ~40 KB/sec = 2.4 MB/minute = 144 MB/hour per receiver!
    
    Only enable if you have sufficient storage.
    """
    logger.debug(
        f"Audio frame: {frame.websdr_id}, "
        f"codec={frame.codec}, "
        f"size={len(frame.adpcm_data)} bytes"
    )
    
    # TODO: Insert into audio_captures table
    # from ..models.acquisitions import AudioCapture
    # AudioCapture.create(
    #     websdr_id=frame.websdr_id,
    #     timestamp=frame.timestamp,
    #     sample_rate=frame.sample_rate,
    #     codec=frame.codec,
    #     audio_data=frame.adpcm_data
    # )


@shared_task(name="rf_acquisition.acquire_openwebrx_all")
def acquire_openwebrx_all(
    duration_seconds: int = 300,
    save_fft: bool = True,
    save_audio: bool = False,
) -> Dict:
    """
    Acquire from ALL 7 OpenWebRX receivers simultaneously.
    
    Args:
        duration_seconds: Acquisition duration (default 300s = 5 minutes)
        save_fft: Save FFT frames (default True)
        save_audio: Save audio frames (default False, uses ~1 GB/hour!)
    
    Returns:
        Dict with aggregate statistics:
        {
            "duration": int,
            "num_receivers": int,
            "total_fft_frames": int,
            "total_audio_frames": int,
            "receivers": {
                "http://sdr1...": {...},
                "http://sdr3...": {...},
                ...
            }
        }
    """
    logger.info(
        f"Starting simultaneous acquisition from {len(OPENWEBRX_URLS)} receivers, "
        f"duration={duration_seconds}s"
    )
    
    try:
        result = asyncio.run(
            _acquire_async_all(duration_seconds, save_fft, save_audio)
        )
        
        logger.info(
            f"✅ Simultaneous acquisition complete: "
            f"{result['total_fft_frames']} FFT total, "
            f"{result['total_audio_frames']} Audio total"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Simultaneous acquisition failed: {e}", exc_info=True)
        return {
            "duration": duration_seconds,
            "num_receivers": len(OPENWEBRX_URLS),
            "total_fft_frames": 0,
            "total_audio_frames": 0,
            "success": False,
            "error": str(e),
        }


async def _acquire_async_all(
    duration_seconds: int,
    save_fft: bool,
    save_audio: bool,
) -> Dict:
    """
    Async worker for simultaneous multi-receiver acquisition.
    
    Runs 7 WebSocket connections in parallel using asyncio.gather().
    """
    # Create tasks for all receivers
    tasks = [
        _acquire_async_single(url, duration_seconds, save_fft, save_audio)
        for url in OPENWEBRX_URLS
    ]
    
    # Run all simultaneously
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    receivers_dict = {}
    total_fft = 0
    total_audio = 0
    errors = 0
    
    for url, result in zip(OPENWEBRX_URLS, results):
        if isinstance(result, Exception):
            logger.error(f"Receiver {url} failed: {result}")
            receivers_dict[url] = {
                "success": False,
                "error": str(result)
            }
            errors += 1
        else:
            receivers_dict[url] = result
            total_fft += result.get("fft_frames", 0)
            total_audio += result.get("audio_frames", 0)
            if not result.get("success", False):
                errors += 1
    
    return {
        "duration": duration_seconds,
        "num_receivers": len(OPENWEBRX_URLS),
        "total_fft_frames": total_fft,
        "total_audio_frames": total_audio,
        "errors": errors,
        "success": errors == 0,
        "receivers": receivers_dict,
    }


@shared_task(name="rf_acquisition.health_check_openwebrx")
def health_check_openwebrx() -> Dict[str, bool]:
    """
    Check connectivity to all OpenWebRX receivers.
    
    Returns:
        Dict mapping URL to online status:
        {
            "http://sdr1.ik1jns.it:8076": True,
            "http://sdr3.ik1jns.it:8073": False,
            ...
        }
    """
    logger.info(f"Health check: {len(OPENWEBRX_URLS)} OpenWebRX receivers")
    
    try:
        result = asyncio.run(_health_check_async())
        
        online_count = sum(1 for status in result.values() if status)
        logger.info(
            f"✅ Health check complete: {online_count}/{len(OPENWEBRX_URLS)} online"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        return {url: False for url in OPENWEBRX_URLS}


async def _health_check_async() -> Dict[str, bool]:
    """
    Async worker for health check.
    
    Tries to connect to each WebSDR WebSocket with a short timeout.
    """
    import aiohttp
    
    async def check_single(url: str) -> tuple[str, bool]:
        """Check single WebSDR."""
        try:
            # Try HTTP GET first (faster than WebSocket)
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    is_online = 200 <= response.status < 400
                    logger.debug(
                        f"Health check: {url} → {response.status} "
                        f"({'ONLINE' if is_online else 'OFFLINE'})"
                    )
                    return url, is_online
        
        except Exception as e:
            logger.warning(f"Health check: {url} → OFFLINE ({type(e).__name__})")
            return url, False
    
    # Check all simultaneously
    tasks = [check_single(url) for url in OPENWEBRX_URLS]
    results = await asyncio.gather(*tasks)
    
    return dict(results)
