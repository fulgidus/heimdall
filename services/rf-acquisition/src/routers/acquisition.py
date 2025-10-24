"""FastAPI endpoints for RF acquisition."""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from celery.result import AsyncResult

from ..models.websdrs import (
    AcquisitionRequest,
    AcquisitionTaskResponse,
    AcquisitionStatusResponse,
    WebSDRConfig,
)
from ..tasks.acquire_iq import acquire_iq, health_check_websdrs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/acquisition", tags=["acquisition"])


# WebSDRs - Northwestern Italy (Piedmont & Liguria regions)
# Source: WEBSDRS.md - Strategic network for triangulation in Northern Italy
DEFAULT_WEBSDRS = [
    {
        "id": 1,
        "name": "Aquila di Giaveno",
        "url": "http://sdr1.ik1jns.it:8076/",
        "location_name": "Giaveno, Italy",
        "latitude": 45.02,
        "longitude": 7.29,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
    {
        "id": 2,
        "name": "Montanaro",
        "url": "http://cbfenis.ddns.net:43510/",
        "location_name": "Montanaro, Italy",
        "latitude": 45.234,
        "longitude": 7.857,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
    {
        "id": 3,
        "name": "Torino",
        "url": "http://vst-aero.it:8073/",
        "location_name": "Torino, Italy",
        "latitude": 45.044,
        "longitude": 7.672,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
    {
        "id": 4,
        "name": "Coazze",
        "url": "http://94.247.189.130:8076/",
        "location_name": "Coazze, Italy",
        "latitude": 45.03,
        "longitude": 7.27,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
    {
        "id": 5,
        "name": "Passo del Giovi",
        "url": "http://iz1mlt.ddns.net:8074/",
        "location_name": "Passo del Giovi, Italy",
        "latitude": 44.561,
        "longitude": 8.956,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
    {
        "id": 6,
        "name": "Genova",
        "url": "http://iq1zw.ddns.net:42154/",
        "location_name": "Genova, Italy",
        "latitude": 44.395,
        "longitude": 8.956,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
    {
        "id": 7,
        "name": "Milano - Baggio",
        "url": "http://iu2mch.duckdns.org:8073/",
        "location_name": "Milano (Baggio), Italy",
        "latitude": 45.478,
        "longitude": 9.123,
        "is_active": True,
        "timeout_seconds": 30,
        "retry_count": 3
    },
]


def get_websdrs_config() -> list[dict]:
    """Get WebSDR configuration."""
    # TODO: Load from database
    return DEFAULT_WEBSDRS


@router.post("/acquire", response_model=AcquisitionTaskResponse)
async def trigger_acquisition(
    request: AcquisitionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger an IQ data acquisition from WebSDR receivers.
    
    Returns:
        Task ID and initial status
    """
    try:
        logger.info(
            "Triggering acquisition at %.2f MHz for %.1f seconds",
            request.frequency_mhz,
            request.duration_seconds
        )
        
        # Get WebSDR configs
        websdrs_config = get_websdrs_config()
        
        # Filter to requested receivers if specified
        if request.websdrs:
            websdrs_config = [
                w for w in websdrs_config if w['id'] in request.websdrs
            ]
        
        if not websdrs_config:
            raise HTTPException(
                status_code=400,
                detail="No active WebSDRs available for acquisition"
            )
        
        # Queue Celery task
        task = acquire_iq.delay(
            frequency_mhz=request.frequency_mhz,
            duration_seconds=request.duration_seconds,
            start_time_iso=request.start_time.isoformat(),
            websdrs_config_list=websdrs_config,
            sample_rate_khz=12.5,
        )
        
        logger.info("Queued acquisition task: %s", task.id)
        
        return AcquisitionTaskResponse(
            task_id=str(task.id),
            status="PENDING",
            message=f"Acquisition task queued for {len(websdrs_config)} WebSDR receivers",
            frequency_mhz=request.frequency_mhz,
            websdrs_count=len(websdrs_config)
        )
    
    except Exception as e:
        logger.exception("Error triggering acquisition: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering acquisition: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=AcquisitionStatusResponse)
async def get_acquisition_status(task_id: str):
    """
    Get status of an ongoing acquisition task.
    
    Args:
        task_id: Celery task ID
    
    Returns:
        Task status and progress information
    """
    try:
        result = AsyncResult(task_id)
        
        status_map = {
            'PENDING': 'PENDING',
            'STARTED': 'PROGRESS',
            'PROGRESS': 'PROGRESS',
            'SUCCESS': 'SUCCESS',
            'FAILURE': 'FAILURE',
            'REVOKED': 'REVOKED',
            'RETRY': 'PROGRESS',
        }
        
        mapped_status = status_map.get(result.state, result.state)
        
        # Extract progress info
        if result.state == 'PROGRESS':
            info = result.info if isinstance(result.info, dict) else {}
            progress = info.get('progress', 0)
            message = info.get('status', 'Processing...')
            measurements_collected = info.get('successful', 0)
        elif result.state == 'SUCCESS':
            info = result.result if isinstance(result.result, dict) else {}
            progress = 100
            message = "Acquisition complete"
            measurements_collected = info.get('measurements_count', 0)
        else:
            info = {}
            progress = 0 if result.state == 'PENDING' else 50
            message = f"Task state: {result.state}"
            measurements_collected = 0
        
        return AcquisitionStatusResponse(
            task_id=task_id,
            status=mapped_status,
            progress=progress,
            message=message,
            measurements_collected=measurements_collected,
            errors=info.get('errors', None),
            result=result.result if result.state == 'SUCCESS' else None
        )
    
    except Exception as e:
        logger.exception("Error getting acquisition status: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get("/websdrs", response_model=list[dict])
async def list_websdrs():
    """
    List all configured WebSDR receivers.
    
    Returns:
        List of WebSDR configurations
    """
    return get_websdrs_config()


@router.get("/websdrs/health")
async def check_websdrs_health():
    """
    Check health status of all WebSDR receivers with metrics.
    
    Returns:
        Dict mapping WebSDR ID to health status with SNR, uptime, last contact
    """
    try:
        from ..storage.db_manager import DatabaseManager
        from ..tasks.uptime_monitor import calculate_uptime_percentage
        
        # Get WebSDR configs
        websdrs_config = get_websdrs_config()
        logger.debug(f"WebSDR configs loaded: {len(websdrs_config)} SDRs")
        
        # Run health check task
        task = health_check_websdrs.delay()
        logger.debug(f"Health check task submitted: {task.id}")
        
        result = task.get(timeout=60)
        logger.info(f"Health check task result: {result}")
        
        # Get SNR statistics from database
        db_manager = DatabaseManager()
        snr_stats = db_manager.get_snr_statistics() if db_manager else {}
        logger.debug(f"SNR statistics retrieved: {len(snr_stats) if snr_stats else 0} entries")
        
        # Format response with detailed status information
        health_status = {}
        check_time = datetime.utcnow().isoformat()
        
        for ws_config in websdrs_config:
            ws_id = ws_config['id']
            # IMPORTANT: Celery serializes dict keys as strings!
            is_online = result.get(str(ws_id), False)
            
            # Get SNR and uptime from stats
            sdr_stats = snr_stats.get(ws_id, {}) if snr_stats else {}
            avg_snr = sdr_stats.get('avg_snr_db', None)
            
            # Calculate uptime from database history (last 24 hours)
            uptime = calculate_uptime_percentage(ws_id, hours=24)
            
            health_status[ws_id] = {
                'websdr_id': ws_id,
                'name': ws_config['name'],
                'status': 'online' if is_online else 'offline',
                'last_check': check_time,
                'uptime': round(uptime, 1),
                'avg_snr': round(avg_snr, 2) if avg_snr is not None else None,
            }
            
            if not is_online:
                health_status[ws_id]['error_message'] = 'Health check failed or timed out'
        
        logger.info(f"Health check response ready with metrics")
        return health_status
    
    except Exception as e:
        logger.exception("Error checking WebSDR health: %s", str(e))
        
        # Return offline status for all WebSDRs on error
        websdrs_config = get_websdrs_config()
        check_time = datetime.utcnow().isoformat()
        
        health_status = {}
        for ws_config in websdrs_config:
            health_status[ws_config['id']] = {
                'websdr_id': ws_config['id'],
                'name': ws_config['name'],
                'status': 'offline',
                'last_check': check_time,
                'error_message': f'Health check error: {str(e)}'
            }
        
        return health_status


@router.get("/config")
async def get_configuration():
    """
    Get acquisition service configuration.
    
    Returns:
        Service configuration details
    """
    return {
        'service': 'rf-acquisition',
        'version': '0.1.0',
        'capabilities': [
            'simultaneous-acquisition',
            'iq-processing',
            'signal-metrics'
        ],
        'websdrs_count': len(get_websdrs_config()),
        'max_duration_seconds': 300,
        'default_sample_rate_khz': 12.5,
    }
