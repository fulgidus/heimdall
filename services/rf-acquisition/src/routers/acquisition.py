"""FastAPI endpoints for RF acquisition."""

import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from celery.result import AsyncResult

from ..models.websdrs import (
    AcquisitionRequest,
    AcquisitionTaskResponse,
    AcquisitionStatusResponse,
    WebSDRConfig,
    WebSDRCreateRequest,
    WebSDRUpdateRequest,
    WebSDRResponse,
)
from ..tasks.acquire_iq import acquire_iq, health_check_websdrs
from ..storage.db_manager import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/acquisition", tags=["acquisition"])


def get_websdrs_config() -> list[dict]:
    """
    Get WebSDR configuration from database.
    
    Returns list of active WebSDR configurations suitable for acquisition tasks.
    """
    db_manager = get_db_manager()
    active_stations = db_manager.get_active_websdrs()
    
    # Convert ORM models to dicts compatible with acquisition tasks
    websdrs_config = []
    for idx, station in enumerate(active_stations, start=1):
        config = {
            "id": idx,  # Sequential ID for compatibility
            "name": station.name,
            "url": station.url,
            "location_name": station.location_description or f"{station.name}, {station.country or 'Italy'}",
            "latitude": float(station.latitude),
            "longitude": float(station.longitude),
            "is_active": station.is_active,
            "timeout_seconds": station.timeout_seconds or 30,
            "retry_count": station.retry_count or 3,
        }
        websdrs_config.append(config)
    
    logger.debug(f"Loaded {len(websdrs_config)} active WebSDR configurations from database")
    return websdrs_config


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
    List all configured WebSDR receivers from database.
    
    Returns:
        List of WebSDR configurations from database
    """
    websdrs = get_websdrs_config()
    logger.info(f"Returning {len(websdrs)} WebSDR stations from database")
    return websdrs


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


# ============================================================================
# OpenWebRX Acquisition Endpoints
# ============================================================================

from pydantic import BaseModel, Field
from ..tasks.acquire_openwebrx import (
    acquire_openwebrx_single,
    acquire_openwebrx_all,
    health_check_openwebrx,
)


class AcquireOpenWebRXRequest(BaseModel):
    """Request to start OpenWebRX acquisition."""
    websdr_url: Optional[str] = Field(None, description="Single WebSDR URL (if None, acquires from all 7)")
    duration_seconds: int = Field(60, ge=10, le=3600, description="Acquisition duration (10-3600 seconds)")
    save_fft: bool = Field(True, description="Save FFT frames to database")
    save_audio: bool = Field(False, description="Save audio frames (uses lots of storage!)")


class AcquireOpenWebRXResponse(BaseModel):
    """Response from acquisition trigger."""
    task_id: str
    message: str
    websdr_url: Optional[str] = None
    duration_seconds: int
    estimated_completion_time: str


@router.post("/openwebrx/acquire", response_model=AcquireOpenWebRXResponse)
async def trigger_openwebrx_acquisition(request: AcquireOpenWebRXRequest):
    """
    Trigger OpenWebRX data acquisition via Celery task.
    
    **Single WebSDR mode:**
    - Set `websdr_url` to specific URL (e.g., "http://sdr1.ik1jns.it:8076")
    
    **Multi-WebSDR mode:**
    - Leave `websdr_url` as null → acquires from all 7 receivers
    
    **Returns:**
    - Celery task ID for status tracking
    - Estimated completion time
    """
    from datetime import datetime, timedelta
    
    # Trigger appropriate Celery task
    if request.websdr_url:
        # Single WebSDR acquisition
        task = acquire_openwebrx_single.delay(
            websdr_url=request.websdr_url,
            duration_seconds=request.duration_seconds,
            save_fft=request.save_fft,
            save_audio=request.save_audio
        )
        
        message = f"OpenWebRX acquisition started for {request.websdr_url}"
        logger.info(f"📡 {message} - Task ID: {task.id}")
    
    else:
        # Multi-WebSDR acquisition (all 7)
        task = acquire_openwebrx_all.delay(
            duration_seconds=request.duration_seconds,
            save_fft=request.save_fft,
            save_audio=request.save_audio
        )
        
        message = "OpenWebRX acquisition started for ALL 7 receivers"
        logger.info(f"📡 {message} - Task ID: {task.id}")
    
    # Calculate estimated completion
    completion_time = datetime.utcnow() + timedelta(seconds=request.duration_seconds + 10)
    
    return AcquireOpenWebRXResponse(
        task_id=task.id,
        message=message,
        websdr_url=request.websdr_url,
        duration_seconds=request.duration_seconds,
        estimated_completion_time=completion_time.isoformat()
    )


@router.get("/openwebrx/status/{task_id}")
async def get_openwebrx_task_status(task_id: str):
    """
    Get status of OpenWebRX acquisition task.
    
    **States:**
    - PENDING: Task queued but not started
    - STARTED: Task running
    - SUCCESS: Task completed successfully
    - FAILURE: Task failed with error
    
    **Returns:**
    - state: Current task state
    - result: Task result (if completed)
    - info: Progress info (if running)
    """
    task = AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "state": task.state,
    }
    
    if task.state == "PENDING":
        response["info"] = "Task is queued and waiting to start"
    
    elif task.state == "STARTED":
        response["info"] = task.info or "Task is running..."
    
    elif task.state == "SUCCESS":
        response["result"] = task.result
        response["info"] = "Task completed successfully"
    
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
        response["info"] = "Task failed"
    
    else:
        response["info"] = task.info
    
    logger.debug(f"📊 Task {task_id} status: {task.state}")
    
    return response


@router.post("/openwebrx/health-check")
async def trigger_openwebrx_health_check():
    """
    Trigger health check for all 7 OpenWebRX receivers.
    
    This is a SYNCHRONOUS call (not Celery task) because health checks
    should be fast (~3 seconds total for all 7 receivers).
    
    **Returns:**
    - Dict mapping URL → online status (True/False)
    - Example: {"http://sdr1.ik1jns.it:8076": True, ...}
    """
    logger.info("🏥 OpenWebRX health check triggered")
    
    try:
        # Call Celery task and wait for result (short timeout)
        task = health_check_openwebrx.apply_async()
        result = task.get(timeout=15)  # 15 second timeout
        
        online_count = sum(1 for status in result.values() if status)
        logger.info(f"✅ Health check complete: {online_count}/7 receivers online")
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "online_count": online_count,
            "total_count": 7,
            "receivers": result
        }
    
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


# ============================================================================
# WebSDR CRUD Endpoints
# ============================================================================

@router.post("/websdrs", response_model=WebSDRResponse, status_code=201)
async def create_websdr(request: WebSDRCreateRequest):
    """
    Create a new WebSDR station.
    
    Args:
        request: WebSDR creation request with all required fields
    
    Returns:
        Created WebSDR station details
    
    Raises:
        HTTPException 400: If station name already exists or validation fails
        HTTPException 500: If database error occurs
    """
    try:
        db_manager = get_db_manager()
        
        # Create the station
        station = db_manager.create_websdr(
            name=request.name,
            url=request.url,
            latitude=request.latitude,
            longitude=request.longitude,
            location_description=request.location_description,
            country=request.country,
            admin_email=request.admin_email,
            altitude_asl=request.altitude_asl,
            timeout_seconds=request.timeout_seconds,
            retry_count=request.retry_count,
            is_active=request.is_active
        )
        
        if not station:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create WebSDR station. Name '{request.name}' may already exist."
            )
        
        logger.info(f"Created WebSDR station: {station.name} (ID: {station.id})")
        return WebSDRResponse.model_validate(station)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating WebSDR station: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/websdrs/{station_id}", response_model=WebSDRResponse)
async def get_websdr(station_id: str):
    """
    Get a specific WebSDR station by ID.
    
    Args:
        station_id: UUID of the WebSDR station
    
    Returns:
        WebSDR station details
    
    Raises:
        HTTPException 404: If station not found
        HTTPException 500: If database error occurs
    """
    try:
        db_manager = get_db_manager()
        station = db_manager.get_websdr_by_id(station_id)
        
        if not station:
            raise HTTPException(
                status_code=404,
                detail=f"WebSDR station with ID {station_id} not found"
            )
        
        return WebSDRResponse.model_validate(station)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving WebSDR station: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.put("/websdrs/{station_id}", response_model=WebSDRResponse)
async def update_websdr(station_id: str, request: WebSDRUpdateRequest):
    """
    Update an existing WebSDR station.
    
    Args:
        station_id: UUID of the WebSDR station to update
        request: Fields to update (only provided fields will be updated)
    
    Returns:
        Updated WebSDR station details
    
    Raises:
        HTTPException 404: If station not found
        HTTPException 400: If update validation fails (e.g., duplicate name)
        HTTPException 500: If database error occurs
    """
    try:
        db_manager = get_db_manager()
        
        # Only include fields that were actually provided
        update_data = request.model_dump(exclude_unset=True)
        
        if not update_data:
            raise HTTPException(
                status_code=400,
                detail="No fields provided for update"
            )
        
        station = db_manager.update_websdr(station_id, **update_data)
        
        if not station:
            raise HTTPException(
                status_code=404,
                detail=f"WebSDR station with ID {station_id} not found or update failed"
            )
        
        logger.info(f"Updated WebSDR station: {station.name} (ID: {station.id})")
        return WebSDRResponse.model_validate(station)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating WebSDR station: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/websdrs/{station_id}", status_code=204)
async def delete_websdr(station_id: str, hard_delete: bool = False):
    """
    Delete a WebSDR station (soft delete by default).
    
    Args:
        station_id: UUID of the WebSDR station to delete
        hard_delete: If True, permanently delete; if False, set is_active=False
    
    Returns:
        No content on success
    
    Raises:
        HTTPException 404: If station not found
        HTTPException 500: If database error occurs
    """
    try:
        db_manager = get_db_manager()
        
        success = db_manager.delete_websdr(station_id, soft_delete=not hard_delete)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"WebSDR station with ID {station_id} not found"
            )
        
        action = "Hard deleted" if hard_delete else "Soft deleted"
        logger.info(f"{action} WebSDR station: {station_id}")
        
        return None  # 204 No Content
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting WebSDR station: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/websdrs-all", response_model=List[WebSDRResponse])
async def list_all_websdrs(include_inactive: bool = False):
    """
    List all WebSDR stations with full details.
    
    Args:
        include_inactive: If True, include inactive stations; if False, only active
    
    Returns:
        List of WebSDR station details
    """
    try:
        db_manager = get_db_manager()
        
        if include_inactive:
            stations = db_manager.get_all_websdrs()
        else:
            stations = db_manager.get_active_websdrs()
        
        return [WebSDRResponse.model_validate(station) for station in stations]
    
    except Exception as e:
        logger.exception(f"Error listing WebSDR stations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

