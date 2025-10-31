"""Import/Export API endpoints for Heimdall SDR.

Provides endpoints to export system state to .heimdall files and import them back.
"""

import logging
import json
from typing import Optional, List
from fastapi import APIRouter, HTTPException
import asyncpg

from ..models.import_export import (
    ExportRequest,
    ImportRequest,
    ExportResponse,
    ImportResponse,
    MetadataResponse,
    HeimdallFile,
    ExportMetadata,
    ExportSections,
    SectionSizes,
    ExportedSource,
    ExportedWebSDR,
    ExportedSession,
    UserSettings,
)
from ..db import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/import-export", tags=["import-export"])


@router.get("/export/metadata", response_model=MetadataResponse)
async def get_export_metadata():
    """
    Get metadata about available data for export.
    
    Returns counts of sources, WebSDRs, sessions, and models.
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        # Count sources
        sources_count = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.known_sources"
        )
        
        # Count WebSDRs
        websdrs_count = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.websdr_stations"
        )
        
        # Count sessions
        sessions_count = await conn.fetchval(
            "SELECT COUNT(*) FROM heimdall.recording_sessions"
        )
        
        # Estimate section sizes (rough estimates)
        settings_size = 256  # Fixed size for settings
        sources_size = sources_count * 500 if sources_count else 0
        websdrs_size = websdrs_count * 400 if websdrs_count else 0
        sessions_size = sessions_count * 600 if sessions_count else 0
        
        return MetadataResponse(
            sources_count=sources_count or 0,
            websdrs_count=websdrs_count or 0,
            sessions_count=sessions_count or 0,
            has_training_model=False,  # TODO: Check for models
            has_inference_model=False,  # TODO: Check for models
            estimated_sizes=SectionSizes(
                settings=settings_size,
                sources=sources_size,
                websdrs=websdrs_size,
                sessions=sessions_size,
                training_model=0,
                inference_model=0,
            ),
        )


@router.post("/export", response_model=ExportResponse)
async def export_data(request: ExportRequest):
    """
    Export selected data sections to a .heimdall file format.
    
    Allows selective export of sources, WebSDRs, sessions, and models.
    """
    pool = get_pool()
    sections = ExportSections()
    
    async with pool.acquire() as conn:
        # Export settings if requested
        if request.include_settings:
            sections.settings = UserSettings()  # Default settings for now
        
        # Export sources if requested
        if request.include_sources:
            rows = await conn.fetch("""
                SELECT 
                    id, name, description, frequency_hz, latitude, longitude,
                    power_dbm, source_type, is_validated, error_margin_meters,
                    created_at, updated_at
                FROM heimdall.known_sources
                ORDER BY created_at DESC
            """)
            
            sources = []
            for row in rows:
                sources.append(ExportedSource(
                    id=str(row["id"]),
                    name=row["name"],
                    description=row["description"],
                    frequency_hz=row["frequency_hz"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    power_dbm=row["power_dbm"],
                    source_type=row["source_type"],
                    is_validated=row["is_validated"],
                    error_margin_meters=row["error_margin_meters"],
                    created_at=row["created_at"].isoformat(),
                    updated_at=row["updated_at"].isoformat(),
                ))
            sections.sources = sources
        
        # Export WebSDRs if requested
        if request.include_websdrs:
            rows = await conn.fetch("""
                SELECT 
                    id, name, url, location_description, latitude, longitude,
                    altitude_asl, country, admin_email as operator, is_active,
                    timeout_seconds, retry_count, created_at, updated_at
                FROM heimdall.websdr_stations
                ORDER BY created_at DESC
            """)
            
            websdrs = []
            for row in rows:
                websdrs.append(ExportedWebSDR(
                    id=str(row["id"]),
                    name=row["name"],
                    url=row["url"],
                    location_description=row["location_description"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    altitude_meters=float(row["altitude_asl"]) if row["altitude_asl"] else None,
                    country=row["country"],
                    operator=row["operator"],
                    is_active=row["is_active"],
                    timeout_seconds=row["timeout_seconds"],
                    retry_count=row["retry_count"],
                    created_at=row["created_at"].isoformat(),
                    updated_at=row["updated_at"].isoformat(),
                ))
            sections.websdrs = websdrs
        
        # Export sessions if requested
        if request.include_sessions:
            rows = await conn.fetch("""
                SELECT 
                    id, known_source_id, session_name, session_start,
                    session_end, duration_seconds, celery_task_id,
                    status, approval_status, notes, created_at, updated_at
                FROM heimdall.recording_sessions
                ORDER BY session_start DESC
            """)
            
            sessions = []
            for row in rows:
                # Count measurements for this session
                measurements_count = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM heimdall.measurements m
                    WHERE m.created_at >= $1
                    AND ($2 IS NULL OR m.created_at <= $2)
                """, row["session_start"], row["session_end"])
                
                sessions.append(ExportedSession(
                    id=str(row["id"]),
                    known_source_id=str(row["known_source_id"]),
                    session_name=row["session_name"],
                    session_start=row["session_start"].isoformat(),
                    session_end=row["session_end"].isoformat() if row["session_end"] else None,
                    duration_seconds=row["duration_seconds"],
                    celery_task_id=row["celery_task_id"],
                    status=row["status"],
                    approval_status=row["approval_status"],
                    notes=row["notes"],
                    created_at=row["created_at"].isoformat(),
                    updated_at=row["updated_at"].isoformat(),
                    measurements_count=measurements_count or 0,
                ))
            sections.sessions = sessions
        
        # TODO: Export models if requested
        # if request.include_training_model:
        #     sections.training_model = ...
        # if request.include_inference_model:
        #     sections.inference_model = ...
    
    # Calculate section sizes
    section_sizes = SectionSizes()
    if sections.settings:
        section_sizes.settings = len(json.dumps(sections.settings.model_dump()))
    if sections.sources:
        section_sizes.sources = len(json.dumps([s.model_dump() for s in sections.sources]))
    if sections.websdrs:
        section_sizes.websdrs = len(json.dumps([w.model_dump() for w in sections.websdrs]))
    if sections.sessions:
        section_sizes.sessions = len(json.dumps([s.model_dump() for s in sections.sessions]))
    
    # Create metadata
    metadata = ExportMetadata(
        creator=request.creator,
        description=request.description,
        section_sizes=section_sizes,
    )
    
    # Create the complete file
    heimdall_file = HeimdallFile(
        metadata=metadata,
        sections=sections,
    )
    
    # Calculate total size
    file_json = heimdall_file.model_dump_json()
    total_size = len(file_json)
    
    return ExportResponse(
        file=heimdall_file,
        size_bytes=total_size,
    )


@router.post("/import", response_model=ImportResponse)
async def import_data(request: ImportRequest):
    """
    Import data from a .heimdall file format.
    
    Supports selective import with conflict resolution.
    """
    pool = get_pool()
    imported_counts = {
        "settings": 0,
        "sources": 0,
        "websdrs": 0,
        "sessions": 0,
        "training_model": 0,
        "inference_model": 0,
    }
    errors = []
    
    async with pool.acquire() as conn:
        # Start transaction
        async with conn.transaction():
            try:
                # Import settings
                if request.import_settings and request.heimdall_file.sections.settings:
                    # TODO: Implement settings import
                    imported_counts["settings"] = 1
                
                # Import sources
                if request.import_sources and request.heimdall_file.sections.sources:
                    for source in request.heimdall_file.sections.sources:
                        try:
                            if request.overwrite_existing:
                                # Try to update existing or insert new
                                await conn.execute("""
                                    INSERT INTO heimdall.known_sources 
                                    (id, name, description, frequency_hz, latitude, longitude,
                                     power_dbm, source_type, is_validated, error_margin_meters,
                                     created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                                    ON CONFLICT (id) DO UPDATE SET
                                        name = EXCLUDED.name,
                                        description = EXCLUDED.description,
                                        frequency_hz = EXCLUDED.frequency_hz,
                                        latitude = EXCLUDED.latitude,
                                        longitude = EXCLUDED.longitude,
                                        power_dbm = EXCLUDED.power_dbm,
                                        source_type = EXCLUDED.source_type,
                                        is_validated = EXCLUDED.is_validated,
                                        error_margin_meters = EXCLUDED.error_margin_meters,
                                        updated_at = NOW()
                                """, 
                                    source.id, source.name, source.description,
                                    source.frequency_hz, source.latitude, source.longitude,
                                    source.power_dbm, source.source_type, source.is_validated,
                                    source.error_margin_meters, source.created_at, source.updated_at
                                )
                            else:
                                # Insert only if doesn't exist
                                await conn.execute("""
                                    INSERT INTO heimdall.known_sources 
                                    (id, name, description, frequency_hz, latitude, longitude,
                                     power_dbm, source_type, is_validated, error_margin_meters,
                                     created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                                    ON CONFLICT (id) DO NOTHING
                                """,
                                    source.id, source.name, source.description,
                                    source.frequency_hz, source.latitude, source.longitude,
                                    source.power_dbm, source.source_type, source.is_validated,
                                    source.error_margin_meters, source.created_at, source.updated_at
                                )
                            imported_counts["sources"] += 1
                        except Exception as e:
                            errors.append(f"Error importing source {source.name}: {str(e)}")
                
                # Import WebSDRs
                if request.import_websdrs and request.heimdall_file.sections.websdrs:
                    for websdr in request.heimdall_file.sections.websdrs:
                        try:
                            if request.overwrite_existing:
                                await conn.execute("""
                                    INSERT INTO heimdall.websdr_stations 
                                    (id, name, url, location_description, latitude, longitude,
                                     altitude_asl, country, admin_email, is_active,
                                     timeout_seconds, retry_count, created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                                    ON CONFLICT (id) DO UPDATE SET
                                        name = EXCLUDED.name,
                                        url = EXCLUDED.url,
                                        location_description = EXCLUDED.location_description,
                                        latitude = EXCLUDED.latitude,
                                        longitude = EXCLUDED.longitude,
                                        altitude_asl = EXCLUDED.altitude_asl,
                                        country = EXCLUDED.country,
                                        admin_email = EXCLUDED.admin_email,
                                        is_active = EXCLUDED.is_active,
                                        timeout_seconds = EXCLUDED.timeout_seconds,
                                        retry_count = EXCLUDED.retry_count,
                                        updated_at = NOW()
                                """,
                                    websdr.id, websdr.name, websdr.url,
                                    websdr.location_description, websdr.latitude, websdr.longitude,
                                    websdr.altitude_meters, websdr.country, websdr.operator,
                                    websdr.is_active, websdr.timeout_seconds, websdr.retry_count,
                                    websdr.created_at, websdr.updated_at
                                )
                            else:
                                await conn.execute("""
                                    INSERT INTO heimdall.websdr_stations 
                                    (id, name, url, location_description, latitude, longitude,
                                     altitude_asl, country, admin_email, is_active,
                                     timeout_seconds, retry_count, created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                                    ON CONFLICT (id) DO NOTHING
                                """,
                                    websdr.id, websdr.name, websdr.url,
                                    websdr.location_description, websdr.latitude, websdr.longitude,
                                    websdr.altitude_meters, websdr.country, websdr.operator,
                                    websdr.is_active, websdr.timeout_seconds, websdr.retry_count,
                                    websdr.created_at, websdr.updated_at
                                )
                            imported_counts["websdrs"] += 1
                        except Exception as e:
                            errors.append(f"Error importing WebSDR {websdr.name}: {str(e)}")
                
                # Import sessions
                if request.import_sessions and request.heimdall_file.sections.sessions:
                    for session in request.heimdall_file.sections.sessions:
                        try:
                            if request.overwrite_existing:
                                await conn.execute("""
                                    INSERT INTO heimdall.recording_sessions 
                                    (id, known_source_id, session_name, session_start,
                                     session_end, duration_seconds, celery_task_id,
                                     status, approval_status, notes, created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                                    ON CONFLICT (id) DO UPDATE SET
                                        known_source_id = EXCLUDED.known_source_id,
                                        session_name = EXCLUDED.session_name,
                                        session_start = EXCLUDED.session_start,
                                        session_end = EXCLUDED.session_end,
                                        duration_seconds = EXCLUDED.duration_seconds,
                                        celery_task_id = EXCLUDED.celery_task_id,
                                        status = EXCLUDED.status,
                                        approval_status = EXCLUDED.approval_status,
                                        notes = EXCLUDED.notes,
                                        updated_at = NOW()
                                """,
                                    session.id, session.known_source_id, session.session_name,
                                    session.session_start, session.session_end, session.duration_seconds,
                                    session.celery_task_id, session.status, session.approval_status,
                                    session.notes, session.created_at, session.updated_at
                                )
                            else:
                                await conn.execute("""
                                    INSERT INTO heimdall.recording_sessions 
                                    (id, known_source_id, session_name, session_start,
                                     session_end, duration_seconds, celery_task_id,
                                     status, approval_status, notes, created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                                    ON CONFLICT (id) DO NOTHING
                                """,
                                    session.id, session.known_source_id, session.session_name,
                                    session.session_start, session.session_end, session.duration_seconds,
                                    session.celery_task_id, session.status, session.approval_status,
                                    session.notes, session.created_at, session.updated_at
                                )
                            imported_counts["sessions"] += 1
                        except Exception as e:
                            errors.append(f"Error importing session {session.session_name}: {str(e)}")
                
                # TODO: Import models
                # if request.import_training_model and request.heimdall_file.sections.training_model:
                #     ...
                # if request.import_inference_model and request.heimdall_file.sections.inference_model:
                #     ...
                
            except Exception as e:
                logger.error(f"Import failed: {e}")
                errors.append(f"Transaction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
    
    success = len(errors) == 0 or sum(imported_counts.values()) > 0
    message = "Import completed successfully" if success else "Import completed with errors"
    
    return ImportResponse(
        success=success,
        message=message,
        imported_counts=imported_counts,
        errors=errors,
    )
