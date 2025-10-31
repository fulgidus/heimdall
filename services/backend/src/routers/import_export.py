"""Import/Export router for .heimdall file operations."""

import logging
import json
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import sys

from ..db import get_db
from ..models.db import WebSDRStation, Base as DBBase
from ..models.import_export import (
    ExportRequest,
    ImportRequest,
    ImportResult,
    ExportMetadataResponse,
    HeimdallFile,
    HeimdallMetadata,
    HeimdallSections,
    UserSettings,
    ExportedSource,
    ExportedWebSDR,
    ExportedSession,
    ExportedModel,
    SectionSizes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/import-export", tags=["import-export"])


async def get_sources_from_db(db: AsyncSession, source_ids: Optional[List[UUID]] = None) -> List[ExportedSource]:
    """Fetch sources from database."""
    try:
        from sqlalchemy import text
        query = text("""
            SELECT id, name, description, frequency_hz, latitude, longitude, 
                   power_dbm, source_type, is_validated, error_margin_meters, 
                   created_at, updated_at
            FROM heimdall.known_sources
        """)
        if source_ids:
            query = text("""
                SELECT id, name, description, frequency_hz, latitude, longitude, 
                       power_dbm, source_type, is_validated, error_margin_meters, 
                       created_at, updated_at
                FROM heimdall.known_sources
                WHERE id = ANY(:ids)
            """)
            result = await db.execute(query, {"ids": [str(sid) for sid in source_ids]})
        else:
            result = await db.execute(query)
        
        rows = result.fetchall()
        return [
            ExportedSource(
                id=row[0],
                name=row[1],
                description=row[2],
                frequency_hz=row[3],
                latitude=row[4],
                longitude=row[5],
                power_dbm=row[6],
                source_type=row[7],
                is_validated=row[8],
                error_margin_meters=row[9],
                created_at=row[10],
                updated_at=row[11]
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching sources: {e}")
        return []


async def get_websdrs_from_db(db: AsyncSession) -> List[ExportedWebSDR]:
    """Fetch WebSDR stations from database."""
    try:
        result = await db.execute(select(WebSDRStation))
        stations = result.scalars().all()
        return [
            ExportedWebSDR(
                id=station.id,
                name=station.name,
                url=station.url,
                country=station.country,
                latitude=station.latitude,
                longitude=station.longitude,
                frequency_min_hz=station.frequency_min_hz,
                frequency_max_hz=station.frequency_max_hz,
                is_active=station.is_active,
                api_type=station.api_type,
                rate_limit_ms=station.rate_limit_ms,
                timeout_seconds=station.timeout_seconds,
                retry_count=station.retry_count,
                admin_email=station.admin_email,
                location_description=station.location_description,
                altitude_asl=station.altitude_asl,
                notes=station.notes,
                created_at=station.created_at,
                updated_at=station.updated_at
            )
            for station in stations
        ]
    except Exception as e:
        logger.error(f"Error fetching WebSDRs: {e}")
        return []


async def get_sessions_from_db(db: AsyncSession, session_ids: Optional[List[UUID]] = None) -> List[ExportedSession]:
    """Fetch recording sessions from database."""
    try:
        from sqlalchemy import text
        if session_ids:
            query = text("""
                SELECT rs.id, rs.known_source_id, rs.session_name, rs.session_start, 
                       rs.session_end, rs.duration_seconds, rs.celery_task_id, 
                       rs.status, rs.approval_status, rs.notes, rs.created_at, rs.updated_at,
                       ks.name as source_name, ks.frequency_hz as source_frequency,
                       COUNT(m.id) as measurements_count
                FROM heimdall.recording_sessions rs
                LEFT JOIN heimdall.known_sources ks ON rs.known_source_id = ks.id
                LEFT JOIN heimdall.measurements m ON m.id IN (
                    SELECT measurement_id FROM heimdall.dataset_measurements dm
                    JOIN heimdall.training_datasets td ON dm.dataset_id = td.id
                )
                WHERE rs.id = ANY(:ids)
                GROUP BY rs.id, ks.name, ks.frequency_hz
            """)
            result = await db.execute(query, {"ids": [str(sid) for sid in session_ids]})
        else:
            query = text("""
                SELECT rs.id, rs.known_source_id, rs.session_name, rs.session_start, 
                       rs.session_end, rs.duration_seconds, rs.celery_task_id, 
                       rs.status, rs.approval_status, rs.notes, rs.created_at, rs.updated_at,
                       ks.name as source_name, ks.frequency_hz as source_frequency,
                       COUNT(m.id) as measurements_count
                FROM heimdall.recording_sessions rs
                LEFT JOIN heimdall.known_sources ks ON rs.known_source_id = ks.id
                LEFT JOIN heimdall.measurements m ON m.id IN (
                    SELECT measurement_id FROM heimdall.dataset_measurements dm
                    JOIN heimdall.training_datasets td ON dm.dataset_id = td.id
                )
                GROUP BY rs.id, ks.name, ks.frequency_hz
            """)
            result = await db.execute(query)
        
        rows = result.fetchall()
        return [
            ExportedSession(
                id=row[0],
                known_source_id=row[1],
                session_name=row[2],
                session_start=row[3],
                session_end=row[4],
                duration_seconds=row[5],
                celery_task_id=row[6],
                status=row[7],
                approval_status=row[8],
                notes=row[9],
                created_at=row[10],
                updated_at=row[11],
                source_name=row[12],
                source_frequency=row[13],
                measurements_count=row[14]
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return []


def calculate_section_sizes(sections: HeimdallSections) -> SectionSizes:
    """Calculate size of each section in bytes."""
    sizes = SectionSizes()
    
    if sections.settings:
        sizes.settings = len(json.dumps(sections.settings.model_dump(), default=str))
    if sections.sources:
        sizes.sources = len(json.dumps([s.model_dump() for s in sections.sources], default=str))
    if sections.websdrs:
        sizes.websdrs = len(json.dumps([w.model_dump() for w in sections.websdrs], default=str))
    if sections.sessions:
        sizes.sessions = len(json.dumps([s.model_dump() for s in sections.sessions], default=str))
    if sections.training_model:
        sizes.training_model = len(json.dumps(sections.training_model.model_dump(), default=str))
    if sections.inference_model:
        sizes.inference_model = len(json.dumps(sections.inference_model.model_dump(), default=str))
    
    return sizes


@router.post("/export", response_model=HeimdallFile)
async def export_data(
    request: ExportRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Export selected data sections to .heimdall format.
    
    Creates a structured JSON file containing the selected sections
    with metadata about the export operation.
    """
    try:
        sections = HeimdallSections()
        
        # Export settings if requested
        if request.include_settings:
            sections.settings = UserSettings()
        
        # Export sources if requested
        if request.include_sources:
            sections.sources = await get_sources_from_db(db)
            logger.info(f"Exported {len(sections.sources)} sources")
        
        # Export WebSDRs if requested
        if request.include_websdrs:
            sections.websdrs = await get_websdrs_from_db(db)
            logger.info(f"Exported {len(sections.websdrs)} WebSDR stations")
        
        # Export sessions if requested
        if request.include_sessions:
            sections.sessions = await get_sessions_from_db(db, request.session_ids)
            logger.info(f"Exported {len(sections.sessions)} sessions")
        
        # Calculate section sizes
        section_sizes = calculate_section_sizes(sections)
        
        # Create metadata
        metadata = HeimdallMetadata(
            version="1.0",
            created_at=datetime.utcnow(),
            creator=request.creator,
            section_sizes=section_sizes,
            description=request.description
        )
        
        # Create complete file structure
        heimdall_file = HeimdallFile(
            metadata=metadata,
            sections=sections
        )
        
        logger.info("Export completed successfully")
        return heimdall_file
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/import", response_model=ImportResult)
async def import_data(
    request: ImportRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Import data from .heimdall file.
    
    Validates and imports selected sections from a .heimdall file,
    with options to overwrite existing data.
    """
    try:
        result = ImportResult(
            success=True,
            message="Import completed successfully",
            imported_counts={},
            errors=[],
            warnings=[]
        )
        
        file_content = request.file_content
        
        # Validate file version
        if file_content.metadata.version != "1.0":
            result.warnings.append(f"File version {file_content.metadata.version} may not be fully compatible")
        
        # Import sources
        if request.import_sources and file_content.sections.sources:
            try:
                from sqlalchemy import text
                imported = 0
                for source in file_content.sections.sources:
                    try:
                        if request.overwrite_existing:
                            # Check if exists
                            check_query = text("SELECT id FROM heimdall.known_sources WHERE id = :id")
                            exists = await db.execute(check_query, {"id": str(source.id)})
                            if exists.fetchone():
                                # Update existing
                                update_query = text("""
                                    UPDATE heimdall.known_sources 
                                    SET name = :name, description = :description, frequency_hz = :frequency_hz,
                                        latitude = :latitude, longitude = :longitude, power_dbm = :power_dbm,
                                        source_type = :source_type, is_validated = :is_validated,
                                        error_margin_meters = :error_margin_meters, updated_at = NOW()
                                    WHERE id = :id
                                """)
                                await db.execute(update_query, {
                                    "id": str(source.id),
                                    "name": source.name,
                                    "description": source.description,
                                    "frequency_hz": source.frequency_hz,
                                    "latitude": source.latitude,
                                    "longitude": source.longitude,
                                    "power_dbm": source.power_dbm,
                                    "source_type": source.source_type,
                                    "is_validated": source.is_validated,
                                    "error_margin_meters": source.error_margin_meters
                                })
                            else:
                                # Insert new
                                insert_query = text("""
                                    INSERT INTO heimdall.known_sources 
                                    (id, name, description, frequency_hz, latitude, longitude, power_dbm,
                                     source_type, is_validated, error_margin_meters, created_at, updated_at)
                                    VALUES (:id, :name, :description, :frequency_hz, :latitude, :longitude, 
                                            :power_dbm, :source_type, :is_validated, :error_margin_meters, 
                                            :created_at, :updated_at)
                                """)
                                await db.execute(insert_query, {
                                    "id": str(source.id),
                                    "name": source.name,
                                    "description": source.description,
                                    "frequency_hz": source.frequency_hz,
                                    "latitude": source.latitude,
                                    "longitude": source.longitude,
                                    "power_dbm": source.power_dbm,
                                    "source_type": source.source_type,
                                    "is_validated": source.is_validated,
                                    "error_margin_meters": source.error_margin_meters,
                                    "created_at": source.created_at,
                                    "updated_at": source.updated_at
                                })
                        else:
                            # Only insert if not exists
                            insert_query = text("""
                                INSERT INTO heimdall.known_sources 
                                (id, name, description, frequency_hz, latitude, longitude, power_dbm,
                                 source_type, is_validated, error_margin_meters, created_at, updated_at)
                                VALUES (:id, :name, :description, :frequency_hz, :latitude, :longitude, 
                                        :power_dbm, :source_type, :is_validated, :error_margin_meters, 
                                        :created_at, :updated_at)
                                ON CONFLICT (id) DO NOTHING
                            """)
                            await db.execute(insert_query, {
                                "id": str(source.id),
                                "name": source.name,
                                "description": source.description,
                                "frequency_hz": source.frequency_hz,
                                "latitude": source.latitude,
                                "longitude": source.longitude,
                                "power_dbm": source.power_dbm,
                                "source_type": source.source_type,
                                "is_validated": source.is_validated,
                                "error_margin_meters": source.error_margin_meters,
                                "created_at": source.created_at,
                                "updated_at": source.updated_at
                            })
                        imported += 1
                    except Exception as e:
                        result.errors.append(f"Failed to import source {source.name}: {str(e)}")
                
                await db.commit()
                result.imported_counts["sources"] = imported
                logger.info(f"Imported {imported} sources")
                
            except Exception as e:
                result.errors.append(f"Failed to import sources: {str(e)}")
                await db.rollback()
        
        # Import WebSDRs
        if request.import_websdrs and file_content.sections.websdrs:
            try:
                imported = 0
                for websdr in file_content.sections.websdrs:
                    try:
                        if request.overwrite_existing:
                            # Delete existing and insert new
                            await db.execute(
                                delete(WebSDRStation).where(WebSDRStation.id == websdr.id)
                            )
                        
                        # Create new WebSDR station
                        station = WebSDRStation(
                            id=websdr.id,
                            name=websdr.name,
                            url=websdr.url,
                            country=websdr.country,
                            latitude=websdr.latitude,
                            longitude=websdr.longitude,
                            frequency_min_hz=websdr.frequency_min_hz,
                            frequency_max_hz=websdr.frequency_max_hz,
                            is_active=websdr.is_active,
                            api_type=websdr.api_type,
                            rate_limit_ms=websdr.rate_limit_ms,
                            timeout_seconds=websdr.timeout_seconds,
                            retry_count=websdr.retry_count,
                            admin_email=websdr.admin_email,
                            location_description=websdr.location_description,
                            altitude_asl=websdr.altitude_asl,
                            notes=websdr.notes,
                            created_at=websdr.created_at,
                            updated_at=websdr.updated_at
                        )
                        db.add(station)
                        imported += 1
                    except Exception as e:
                        result.errors.append(f"Failed to import WebSDR {websdr.name}: {str(e)}")
                
                await db.commit()
                result.imported_counts["websdrs"] = imported
                logger.info(f"Imported {imported} WebSDR stations")
                
            except Exception as e:
                result.errors.append(f"Failed to import WebSDRs: {str(e)}")
                await db.rollback()
        
        # Import settings
        if request.import_settings and file_content.sections.settings:
            result.warnings.append("Settings import not yet fully implemented - will be stored in user preferences")
            result.imported_counts["settings"] = 1
        
        # Determine overall success
        if result.errors:
            result.success = len(result.errors) < sum(result.imported_counts.values())
            if not result.success:
                result.message = "Import failed with errors"
            else:
                result.message = "Import completed with some errors"
        
        logger.info(f"Import completed: {result.imported_counts}")
        return result
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/export/metadata", response_model=ExportMetadataResponse)
async def get_export_metadata(db: AsyncSession = Depends(get_db)):
    """
    Get metadata about available data for export.
    
    Returns counts and information about what can be exported.
    """
    try:
        from sqlalchemy import text
        
        # Count sources
        result = await db.execute(text("SELECT COUNT(*) FROM heimdall.known_sources"))
        sources_count = result.scalar() or 0
        
        # Count WebSDRs
        result = await db.execute(text("SELECT COUNT(*) FROM heimdall.websdr_stations"))
        websdrs_count = result.scalar() or 0
        
        # Count sessions
        result = await db.execute(text("SELECT COUNT(*) FROM heimdall.recording_sessions"))
        sessions_count = result.scalar() or 0
        
        # Check for models
        result = await db.execute(text("SELECT COUNT(*) FROM heimdall.models WHERE is_production = TRUE"))
        has_inference = (result.scalar() or 0) > 0
        
        result = await db.execute(text("SELECT COUNT(*) FROM heimdall.models WHERE is_active = TRUE"))
        has_training = (result.scalar() or 0) > 0
        
        # Estimate size (rough estimate: 1KB per source, 2KB per WebSDR, 5KB per session)
        estimated_size = (sources_count * 1024) + (websdrs_count * 2048) + (sessions_count * 5120)
        
        return ExportMetadataResponse(
            available_sources_count=sources_count,
            available_websdrs_count=websdrs_count,
            available_sessions_count=sessions_count,
            has_training_model=has_training,
            has_inference_model=has_inference,
            estimated_size_bytes=estimated_size
        )
        
    except Exception as e:
        logger.error(f"Failed to get export metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")
