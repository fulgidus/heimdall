"""Import/Export API endpoints for Heimdall SDR.

Provides endpoints to export system state to .heimdall files and import them back.
"""

import base64
import json
import logging

from fastapi import APIRouter, HTTPException

from ..db import get_pool
from ..models.import_export import (
    ExportedModel,
    ExportedSampleSet,
    ExportedSession,
    ExportedSource,
    ExportedWebSDR,
    ExportMetadata,
    ExportRequest,
    ExportResponse,
    ExportSections,
    HeimdallFile,
    ImportRequest,
    ImportResponse,
    MetadataResponse,
    SectionSizes,
    UserSettings,
)
from ..storage.minio_client import MinIOClient
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/import-export", tags=["import-export"])


@router.get("/export/metadata", response_model=MetadataResponse)
async def get_export_metadata():
    """
    Get metadata about available data for export.

    Returns counts and lists of sources, WebSDRs, sessions, sample sets, and models.
    """
    pool = get_pool()

    async with pool.acquire() as conn:
        # Count sources
        sources_count = await conn.fetchval("SELECT COUNT(*) FROM heimdall.known_sources")

        # Count WebSDRs
        websdrs_count = await conn.fetchval("SELECT COUNT(*) FROM heimdall.websdr_stations")

        # Count sessions
        sessions_count = await conn.fetchval("SELECT COUNT(*) FROM heimdall.recording_sessions")

        # Get available sample sets
        sample_sets_rows = await conn.fetch(
            """
            SELECT 
                sd.id, 
                sd.name, 
                COALESCE(COUNT(mf.recording_session_id), 0) as num_samples,
                sd.created_at
            FROM heimdall.synthetic_datasets sd
            LEFT JOIN heimdall.measurement_features mf ON mf.dataset_id = sd.id
            GROUP BY sd.id, sd.name, sd.created_at
            ORDER BY sd.created_at DESC
        """
        )
        
        sample_sets = [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "num_samples": row["num_samples"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in sample_sets_rows
        ]

        # Get available models
        models_rows = await conn.fetch(
            """
            SELECT 
                id, 
                model_name, 
                COALESCE(version, 1) as version,
                created_at,
                onnx_model_location
            FROM heimdall.models
            ORDER BY created_at DESC
        """
        )
        
        models = [
            {
                "id": str(row["id"]),
                "model_name": row["model_name"],
                "version": row["version"],
                "created_at": row["created_at"].isoformat(),
                "has_onnx": row["onnx_model_location"] is not None,
            }
            for row in models_rows
        ]

        # Estimate section sizes (rough estimates)
        settings_size = 256
        sources_size = sources_count * 500 if sources_count else 0
        websdrs_size = websdrs_count * 400 if websdrs_count else 0
        sessions_size = sessions_count * 600 if sessions_count else 0
        # Sample size calculation:
        # - 7 receivers × 593 bytes (17 features × 4 stats each × 8 bytes + metadata)
        # - Ground truth + sample metadata: ~136 bytes
        # - JSONB overhead (PostgreSQL): ~1.3x
        # Total: ~5600 bytes per sample
        sample_sets_size = sum(s["num_samples"] * 5600 for s in sample_sets)
        models_size = len(models) * 50000000  # ~50MB per model (rough estimate)

        return MetadataResponse(
            sources_count=sources_count or 0,
            websdrs_count=websdrs_count or 0,
            sessions_count=sessions_count or 0,
            sample_sets=sample_sets,
            models=models,
            estimated_sizes=SectionSizes(
                settings=settings_size,
                sources=sources_size,
                websdrs=websdrs_size,
                sessions=sessions_size,
                sample_sets=sample_sets_size,
                models=models_size,
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
            rows = await conn.fetch(
                """
                SELECT
                    id, name, description, frequency_hz, latitude, longitude,
                    power_dbm, source_type, is_validated, error_margin_meters,
                    created_at, updated_at
                FROM heimdall.known_sources
                ORDER BY created_at DESC
            """
            )

            sources = []
            for row in rows:
                sources.append(
                    ExportedSource(
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
                    )
                )
            sections.sources = sources

        # Export WebSDRs if requested
        if request.include_websdrs:
            rows = await conn.fetch(
                """
                SELECT
                    id, name, url, location_description, latitude, longitude,
                    altitude_asl, country, admin_email as operator, is_active,
                    timeout_seconds, retry_count, created_at, updated_at
                FROM heimdall.websdr_stations
                ORDER BY created_at DESC
            """
            )

            websdrs = []
            for row in rows:
                websdrs.append(
                    ExportedWebSDR(
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
                    )
                )
            sections.websdrs = websdrs

        # Export sessions if requested
        if request.include_sessions:
            rows = await conn.fetch(
                """
                SELECT
                    id, known_source_id, session_name, session_start,
                    session_end, duration_seconds, celery_task_id,
                    status, approval_status, notes, created_at, updated_at
                FROM heimdall.recording_sessions
                ORDER BY session_start DESC
            """
            )

            sessions = []
            for row in rows:
                # Count measurements for this session
                measurements_count = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM heimdall.measurements m
                    WHERE m.created_at >= $1
                    AND ($2 IS NULL OR m.created_at <= $2)
                """,
                    row["session_start"],
                    row["session_end"],
                )

                sessions.append(
                    ExportedSession(
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
                    )
                )
            sections.sessions = sessions

        # Export sample sets if requested
        if request.sample_set_ids:
            sample_sets = []
            for dataset_id in request.sample_set_ids:
                # Get dataset metadata
                dataset_row = await conn.fetchrow(
                    """
                    SELECT 
                        sd.id, sd.name, sd.description, sd.config, 
                        sd.quality_metrics, sd.created_at,
                        COALESCE(COUNT(mf.recording_session_id), 0) as num_samples
                    FROM heimdall.synthetic_datasets sd
                    LEFT JOIN heimdall.measurement_features mf ON mf.dataset_id = sd.id
                    WHERE sd.id = $1
                    GROUP BY sd.id, sd.name, sd.description, sd.config, 
                             sd.quality_metrics, sd.created_at
                    """,
                    dataset_id,
                )
                
                if not dataset_row:
                    logger.warning(f"Sample set {dataset_id} not found, skipping")
                    continue

                # Get sample data (limit to reasonable size for export)
                samples_rows = await conn.fetch(
                    """
                    SELECT 
                        tx_lat, tx_lon, tx_power_dbm, frequency_hz,
                        extraction_metadata, mean_snr_db, overall_confidence, gdop
                    FROM heimdall.measurement_features
                    WHERE dataset_id = $1
                    LIMIT 10000
                    """,
                    dataset_id,
                )
                
                samples = []
                for sample_row in samples_rows:
                    samples.append({
                        "tx_lat": float(sample_row["tx_lat"]),
                        "tx_lon": float(sample_row["tx_lon"]),
                        "tx_power_dbm": float(sample_row["tx_power_dbm"]) if sample_row["tx_power_dbm"] else None,
                        "frequency_hz": float(sample_row["frequency_hz"]) if sample_row["frequency_hz"] else None,
                        "mean_snr_db": float(sample_row["mean_snr_db"]) if sample_row["mean_snr_db"] else None,
                        "overall_confidence": float(sample_row["overall_confidence"]) if sample_row["overall_confidence"] else None,
                        "gdop": float(sample_row["gdop"]) if sample_row["gdop"] else None,
                        "extraction_metadata": sample_row["extraction_metadata"],
                    })

                sample_sets.append(
                    ExportedSampleSet(
                        id=str(dataset_row["id"]),
                        name=dataset_row["name"],
                        description=dataset_row["description"],
                        num_samples=dataset_row["num_samples"],
                        config=dataset_row["config"],
                        quality_metrics=dataset_row["quality_metrics"],
                        created_at=dataset_row["created_at"].isoformat(),
                        samples=samples[:10000],  # Limit to 10k samples max
                    )
                )
            
            sections.sample_sets = sample_sets if sample_sets else None

        # Export models if requested
        if request.model_ids:
            minio_client = MinIOClient(
                endpoint_url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name="heimdall-models",
            )
            
            models = []
            for model_id in request.model_ids:
                # Get model metadata
                model_row = await conn.fetchrow(
                    """
                    SELECT 
                        id, model_name, COALESCE(version, 1) as version, model_type,
                        onnx_model_location, accuracy_meters, hyperparameters,
                        training_metrics, created_at
                    FROM heimdall.models
                    WHERE id = $1
                    """,
                    model_id,
                )
                
                if not model_row:
                    logger.warning(f"Model {model_id} not found, skipping")
                    continue

                # Download ONNX model from MinIO if available
                onnx_base64 = None
                if model_row["onnx_model_location"]:
                    try:
                        # Parse s3://bucket/key format
                        s3_path = model_row["onnx_model_location"]
                        if s3_path.startswith("s3://"):
                            s3_path = s3_path[5:]
                        
                        parts = s3_path.split("/", 1)
                        bucket = parts[0]
                        key = parts[1] if len(parts) > 1 else ""
                        
                        # Download from MinIO
                        response = minio_client.s3_client.get_object(Bucket=bucket, Key=key)
                        onnx_bytes = response["Body"].read()
                        onnx_base64 = base64.b64encode(onnx_bytes).decode("utf-8")
                        logger.info(f"Downloaded ONNX model from {s3_path} ({len(onnx_bytes)} bytes)")
                    except Exception as e:
                        logger.error(f"Failed to download ONNX model for {model_id}: {e}")

                models.append(
                    ExportedModel(
                        id=str(model_row["id"]),
                        model_name=model_row["model_name"],
                        version=model_row["version"],
                        model_type=model_row["model_type"],
                        created_at=model_row["created_at"].isoformat(),
                        onnx_model_base64=onnx_base64,
                        accuracy_meters=model_row["accuracy_meters"],
                        hyperparameters=model_row["hyperparameters"],
                        training_metrics=model_row["training_metrics"],
                    )
                )
            
            sections.models = models if models else None

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
    if sections.sample_sets:
        section_sizes.sample_sets = len(json.dumps([s.model_dump() for s in sections.sample_sets]))
    if sections.models:
        section_sizes.models = len(json.dumps([m.model_dump() for m in sections.models]))

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
        "sample_sets": 0,
        "models": 0,
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
                                await conn.execute(
                                    """
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
                                    source.id,
                                    source.name,
                                    source.description,
                                    source.frequency_hz,
                                    source.latitude,
                                    source.longitude,
                                    source.power_dbm,
                                    source.source_type,
                                    source.is_validated,
                                    source.error_margin_meters,
                                    source.created_at,
                                    source.updated_at,
                                )
                            else:
                                # Insert only if doesn't exist
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.known_sources
                                    (id, name, description, frequency_hz, latitude, longitude,
                                     power_dbm, source_type, is_validated, error_margin_meters,
                                     created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                                    ON CONFLICT (id) DO NOTHING
                                """,
                                    source.id,
                                    source.name,
                                    source.description,
                                    source.frequency_hz,
                                    source.latitude,
                                    source.longitude,
                                    source.power_dbm,
                                    source.source_type,
                                    source.is_validated,
                                    source.error_margin_meters,
                                    source.created_at,
                                    source.updated_at,
                                )
                            imported_counts["sources"] += 1
                        except Exception as e:
                            errors.append(f"Error importing source {source.name}: {str(e)}")

                # Import WebSDRs
                if request.import_websdrs and request.heimdall_file.sections.websdrs:
                    for websdr in request.heimdall_file.sections.websdrs:
                        try:
                            if request.overwrite_existing:
                                await conn.execute(
                                    """
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
                                    websdr.id,
                                    websdr.name,
                                    websdr.url,
                                    websdr.location_description,
                                    websdr.latitude,
                                    websdr.longitude,
                                    websdr.altitude_meters,
                                    websdr.country,
                                    websdr.operator,
                                    websdr.is_active,
                                    websdr.timeout_seconds,
                                    websdr.retry_count,
                                    websdr.created_at,
                                    websdr.updated_at,
                                )
                            else:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.websdr_stations
                                    (id, name, url, location_description, latitude, longitude,
                                     altitude_asl, country, admin_email, is_active,
                                     timeout_seconds, retry_count, created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                                    ON CONFLICT (id) DO NOTHING
                                """,
                                    websdr.id,
                                    websdr.name,
                                    websdr.url,
                                    websdr.location_description,
                                    websdr.latitude,
                                    websdr.longitude,
                                    websdr.altitude_meters,
                                    websdr.country,
                                    websdr.operator,
                                    websdr.is_active,
                                    websdr.timeout_seconds,
                                    websdr.retry_count,
                                    websdr.created_at,
                                    websdr.updated_at,
                                )
                            imported_counts["websdrs"] += 1
                        except Exception as e:
                            errors.append(f"Error importing WebSDR {websdr.name}: {str(e)}")

                # Import sessions
                if request.import_sessions and request.heimdall_file.sections.sessions:
                    for session in request.heimdall_file.sections.sessions:
                        try:
                            if request.overwrite_existing:
                                await conn.execute(
                                    """
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
                                    session.id,
                                    session.known_source_id,
                                    session.session_name,
                                    session.session_start,
                                    session.session_end,
                                    session.duration_seconds,
                                    session.celery_task_id,
                                    session.status,
                                    session.approval_status,
                                    session.notes,
                                    session.created_at,
                                    session.updated_at,
                                )
                            else:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.recording_sessions
                                    (id, known_source_id, session_name, session_start,
                                     session_end, duration_seconds, celery_task_id,
                                     status, approval_status, notes, created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                                    ON CONFLICT (id) DO NOTHING
                                """,
                                    session.id,
                                    session.known_source_id,
                                    session.session_name,
                                    session.session_start,
                                    session.session_end,
                                    session.duration_seconds,
                                    session.celery_task_id,
                                    session.status,
                                    session.approval_status,
                                    session.notes,
                                    session.created_at,
                                    session.updated_at,
                                )
                            imported_counts["sessions"] += 1
                        except Exception as e:
                            errors.append(
                                f"Error importing session {session.session_name}: {str(e)}"
                            )

                # Import sample sets
                if request.import_sample_sets and request.heimdall_file.sections.sample_sets:
                    for sample_set in request.heimdall_file.sections.sample_sets:
                        try:
                            # Insert or update dataset metadata
                            if request.overwrite_existing:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.synthetic_datasets
                                    (id, name, description, num_samples, config, 
                                     quality_metrics, created_at)
                                    VALUES ($1, $2, $3, $4, CAST($5 AS jsonb), CAST($6 AS jsonb), $7)
                                    ON CONFLICT (id) DO UPDATE SET
                                        name = EXCLUDED.name,
                                        description = EXCLUDED.description,
                                        num_samples = EXCLUDED.num_samples,
                                        config = EXCLUDED.config,
                                        quality_metrics = EXCLUDED.quality_metrics,
                                        updated_at = NOW()
                                    """,
                                    sample_set.id,
                                    sample_set.name,
                                    sample_set.description,
                                    sample_set.num_samples,
                                    json.dumps(sample_set.config) if sample_set.config else None,
                                    json.dumps(sample_set.quality_metrics) if sample_set.quality_metrics else None,
                                    sample_set.created_at,
                                )
                            else:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.synthetic_datasets
                                    (id, name, description, num_samples, config, 
                                     quality_metrics, created_at)
                                    VALUES ($1, $2, $3, $4, CAST($5 AS jsonb), CAST($6 AS jsonb), $7)
                                    ON CONFLICT (id) DO NOTHING
                                    """,
                                    sample_set.id,
                                    sample_set.name,
                                    sample_set.description,
                                    sample_set.num_samples,
                                    json.dumps(sample_set.config) if sample_set.config else None,
                                    json.dumps(sample_set.quality_metrics) if sample_set.quality_metrics else None,
                                    sample_set.created_at,
                                )

                            # Import sample data if present
                            if sample_set.samples:
                                for sample in sample_set.samples:
                                    try:
                                        await conn.execute(
                                            """
                                            INSERT INTO heimdall.measurement_features
                                            (dataset_id, tx_latitude, tx_longitude, tx_power_dbm,
                                             extraction_metadata, mean_snr_db, overall_confidence, gdop)
                                            VALUES ($1, $2, $3, $4, CAST($5 AS jsonb), $6, $7, $8)
                                            ON CONFLICT DO NOTHING
                                            """,
                                            sample_set.id,
                                            sample.get("tx_lat"),
                                            sample.get("tx_lon"),
                                            sample.get("tx_power_dbm"),
                                            json.dumps(sample.get("extraction_metadata")) if sample.get("extraction_metadata") else None,
                                            sample.get("mean_snr_db"),
                                            sample.get("overall_confidence"),
                                            sample.get("gdop"),
                                        )
                                    except Exception as e:
                                        logger.warning(f"Error importing sample: {e}")
                                        # Continue with other samples even if one fails

                            imported_counts["sample_sets"] += 1
                        except Exception as e:
                            errors.append(f"Error importing sample set {sample_set.name}: {str(e)}")

                # Import models
                if request.import_models and request.heimdall_file.sections.models:
                    minio_client = MinIOClient(
                        endpoint_url=settings.minio_url,
                        access_key=settings.minio_access_key,
                        secret_key=settings.minio_secret_key,
                        bucket_name="heimdall-models",
                    )
                    
                    for model in request.heimdall_file.sections.models:
                        try:
                            # Upload ONNX model to MinIO if present
                            onnx_location = None
                            if model.onnx_model_base64:
                                try:
                                    import io
                                    onnx_bytes = base64.b64decode(model.onnx_model_base64)
                                    onnx_key = f"imported/{model.model_name}-v{model.version}.onnx"
                                    
                                    minio_client.s3_client.put_object(
                                        Bucket="heimdall-models",
                                        Key=onnx_key,
                                        Body=io.BytesIO(onnx_bytes),
                                        ContentLength=len(onnx_bytes),
                                    )
                                    
                                    onnx_location = f"s3://heimdall-models/{onnx_key}"
                                    logger.info(f"Uploaded ONNX model to {onnx_location}")
                                except Exception as e:
                                    logger.error(f"Failed to upload ONNX model: {e}")
                                    errors.append(f"Failed to upload ONNX for {model.model_name}: {str(e)}")

                            # Insert or update model metadata
                            if request.overwrite_existing:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.models
                                    (id, model_name, version, model_type, onnx_model_location,
                                     accuracy_meters, hyperparameters, training_metrics, 
                                     is_active, created_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, CAST($7 AS jsonb), 
                                            CAST($8 AS jsonb), FALSE, $9)
                                    ON CONFLICT (id) DO UPDATE SET
                                        model_name = EXCLUDED.model_name,
                                        version = EXCLUDED.version,
                                        model_type = EXCLUDED.model_type,
                                        onnx_model_location = EXCLUDED.onnx_model_location,
                                        accuracy_meters = EXCLUDED.accuracy_meters,
                                        hyperparameters = EXCLUDED.hyperparameters,
                                        training_metrics = EXCLUDED.training_metrics,
                                        updated_at = NOW()
                                    """,
                                    model.id,
                                    f"{model.model_name} (Imported)",
                                    model.version,
                                    model.model_type,
                                    onnx_location,
                                    model.accuracy_meters,
                                    json.dumps(model.hyperparameters) if model.hyperparameters else None,
                                    json.dumps(model.training_metrics) if model.training_metrics else None,
                                    model.created_at,
                                )
                            else:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.models
                                    (id, model_name, version, model_type, onnx_model_location,
                                     accuracy_meters, hyperparameters, training_metrics, 
                                     is_active, created_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, CAST($7 AS jsonb), 
                                            CAST($8 AS jsonb), FALSE, $9)
                                    ON CONFLICT (id) DO NOTHING
                                    """,
                                    model.id,
                                    f"{model.model_name} (Imported)",
                                    model.version,
                                    model.model_type,
                                    onnx_location,
                                    model.accuracy_meters,
                                    json.dumps(model.hyperparameters) if model.hyperparameters else None,
                                    json.dumps(model.training_metrics) if model.training_metrics else None,
                                    model.created_at,
                                )
                            
                            imported_counts["models"] += 1
                        except Exception as e:
                            errors.append(f"Error importing model {model.model_name}: {str(e)}")

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
