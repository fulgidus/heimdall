"""Import/Export API endpoints for Heimdall SDR.

Provides endpoints to export system state to .heimdall files and import them back.
"""

import base64
import json
import logging

import asyncpg
from fastapi import APIRouter, HTTPException

from ..db import get_pool
from ..models.import_export import (
    AvailableAudioLibrary,
    ExportedAudioChunk,
    ExportedAudioLibrary,
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
    SampleSetExportConfig,
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

        # Get available sample sets with accurate size calculations including IQ data
        # Step 1: Get counts efficiently (no pg_column_size which is slow on large datasets)
        sample_sets_rows = await conn.fetch(
            """
            SELECT 
                sd.id, 
                sd.name, 
                sd.created_at,
                COALESCE(COUNT(DISTINCT mf.recording_session_id), 0) as num_samples,
                COALESCE(
                    (SELECT COUNT(*) FROM heimdall.synthetic_iq_samples WHERE dataset_id = sd.id),
                    0
                ) as num_iq_samples
            FROM heimdall.synthetic_datasets sd
            LEFT JOIN heimdall.measurement_features mf ON mf.dataset_id = sd.id
            GROUP BY sd.id, sd.name, sd.created_at
            ORDER BY sd.created_at DESC
        """
        )
        
        # Step 2: Calculate average feature size from a statistical sample (100 rows max)
        # This is ~1000x faster than calculating on all rows
        avg_feature_size = await conn.fetchval(
            """
            SELECT COALESCE(
                CEIL(
                    AVG(
                        pg_column_size(recording_session_id) +
                        pg_column_size(dataset_id) +
                        pg_column_size(tx_latitude) +
                        pg_column_size(tx_longitude) +
                        pg_column_size(tx_power_dbm) +
                        pg_column_size(extraction_metadata) +
                        pg_column_size(mean_snr_db) +
                        pg_column_size(overall_confidence) +
                        pg_column_size(gdop) +
                        pg_column_size(created_at)
                    ) * 1.25  -- JSON overhead
                )::bigint,
                433  -- Fallback based on known data
            )
            FROM (
                SELECT * FROM heimdall.measurement_features 
                TABLESAMPLE SYSTEM (1)  -- Sample ~1% of pages (fast)
                LIMIT 100
            ) sample
        """
        )
        
        # Use the sampled average feature size for all datasets
        # Each IQ sample: ~13MB (7 receivers × 350KB IQ data × 1.33 base64 + metadata)
        # Real-world: UHF dataset = ~87GB for 6706 samples ≈ 13MB/sample
        estimated_size_per_iq = 13_000_000
        
        sample_sets = [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "num_samples": row["num_samples"],
                "num_iq_samples": row["num_iq_samples"],
                "created_at": row["created_at"].isoformat(),
                "estimated_size_per_feature": avg_feature_size,
                "estimated_size_per_iq": estimated_size_per_iq,
                "estimated_size_bytes": (
                    row["num_samples"] * avg_feature_size +
                    row["num_iq_samples"] * estimated_size_per_iq
                ),
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

        # Get available audio library entries with actual chunk sizes
        audio_library_rows = await conn.fetch(
            """
            SELECT 
                al.id, 
                al.filename, 
                al.category,
                al.file_size_bytes,
                al.duration_seconds,
                al.total_chunks,
                al.created_at,
                COALESCE(SUM(ac.file_size_bytes), 0) as chunks_total_bytes
            FROM heimdall.audio_library al
            LEFT JOIN heimdall.audio_chunks ac ON ac.audio_id = al.id
            WHERE al.processing_status = 'READY'
            GROUP BY al.id, al.filename, al.category, al.file_size_bytes, 
                     al.duration_seconds, al.total_chunks, al.created_at
            ORDER BY al.created_at DESC
        """
        )
        
        audio_library = [
            {
                "id": str(row["id"]),
                "filename": row["filename"],
                "category": row["category"],
                "duration_seconds": float(row["duration_seconds"]),
                "total_chunks": row["total_chunks"],
                "file_size_bytes": row["file_size_bytes"],  # Original file size (for reference)
                "chunks_total_bytes": row["chunks_total_bytes"],  # Actual preprocessed chunks size
                "created_at": row["created_at"].isoformat(),
            }
            for row in audio_library_rows
        ]

        # Estimate section sizes
        settings_size = 256
        sources_size = sources_count * 500 if sources_count else 0
        websdrs_size = websdrs_count * 400 if websdrs_count else 0
        sessions_size = sessions_count * 600 if sessions_count else 0
        
        # Sample sets: Use accurate calculation from database query (includes IQ data if present)
        # Real-world: ~90GB total datasets (85GB largest + 5GB others)
        sample_sets_size = sum(s["estimated_size_bytes"] for s in sample_sets)
        
        # Models: Use actual file size from MinIO if available, otherwise estimate
        # Real-world: ONNX models range from 50MB (small) to 500MB (large ResNet-based)
        models_size = 0
        for model in models:
            if model["has_onnx"]:
                # Average ONNX model size: ~200MB (ResNet-18 backbone + heads)
                # Base64 encoding adds 33% overhead: 200MB * 1.33 ≈ 266MB
                models_size += 266_000_000
            else:
                # Metadata only (no ONNX file): ~5KB per model
                models_size += 5_000
        
        # Audio library: Calculate from actual chunk sizes (preprocessed .npy files)
        # Real-world: Original audio ~800MB WAV + ~80MB MP3 = ~880MB
        # Chunks are resampled to 200kHz mono .npy files (typically larger than original)
        # Base64 encoding adds 33% overhead, JSON structure adds ~10% overhead
        audio_library_size = sum(
            int(al["chunks_total_bytes"] * 1.43)  # Use actual chunk bytes, not original file size
            for al in audio_library
        )

        return MetadataResponse(
            sources_count=sources_count or 0,
            websdrs_count=websdrs_count or 0,
            sessions_count=sessions_count or 0,
            sample_sets=sample_sets,
            models=models,
            audio_library=audio_library,
            estimated_sizes=SectionSizes(
                settings=settings_size,
                sources=sources_size,
                websdrs=websdrs_size,
                sessions=sessions_size,
                sample_sets=sample_sets_size,
                models=models_size,
                audio_library=int(audio_library_size),
            ),
        )


@router.post("/export", response_model=ExportResponse)
async def export_data(request: ExportRequest):
    """
    Export selected data sections to a .heimdall file format.

    Allows selective export of sources, WebSDRs, sessions, and models.
    """
    try:
        pool = get_pool()
        sections = ExportSections()
    except Exception as e:
        logger.error(f"Failed to initialize export: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export initialization failed: {str(e)}")

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
        if request.sample_set_configs:
            sample_sets = []
            for config_raw in request.sample_set_configs:
                try:
                    # Ensure config is a Pydantic model instance
                    # FastAPI should do this automatically, but being defensive for robustness
                    config: SampleSetExportConfig
                    if isinstance(config_raw, dict):
                        config = SampleSetExportConfig.model_validate(config_raw)
                    else:
                        config = config_raw
                    
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
                        config.dataset_id,
                    )
                    
                    if not dataset_row:
                        logger.warning(f"Sample set {config.dataset_id} not found, skipping")
                        continue

                    # Determine actual export range
                    total_samples = dataset_row["num_samples"]
                    offset = config.sample_offset
                    limit = config.sample_limit if config.sample_limit else (total_samples - offset)
                    
                    # Validate range
                    if offset >= total_samples:
                        logger.warning(
                            f"Sample set {config.dataset_id}: offset {offset} >= total {total_samples}, skipping"
                        )
                        continue
                    
                    # Adjust limit if it exceeds available samples
                    actual_limit = min(limit, total_samples - offset)

                    # Get COMPLETE feature data from measurement_features
                    features_rows = await conn.fetch(
                        """
                        SELECT 
                            recording_session_id, timestamp, receiver_features,
                            tx_latitude, tx_longitude, tx_altitude_m, tx_power_dbm, tx_known,
                            extraction_metadata, overall_confidence, mean_snr_db,
                            num_receivers_detected, gdop, extraction_failed, error_message,
                            created_at, num_receivers_in_sample
                        FROM heimdall.measurement_features
                        WHERE dataset_id = $1
                        ORDER BY created_at ASC
                        OFFSET $2
                        LIMIT $3
                        """,
                        config.dataset_id,
                        offset,
                        actual_limit,
                    )
                    
                    features = []
                    for feat_row in features_rows:
                        # Parse JSON fields
                        receiver_features = feat_row["receiver_features"]
                        if isinstance(receiver_features, str):
                            receiver_features = json.loads(receiver_features)
                        # receiver_features is a JSONB array, each element might be a JSON string
                        if receiver_features and isinstance(receiver_features, list):
                            receiver_features = [
                                json.loads(rf) if isinstance(rf, str) else rf
                                for rf in receiver_features
                            ]
                        
                        extraction_metadata = feat_row["extraction_metadata"]
                        if isinstance(extraction_metadata, str):
                            extraction_metadata = json.loads(extraction_metadata)
                        
                        features.append({
                            "recording_session_id": str(feat_row["recording_session_id"]),
                            "timestamp": feat_row["timestamp"].isoformat(),
                            "receiver_features": receiver_features,
                            "tx_latitude": float(feat_row["tx_latitude"]) if feat_row["tx_latitude"] else None,
                            "tx_longitude": float(feat_row["tx_longitude"]) if feat_row["tx_longitude"] else None,
                            "tx_altitude_m": float(feat_row["tx_altitude_m"]) if feat_row["tx_altitude_m"] else None,
                            "tx_power_dbm": float(feat_row["tx_power_dbm"]) if feat_row["tx_power_dbm"] else None,
                            "tx_known": feat_row["tx_known"],
                            "extraction_metadata": extraction_metadata,
                            "overall_confidence": float(feat_row["overall_confidence"]),
                            "mean_snr_db": float(feat_row["mean_snr_db"]) if feat_row["mean_snr_db"] else None,
                            "num_receivers_detected": feat_row["num_receivers_detected"],
                            "gdop": float(feat_row["gdop"]) if feat_row["gdop"] else None,
                            "extraction_failed": feat_row["extraction_failed"],
                            "error_message": feat_row["error_message"],
                            "created_at": feat_row["created_at"].isoformat(),
                            "num_receivers_in_sample": feat_row["num_receivers_in_sample"],
                        })
                    
                    # Get COMPLETE IQ data from synthetic_iq_samples
                    iq_samples = []
                    if config.include_iq_data:
                        logger.info(f"Fetching IQ samples for dataset {config.dataset_id}")
                        iq_rows = await conn.fetch(
                            """
                            SELECT 
                                id, dataset_id, sample_idx, timestamp,
                                tx_lat, tx_lon, tx_alt, tx_power_dbm, frequency_hz,
                                receivers_metadata, num_receivers, gdop, mean_snr_db, overall_confidence,
                                iq_metadata, iq_storage_paths, created_at
                            FROM heimdall.synthetic_iq_samples
                            WHERE dataset_id = $1
                            ORDER BY sample_idx ASC
                            OFFSET $2
                            LIMIT $3
                            """,
                            config.dataset_id,
                            offset,
                            actual_limit,
                        )
                        
                        # Initialize MinIO client for IQ data download
                        minio_client = MinIOClient(
                            endpoint_url=settings.MINIO_ENDPOINT,
                            access_key=settings.MINIO_ACCESS_KEY,
                            secret_key=settings.MINIO_SECRET_KEY,
                            bucket_name="heimdall-synthetic-iq",
                        )
                        
                        for iq_row in iq_rows:
                            # Parse JSON fields
                            receivers_metadata = iq_row["receivers_metadata"]
                            if isinstance(receivers_metadata, str):
                                receivers_metadata = json.loads(receivers_metadata)
                            
                            iq_metadata = iq_row["iq_metadata"]
                            if isinstance(iq_metadata, str):
                                iq_metadata = json.loads(iq_metadata)
                            
                            iq_storage_paths = iq_row["iq_storage_paths"]
                            if isinstance(iq_storage_paths, str):
                                iq_storage_paths = json.loads(iq_storage_paths)
                            
                            # Download IQ binary data from MinIO and encode to base64
                            iq_data_list = []
                            for receiver_id, s3_path in iq_storage_paths.items():
                                try:
                                    # Parse s3://bucket/key format or assume heimdall-synthetic-iq bucket
                                    if s3_path.startswith("s3://"):
                                        s3_path = s3_path[5:]
                                        parts = s3_path.split("/", 1)
                                        bucket = parts[0]
                                        key = parts[1] if len(parts) > 1 else ""
                                    else:
                                        # Assume default bucket
                                        bucket = "heimdall-synthetic-iq"
                                        key = s3_path
                                    
                                    # Download IQ data
                                    response = minio_client.s3_client.get_object(Bucket=bucket, Key=key)
                                    iq_bytes = response["Body"].read()
                                    iq_base64 = base64.b64encode(iq_bytes).decode("utf-8")
                                    
                                    iq_data_list.append({
                                        "receiver_id": receiver_id,
                                        "iq_data_base64": iq_base64,
                                    })
                                    logger.debug(f"Downloaded IQ data for {receiver_id}: {len(iq_bytes)} bytes")
                                except Exception as e:
                                    logger.error(f"Failed to download IQ data for {receiver_id} at {s3_path}: {e}")
                                    # Continue with other receivers even if one fails
                            
                            iq_samples.append({
                                "id": str(iq_row["id"]),
                                "sample_idx": iq_row["sample_idx"],
                                "timestamp": iq_row["timestamp"].isoformat(),
                                "tx_lat": float(iq_row["tx_lat"]),
                                "tx_lon": float(iq_row["tx_lon"]),
                                "tx_alt": float(iq_row["tx_alt"]),
                                "tx_power_dbm": float(iq_row["tx_power_dbm"]),
                                "frequency_hz": iq_row["frequency_hz"],
                                "num_receivers": iq_row["num_receivers"],
                                "gdop": float(iq_row["gdop"]) if iq_row["gdop"] else None,
                                "mean_snr_db": float(iq_row["mean_snr_db"]) if iq_row["mean_snr_db"] else None,
                                "overall_confidence": float(iq_row["overall_confidence"]) if iq_row["overall_confidence"] else None,
                                "receivers_metadata": receivers_metadata,
                                "iq_metadata": iq_metadata,
                                "iq_storage_paths": iq_storage_paths,
                                "iq_data": iq_data_list,
                                "created_at": iq_row["created_at"].isoformat(),
                            })
                    else:
                        logger.info(f"Skipping IQ data download (include_iq_data=False)")

                    # Parse JSON fields if they're strings (asyncpg sometimes returns jsonb as strings)
                    dataset_config = dataset_row["config"]
                    if isinstance(dataset_config, str):
                        dataset_config = json.loads(dataset_config)
                    
                    quality_metrics = dataset_row["quality_metrics"]
                    if isinstance(quality_metrics, str):
                        quality_metrics = json.loads(quality_metrics)
                    
                    sample_sets.append(
                        ExportedSampleSet(
                            id=str(dataset_row["id"]),
                            name=dataset_row["name"],
                            description=dataset_row["description"],
                            num_samples=dataset_row["num_samples"],
                            config=dataset_config,
                            quality_metrics=quality_metrics,
                            created_at=dataset_row["created_at"].isoformat(),
                            features=features,
                            iq_samples=iq_samples if config.include_iq_data else None,
                            num_exported_features=len(features),
                            num_exported_iq_samples=len(iq_samples) if config.include_iq_data else 0,
                            export_range={"offset": offset, "limit": actual_limit},
                        )
                    )
                except asyncpg.QueryCanceledError as e:
                    logger.error(f"Query timeout for sample set {config.dataset_id}: {e}")
                    raise HTTPException(
                        status_code=504,
                        detail=f"Export timed out for sample set {config.dataset_id}. Try reducing sample range.",
                    )
                except MemoryError as e:
                    logger.error(f"Memory exhausted for sample set {config.dataset_id}: {e}")
                    raise HTTPException(
                        status_code=507,
                        detail=f"Insufficient memory to export sample set {config.dataset_id}. Try reducing sample range.",
                    )
                except asyncpg.PostgresError as e:
                    logger.error(f"Database error for sample set {config.dataset_id}: {e}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Database error exporting sample set {config.dataset_id}: {str(e)}",
                    )
                except Exception as e:
                    logger.error(f"Unexpected error for sample set {config.dataset_id}: {e}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to export sample set {config.dataset_id}: {str(e)}",
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

                # Parse JSON fields if they're strings (PostgreSQL JSONB returns as strings)
                hyperparameters = model_row["hyperparameters"]
                if isinstance(hyperparameters, str):
                    hyperparameters = json.loads(hyperparameters)
                
                training_metrics = model_row["training_metrics"]
                if isinstance(training_metrics, str):
                    training_metrics = json.loads(training_metrics)
                
                models.append(
                    ExportedModel(
                        id=str(model_row["id"]),
                        model_name=model_row["model_name"],
                        version=model_row["version"],
                        model_type=model_row["model_type"],
                        created_at=model_row["created_at"].isoformat(),
                        onnx_model_base64=onnx_base64,
                        accuracy_meters=model_row["accuracy_meters"],
                        hyperparameters=hyperparameters,
                        training_metrics=training_metrics,
                    )
                )
            
            sections.models = models if models else None

        # Export audio library if requested
        if request.audio_library_ids:
            minio_audio_client = MinIOClient(
                endpoint_url=settings.minio_url,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                bucket_name="heimdall-audio-chunks",
            )
            
            audio_libraries = []
            for audio_id in request.audio_library_ids:
                # Get audio library metadata
                audio_row = await conn.fetchrow(
                    """
                    SELECT 
                        id, filename, category, tags, file_size_bytes,
                        duration_seconds, sample_rate, channels, audio_format,
                        processing_status, total_chunks, enabled,
                        created_at, updated_at
                    FROM heimdall.audio_library
                    WHERE id = $1
                    """,
                    audio_id,
                )
                
                if not audio_row:
                    logger.warning(f"Audio library entry {audio_id} not found, skipping")
                    continue
                
                # Get audio chunks for this audio file
                chunks_rows = await conn.fetch(
                    """
                    SELECT 
                        id, chunk_index, duration_seconds, sample_rate,
                        num_samples, file_size_bytes, original_offset_seconds,
                        rms_amplitude, minio_path, created_at
                    FROM heimdall.audio_chunks
                    WHERE audio_id = $1
                    ORDER BY chunk_index ASC
                    """,
                    audio_id,
                )
                
                chunks = []
                for chunk_row in chunks_rows:
                    # Download chunk data from MinIO
                    audio_data_base64 = None
                    minio_path = chunk_row["minio_path"]
                    
                    if minio_path:
                        try:
                            # Parse s3://bucket/key format
                            if minio_path.startswith("s3://"):
                                minio_path = minio_path[5:]
                            
                            parts = minio_path.split("/", 1)
                            bucket = parts[0]
                            key = parts[1] if len(parts) > 1 else ""
                            
                            # Download from MinIO
                            response = minio_audio_client.s3_client.get_object(Bucket=bucket, Key=key)
                            audio_bytes = response["Body"].read()
                            audio_data_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                            logger.info(f"Downloaded audio chunk from {minio_path} ({len(audio_bytes)} bytes)")
                        except Exception as e:
                            logger.error(f"Failed to download audio chunk {chunk_row['id']}: {e}")
                            # Continue with other chunks even if one fails
                    
                    chunks.append(
                        ExportedAudioChunk(
                            id=str(chunk_row["id"]),
                            chunk_index=chunk_row["chunk_index"],
                            duration_seconds=float(chunk_row["duration_seconds"]),
                            sample_rate=chunk_row["sample_rate"],
                            num_samples=chunk_row["num_samples"],
                            file_size_bytes=chunk_row["file_size_bytes"],
                            original_offset_seconds=float(chunk_row["original_offset_seconds"]),
                            rms_amplitude=float(chunk_row["rms_amplitude"]) if chunk_row["rms_amplitude"] else None,
                            created_at=chunk_row["created_at"].isoformat(),
                            audio_data_base64=audio_data_base64,
                        )
                    )
                
                audio_libraries.append(
                    ExportedAudioLibrary(
                        id=str(audio_row["id"]),
                        filename=audio_row["filename"],
                        category=audio_row["category"],
                        tags=audio_row["tags"] if audio_row["tags"] else None,
                        file_size_bytes=audio_row["file_size_bytes"],
                        duration_seconds=float(audio_row["duration_seconds"]),
                        sample_rate=audio_row["sample_rate"],
                        channels=audio_row["channels"],
                        audio_format=audio_row["audio_format"],
                        processing_status=audio_row["processing_status"],
                        total_chunks=audio_row["total_chunks"],
                        enabled=audio_row["enabled"],
                        created_at=audio_row["created_at"].isoformat(),
                        updated_at=audio_row["updated_at"].isoformat(),
                        chunks=chunks if chunks else None,
                    )
                )
            
            sections.audio_library = audio_libraries if audio_libraries else None

    # Calculate section sizes efficiently (without full serialization)
    section_sizes = SectionSizes()
    
    # Small sections: use actual serialization (minimal memory impact)
    if sections.settings:
        section_sizes.settings = len(json.dumps(sections.settings.model_dump()))
    if sections.sources:
        section_sizes.sources = len(json.dumps([s.model_dump() for s in sections.sources]))
    if sections.websdrs:
        section_sizes.websdrs = len(json.dumps([w.model_dump() for w in sections.websdrs]))
    if sections.sessions:
        section_sizes.sessions = len(json.dumps([s.model_dump() for s in sections.sessions]))
    
    # Large sections: estimate based on sample + count
    if sections.sample_sets:
        # Estimate: serialize first sample set, multiply by count with overhead factor
        if len(sections.sample_sets) > 0:
            first_sample_size = len(json.dumps(sections.sample_sets[0].model_dump()))
            section_sizes.sample_sets = first_sample_size * len(sections.sample_sets)
        else:
            section_sizes.sample_sets = 0
    
    if sections.models:
        # Models with ONNX can be huge - estimate per model
        total_model_size = 0
        for model in sections.models:
            # Base metadata: ~500 bytes
            base_size = 500
            # ONNX model if present
            if model.onnx_model_base64:
                base_size += len(model.onnx_model_base64)
            total_model_size += base_size
        section_sizes.models = total_model_size
    
    if sections.audio_library:
        # Audio chunks can be huge - estimate per audio library item
        total_audio_size = 0
        for audio in sections.audio_library:
            # Base metadata: ~500 bytes
            base_size = 500
            # Chunks if present
            if audio.chunks:
                for chunk in audio.chunks:
                    chunk_size = 500  # metadata
                    if chunk.audio_data_base64:
                        chunk_size += len(chunk.audio_data_base64)
                    base_size += chunk_size
            total_audio_size += base_size
        section_sizes.audio_library = total_audio_size

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
        "audio_library": 0,
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

                # Import audio library
                if request.import_audio_library and request.heimdall_file.sections.audio_library:
                    minio_audio_client = MinIOClient(
                        endpoint_url=settings.minio_url,
                        access_key=settings.minio_access_key,
                        secret_key=settings.minio_secret_key,
                        bucket_name="heimdall-audio-chunks",
                    )
                    
                    for audio_lib in request.heimdall_file.sections.audio_library:
                        try:
                            # Insert or update audio library metadata
                            if request.overwrite_existing:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.audio_library
                                    (id, filename, category, tags, file_size_bytes,
                                     duration_seconds, sample_rate, channels, audio_format,
                                     processing_status, total_chunks, enabled,
                                     created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                                    ON CONFLICT (id) DO UPDATE SET
                                        filename = EXCLUDED.filename,
                                        category = EXCLUDED.category,
                                        tags = EXCLUDED.tags,
                                        file_size_bytes = EXCLUDED.file_size_bytes,
                                        duration_seconds = EXCLUDED.duration_seconds,
                                        sample_rate = EXCLUDED.sample_rate,
                                        channels = EXCLUDED.channels,
                                        audio_format = EXCLUDED.audio_format,
                                        processing_status = EXCLUDED.processing_status,
                                        total_chunks = EXCLUDED.total_chunks,
                                        enabled = EXCLUDED.enabled,
                                        updated_at = NOW()
                                    """,
                                    audio_lib.id,
                                    audio_lib.filename,
                                    audio_lib.category,
                                    audio_lib.tags,
                                    audio_lib.file_size_bytes,
                                    audio_lib.duration_seconds,
                                    audio_lib.sample_rate,
                                    audio_lib.channels,
                                    audio_lib.audio_format,
                                    audio_lib.processing_status,
                                    audio_lib.total_chunks,
                                    audio_lib.enabled,
                                    audio_lib.created_at,
                                    audio_lib.updated_at,
                                )
                            else:
                                await conn.execute(
                                    """
                                    INSERT INTO heimdall.audio_library
                                    (id, filename, category, tags, file_size_bytes,
                                     duration_seconds, sample_rate, channels, audio_format,
                                     processing_status, total_chunks, enabled,
                                     created_at, updated_at)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                                    ON CONFLICT (id) DO NOTHING
                                    """,
                                    audio_lib.id,
                                    audio_lib.filename,
                                    audio_lib.category,
                                    audio_lib.tags,
                                    audio_lib.file_size_bytes,
                                    audio_lib.duration_seconds,
                                    audio_lib.sample_rate,
                                    audio_lib.channels,
                                    audio_lib.audio_format,
                                    audio_lib.processing_status,
                                    audio_lib.total_chunks,
                                    audio_lib.enabled,
                                    audio_lib.created_at,
                                    audio_lib.updated_at,
                                )
                            
                            # Import audio chunks if present
                            if audio_lib.chunks:
                                for chunk in audio_lib.chunks:
                                    try:
                                        # Upload chunk to MinIO if audio data is present
                                        minio_path = None
                                        if chunk.audio_data_base64:
                                            try:
                                                import io
                                                audio_bytes = base64.b64decode(chunk.audio_data_base64)
                                                chunk_key = f"imported/{audio_lib.id}/{chunk.chunk_index:04d}.npy"
                                                
                                                minio_audio_client.s3_client.put_object(
                                                    Bucket="heimdall-audio-chunks",
                                                    Key=chunk_key,
                                                    Body=io.BytesIO(audio_bytes),
                                                    ContentLength=len(audio_bytes),
                                                )
                                                
                                                minio_path = f"s3://heimdall-audio-chunks/{chunk_key}"
                                                logger.info(f"Uploaded audio chunk to {minio_path}")
                                            except Exception as e:
                                                logger.error(f"Failed to upload audio chunk: {e}")
                                                errors.append(f"Failed to upload audio chunk for {audio_lib.filename}: {str(e)}")
                                        
                                        # Insert chunk metadata
                                        if request.overwrite_existing:
                                            await conn.execute(
                                                """
                                                INSERT INTO heimdall.audio_chunks
                                                (id, audio_id, chunk_index, duration_seconds, sample_rate,
                                                 num_samples, file_size_bytes, original_offset_seconds,
                                                 rms_amplitude, minio_path, created_at)
                                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                                                ON CONFLICT (id) DO UPDATE SET
                                                    chunk_index = EXCLUDED.chunk_index,
                                                    duration_seconds = EXCLUDED.duration_seconds,
                                                    sample_rate = EXCLUDED.sample_rate,
                                                    num_samples = EXCLUDED.num_samples,
                                                    file_size_bytes = EXCLUDED.file_size_bytes,
                                                    original_offset_seconds = EXCLUDED.original_offset_seconds,
                                                    rms_amplitude = EXCLUDED.rms_amplitude,
                                                    minio_path = EXCLUDED.minio_path
                                                """,
                                                chunk.id,
                                                audio_lib.id,
                                                chunk.chunk_index,
                                                chunk.duration_seconds,
                                                chunk.sample_rate,
                                                chunk.num_samples,
                                                chunk.file_size_bytes,
                                                chunk.original_offset_seconds,
                                                chunk.rms_amplitude,
                                                minio_path,
                                                chunk.created_at,
                                            )
                                        else:
                                            await conn.execute(
                                                """
                                                INSERT INTO heimdall.audio_chunks
                                                (id, audio_id, chunk_index, duration_seconds, sample_rate,
                                                 num_samples, file_size_bytes, original_offset_seconds,
                                                 rms_amplitude, minio_path, created_at)
                                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                                                ON CONFLICT (id) DO NOTHING
                                                """,
                                                chunk.id,
                                                audio_lib.id,
                                                chunk.chunk_index,
                                                chunk.duration_seconds,
                                                chunk.sample_rate,
                                                chunk.num_samples,
                                                chunk.file_size_bytes,
                                                chunk.original_offset_seconds,
                                                chunk.rms_amplitude,
                                                minio_path,
                                                chunk.created_at,
                                            )
                                    except Exception as e:
                                        logger.warning(f"Error importing audio chunk {chunk.chunk_index}: {e}")
                                        # Continue with other chunks even if one fails
                            
                            imported_counts["audio_library"] += 1
                        except Exception as e:
                            errors.append(f"Error importing audio library {audio_lib.filename}: {str(e)}")

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


@router.post("/export/async")
async def export_data_async(request: ExportRequest):
    """
    Initiate asynchronous export with progress tracking.
    
    Returns immediately with a task_id. Client should listen to WebSocket
    for progress updates and completion notification with download URL.
    """
    from ..tasks.export_task import export_async
    
    # Submit Celery task
    task = export_async.delay(request.dict())
    
    return {
        "status": "processing",
        "task_id": task.id,
        "message": "Export started. Listen to WebSocket for progress updates."
    }


@router.get("/export/{task_id}/status")
async def get_export_status(task_id: str):
    """
    Get status of an ongoing export task.
    
    Returns task state, progress information, and download URL when complete.
    """
    from celery.result import AsyncResult
    
    try:
        result = AsyncResult(task_id)
        
        status_map = {
            "PENDING": "pending",
            "STARTED": "processing",
            "PROGRESS": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "REVOKED": "cancelled",
            "RETRY": "processing",
        }
        
        mapped_status = status_map.get(result.state, result.state.lower())
        
        # Extract progress info
        if result.state == "PROGRESS":
            info = result.info if isinstance(result.info, dict) else {}
            progress = info.get("progress", 0)
            message = info.get("status", "Processing export...")
            section_sizes = info.get("section_sizes")
        elif result.state == "SUCCESS":
            info = result.result if isinstance(result.result, dict) else {}
            progress = 100
            message = "Export completed successfully"
            section_sizes = info.get("section_sizes")
            download_url = f"/api/import-export/download/{task_id}"
        elif result.state == "FAILURE":
            progress = 0
            message = f"Export failed: {str(result.info)}"
            section_sizes = None
            download_url = None
        else:
            progress = 0 if result.state == "PENDING" else 50
            message = f"Task state: {result.state}"
            section_sizes = None
            download_url = None
        
        response = {
            "task_id": task_id,
            "status": mapped_status,
            "progress": progress,
            "message": message,
        }
        
        if section_sizes:
            response["section_sizes"] = section_sizes
        
        if result.state == "SUCCESS":
            response["download_url"] = download_url
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get export status for {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve export status: {str(e)}"
        )


@router.post("/export/{task_id}/cancel")
async def cancel_export(task_id: str):
    """
    Cancel an ongoing export task.
    
    Terminates the Celery task and publishes a cancellation event via WebSocket.
    """
    from celery.result import AsyncResult
    from ..events.publisher import get_event_publisher
    
    try:
        # Get task result
        task_result = AsyncResult(task_id)
        
        # Check if task exists and is running
        task_state = task_result.state
        
        if task_state in ['SUCCESS', 'FAILURE']:
            return {
                "status": "already_completed",
                "task_id": task_id,
                "message": f"Task already completed with state: {task_state}"
            }
        
        if task_state == 'REVOKED':
            return {
                "status": "already_cancelled",
                "task_id": task_id,
                "message": "Task was already cancelled"
            }
        
        # Revoke task (terminate=True to kill running task)
        task_result.revoke(terminate=True)
        
        logger.info(f"Cancelled export task {task_id} (state was: {task_state})")
        
        # Publish cancellation event via WebSocket
        publisher = get_event_publisher()
        publisher.publish_export_cancelled(
            task_id=task_id,
            status='cancelled'
        )
        
        # Try to delete any partial export file from MinIO
        try:
            minio_client = MinIOClient(
                endpoint_url=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                bucket_name="heimdall-exports"
            )
            s3_path = f"exports/export-{task_id}.heimdall"
            minio_client.delete_object(s3_path)
            logger.info(f"Deleted partial export file: {s3_path}")
        except Exception as e:
            logger.debug(f"No partial file to delete or delete failed: {e}")
        
        return {
            "status": "cancelled",
            "task_id": task_id,
            "message": "Export task cancelled successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel export task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel export: {str(e)}"
        )


@router.get("/download/{task_id}")
async def download_export(task_id: str):
    """
    Download exported .heimdall file and auto-delete after successful download.
    """
    from fastapi.responses import StreamingResponse
    
    minio_client = MinIOClient(
        endpoint_url=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket_name="heimdall-exports"
    )
    
    filename = f"export-{task_id}.heimdall"
    s3_path = f"exports/{filename}"
    
    try:
        # Stream file from MinIO
        response = minio_client.s3_client.get_object(
            Bucket="heimdall-exports",
            Key=s3_path
        )
        
        file_content = response["Body"].read()
        file_size = len(file_content)
        
        logger.info(f"Serving export file {filename} ({file_size} bytes)")
        
        # Delete file after successful download
        try:
            minio_client.delete_object(s3_path)
            logger.info(f"Deleted export file after download: {s3_path}")
        except Exception as e:
            logger.error(f"Failed to delete export file {s3_path}: {e}")
        
        # Return file as streaming response
        return StreamingResponse(
            iter([file_content]),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(file_size)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to download export {task_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Export file not found or has expired: {task_id}"
        )
