"""
Celery task for asynchronous .heimdall file export with progress tracking.

This task generates large export files incrementally, publishing progress updates
via RabbitMQ/WebSocket so the frontend can show real-time progress.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import asyncpg
from celery import Task, shared_task

from ..config import settings
from ..db import get_pool
from ..events.publisher import get_event_publisher
from ..models.import_export import (
    ExportedFeature,
    ExportedIQSample,
    ExportedModel,
    ExportedSampleSet,
    ExportedSession,
    ExportedSource,
    ExportedWebSDR,
    ExportMetadata,
    ExportRequest,
    ExportSections,
    HeimdallFile,
    IQData,
    SectionSizes,
    UserSettings,
)
from ..storage.minio_client import MinIOClient

logger = logging.getLogger(__name__)


class ExportTask(Task):
    """Base task with progress tracking capabilities."""
    
    def update_progress(self, stage: str, current: int, total: int, message: str, **kwargs):
        """Publish progress update via WebSocket."""
        publisher = get_event_publisher()
        publisher.publish_export_progress(
            task_id=self.request.id,
            stage=stage,
            current=current,
            total=total,
            message=message,
            **kwargs
        )


@shared_task(bind=True, base=ExportTask, name="export_async")
def export_async(self: ExportTask, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronously export .heimdall file with progress tracking.
    
    Args:
        request_data: ExportRequest as dict
        
    Returns:
        Dict with 'status', 'download_url', 'file_size_bytes'
    """
    task_id = self.request.id
    publisher = get_event_publisher()
    
    try:
        logger.info(f"Starting async export task {task_id}")
        
        # Parse request
        request = ExportRequest(**request_data)
        
        # Use asyncio to run async database queries
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(_export_data_async(self, request))
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Export task {task_id} failed: {e}", exc_info=True)
        publisher.publish_export_completed(
            task_id=task_id,
            status='failed',
            error_message=str(e)
        )
        raise


async def _export_data_async(task: ExportTask, request: ExportRequest) -> Dict[str, Any]:
    """Async implementation of export logic."""
    task_id = task.request.id
    publisher = get_event_publisher()
    pool = get_pool()
    
    sections = ExportSections()
    section_sizes = SectionSizes()
    
    async with pool.acquire() as conn:
        # Stage 1: Settings
        if request.include_settings:
            task.update_progress("settings", 0, 1, "Exporting settings...")
            sections.settings = UserSettings()  # Default for now
            section_sizes.settings = len(json.dumps(sections.settings.dict()))
            task.update_progress("settings", 1, 1, "Settings exported")
        
        # Stage 2: Sources
        if request.include_sources:
            task.update_progress("sources", 0, 1, "Counting sources...")
            sources_count = await conn.fetchval("SELECT COUNT(*) FROM heimdall.known_sources")
            
            if sources_count > 0:
                task.update_progress("sources", 0, sources_count, f"Exporting {sources_count} sources...")
                
                sources_rows = await conn.fetch("""
                    SELECT id, name, description, frequency_hz, latitude, longitude,
                           power_dbm, source_type, is_validated, error_margin_meters,
                           created_at, updated_at
                    FROM heimdall.known_sources
                    ORDER BY created_at DESC
                """)
                
                sections.sources = [
                    ExportedSource(
                        id=str(row["id"]),
                        name=row["name"],
                        description=row["description"],
                        frequency_hz=row["frequency_hz"],
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        power_dbm=float(row["power_dbm"]) if row["power_dbm"] else None,
                        source_type=row["source_type"],
                        is_validated=row["is_validated"],
                        error_margin_meters=float(row["error_margin_meters"]) if row["error_margin_meters"] else None,
                        created_at=row["created_at"].isoformat(),
                        updated_at=row["updated_at"].isoformat(),
                    )
                    for row in sources_rows
                ]
                
                section_sizes.sources = len(json.dumps([s.dict() for s in sections.sources]))
                task.update_progress("sources", sources_count, sources_count, f"Exported {sources_count} sources")
        
        # Stage 3: WebSDRs
        if request.include_websdrs:
            task.update_progress("websdrs", 0, 1, "Counting WebSDR stations...")
            websdrs_count = await conn.fetchval("SELECT COUNT(*) FROM heimdall.websdr_stations")
            
            if websdrs_count > 0:
                task.update_progress("websdrs", 0, websdrs_count, f"Exporting {websdrs_count} WebSDR stations...")
                
                websdrs_rows = await conn.fetch("""
                    SELECT id, name, url, location_description, latitude, longitude,
                           altitude_meters, country, operator, is_active,
                           timeout_seconds, retry_count, created_at, updated_at
                    FROM heimdall.websdr_stations
                    ORDER BY created_at DESC
                """)
                
                sections.websdrs = [
                    ExportedWebSDR(
                        id=str(row["id"]),
                        name=row["name"],
                        url=row["url"],
                        location_description=row["location_description"],
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        altitude_meters=float(row["altitude_meters"]) if row["altitude_meters"] else None,
                        country=row["country"],
                        operator=row["operator"],
                        is_active=row["is_active"],
                        timeout_seconds=row["timeout_seconds"],
                        retry_count=row["retry_count"],
                        created_at=row["created_at"].isoformat(),
                        updated_at=row["updated_at"].isoformat(),
                    )
                    for row in websdrs_rows
                ]
                
                section_sizes.websdrs = len(json.dumps([w.dict() for w in sections.websdrs]))
                task.update_progress("websdrs", websdrs_count, websdrs_count, f"Exported {websdrs_count} WebSDRs")
        
        # Stage 4: Sample Sets (the big one!)
        if request.sample_set_configs:
            total_datasets = len(request.sample_set_configs)
            sections.sample_sets = []
            
            for idx, config in enumerate(request.sample_set_configs, 1):
                task.update_progress(
                    "sample_sets",
                    idx - 1,
                    total_datasets,
                    f"Exporting dataset {idx}/{total_datasets}: {config.dataset_id[:8]}..."
                )
                
                # Export this dataset (with streaming progress for large datasets)
                exported_set = await _export_sample_set(task, conn, config, idx, total_datasets)
                sections.sample_sets.append(exported_set)
                
                task.update_progress(
                    "sample_sets",
                    idx,
                    total_datasets,
                    f"Exported dataset {idx}/{total_datasets}"
                )
            
            section_sizes.sample_sets = len(json.dumps([s.dict() for s in sections.sample_sets]))
    
    # Stage 5: Finalize and save to MinIO
    task.update_progress("finalizing", 0, 1, "Finalizing export file...")
    
    metadata = ExportMetadata(
        version="1.0",
        created_at=datetime.utcnow(),
        creator=request.creator,
        section_sizes=section_sizes,
        description=request.description
    )
    
    heimdall_file = HeimdallFile(
        metadata=metadata,
        sections=sections
    )
    
    # Serialize to JSON
    file_json = heimdall_file.json(indent=2)
    file_bytes = file_json.encode('utf-8')
    file_size = len(file_bytes)
    
    # Upload to MinIO
    minio_client = MinIOClient(
        endpoint_url=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        bucket_name="heimdall-exports"
    )
    
    filename = f"export-{task_id}.heimdall"
    minio_path = f"exports/{filename}"
    
    task.update_progress("finalizing", 0, 1, "Uploading to storage...")
    
    success, result = minio_client.upload_bytes(
        file_bytes,
        minio_path,
        content_type="application/json"
    )
    
    if not success:
        raise Exception(f"Failed to upload export file: {result}")
    
    download_url = f"/api/import-export/download/{task_id}"
    
    task.update_progress("finalizing", 1, 1, "Export complete!")
    
    publisher.publish_export_completed(
        task_id=task_id,
        status='completed',
        download_url=download_url,
        file_size_bytes=file_size
    )
    
    return {
        'status': 'completed',
        'download_url': download_url,
        'file_size_bytes': file_size,
        'task_id': task_id
    }


async def _export_sample_set(
    task: ExportTask,
    conn: asyncpg.Connection,
    config,
    dataset_num: int,
    total_datasets: int
) -> ExportedSampleSet:
    """Export a single sample set with granular progress updates."""
    
    # Get dataset metadata
    dataset_row = await conn.fetchrow("""
        SELECT id, name, description, num_samples, config, quality_metrics, created_at
        FROM heimdall.synthetic_datasets
        WHERE id = $1
    """, config.dataset_id)
    
    if not dataset_row:
        raise ValueError(f"Dataset {config.dataset_id} not found")
    
    total_samples = dataset_row["num_samples"]
    offset = config.sample_offset
    limit = config.sample_limit if config.sample_limit else (total_samples - offset)
    actual_limit = min(limit, total_samples - offset)
    
    # Export features
    task.update_progress(
        "sample_sets",
        dataset_num - 1,
        total_datasets,
        f"Dataset {dataset_num}: Fetching {actual_limit} features..."
    )
    
    features_rows = await conn.fetch("""
        SELECT recording_session_id, timestamp, receiver_features,
               tx_latitude, tx_longitude, tx_altitude_m, tx_power_dbm, tx_known,
               extraction_metadata, overall_confidence, mean_snr_db,
               num_receivers_detected, gdop, extraction_failed, error_message,
               created_at, num_receivers_in_sample
        FROM heimdall.measurement_features
        WHERE dataset_id = $1
        ORDER BY created_at ASC
        OFFSET $2
        LIMIT $3
    """, config.dataset_id, offset, actual_limit)
    
    features = []
    for feat_idx, feat_row in enumerate(features_rows, 1):
        # Update progress every 50 samples
        if feat_idx % 50 == 0:
            task.update_progress(
                "sample_sets",
                dataset_num - 1,
                total_datasets,
                f"Dataset {dataset_num}: Processing feature {feat_idx}/{actual_limit}..."
            )
        
        # Parse JSON fields
        receiver_features = feat_row["receiver_features"]
        if isinstance(receiver_features, str):
            receiver_features = json.loads(receiver_features)
        if receiver_features and isinstance(receiver_features, list):
            receiver_features = [
                json.loads(rf) if isinstance(rf, str) else rf
                for rf in receiver_features
            ]
        
        extraction_metadata = feat_row["extraction_metadata"]
        if isinstance(extraction_metadata, str):
            extraction_metadata = json.loads(extraction_metadata)
        
        features.append(ExportedFeature(
            recording_session_id=str(feat_row["recording_session_id"]),
            timestamp=feat_row["timestamp"].isoformat(),
            receiver_features=receiver_features,
            tx_latitude=float(feat_row["tx_latitude"]) if feat_row["tx_latitude"] else None,
            tx_longitude=float(feat_row["tx_longitude"]) if feat_row["tx_longitude"] else None,
            tx_altitude_m=float(feat_row["tx_altitude_m"]) if feat_row["tx_altitude_m"] else None,
            tx_power_dbm=float(feat_row["tx_power_dbm"]) if feat_row["tx_power_dbm"] else None,
            tx_known=feat_row["tx_known"],
            extraction_metadata=extraction_metadata,
            overall_confidence=float(feat_row["overall_confidence"]),
            mean_snr_db=float(feat_row["mean_snr_db"]) if feat_row["mean_snr_db"] else None,
            num_receivers_detected=feat_row["num_receivers_detected"],
            gdop=float(feat_row["gdop"]) if feat_row["gdop"] else None,
            extraction_failed=feat_row["extraction_failed"],
            error_message=feat_row["error_message"],
            created_at=feat_row["created_at"].isoformat(),
            num_receivers_in_sample=feat_row["num_receivers_in_sample"],
        ))
    
    # Export IQ samples if requested
    iq_samples = []
    if config.include_iq_data:
        task.update_progress(
            "sample_sets",
            dataset_num - 1,
            total_datasets,
            f"Dataset {dataset_num}: Fetching {actual_limit} IQ samples (this may take a while)..."
        )
        
        iq_rows = await conn.fetch("""
            SELECT id, dataset_id, sample_idx, timestamp,
                   tx_lat, tx_lon, tx_alt, tx_power_dbm, frequency_hz,
                   receivers_metadata, num_receivers, gdop, mean_snr_db, overall_confidence,
                   iq_metadata, iq_storage_paths, created_at
            FROM heimdall.synthetic_iq_samples
            WHERE dataset_id = $1
            ORDER BY sample_idx ASC
            OFFSET $2
            LIMIT $3
        """, config.dataset_id, offset, actual_limit)
        
        # Initialize MinIO client for IQ data download
        minio_client = MinIOClient(
            endpoint_url=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            bucket_name="rf-signals"
        )
        
        for iq_idx, iq_row in enumerate(iq_rows, 1):
            # Update progress every 10 IQ samples (they're large)
            if iq_idx % 10 == 0:
                task.update_progress(
                    "sample_sets",
                    dataset_num - 1,
                    total_datasets,
                    f"Dataset {dataset_num}: Downloading IQ sample {iq_idx}/{actual_limit}..."
                )
            
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
            
            # Download IQ data from MinIO
            iq_data_list = []
            for receiver_id, s3_path in iq_storage_paths.items():
                try:
                    response = minio_client.s3_client.get_object(
                        Bucket="rf-signals",
                        Key=s3_path
                    )
                    iq_bytes = response["Body"].read()
                    iq_base64 = base64.b64encode(iq_bytes).decode("utf-8")
                    
                    iq_data_list.append(IQData(
                        receiver_id=receiver_id,
                        iq_data_base64=iq_base64
                    ))
                except Exception as e:
                    logger.error(f"Failed to download IQ data for {receiver_id} at {s3_path}: {e}")
            
            iq_samples.append(ExportedIQSample(
                id=str(iq_row["id"]),
                sample_idx=iq_row["sample_idx"],
                timestamp=iq_row["timestamp"].isoformat(),
                tx_lat=float(iq_row["tx_lat"]),
                tx_lon=float(iq_row["tx_lon"]),
                tx_alt=float(iq_row["tx_alt"]),
                tx_power_dbm=float(iq_row["tx_power_dbm"]),
                frequency_hz=iq_row["frequency_hz"],
                num_receivers=iq_row["num_receivers"],
                gdop=float(iq_row["gdop"]) if iq_row["gdop"] else None,
                mean_snr_db=float(iq_row["mean_snr_db"]) if iq_row["mean_snr_db"] else None,
                overall_confidence=float(iq_row["overall_confidence"]) if iq_row["overall_confidence"] else None,
                receivers_metadata=receivers_metadata,
                iq_metadata=iq_metadata,
                iq_storage_paths=iq_storage_paths,
                iq_data=iq_data_list,
                created_at=iq_row["created_at"].isoformat(),
            ))
    
    # Parse dataset config and quality_metrics
    dataset_config = dataset_row["config"]
    if isinstance(dataset_config, str):
        dataset_config = json.loads(dataset_config)
    
    quality_metrics = dataset_row["quality_metrics"]
    if isinstance(quality_metrics, str):
        quality_metrics = json.loads(quality_metrics)
    
    return ExportedSampleSet(
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
