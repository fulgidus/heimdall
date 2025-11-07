"""Celery tasks module."""

from .acquire_iq import (
    acquire_iq,
    health_check_websdrs,
    save_measurements_to_minio,
    save_measurements_to_timescaledb,
)
from .audio_preprocessing import preprocess_audio_file
from .batch_feature_extraction import (
    backfill_all_features,
    batch_feature_extraction_task,
)
from .comprehensive_health_monitor import monitor_comprehensive_health
from .minio_lifecycle import cleanup_orphan_files, get_storage_stats
from .services_health_monitor import monitor_services_health
from .uptime_monitor import (
    calculate_uptime_percentage,
    monitor_websdrs_uptime,
)

__all__ = [
    "acquire_iq",
    "save_measurements_to_minio",
    "save_measurements_to_timescaledb",
    "health_check_websdrs",
    "monitor_websdrs_uptime",
    "calculate_uptime_percentage",
    "monitor_services_health",
    "monitor_comprehensive_health",
    "batch_feature_extraction_task",
    "backfill_all_features",
    "preprocess_audio_file",
    "cleanup_orphan_files",
    "get_storage_stats",
]
