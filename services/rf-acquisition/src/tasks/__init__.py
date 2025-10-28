"""Celery tasks module."""

from .acquire_iq import (
    acquire_iq,
    save_measurements_to_minio,
    save_measurements_to_timescaledb,
    health_check_websdrs,
)
from .uptime_monitor import (
    monitor_websdrs_uptime,
    calculate_uptime_percentage,
)
from .acquire_openwebrx import (
    acquire_openwebrx_single,
    acquire_openwebrx_all,
    health_check_openwebrx,
)

__all__ = [
    "acquire_iq",
    "save_measurements_to_minio",
    "save_measurements_to_timescaledb",
    "health_check_websdrs",
    "monitor_websdrs_uptime",
    "calculate_uptime_percentage",
    "acquire_openwebrx_single",
    "acquire_openwebrx_all",
    "health_check_openwebrx",
]
