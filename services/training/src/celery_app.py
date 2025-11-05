"""
Celery application instance for training service.

This module creates a single Celery app instance that is shared
across the training service. Import this module to access the
configured celery_app instance.
"""

from celery import Celery
from .config import settings

SERVICE_NAME = "training"

# Create Celery app instance
celery_app = Celery(
    SERVICE_NAME,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend_url
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 6,  # 6 hours for training
    task_soft_time_limit=3600 * 5.5,  # 5.5 hours
)

# Auto-discover tasks in tasks module
celery_app.autodiscover_tasks(['src.tasks'])
