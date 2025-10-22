"""Celery tasks for RF acquisition orchestration"""
import os
import json
from datetime import datetime
from celery import Celery
from kombu import Exchange, Queue

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

celery_app = Celery(
    "data_ingestion",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Configure task routing
celery_app.conf.task_routes = {
    "services.data_ingestion_web.tasks.trigger_acquisition": {
        "queue": "acquisition.websdr-fetch",
        "routing_key": "acquisition.websdr-fetch",
    }
}

# Configure queues
celery_app.conf.queues = (
    Queue(
        "acquisition.websdr-fetch",
        Exchange("acquisition", type="direct"),
        routing_key="acquisition.websdr-fetch",
    ),
)

# General configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=300,  # 5 minutes soft timeout
)


@celery_app.task(bind=True, name="data_ingestion.trigger_acquisition")
def trigger_acquisition(
    self,
    session_id: int,
    frequency_mhz: float,
    duration_seconds: int,
) -> dict:
    """
    Trigger RF acquisition from WebSDR receivers.
    This task calls the rf-acquisition service API.
    """
    import requests
    from .repository import SessionRepository
    from .database import SessionLocal

    db = SessionLocal()
    repo = SessionRepository()

    try:
        # Update session as processing
        from .models.session import SessionStatus
        session = repo.update_status(
            db,
            session_id,
            status=SessionStatus.PROCESSING,
            celery_task_id=self.request.id,
            started_at=datetime.utcnow(),
        )
        
        # Call RF acquisition API
        # The rf-acquisition service is available at http://rf-acquisition:8001 in docker-compose
        rf_api_url = os.getenv("RF_ACQUISITION_API_URL", "http://rf-acquisition:8001")
        
        response = requests.post(
            f"{rf_api_url}/api/acquire",
            json={
                "frequency_mhz": frequency_mhz,
                "duration_seconds": duration_seconds,
            },
            timeout=duration_seconds + 60,  # Give extra time for timeout
        )
        response.raise_for_status()
        
        acquisition_data = response.json()
        
        # Mark as completed with results
        repo.update_completed(
            db,
            session_id,
            result_metadata=acquisition_data.get("metadata", {}),
            minio_path=acquisition_data.get("minio_path", ""),
        )
        
        return {
            "status": "completed",
            "session_id": session_id,
            "task_id": self.request.id,
            "result": acquisition_data,
        }

    except requests.exceptions.RequestException as e:
        error_msg = f"RF acquisition failed: {str(e)}"
        repo.update_failed(db, session_id, error_msg)
        return {
            "status": "failed",
            "session_id": session_id,
            "task_id": self.request.id,
            "error": error_msg,
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        repo.update_failed(db, session_id, error_msg)
        return {
            "status": "failed",
            "session_id": session_id,
            "task_id": self.request.id,
            "error": error_msg,
        }
    finally:
        db.close()
