#!/usr/bin/env python3
"""
Script to re-queue a training job that is stuck in 'queued' status.
This will send a new Celery task for the specified job ID.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from celery import Celery

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "requeue_script",
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND
)


def requeue_training_job(job_id: str):
    """Send a new Celery task to start the training job."""
    print(f"Re-queueing training job: {job_id}")
    
    task = celery_app.send_task(
        'src.tasks.training_task.start_training_job',
        args=[job_id],
        queue='training'
    )
    
    print(f"✅ Task sent successfully!")
    print(f"   Task ID: {task.id}")
    print(f"   Queue: training")
    print(f"   Job ID: {job_id}")
    
    return task


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python requeue_training_job.py <job_id>")
        print("\nExample:")
        print("  python requeue_training_job.py 5baa8902-211b-4f8d-b6cc-3110984b0b2a")
        sys.exit(1)
    
    job_id = sys.argv[1]
    requeue_training_job(job_id)
