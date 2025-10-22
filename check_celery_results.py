#!/usr/bin/env python3
"""Check Celery task results via result backend."""

import sys
sys.path.insert(0, 'services/rf-acquisition')

from celery import Celery
from celery.result import AsyncResult
import time

# Initialize Celery
celery = Celery('rf_acquisition')
celery.config_from_object('src.config:CeleryConfig')

# Recent task IDs from our load test (use your actual IDs)
task_ids = [
    '90e448a8-0f2d-4b0d-9275-1fbf59d4b795',
    '7ea715df-a959-46f7-ab58-86bb09be1ef8',
]

for task_id in task_ids:
    result = AsyncResult(task_id, app=celery)
    print(f"Task: {task_id}")
    print(f"  State: {result.state}")
    print(f"  Info: {result.info}")
    print(f"  Result: {result.result}")
    print()
