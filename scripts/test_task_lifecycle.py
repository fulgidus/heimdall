#!/usr/bin/env python3
"""Test the complete task lifecycle."""

import requests
import time

# Submit
resp = requests.post('http://localhost:8001/api/v1/acquisition/acquire', json={
    'frequency_mhz': 145.5,
    'duration_seconds': 3
})
task_id = resp.json()['task_id']
print(f'Task: {task_id}')

# Wait for completion (WebSDR timeout ~40s + processing)
for i in range(15):
    time.sleep(3)
    status = requests.get(f'http://localhost:8001/api/v1/acquisition/status/{task_id}').json()
    print(f'[{i}] Status: {status["status"]} | Progress: {status.get("progress", "N/A")}')
    if status['status'] in ['SUCCESS', 'FAILURE', 'PARTIAL_FAILURE']:
        print(f'\nFINAL STATUS: {status["status"]}')
        print(f'Full response: {status}')
        break
else:
    print('\nTimeout: Task did not complete within 45 seconds')
