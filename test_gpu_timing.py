import requests
import time

BASE_URL = "http://localhost:8002"

print("Submitting 5 sample job...")
response = requests.post(
    f"{BASE_URL}/datasets/synthetic",
    json={
        "num_samples": 5,
        "dataset_type": "iq_processed",
        "receivers": "fixed"
    }
)
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# Wait for completion
while True:
    status_resp = requests.get(f"{BASE_URL}/tasks/{task_id}")
    status = status_resp.json()["status"]
    if status in ["completed", "failed"]:
        print(f"Final status: {status}")
        break
    time.sleep(0.5)

print("Done!")
