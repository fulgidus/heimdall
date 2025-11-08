#!/usr/bin/env python3
"""
Test training job submission for all 13 model architectures.
This validates that the training service can initialize and start training for each model.

Run: python3 test_all_models.py
"""

import requests
import time
import json
from typing import Dict, List

# All 13 models from MODEL_REGISTRY
ALL_MODELS = [
    "triangulation_model",
    "localization_net_convnext_large",
    "localization_net_vit",
    "iq_resnet18",
    "iq_resnet50",
    "iq_resnet101",
    "iq_vggnet",
    "iq_efficientnet_b4",
    "iq_transformer",
    "iq_wavenet",
    "iq_hybrid",
    "heimdall_net",
    "localization_ensemble_flagship"
]

BACKEND_URL = "http://localhost:8002"  # Training service port
DATASET_ID = "22a722a3-5508-48a9-8657-d75bbe5629bd"  # Synthetic dataset "Basic" with 1000 samples


def submit_training_job(model_architecture: str) -> Dict:
    """Submit a training job for a given model architecture."""
    payload = {
        "job_name": f"test_{model_architecture}",
        "model_architecture": model_architecture,
        "dataset_id": DATASET_ID,
        "total_epochs": 1,  # Minimal epochs just to test initialization
        "batch_size": 16,
        "learning_rate": 0.001,
        "use_gpu": True
    }
    
    response = requests.post(f"{BACKEND_URL}/api/v1/jobs/training", json=payload)
    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str) -> Dict:
    """Get the status of a training job."""
    response = requests.get(f"{BACKEND_URL}/api/v1/jobs/training/{job_id}")
    response.raise_for_status()
    return response.json()


def test_all_models():
    """Test training job submission for all 13 models."""
    print("\n" + "="*80)
    print("BULK TRAINING TEST: All 13 Model Architectures")
    print("="*80)
    print(f"Testing {len(ALL_MODELS)} models with dataset {DATASET_ID}")
    print("="*80 + "\n")
    
    results = []
    
    for i, model_arch in enumerate(ALL_MODELS, 1):
        print(f"[{i}/{len(ALL_MODELS)}] Testing: {model_arch}...", end=" ", flush=True)
        
        try:
            # Submit job
            job_response = submit_training_job(model_arch)
            job_id = job_response["id"]  # Use "id" not "job_id"
            
            # Wait a bit for job to start
            time.sleep(2)
            
            # Check job status
            status = get_job_status(job_id)
            job_status = status.get("status", "unknown")
            
            if job_status in ["pending", "running", "completed"]:
                print(f"✓ SUCCESS (status: {job_status})")
                results.append({
                    "model": model_arch,
                    "status": "SUCCESS",
                    "job_id": job_id,
                    "job_status": job_status
                })
            else:
                print(f"✗ FAILED (status: {job_status})")
                results.append({
                    "model": model_arch,
                    "status": "FAILED",
                    "job_id": job_id,
                    "job_status": job_status,
                    "error": status.get("error_message", "Unknown error")
                })
                
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:100]}")
            results.append({
                "model": model_arch,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    print(f"Total models: {len(ALL_MODELS)}")
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"✗ Errors: {error_count}")
    print("="*80)
    
    # Show failures
    failures = [r for r in results if r["status"] != "SUCCESS"]
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"\n  Model: {failure['model']}")
            print(f"  Status: {failure['status']}")
            if "error" in failure:
                print(f"  Error: {failure['error'][:200]}")
            if "job_status" in failure:
                print(f"  Job Status: {failure['job_status']}")
    
    print("\n" + "="*80)
    print(f"RESULT: {success_count}/{len(ALL_MODELS)} models working")
    print("="*80 + "\n")
    
    return success_count == len(ALL_MODELS)


if __name__ == "__main__":
    success = test_all_models()
    exit(0 if success else 1)
