#!/usr/bin/env python3
"""
Test script to verify the three model fixes by triggering training jobs.

Fixed models:
1. IQTransformer (iq_transformer) - Fixed sequence length handling
2. LocalizationNetViT (localization_net_vit) - Fixed forward() signature
3. LocalizationNet ConvNeXt (localization_net_convnext_large) - Fixed forward() signature

This script:
1. Submits training jobs for all three models
2. Monitors their progress via WebSocket
3. Reports if they start successfully (first epoch begins)
"""

import asyncio
import json
import sys
import time
from typing import Dict, List

import httpx
import websockets

BACKEND_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001"

# Dataset ID to use for testing (from database)
DATASET_ID = "22a722a3-5508-48a9-8657-d75bbe5629bd"

# Models to test with their configurations
MODELS_TO_TEST = [
    {
        "name": "IQTransformer - Fixed Sequence Length",
        "config": {
            "job_name": "test_iq_transformer_fixed",
            "config": {
                "dataset_ids": [DATASET_ID],  # Required field
                "model_architecture": "iq_transformer",
                "epochs": 2,  # Just 2 epochs to verify it starts
                "batch_size": 16,
                "learning_rate": 0.001,
                "accelerator": "cpu",
            }
        }
    },
    {
        "name": "LocalizationNetViT - Fixed Forward Signature",
        "config": {
            "job_name": "test_localization_net_vit_fixed",
            "config": {
                "dataset_ids": [DATASET_ID],  # Required field
                "model_architecture": "localization_net_vit",
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 0.001,
                "accelerator": "cpu",
            }
        }
    },
    {
        "name": "LocalizationNet ConvNeXt - Fixed Forward Signature",
        "config": {
            "job_name": "test_localization_net_convnext_fixed",
            "config": {
                "dataset_ids": [DATASET_ID],  # Required field
                "model_architecture": "localization_net_convnext_large",
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 0.001,
                "accelerator": "cpu",
            }
        }
    }
]


async def submit_training_job(client: httpx.AsyncClient, job_config: dict) -> str:
    """Submit a training job and return the job ID."""
    response = await client.post(
        f"{BACKEND_URL}/api/v1/training/jobs",
        json=job_config,
        timeout=30.0
    )
    response.raise_for_status()
    data = response.json()
    print(f"  Response data: {data}")  # Debug: see what keys we get
    # Try both 'id' and 'job_id' keys
    return data.get("id") or data.get("job_id")


async def monitor_job_websocket(job_id: str, timeout: int = 180) -> Dict:
    """
    Monitor a training job via WebSocket until it starts training or fails.
    
    Returns dict with:
    - success: bool (True if training started, False if failed)
    - status: str (final status)
    - error: str or None (error message if failed)
    """
    ws_url = f"{WS_URL}/api/v1/ws/training/{job_id}"
    
    print(f"  Connecting to WebSocket: {ws_url}")
    
    start_time = time.time()
    result = {
        "success": False,
        "status": "unknown",
        "error": None,
        "started_training": False,
    }
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("  ✓ WebSocket connected")
            
            while time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    
                    event_type = data.get("type")
                    
                    if event_type == "status_update":
                        status = data.get("status")
                        result["status"] = status
                        print(f"  Status: {status}")
                        
                        if status == "failed":
                            result["success"] = False
                            result["error"] = data.get("error", "Unknown error")
                            print(f"  ✗ Job failed: {result['error']}")
                            return result
                    
                    elif event_type == "epoch_start":
                        epoch = data.get("epoch", 0)
                        print(f"  ✓ Epoch {epoch} started - Training is working!")
                        result["success"] = True
                        result["started_training"] = True
                        result["status"] = "running"
                        return result
                    
                    elif event_type == "training_complete":
                        print("  ✓ Training completed!")
                        result["success"] = True
                        result["status"] = "completed"
                        return result
                    
                    elif event_type == "error":
                        result["success"] = False
                        result["error"] = data.get("error", "Unknown error")
                        result["status"] = "failed"
                        print(f"  ✗ Error: {result['error']}")
                        return result
                        
                except asyncio.TimeoutError:
                    # No message received, continue waiting
                    elapsed = time.time() - start_time
                    print(f"  Waiting... ({int(elapsed)}s elapsed)")
                    continue
            
            print(f"  ⚠ Timeout after {timeout}s")
            result["error"] = f"Timeout after {timeout}s"
            return result
            
    except Exception as e:
        print(f"  ✗ WebSocket error: {e}")
        result["error"] = str(e)
        return result


async def test_model(client: httpx.AsyncClient, model_info: dict) -> Dict:
    """Test a single model by submitting job and monitoring progress."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_info['name']}")
    print(f"{'='*70}")
    
    try:
        # Submit job
        print("Submitting training job...")
        job_id = await submit_training_job(client, model_info["config"])
        print(f"✓ Job created: {job_id}")
        
        # Monitor progress
        print("Monitoring job progress...")
        result = await monitor_job_websocket(job_id)
        
        return {
            "model_name": model_info["name"],
            "job_id": job_id,
            **result
        }
        
    except Exception as e:
        print(f"✗ Failed to test model: {e}")
        return {
            "model_name": model_info["name"],
            "job_id": None,
            "success": False,
            "error": str(e)
        }


async def main():
    """Run tests for all three fixed models."""
    print("="*70)
    print("TESTING FIXED MODELS")
    print("="*70)
    print(f"\nBackend URL: {BACKEND_URL}")
    print(f"WebSocket URL: {WS_URL}")
    print(f"\nModels to test: {len(MODELS_TO_TEST)}")
    
    results: List[Dict] = []
    
    async with httpx.AsyncClient() as client:
        # Test each model sequentially
        for model_info in MODELS_TO_TEST:
            result = await test_model(client, model_info)
            results.append(result)
            
            # Small delay between jobs
            await asyncio.sleep(2)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r["success"])
    
    for result in results:
        status_icon = "✅" if result["success"] else "❌"
        print(f"\n{status_icon} {result['model_name']}")
        print(f"   Job ID: {result.get('job_id', 'N/A')}")
        print(f"   Status: {result.get('status', 'unknown')}")
        if result.get("error"):
            print(f"   Error: {result['error']}")
        if result.get("started_training"):
            print(f"   ✓ Successfully started training (verified by epoch start)")
    
    print(f"\n{'='*70}")
    print(f"Results: {success_count}/{len(MODELS_TO_TEST)} models working")
    print(f"{'='*70}")
    
    if success_count == len(MODELS_TO_TEST):
        print("\n🎉 All fixes verified! All models can now train successfully.")
        return 0
    else:
        print(f"\n⚠ {len(MODELS_TO_TEST) - success_count} model(s) still have issues.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
