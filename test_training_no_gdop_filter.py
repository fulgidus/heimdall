#!/usr/bin/env python3
"""
Test training with triangulation_model and NO GDOP filtering.

This should demonstrate that the RF fixes work by showing improved RMSE.
"""

import requests
import time
import json

BACKEND_URL = "http://localhost:8001"
DATASET_ID = "fb242aa2-e605-4529-8118-088c55862efc"  # rf_features_fixed

def submit_training_job():
    """Submit training job with triangulation_model and NO GDOP filter."""
    
    job_request = {
        "job_name": "triangulation_no_gdop_filter_test",
        "config": {
            "model_architecture": "triangulation_model",
            "dataset_ids": [DATASET_ID],  # List of dataset IDs
            "epochs": 10,  # Short test run
            "batch_size": 16,
            "learning_rate": 0.001,
            "accelerator": "gpu",
            "devices": 1,
            "num_workers": 0,
            "early_stop_patience": 5,
            
            # CRITICAL: Disable GDOP filtering
            "max_gdop": 999.0,  # Accept all samples regardless of GDOP
            "min_snr_db": -999.0,  # No SNR filter
            
            # Use GPU preloading for speed
            "preload_to_gpu": True
        }
    }
    
    print(f"🚀 Submitting training job: {job_request['job_name']}")
    print(f"   Model: {job_request['config']['model_architecture']}")
    print(f"   Dataset: {DATASET_ID}")
    print(f"   Epochs: {job_request['config']['epochs']}")
    print(f"   GDOP filter: DISABLED (max={job_request['config']['max_gdop']})")
    print()
    
    response = requests.post(
        f"{BACKEND_URL}/api/v1/training/jobs",
        json=job_request
    )
    
    if response.status_code not in [200, 201]:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(response.text)
        return None
    
    job_data = response.json()
    job_id = job_data["id"]
    
    print(f"✅ Job submitted successfully!")
    print(f"   Job ID: {job_id}")
    print(f"   Status: {job_data['status']}")
    print()
    
    return job_id


def monitor_job(job_id, check_interval=5):
    """Monitor training job until completion."""
    
    print(f"📊 Monitoring job {job_id}...")
    print(f"   Checking every {check_interval}s")
    print()
    
    prev_status = None
    prev_epoch = None
    
    while True:
        try:
            response = requests.get(f"{BACKEND_URL}/api/v1/training/jobs/{job_id}")
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch job status: {response.status_code}")
                break
            
            job = response.json()
            status = job.get("status", "unknown")
            current_epoch = job.get("current_epoch", 0)
            total_epochs = job.get("total_epochs", 0)
            train_loss = job.get("train_loss")
            val_loss = job.get("val_loss")
            train_accuracy = job.get("train_accuracy")
            val_accuracy = job.get("val_accuracy")
            train_samples = job.get("train_samples", 0)
            val_samples = job.get("val_samples", 0)
            
            # Print status updates
            if status != prev_status or current_epoch != prev_epoch:
                if status == "running":
                    print(f"⚙️  Epoch {current_epoch}/{total_epochs}")
                    if train_samples > 0:
                        print(f"   📈 Samples: {train_samples} train + {val_samples} val")
                    if train_loss is not None:
                        print(f"   📉 Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
                    if train_accuracy is not None:
                        print(f"   🎯 Train RMSE: {train_accuracy:.2f}m, Val RMSE: {val_accuracy:.2f}m")
                    print()
                elif status == "completed":
                    print(f"✅ Training completed!")
                    print(f"   Final epoch: {current_epoch}/{total_epochs}")
                    print(f"   Final train loss: {train_loss:.6f}")
                    print(f"   Final val loss: {val_loss:.6f}")
                    print(f"   Final train RMSE: {train_accuracy:.2f}m")
                    print(f"   Final val RMSE: {val_accuracy:.2f}m")
                    print()
                    
                    # Check if RF fixes worked (expect < 1km RMSE vs 67-70km baseline)
                    if val_accuracy < 1000:
                        print(f"🎉 SUCCESS! Val RMSE ({val_accuracy:.2f}m) < 1km!")
                        print(f"   RF fixes WORKING! (Baseline was 67-70km)")
                    else:
                        print(f"⚠️  Val RMSE ({val_accuracy:.2f}m) still high")
                        print(f"   Expected < 1km with RF fixes applied")
                    
                    return job
                    
                elif status == "failed":
                    print(f"❌ Training failed!")
                    error = job.get("error_message", "Unknown error")
                    print(f"   Error: {error}")
                    return job
                
                prev_status = status
                prev_epoch = current_epoch
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n⏸️  Monitoring interrupted")
            break
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            break
    
    return None


if __name__ == "__main__":
    # Submit training job
    job_id = submit_training_job()
    
    if not job_id:
        print("❌ Failed to submit training job")
        exit(1)
    
    # Monitor until completion
    final_job = monitor_job(job_id)
    
    if final_job and final_job.get("status") == "completed":
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(json.dumps(final_job, indent=2))
        exit(0)
    else:
        print("\n" + "="*60)
        print("TRAINING FAILED OR INTERRUPTED")
        print("="*60)
        exit(1)
