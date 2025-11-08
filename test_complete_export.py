#!/usr/bin/env python3
"""
Test complete export functionality with precise size calculation.
"""

import requests
import json
import time

BACKEND_URL = "http://localhost:8001"

def test_complete_export():
    """Test export with models, datasets, and audio."""
    
    # Get IDs from database
    model_id = "ae7eb463-6c3a-47fd-bf30-5cfc0be85c88"
    dataset_id = "0d3b82ec-edba-4721-842c-c45d60a0f795"
    audio_id = "0337d729-60f7-4836-82da-a3ac3c95892d"
    
    print("=" * 80)
    print("Testing Complete Export with Precise Size Calculation")
    print("=" * 80)
    
    # Create export request
    export_request = {
        "creator": {
            "username": "test_user",
            "name": "Test User"
        },
        "description": "Complete export test with all sections",
        "include_settings": True,
        "include_sources": True,
        "include_websdrs": True,
        "model_ids": [model_id],
        "sample_set_configs": [
            {
                "dataset_id": dataset_id,
                "sample_offset": 0,
                "sample_limit": 2,  # Just 2 samples to test
                "include_iq_data": True
            }
        ],
        "audio_library_ids": [audio_id]
    }
    
    print("\n1. Submitting export request...")
    print(f"   - Models: 1 ({model_id[:8]}...)")
    print(f"   - Datasets: 1 ({dataset_id[:8]}..., 2 samples with IQ)")
    print(f"   - Audio: 1 ({audio_id[:8]}...)")
    
    response = requests.post(
        f"{BACKEND_URL}/api/import-export/export/async",
        json=export_request
    )
    
    if response.status_code != 200:
        print(f"\n❌ Failed to submit export: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    print(f"\n✅ Export task submitted: {task_id}")
    
    # Poll for completion
    print("\n2. Waiting for export to complete...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        time.sleep(2)
        
        status_response = requests.get(
            f"{BACKEND_URL}/api/import-export/export/{task_id}/status"
        )
        
        if status_response.status_code != 200:
            print(f"\n❌ Failed to get status: {status_response.status_code}")
            return
        
        status = status_response.json()
        state = status["state"]
        progress = status.get("progress", {})
        
        if progress:
            stage = progress.get("stage", "unknown")
            current = progress.get("current", 0)
            total = progress.get("total", 0)
            message = progress.get("message", "")
            print(f"   [{stage}] {current}/{total} - {message}")
        
        if state == "SUCCESS":
            print("\n✅ Export completed successfully!")
            print(f"\n3. Export Results:")
            print(f"   Download URL: {status['result']['download_url']}")
            print(f"   File size: {status['result']['file_size_bytes']:,} bytes")
            
            # Show section sizes
            if "section_sizes" in status["result"]:
                sizes = status["result"]["section_sizes"]
                print(f"\n4. Section Sizes (Precise Calculation):")
                
                total_sections = 0
                for section, size in sizes.items():
                    if size and size > 0:
                        print(f"   - {section:20s}: {size:>12,} bytes ({size / 1024 / 1024:.2f} MB)")
                        total_sections += size
                
                print(f"   {'=' * 35}")
                print(f"   {'TOTAL SECTIONS':20s}: {total_sections:>12,} bytes ({total_sections / 1024 / 1024:.2f} MB)")
                print(f"   {'ACTUAL FILE':20s}: {status['result']['file_size_bytes']:>12,} bytes ({status['result']['file_size_bytes'] / 1024 / 1024:.2f} MB)")
                
                # Calculate accuracy
                accuracy = (total_sections / status['result']['file_size_bytes']) * 100
                print(f"   {'SIZE ACCURACY':20s}: {accuracy:.1f}%")
            
            return
        
        elif state == "FAILURE":
            print(f"\n❌ Export failed!")
            print(f"   Error: {status.get('error', 'Unknown error')}")
            return
    
    print(f"\n❌ Export timed out after {max_wait} seconds")

if __name__ == "__main__":
    test_complete_export()
