#!/usr/bin/env python3
"""
Validate precise size calculation implementation.
Test with smaller dataset to verify section size accuracy.
"""

import requests
import json
import time

BACKEND_URL = "http://localhost:8001"

def test_small_export_with_size_validation():
    """Test export with small audio file to validate size calculation."""
    
    # IDs from database
    model_id = "ae7eb463-6c3a-47fd-bf30-5cfc0be85c88"  # ttt model
    dataset_id = "0d3b82ec-edba-4721-842c-c45d60a0f795"  # VHF dataset
    audio_id = "91f8f915-da00-4106-a52b-ccdae1765807"  # so-fresh.mp3 (96 chunks)
    
    print("=" * 80)
    print("Testing Precise Size Calculation")
    print("=" * 80)
    print(f"\n📊 Test Configuration:")
    print(f"   - Model: 1 (ONNX ~134KB)")
    print(f"   - Dataset: 2 IQ samples (~2MB)")
    print(f"   - Audio: 96 chunks (~3MB)")
    print(f"   - Sources: Yes")
    print(f"   - WebSDRs: Yes")
    
    export_request = {
        "creator": {
            "username": "size_test",
            "name": "Size Test User"
        },
        "description": "Testing precise size calculation with small dataset",
        "include_settings": True,
        "include_sources": True,
        "include_websdrs": True,
        "model_ids": [model_id],
        "sample_set_configs": [
            {
                "dataset_id": dataset_id,
                "sample_offset": 0,
                "sample_limit": 2,
                "include_iq_data": True
            }
        ],
        "audio_library_ids": [audio_id]
    }
    
    print("\n🚀 Submitting export request...")
    response = requests.post(
        f"{BACKEND_URL}/api/import-export/export/async",
        json=export_request
    )
    
    if response.status_code != 200:
        print(f"\n❌ Failed: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    print(f"✅ Task ID: {task_id}")
    
    # Poll for completion
    print("\n⏳ Waiting for export...")
    max_wait = 180
    start_time = time.time()
    last_stage = None
    
    while time.time() - start_time < max_wait:
        time.sleep(2)
        
        status_response = requests.get(
            f"{BACKEND_URL}/api/import-export/export/{task_id}/status"
        )
        
        if status_response.status_code != 200:
            # Task might still be running, continue
            continue
        
        status = status_response.json()
        state = status.get("state")
        progress = status.get("progress", {})
        
        if progress:
            stage = progress.get("stage", "unknown")
            if stage != last_stage:
                print(f"   📍 Stage: {stage}")
                last_stage = stage
        
        if state == "SUCCESS":
            print("\n✅ Export Completed!")
            result_data = status["result"]
            
            # Display file size
            file_size_mb = result_data["file_size_bytes"] / 1024 / 1024
            print(f"\n📁 File Info:")
            print(f"   Total Size: {file_size_mb:.2f} MB ({result_data['file_size_bytes']:,} bytes)")
            print(f"   Download URL: {result_data['download_url']}")
            
            # Check if section sizes are available
            if "section_sizes" in result_data:
                sizes = result_data["section_sizes"]
                print(f"\n📊 Section Sizes (Precise Calculation):")
                print(f"   {'Section':<20} {'Size (bytes)':>15} {'Size (MB)':>12}")
                print(f"   {'-' * 50}")
                
                total_sections = 0
                for section, size in sorted(sizes.items()):
                    if size and size > 0:
                        size_mb = size / 1024 / 1024
                        print(f"   {section:<20} {size:>15,} {size_mb:>11.2f}")
                        total_sections += size
                
                print(f"   {'-' * 50}")
                total_mb = total_sections / 1024 / 1024
                actual_mb = result_data['file_size_bytes'] / 1024 / 1024
                print(f"   {'CALCULATED TOTAL':<20} {total_sections:>15,} {total_mb:>11.2f}")
                print(f"   {'ACTUAL FILE SIZE':<20} {result_data['file_size_bytes']:>15,} {actual_mb:>11.2f}")
                
                # Calculate accuracy
                if result_data['file_size_bytes'] > 0:
                    accuracy = (total_sections / result_data['file_size_bytes']) * 100
                    difference_mb = abs(actual_mb - total_mb)
                    print(f"\n🎯 Size Calculation Accuracy:")
                    print(f"   Predicted: {accuracy:.2f}%")
                    print(f"   Difference: {difference_mb:.2f} MB")
                    
                    if accuracy >= 95:
                        print(f"   ✅ EXCELLENT (>95%)")
                    elif accuracy >= 90:
                        print(f"   ✅ GOOD (90-95%)")
                    elif accuracy >= 85:
                        print(f"   ⚠️  ACCEPTABLE (85-90%)")
                    else:
                        print(f"   ❌ NEEDS IMPROVEMENT (<85%)")
                
                # Breakdown by type
                print(f"\n📈 Size Breakdown:")
                metadata_size = sizes.get("settings", 0) + sizes.get("sources", 0) + sizes.get("websdrs", 0)
                binary_size = sizes.get("models", 0) + sizes.get("sample_sets", 0) + sizes.get("audio_library", 0)
                
                metadata_mb = metadata_size / 1024 / 1024
                binary_mb = binary_size / 1024 / 1024
                
                print(f"   Metadata (settings, sources, websdrs): {metadata_mb:.2f} MB")
                print(f"   Binary (models, IQ, audio): {binary_mb:.2f} MB")
                
            else:
                print("\n⚠️  Section sizes not available in result")
            
            return True
        
        elif state == "FAILURE":
            print(f"\n❌ Export failed!")
            print(f"   Error: {status.get('error', 'Unknown')}")
            return False
    
    print(f"\n❌ Timeout after {max_wait}s")
    return False

if __name__ == "__main__":
    success = test_small_export_with_size_validation()
    exit(0 if success else 1)
