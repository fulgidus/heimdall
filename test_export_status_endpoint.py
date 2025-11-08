#!/usr/bin/env python3
"""
Test the export status endpoint.

Tests that:
1. Export task submission returns a task_id
2. Status endpoint returns valid status information
3. Status endpoint includes section_sizes when complete
"""

import time
import requests
import json
from typing import Dict, Any, Optional

BASE_URL = "http://localhost:8001"


def submit_export(audio_id: str, model_id: Optional[str] = None) -> Optional[str]:
    """Submit an export job with small audio file."""
    export_request = {
        "creator": {
            "username": "test_user",
            "name": "Status Endpoint Test"
        },
        "description": "Testing status endpoint with small audio file",
        "include_settings": False,
        "include_sources": False,
        "include_websdrs": False,
        "include_sessions": False,
        "sample_set_configs": [],
        "model_ids": [model_id] if model_id else [],
        "audio_library_ids": [audio_id],
    }
    
    print(f"\n📤 Submitting export request...")
    response = requests.post(
        f"{BASE_URL}/api/import-export/export/async",
        json=export_request,
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"❌ Export submission failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    result = response.json()
    task_id = result.get("task_id")
    print(f"✅ Export submitted successfully!")
    print(f"   Task ID: {task_id}")
    return task_id


def check_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Check export task status."""
    response = requests.get(
        f"{BASE_URL}/api/import-export/export/{task_id}/status",
        timeout=10
    )
    
    if response.status_code != 200:
        print(f"❌ Status check failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    return response.json()


def main():
    print("=" * 80)
    print("🧪 EXPORT STATUS ENDPOINT TEST")
    print("=" * 80)
    
    # Test with small audio file (96 chunks)
    # Model: ae7eb463-6c3a-47fd-bf30-5cfc0be85c88
    # Audio: 91f8f915-da00-4106-a52b-ccdae1765807 (so-fresh.mp3, 96 chunks)
    
    audio_id = "91f8f915-da00-4106-a52b-ccdae1765807"
    model_id = "ae7eb463-6c3a-47fd-bf30-5cfc0be85c88"
    
    # Submit export
    task_id = submit_export(audio_id, model_id)
    if not task_id:
        print("\n❌ TEST FAILED: Could not submit export")
        return 1
    
    # Poll status
    print(f"\n⏳ Polling status every 2 seconds...")
    max_polls = 60  # Max 2 minutes
    poll_count = 0
    status_data: Optional[Dict[str, Any]] = None
    
    while poll_count < max_polls:
        time.sleep(2)
        poll_count += 1
        
        status_data = check_status(task_id)
        if not status_data:
            print(f"\n❌ TEST FAILED: Status endpoint returned error")
            return 1
        
        status = status_data.get("status")
        progress = status_data.get("progress", 0)
        message = status_data.get("message", "")
        
        print(f"\r[Poll {poll_count:2d}] Status: {status:12s} | Progress: {progress:3d}% | {message}", end="", flush=True)
        
        if status == "completed":
            print("\n\n✅ Export completed!")
            
            # Check for section_sizes
            section_sizes = status_data.get("section_sizes")
            if section_sizes:
                print(f"\n📊 Section Sizes:")
                print(f"   Settings:      {section_sizes.get('settings', 0):,} bytes")
                print(f"   Sources:       {section_sizes.get('sources', 0):,} bytes")
                print(f"   WebSDRs:       {section_sizes.get('websdrs', 0):,} bytes")
                print(f"   Sessions:      {section_sizes.get('sessions', 0):,} bytes")
                print(f"   Sample Sets:   {section_sizes.get('sample_sets', 0):,} bytes")
                print(f"   Models:        {section_sizes.get('models', 0):,} bytes")
                print(f"   Audio Library: {section_sizes.get('audio_library', 0):,} bytes")
                total = sum(section_sizes.values())
                print(f"   TOTAL:         {total:,} bytes ({total / 1024 / 1024:.2f} MB)")
            else:
                print(f"\n⚠️  WARNING: No section_sizes in response")
                print(f"Full response: {json.dumps(status_data, indent=2)}")
            
            # Check for download URL
            download_url = status_data.get("download_url")
            if download_url:
                print(f"\n📥 Download URL: {download_url}")
            else:
                print(f"\n⚠️  WARNING: No download_url in response")
            
            print("\n✅ TEST PASSED: Status endpoint working correctly!")
            return 0
        
        elif status == "failed":
            print(f"\n\n❌ Export failed: {message}")
            print(f"Full response: {json.dumps(status_data, indent=2)}")
            return 1
        
        elif status == "cancelled":
            print(f"\n\n⚠️  Export cancelled")
            return 1
    
    print(f"\n\n⏱️  Timeout after {max_polls * 2} seconds")
    print(f"Last status: {status_data}")
    return 1


if __name__ == "__main__":
    exit(main())
