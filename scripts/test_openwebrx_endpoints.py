#!/usr/bin/env python3
"""
Test script to verify OpenWebRX endpoints work end-to-end.

This script:
1. Checks if services are running
2. Triggers a short acquisition (10 seconds)
3. Polls task status until complete
4. Displays results

Usage:
    python test_openwebrx_endpoints.py
"""

import requests
import time
import sys
from datetime import datetime

# Configuration
API_GATEWAY_URL = "http://localhost:8000"
RF_ACQUISITION_URL = "http://localhost:8001"

# Test parameters
TEST_WEBSDR_URL = "http://sdr1.ik1jns.it:8076"
TEST_DURATION = 10  # seconds


def print_header(text):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_services():
    """Check if services are running."""
    print_header("1. Checking Services")
    
    services = {
        "API Gateway": f"{API_GATEWAY_URL}/health",
        "RF Acquisition": f"{RF_ACQUISITION_URL}/health",
    }
    
    all_healthy = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✅ {name}: HEALTHY")
            else:
                print(f"❌ {name}: UNHEALTHY (status {response.status_code})")
                all_healthy = False
        except requests.RequestException as e:
            print(f"❌ {name}: UNREACHABLE ({e})")
            all_healthy = False
    
    if not all_healthy:
        print("\n⚠️  Some services are not running!")
        print("Make sure to start:")
        print("  1. API Gateway: cd services/api-gateway && uvicorn src.main:app --reload")
        print("  2. RF Acquisition: cd services/rf-acquisition && uvicorn src.main:app --port 8001 --reload")
        print("  3. Celery Worker: cd services/rf-acquisition && celery -A src.main worker --loglevel=info")
        return False
    
    return True


def trigger_acquisition():
    """Trigger OpenWebRX acquisition."""
    print_header("2. Triggering Acquisition")
    
    payload = {
        "websdr_url": TEST_WEBSDR_URL,
        "duration_seconds": TEST_DURATION,
        "save_fft": False,  # Don't save to DB for testing
        "save_audio": False
    }
    
    print(f"📡 Starting acquisition from: {TEST_WEBSDR_URL}")
    print(f"⏱️  Duration: {TEST_DURATION} seconds")
    
    try:
        # Try API Gateway first
        response = requests.post(
            f"{API_GATEWAY_URL}/api/v1/acquisition/openwebrx/acquire",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Acquisition started!")
            print(f"   Task ID: {result['task_id']}")
            print(f"   Message: {result['message']}")
            print(f"   ETA: {result['estimated_completion_time']}")
            return result['task_id']
        
        else:
            print(f"❌ API Gateway failed (status {response.status_code})")
            print(f"   Response: {response.text}")
            
            # Try RF Acquisition service directly
            print("\n🔄 Trying RF Acquisition service directly...")
            response = requests.post(
                f"{RF_ACQUISITION_URL}/api/v1/acquisition/openwebrx/acquire",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Acquisition started (direct)!")
                print(f"   Task ID: {result['task_id']}")
                return result['task_id']
            else:
                print(f"❌ RF Acquisition also failed: {response.text}")
                return None
    
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def poll_task_status(task_id):
    """Poll task status until completion."""
    print_header("3. Monitoring Task Progress")
    
    print(f"📊 Polling task {task_id}...")
    print(f"   (This will take ~{TEST_DURATION + 5} seconds)")
    
    start_time = time.time()
    poll_interval = 2  # seconds
    max_polls = (TEST_DURATION + 30) // poll_interval
    
    for i in range(max_polls):
        try:
            # Try API Gateway first
            response = requests.get(
                f"{API_GATEWAY_URL}/api/v1/acquisition/openwebrx/status/{task_id}",
                timeout=5
            )
            
            if response.status_code != 200:
                # Try direct
                response = requests.get(
                    f"{RF_ACQUISITION_URL}/api/v1/acquisition/openwebrx/status/{task_id}",
                    timeout=5
                )
            
            if response.status_code == 200:
                result = response.json()
                state = result.get("state")
                elapsed = time.time() - start_time
                
                if state == "PENDING":
                    print(f"   [{elapsed:5.1f}s] ⏳ PENDING: Task queued")
                
                elif state == "STARTED":
                    info = result.get("info", "Running...")
                    print(f"   [{elapsed:5.1f}s] ⚙️  STARTED: {info}")
                
                elif state == "SUCCESS":
                    print(f"\n✅ Task completed successfully in {elapsed:.1f}s!")
                    return result.get("result")
                
                elif state == "FAILURE":
                    error = result.get("error", "Unknown error")
                    print(f"\n❌ Task failed after {elapsed:.1f}s:")
                    print(f"   Error: {error}")
                    return None
                
                else:
                    print(f"   [{elapsed:5.1f}s] ❓ {state}: {result.get('info', '')}")
            
            else:
                print(f"   ⚠️  Status check failed: HTTP {response.status_code}")
        
        except requests.RequestException as e:
            print(f"   ⚠️  Request failed: {e}")
        
        # Wait before next poll
        time.sleep(poll_interval)
    
    print(f"\n⏰ Timeout: Task did not complete in {max_polls * poll_interval}s")
    return None


def display_results(result):
    """Display acquisition results."""
    print_header("4. Results")
    
    if not result:
        print("❌ No results available")
        return
    
    print(f"📊 Acquisition Statistics:")
    print(f"   WebSDR: {result.get('websdr_url', 'N/A')}")
    print(f"   Duration: {result.get('duration', 0)} seconds")
    print(f"   FFT frames: {result.get('fft_frames', 0)}")
    print(f"   Audio frames: {result.get('audio_frames', 0)}")
    print(f"   Text messages: {result.get('text_messages', 0)}")
    print(f"   Errors: {result.get('errors', 0)}")
    print(f"   Success: {result.get('success', False)}")
    
    # Calculate rates
    duration = result.get('duration', 1)
    fft_fps = result.get('fft_frames', 0) / duration
    audio_fps = result.get('audio_frames', 0) / duration
    
    print(f"\n📈 Performance:")
    print(f"   FFT rate: {fft_fps:.1f} fps")
    print(f"   Audio rate: {audio_fps:.1f} fps")
    
    if result.get('success'):
        print("\n✅ All systems operational!")
    else:
        print(f"\n⚠️  Acquisition completed with {result.get('errors', 0)} errors")


def main():
    """Main test flow."""
    print("\n" + "=" * 70)
    print("  OpenWebRX Endpoint Integration Test")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    # Step 1: Check services
    if not check_services():
        print("\n❌ Test aborted: Services not available")
        sys.exit(1)
    
    # Step 2: Trigger acquisition
    task_id = trigger_acquisition()
    if not task_id:
        print("\n❌ Test failed: Could not start acquisition")
        sys.exit(1)
    
    # Step 3: Poll status
    result = poll_task_status(task_id)
    
    # Step 4: Display results
    display_results(result)
    
    # Final verdict
    print("\n" + "=" * 70)
    if result and result.get('success'):
        print("  ✅ TEST PASSED - All endpoints working!")
    else:
        print("  ❌ TEST FAILED - Check logs for details")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
