#!/usr/bin/env python3
"""
Test script to verify WebSDR connectivity fixes.

This script tests:
1. WebSDR list endpoint returns all configured receivers
2. WebSDR health endpoint returns proper format
3. API Gateway correctly proxies requests to rf-acquisition service
"""

import requests
import json
import sys

API_GATEWAY_URL = "http://localhost:8000"
RF_ACQUISITION_URL = "http://localhost:8001"


def test_websdr_list_direct():
    """Test WebSDR list endpoint directly on rf-acquisition."""
    print("Testing WebSDR list endpoint (direct)...")
    response = requests.get(f"{RF_ACQUISITION_URL}/api/v1/acquisition/websdrs")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert isinstance(data, list), "Expected list response"
    assert len(data) == 7, f"Expected 7 WebSDRs, got {len(data)}"
    
    # Check first WebSDR has required fields
    websdr = data[0]
    required_fields = ['id', 'name', 'url', 'location_name', 'latitude', 'longitude', 'is_active']
    for field in required_fields:
        assert field in websdr, f"Missing field: {field}"
    
    print(f"✓ Direct endpoint returns {len(data)} WebSDRs")
    return True


def test_websdr_list_via_gateway():
    """Test WebSDR list endpoint via API Gateway."""
    print("Testing WebSDR list endpoint (via API Gateway)...")
    response = requests.get(f"{API_GATEWAY_URL}/api/v1/acquisition/websdrs")
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert isinstance(data, list), "Expected list response"
    assert len(data) == 7, f"Expected 7 WebSDRs, got {len(data)}"
    
    print(f"✓ API Gateway proxy returns {len(data)} WebSDRs")
    return True


def test_websdr_health_direct():
    """Test WebSDR health endpoint directly on rf-acquisition."""
    print("Testing WebSDR health endpoint (direct)...")
    response = requests.get(f"{RF_ACQUISITION_URL}/api/v1/acquisition/websdrs/health", timeout=120)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert isinstance(data, dict), "Expected dict response"
    
    # Check we have status for all 7 WebSDRs
    assert len(data) == 7, f"Expected 7 WebSDR statuses, got {len(data)}"
    
    # Check format of health status
    for ws_id, status in data.items():
        required_fields = ['websdr_id', 'name', 'status', 'last_check']
        for field in required_fields:
            assert field in status, f"Missing field {field} in WebSDR {ws_id} status"
        
        assert status['status'] in ['online', 'offline'], f"Invalid status: {status['status']}"
    
    print(f"✓ Direct health check returns proper format for all {len(data)} WebSDRs")
    return True


def test_websdr_health_via_gateway():
    """Test WebSDR health endpoint via API Gateway."""
    print("Testing WebSDR health endpoint (via API Gateway)...")
    response = requests.get(f"{API_GATEWAY_URL}/api/v1/acquisition/websdrs/health", timeout=120)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert isinstance(data, dict), "Expected dict response"
    assert len(data) == 7, f"Expected 7 WebSDR statuses, got {len(data)}"
    
    print(f"✓ API Gateway health proxy returns status for all {len(data)} WebSDRs")
    print(f"\nSample WebSDR status:")
    sample_id = list(data.keys())[0]
    print(json.dumps(data[sample_id], indent=2))
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("WebSDR Connectivity Fix - Verification Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("WebSDR List (Direct)", test_websdr_list_direct),
        ("WebSDR List (API Gateway)", test_websdr_list_via_gateway),
        ("WebSDR Health (Direct)", test_websdr_health_direct),
        ("WebSDR Health (API Gateway)", test_websdr_health_via_gateway),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
