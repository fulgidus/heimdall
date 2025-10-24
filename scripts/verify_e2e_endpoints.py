#!/usr/bin/env python3
"""
Quick verification script to test new API endpoints.
Tests that all newly added stub endpoints respond correctly.
"""

import requests
import json
from typing import Dict, Any

# API Gateway URL (adjust if needed)
BASE_URL = "http://localhost:8000"

# Test cases: (method, endpoint, expected_status, description)
TEST_CASES = [
    # Auth endpoints
    ("GET", "/api/v1/auth/me", 200, "Get current user info"),
    ("GET", "/api/v1/auth/check", 200, "Check auth status"),
    
    # Profile endpoints
    ("GET", "/api/v1/profile", 200, "Get user profile"),
    ("GET", "/api/v1/profile/history", 200, "Get user activity history"),
    
    # User endpoints
    ("GET", "/api/v1/user", 200, "Get user info"),
    ("GET", "/api/v1/user/activity", 200, "Get user activity"),
    ("GET", "/api/v1/user/preferences", 200, "Get user preferences"),
    
    # Settings endpoints
    ("GET", "/api/v1/settings", 200, "Get app settings"),
    ("GET", "/api/v1/config", 200, "Get app config"),
    
    # Dashboard endpoints
    ("GET", "/api/v1/stats", 200, "Get dashboard stats"),
    ("GET", "/api/v1/activity", 200, "Get recent activity"),
    ("GET", "/api/v1/recent", 200, "Get recent items"),
    
    # System endpoints
    ("GET", "/api/v1/system/status", 200, "Get system status"),
    ("GET", "/api/v1/system/services", 200, "Get system services"),
    ("GET", "/api/v1/system/metrics", 200, "Get system metrics"),
    
    # Localization endpoints
    ("GET", "/api/v1/localizations", 200, "Get recent localizations"),
    
    # Sessions endpoints
    ("GET", "/api/v1/sessions/analytics", 200, "Get session analytics"),
    
    # Analytics endpoints
    ("GET", "/api/v1/analytics/system", 200, "Get analytics system metrics"),
    ("GET", "/api/v1/analytics/predictions/metrics", 200, "Get prediction metrics"),
    ("GET", "/api/v1/analytics/websdr/performance", 200, "Get WebSDR performance"),
]


def test_endpoint(method: str, endpoint: str, expected_status: int, description: str) -> Dict[str, Any]:
    """Test a single endpoint."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json={}, timeout=5)
        elif method == "PATCH":
            response = requests.patch(url, json={}, timeout=5)
        else:
            return {
                "success": False,
                "endpoint": endpoint,
                "error": f"Unsupported method: {method}"
            }
        
        success = response.status_code == expected_status
        
        return {
            "success": success,
            "endpoint": endpoint,
            "method": method,
            "description": description,
            "status_code": response.status_code,
            "expected_status": expected_status,
            "response_size": len(response.content),
        }
    
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "endpoint": endpoint,
            "error": "Connection refused - is the API gateway running?"
        }
    except Exception as e:
        return {
            "success": False,
            "endpoint": endpoint,
            "error": str(e)
        }


def main():
    """Run all tests and print results."""
    print("=" * 80)
    print("E2E API Endpoint Verification")
    print("=" * 80)
    print(f"\nTesting API Gateway at: {BASE_URL}")
    print(f"Total endpoints to test: {len(TEST_CASES)}\n")
    
    results = []
    passed = 0
    failed = 0
    
    for method, endpoint, expected_status, description in TEST_CASES:
        result = test_endpoint(method, endpoint, expected_status, description)
        results.append(result)
        
        if result.get("success"):
            passed += 1
            status = "✓ PASS"
        else:
            failed += 1
            status = "✗ FAIL"
        
        print(f"{status} {method:6} {endpoint:50} - {description}")
        
        if not result.get("success") and "error" in result:
            print(f"       Error: {result['error']}")
        elif not result.get("success"):
            print(f"       Expected {result.get('expected_status')}, got {result.get('status_code')}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} total")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 All endpoints are working correctly!")
        return 0
    else:
        print(f"\n⚠️  {failed} endpoint(s) failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
