#!/usr/bin/env python3
"""
Manual test script for session update endpoint

This script tests the session update functionality by:
1. Verifying the endpoint exists and accepts the correct parameters
2. Testing validation rules for the RecordingSessionUpdate model

Run this script with:
    python3 services/backend/tests/manual_test_session_update.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.session import RecordingSessionUpdate
from pydantic import ValidationError


def test_model_validation():
    """Test RecordingSessionUpdate model validation"""
    
    print("Testing RecordingSessionUpdate model validation...")
    
    # Test 1: Valid update with all fields
    print("\n1. Valid update with all fields:")
    try:
        update = RecordingSessionUpdate(
            session_name="Test Session",
            notes="Test notes",
            approval_status="approved"
        )
        print(f"✓ Valid: {update.model_dump()}")
    except ValidationError as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 2: Valid update with only session_name
    print("\n2. Valid update with only session_name:")
    try:
        update = RecordingSessionUpdate(session_name="Test Session")
        print(f"✓ Valid: {update.model_dump()}")
    except ValidationError as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 3: Valid update with only notes
    print("\n3. Valid update with only notes:")
    try:
        update = RecordingSessionUpdate(notes="Test notes")
        print(f"✓ Valid: {update.model_dump()}")
    except ValidationError as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 4: Valid update with empty notes
    print("\n4. Valid update with empty notes:")
    try:
        update = RecordingSessionUpdate(notes="")
        print(f"✓ Valid: {update.model_dump()}")
    except ValidationError as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 5: Invalid - empty session_name
    print("\n5. Invalid - empty session_name:")
    try:
        update = RecordingSessionUpdate(session_name="")
        print(f"✗ Should have failed but got: {update.model_dump()}")
        return False
    except ValidationError as e:
        print(f"✓ Correctly rejected: {e.errors()[0]['msg']}")
    
    # Test 6: Invalid - session_name too long
    print("\n6. Invalid - session_name too long:")
    try:
        update = RecordingSessionUpdate(session_name="x" * 256)
        print(f"✗ Should have failed but got: {update.model_dump()}")
        return False
    except ValidationError as e:
        print(f"✓ Correctly rejected: {e.errors()[0]['msg']}")
    
    # Test 7: Invalid approval_status
    print("\n7. Invalid approval_status:")
    try:
        update = RecordingSessionUpdate(approval_status="invalid")
        print(f"✗ Should have failed but got: {update.model_dump()}")
        return False
    except ValidationError as e:
        print(f"✓ Correctly rejected: {e.errors()[0]['msg']}")
    
    # Test 8: Valid approval statuses
    print("\n8. Valid approval statuses:")
    for status in ["pending", "approved", "rejected"]:
        try:
            update = RecordingSessionUpdate(approval_status=status)
            print(f"✓ Valid status '{status}': {update.model_dump()}")
        except ValidationError as e:
            print(f"✗ Failed for status '{status}': {e}")
            return False
    
    print("\n✓ All validation tests passed!")
    return True


def test_endpoint_signature():
    """Test that the endpoint signature is correct"""
    
    print("\nTesting endpoint signature...")
    
    from routers.sessions import router
    from fastapi.routing import APIRoute
    
    # Find the PATCH /{session_id} endpoint
    patch_endpoint = None
    for route in router.routes:
        if isinstance(route, APIRoute):
            if "PATCH" in route.methods and "/{session_id}" in route.path:
                patch_endpoint = route
                break
    
    if not patch_endpoint:
        print("✗ PATCH /{session_id} endpoint not found!")
        return False
    
    print(f"✓ Found endpoint: PATCH {patch_endpoint.path}")
    print(f"  Name: {patch_endpoint.name}")
    print(f"  Response model: {patch_endpoint.response_model}")
    
    # Check the function signature
    import inspect
    sig = inspect.signature(patch_endpoint.endpoint)
    print(f"  Parameters: {list(sig.parameters.keys())}")
    
    if 'session_id' not in sig.parameters:
        print("✗ Missing session_id parameter!")
        return False
    
    if 'session_update' not in sig.parameters:
        print("✗ Missing session_update parameter!")
        return False
    
    print("✓ Endpoint signature is correct!")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Manual Test: Session Update Functionality")
    print("=" * 60)
    
    results = []
    
    # Test model validation
    results.append(test_model_validation())
    
    # Test endpoint signature
    results.append(test_endpoint_signature())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
