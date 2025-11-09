#!/usr/bin/env python3
"""
Test script to debug models endpoint error
"""
import requests
import json

# Get token from Keycloak
token_url = "http://localhost:8080/realms/heimdall/protocol/openid-connect/token"
token_data = {
    "client_id": "heimdall-backend",
    "username": "admin",
    "password": "admin",
    "grant_type": "password"
}

print("1. Getting token from Keycloak...")
try:
    token_response = requests.post(token_url, data=token_data)
    print(f"   Status: {token_response.status_code}")
    
    if token_response.status_code == 200:
        token = token_response.json().get("access_token")
        if token:
            print(f"   Token acquired: {token[:50]}...")
        else:
            print("   ERROR: No access_token in response")
            print(f"   Response: {token_response.json()}")
            exit(1)
    else:
        print(f"   ERROR: {token_response.text}")
        exit(1)
except Exception as e:
    print(f"   EXCEPTION: {e}")
    exit(1)

# Test models endpoint
print("\n2. Testing /api/v1/models endpoint...")
models_url = "http://localhost:8001/api/v1/models?page=1&per_page=50"
headers = {"Authorization": f"Bearer {token}"}

try:
    models_response = requests.get(models_url, headers=headers)
    print(f"   Status: {models_response.status_code}")
    
    if models_response.status_code == 200:
        data = models_response.json()
        print(f"   SUCCESS: Retrieved {data.get('total', 0)} models")
        print(f"   Models: {json.dumps(data, indent=2)[:500]}...")
    else:
        print(f"   ERROR Response:")
        print(f"   {models_response.text}")
except Exception as e:
    print(f"   EXCEPTION: {e}")
    exit(1)

print("\n✅ Test completed successfully!")
