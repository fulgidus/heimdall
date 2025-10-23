#!/usr/bin/env python3
"""
Test della call HTTP all'endpoint /websdrs/health per capire l'eccezione.
"""
import requests
import json

print("Calling /websdrs/health...")
try:
    r = requests.get('http://localhost:8001/api/v1/acquisition/websdrs/health', timeout=120)
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
