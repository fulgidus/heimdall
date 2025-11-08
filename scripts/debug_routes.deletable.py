#!/usr/bin/env python3
import sys

sys.path.insert(0, "/app")

from src import main

print("=" * 80)
print("ALL ROUTES REGISTERED IN API GATEWAY")
print("=" * 80)

for i, r in enumerate(main.app.routes, 1):
    methods = getattr(r, "methods", [])
    print(f"{i}. Path: {r.path}")
    print(f"   Methods: {methods}")
    print(f"   Type: {type(r).__name__}")
    print()

print("=" * 80)
print(f"TOTAL ROUTES: {len(main.app.routes)}")
print("=" * 80)

# Check for proxy routes specifically
proxy_routes = [r for r in main.app.routes if "/api/v1/" in r.path]
print(f"\nProxy routes found: {len(proxy_routes)}")
for r in proxy_routes:
    print(f"  - {r.path}")
