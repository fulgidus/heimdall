#!/usr/bin/env python3
"""Inspect Redis for task state."""

import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

# List all keys
keys = r.keys('*')
print(f"Total Redis keys: {len(keys)}\n")

# Print some keys
for key in keys[:10]:
    val = r.get(key)
    print(f"Key: {key}")
    print(f"Value: {val}\n")
