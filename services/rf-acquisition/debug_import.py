#!/usr/bin/env python
"""Debug script for db.py module loading"""
import sys
import traceback

sys.path.insert(0, '.')

try:
    print("Importing src.models.db...")
    import src.models.db as db
    print(f"Module: {db}")
    print(f"Module file: {db.__file__}")
    print(f"Module contents: {dir(db)}")
    
    # Check if Measurement is in Base registry
    from src.models.db import Base
    print(f"Base: {Base}")
    print(f"Base registry: {Base.registry.mappers}")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
