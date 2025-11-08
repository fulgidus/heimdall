#!/usr/bin/env python3
"""Test script to verify PostgreSQL health check fix."""
import asyncio
from datetime import datetime
from urllib.parse import urlparse
import asyncpg
import os

async def check_postgresql_health():
    """Check PostgreSQL/TimescaleDB health."""
    database_url = os.getenv("DATABASE_URL", "postgresql://heimdall:heimdall123@postgres:5432/heimdall")
    
    try:
        # Create a fresh connection in this event loop
        db_url = urlparse(database_url)
        
        conn = await asyncpg.connect(
            user=db_url.username,
            password=db_url.password,
            database=db_url.path.lstrip("/"),
            host="localhost",  # Use localhost since we're running from host
            port=5433,  # Use mapped port
            timeout=5,
        )
        
        try:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                return {
                    "service": "postgresql",
                    "status": "healthy",
                    "message": "Database connection OK",
                    "type": "database",
                    "last_check": datetime.utcnow().isoformat(),
                }
        finally:
            await conn.close()
            
    except Exception as e:
        return {
            "service": "postgresql",
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "type": "database",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
        }

async def main():
    print("Testing PostgreSQL health check...")
    result = await check_postgresql_health()
    print(f"\nResult:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    if result["status"] == "healthy":
        print("\n✅ Database health check PASSED!")
        return 0
    else:
        print("\n❌ Database health check FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
