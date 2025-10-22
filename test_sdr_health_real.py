#!/usr/bin/env python3
"""
Test HEAD requests to real WebSDR receivers to see what they respond with.
"""
import asyncio
import aiohttp

WEBSDRS = {
    1: "http://sdr1.ik1jns.it:8076/",
    2: "http://alba-sdr.ddns.net:8073/",
    3: "http://sdr.ik8pxu.com:8076/",
    4: "http://balma-martino.ddns.net:8074/",
    5: "http://sdr.k4lyf.com:8076/",
    6: "http://sdr-italy-liguria.ddns.net:8076/",
    7: "http://sdr-piedmont.ddns.net:8076/",
}

async def test_sdr(session, sdr_id, url):
    """Test a single SDR with HEAD request."""
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=False) as response:
            print(f"  HEAD {url}")
            print(f"    ✓ Status: {response.status} ({response.reason})")
            print(f"    ✓ Current logic (< 500): {response.status < 500} ← {'WRONG!' if response.status >= 200 else 'OK'}")
            print(f"    ✓ Headers: {dict(response.headers)}")
            return response.status
    except asyncio.TimeoutError:
        print(f"  HEAD {url}")
        print(f"    ✗ Timeout (10s)")
        return None
    except Exception as e:
        print(f"  HEAD {url}")
        print(f"    ✗ Error: {type(e).__name__}: {e}")
        return None

async def test_all():
    """Test all SDRs."""
    print("=" * 80)
    print("Testing Real WebSDR Health Check Logic")
    print("=" * 80)
    print()
    
    async with aiohttp.ClientSession() as session:
        for sdr_id, url in WEBSDRS.items():
            print(f"SDR #{sdr_id}: {url}")
            await test_sdr(session, sdr_id, url)
            print()
    
    print("=" * 80)
    print("PROBLEM FOUND:")
    print("  Current logic: Returns True if response.status < 500")
    print("  BUT: SDRs respond with 200/301/302/403/404, all < 500!")
    print("  RESULT: We can't distinguish online from offline!")
    print()
    print("BETTER LOGIC:")
    print("  ✓ Try GET / (returns 200 if online, 404 if really down)")
    print("  ✓ Or: Try GET /api/info (specific endpoint)")
    print("  ✓ Or: Check for specific response headers (Server, Content-Type)")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_all())
