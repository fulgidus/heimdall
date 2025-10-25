#!/usr/bin/env python3
"""
Testa che il backend API Gateway e services rispondono correttamente.
Simula esattamente quello che il frontend cercherà di fare.
"""

import sys
import httpx
import json
from datetime import datetime

# Colori per output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_status(status: str, message: str, details: str | None = None):
    """Stampa messaggio di status formattato"""
    if "✅" in status:
        color = Colors.GREEN
    elif "❌" in status:
        color = Colors.RED
    elif "⚠️" in status:
        color = Colors.YELLOW
    else:
        color = Colors.BLUE
    
    print(f"{color}{status} {message}{Colors.END}")
    if details:
        print(f"   {details}")

def test_endpoint(method: str, url: str, description: str):
    """Testa un singolo endpoint"""
    print(f"\n📡 Testing: {description}")
    print(f"   {method} {url}")
    
    try:
        with httpx.Client(timeout=5.0) as client:
            if method == "GET":
                response = client.get(url)
            else:
                response = client.request(method, url)
            
            # Successo
            if 200 <= response.status_code < 300:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        print_status("✅", f"Status {response.status_code} OK", 
                                   f"Received {len(data)} items")
                    else:
                        print_status("✅", f"Status {response.status_code} OK",
                                   f"Data type: {type(data).__name__}")
                    return True
                except:
                    print_status("✅", f"Status {response.status_code} OK", 
                               f"Response size: {len(response.text)} bytes")
                    return True
            else:
                print_status("❌", f"Status {response.status_code}", 
                           response.text[:200])
                return False
                
    except httpx.ConnectError as e:
        print_status("❌", "Connection Error", str(e))
        return False
    except httpx.TimeoutException:
        print_status("❌", "Timeout (5s)", "Backend may be offline or slow")
        return False
    except Exception as e:
        print_status("❌", f"Error: {type(e).__name__}", str(e))
        return False

def main():
    print(f"\n{Colors.BOLD}🔍 Frontend → Backend Connectivity Test{Colors.END}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"   Testing API Gateway on localhost:8000")
    
    results = {
        "passed": [],
        "failed": []
    }
    
    # Test API Gateway accessibility
    print(f"\n{Colors.BOLD}1️⃣  Testing API Gateway (port 8000){Colors.END}")
    
    endpoints = [
        ("GET", "http://localhost:8000/health", "API Gateway health check"),
        ("GET", "http://localhost:8000/api/v1/acquisition/websdrs", "Get all WebSDRs"),
        ("GET", "http://localhost:8000/api/v1/acquisition/websdrs/health", "Check WebSDR health status"),
    ]
    
    for method, url, description in endpoints:
        if test_endpoint(method, url, description):
            results["passed"].append(description)
        else:
            results["failed"].append(description)
    
    # Summary
    print(f"\n{Colors.BOLD}📊 Test Summary{Colors.END}")
    print(f"   ✅ Passed: {len(results['passed'])}/{len(endpoints)}")
    print(f"   ❌ Failed: {len(results['failed'])}/{len(endpoints)}")
    
    if results["passed"]:
        print(f"\n{Colors.GREEN}✅ Passed Tests:{Colors.END}")
        for test in results["passed"]:
            print(f"   • {test}")
    
    if results["failed"]:
        print(f"\n{Colors.RED}❌ Failed Tests:{Colors.END}")
        for test in results["failed"]:
            print(f"   • {test}")
        
        print(f"\n{Colors.YELLOW}💡 Troubleshooting:{Colors.END}")
        print("""
   1. Verify backend is running:
      - Check if port 8000 is listening: netstat -ano | findstr :8000
      - Start backend: python services/rf-acquisition/src/main.py
   
   2. Verify API Gateway is running:
      - Should be at http://localhost:8000/health
      - If 404, restart: docker compose up -d api-gateway
   
   3. Check Docker if using containers:
      - docker ps | grep api-gateway
      - docker logs api-gateway
      - docker compose up -d  (to restart all services)
        """)
        return 1
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All Tests Passed! Frontend can connect to backend.{Colors.END}")
    print("""
   Next steps:
   1. Open http://localhost:3001/websdrs in browser
   2. Press F12 to open DevTools
   3. Go to Console tab and look for log messages starting with 🔧, 🚀, 📡
   4. If you see error logs (❌), share them for further diagnosis
    """)
    return 0

if __name__ == "__main__":
    sys.exit(main())
