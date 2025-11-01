#!/usr/bin/env python3
"""
Test the full stack: Frontend → API Gateway → RF-Acquisition Service

Questo script testa che il percorso completo funziona:
1. API Gateway (port 8000) è online
2. RF-Acquisition Service (port 8001) è online
3. Il path /api/v1/acquisition/websdrs è raggiungibile
4. Dati vengono restituiti correttamente
"""

import sys
from datetime import datetime

import httpx


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def test_endpoint(name: str, url: str, expected_status: int = 200) -> bool:
    """Testa un singolo endpoint"""
    print(f"\n{'='*70}")
    print(f"🔍 Test: {name}")
    print(f"   URL: {url}")

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)

            # Status check
            status_ok = response.status_code == expected_status
            status_symbol = "✅" if status_ok else "❌"
            status_color = Colors.GREEN if status_ok else Colors.RED

            print(
                f"   {status_color}{status_symbol} Status: {response.status_code}{Colors.END}",
                end="",
            )
            if not status_ok:
                print(f" (expected {expected_status})")
                return False
            print()

            # Content check
            try:
                data = response.json()
                print(f"   {Colors.GREEN}✅ Response Type: JSON{Colors.END}")

                if isinstance(data, list):
                    print(
                        f"   {Colors.GREEN}✅ Data Type: Array with {len(data)} items{Colors.END}"
                    )
                    if data:
                        print(f"   {Colors.BLUE}Sample Item (first):{Colors.END}")
                        first_item = data[0]
                        for key, value in list(first_item.items())[:5]:
                            print(f"      • {key}: {str(value)[:50]}")
                elif isinstance(data, dict):
                    print(f"   {Colors.GREEN}✅ Data Type: Object{Colors.END}")
                    print(f"   {Colors.BLUE}Keys:{Colors.END}")
                    for key in list(data.keys())[:5]:
                        print(f"      • {key}")

                return True
            except Exception as e:
                print(f"   {Colors.YELLOW}⚠️  Response is not JSON: {str(e)[:100]}{Colors.END}")
                print(f"   {Colors.BLUE}Raw response:{Colors.END} {response.text[:200]}")
                return True  # Still OK if we got a 200

    except httpx.ConnectError as e:
        print(f"   {Colors.RED}❌ Connection Error{Colors.END}")
        print(f"   {str(e)}")
        return False
    except httpx.TimeoutException:
        print(f"   {Colors.RED}❌ Timeout (5 seconds){Colors.END}")
        print("   Backend may be offline or unresponsive")
        return False
    except Exception as e:
        print(f"   {Colors.RED}❌ Error: {type(e).__name__}{Colors.END}")
        print(f"   {str(e)}")
        return False


def main():
    print(f"\n{Colors.BOLD}{'='*70}")
    print("🧪 Full Stack Test: Frontend → API Gateway → RF-Acquisition")
    print(f"{'='*70}{Colors.END}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    results = {"passed": [], "failed": []}

    # Test 1: API Gateway is online
    print(f"\n{Colors.BOLD}Layer 1: API Gateway (port 8000){Colors.END}")
    if test_endpoint("API Gateway Health Check", "http://localhost:8000/health"):
        results["passed"].append("API Gateway health")
    else:
        results["failed"].append("API Gateway health")

    # Test 2: RF-Acquisition Service is online
    print(f"\n{Colors.BOLD}Layer 2: RF-Acquisition Service (port 8001){Colors.END}")
    if test_endpoint("RF-Acquisition Health Check", "http://localhost:8001/health"):
        results["passed"].append("RF-Acquisition health")
    else:
        results["failed"].append("RF-Acquisition health")

    # Test 3: Direct call to RF-Acquisition /websdrs
    print(f"\n{Colors.BOLD}Layer 2b: Direct RF-Acquisition /websdrs (Bypass Gateway){Colors.END}")
    if test_endpoint(
        "Get WebSDRs (Direct Service)", "http://localhost:8001/api/v1/acquisition/websdrs"
    ):
        results["passed"].append("Direct /websdrs call")
    else:
        results["failed"].append("Direct /websdrs call")

    # Test 4: Through API Gateway (what frontend does)
    print(f"\n{Colors.BOLD}Layer 1+2: Through API Gateway (What Frontend Does){Colors.END}")
    if test_endpoint(
        "Get WebSDRs (Via API Gateway)", "http://localhost:8000/api/v1/acquisition/websdrs"
    ):
        results["passed"].append("Gateway /websdrs call")
    else:
        results["failed"].append("Gateway /websdrs call")

    # Test 5: Health check through gateway
    print(f"\n{Colors.BOLD}Layer 1+2: WebSDR Health Check (Via API Gateway){Colors.END}")
    if test_endpoint(
        "Check WebSDR Health (Via Gateway)",
        "http://localhost:8000/api/v1/acquisition/websdrs/health",
    ):
        results["passed"].append("Gateway /websdrs/health call")
    else:
        results["failed"].append("Gateway /websdrs/health call")

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"📊 Summary{Colors.END}")
    print(f"{'='*70}")
    print(f"{Colors.GREEN}✅ Passed: {len(results['passed'])}{Colors.END}")
    print(f"{Colors.RED}❌ Failed: {len(results['failed'])}{Colors.END}")
    print()

    if results["passed"]:
        print(f"{Colors.GREEN}✅ Passed Tests:{Colors.END}")
        for test in results["passed"]:
            print(f"   • {test}")

    if results["failed"]:
        print(f"\n{Colors.RED}❌ Failed Tests:{Colors.END}")
        for test in results["failed"]:
            print(f"   • {test}")

        print(f"\n{Colors.YELLOW}💡 Troubleshooting:{Colors.END}")
        print(
            """
   Scenario 1: Both Gateway and Service offline
   → Start Docker: docker compose up -d
   → Or start services manually:
      python services/api-gateway/src/main.py
      python services/rf-acquisition/src/main.py

   Scenario 2: Only API Gateway offline (port 8000)
   → docker compose up -d api-gateway

   Scenario 3: Only RF-Acquisition offline (port 8001)
   → docker compose up -d rf-acquisition

   Scenario 4: Services online but 404 on /api/v1/acquisition/websdrs
   → Check that RF-Acquisition service has this endpoint defined
   → Verify: curl http://localhost:8001/docs (Swagger UI)

   Scenario 5: Working but frontend still shows 404
   → Frontend base URL is wrong (should be http://localhost:8000)
   → Frontend cache issue: Ctrl+Shift+R hard reload
   → CORS issue: Check browser console for CORS errors
        """
        )
        return 1

    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All Tests Passed!{Colors.END}")
    print(
        """
   The full stack is working:
   ✅ API Gateway (port 8000) is online
   ✅ RF-Acquisition Service (port 8001) is online
   ✅ /api/v1/acquisition/websdrs endpoint is accessible
   ✅ /api/v1/acquisition/websdrs/health endpoint is accessible

   Frontend should now be able to:
   1. Call http://localhost:8000/api/v1/acquisition/websdrs
   2. Receive JSON array of 7 WebSDRs
   3. Display real data in the WebSDRManagement page

   If frontend still shows mock data or 404:
   - Check browser console (F12) for errors
   - Verify frontend .env has VITE_API_URL=http://localhost:8000
   - Hard reload browser (Ctrl+Shift+R)
   - Check Network tab (F12) to see actual HTTP requests
    """
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
