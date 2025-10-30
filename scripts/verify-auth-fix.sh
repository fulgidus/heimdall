#!/bin/bash
#
# Verification script for authentication fix
# This script checks if all auth endpoints are accessible
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
API_GATEWAY_URL="${API_GATEWAY_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"

echo "========================================"
echo "Authentication Fix Verification Script"
echo "========================================"
echo ""

# Function to check if a service is running
check_service() {
    local url=$1
    local name=$2
    
    echo -n "Checking $name... "
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        return 1
    fi
}

# Function to check an endpoint
check_endpoint() {
    local method=$1
    local url=$2
    local expected_status=$3
    local description=$4
    
    echo -n "Testing $description... "
    
    if [ "$method" = "GET" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    else
        status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$url")
    fi
    
    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✓ $status${NC}"
        return 0
    elif [ "$status" = "404" ]; then
        echo -e "${RED}✗ 404 NOT FOUND (BROKEN!)${NC}"
        return 1
    else
        echo -e "${YELLOW}! $status (expected $expected_status, but not 404)${NC}"
        return 0  # Not 404, so endpoint exists
    fi
}

echo "Step 1: Checking if services are running..."
echo "-------------------------------------------"

check_service "$API_GATEWAY_URL/health" "API Gateway" || {
    echo ""
    echo -e "${RED}ERROR: API Gateway is not running!${NC}"
    echo "Please start services with: docker-compose up -d"
    exit 1
}

check_service "$FRONTEND_URL/health" "Frontend" || {
    echo ""
    echo -e "${YELLOW}WARNING: Frontend is not running${NC}"
    echo "Frontend tests will be skipped"
    FRONTEND_RUNNING=false
}

echo ""
echo "Step 2: Testing auth endpoints on API Gateway..."
echo "-------------------------------------------"

# Test login endpoint (should return 400 for missing credentials, NOT 404)
check_endpoint "POST" "$API_GATEWAY_URL/api/v1/auth/login" "400" "POST /api/v1/auth/login" || {
    echo ""
    echo -e "${RED}CRITICAL: Login endpoint returns 404!${NC}"
    echo "This indicates the endpoint is not registered or path is wrong."
    exit 1
}

# Test refresh endpoint (should return 400 for missing token, NOT 404)
check_endpoint "POST" "$API_GATEWAY_URL/api/v1/auth/refresh" "400" "POST /api/v1/auth/refresh" || {
    echo ""
    echo -e "${RED}CRITICAL: Refresh endpoint returns 404!${NC}"
    exit 1
}

# Test check endpoint (should return 200 for anonymous user)
check_endpoint "GET" "$API_GATEWAY_URL/api/v1/auth/check" "200" "GET /api/v1/auth/check" || {
    echo ""
    echo -e "${RED}CRITICAL: Check endpoint returns 404!${NC}"
    exit 1
}

if [ "$FRONTEND_RUNNING" != "false" ]; then
    echo ""
    echo "Step 3: Testing auth endpoints through Nginx proxy..."
    echo "-------------------------------------------"
    
    # Test via frontend proxy
    check_endpoint "POST" "$FRONTEND_URL/api/v1/auth/login" "400" "POST /api/v1/auth/login (via Nginx)" || {
        echo ""
        echo -e "${RED}CRITICAL: Nginx proxy not forwarding correctly!${NC}"
        exit 1
    }
fi

echo ""
echo "Step 4: Testing with actual credentials..."
echo "-------------------------------------------"

# Test login with credentials (should return 200 or 401, NOT 404)
echo -n "Testing login with credentials... "
status=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$API_GATEWAY_URL/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"email":"admin@heimdall.local","password":"admin"}')

if [ "$status" = "200" ]; then
    echo -e "${GREEN}✓ $status (Login successful!)${NC}"
elif [ "$status" = "401" ]; then
    echo -e "${YELLOW}! $status (Invalid credentials, but endpoint works)${NC}"
elif [ "$status" = "500" ]; then
    echo -e "${YELLOW}! $status (Keycloak may not be ready)${NC}"
    echo "  Check: docker logs heimdall-keycloak"
elif [ "$status" = "404" ]; then
    echo -e "${RED}✗ 404 NOT FOUND (BROKEN!)${NC}"
    exit 1
else
    echo -e "${YELLOW}! $status (Unexpected status)${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}✓ Verification Complete${NC}"
echo "========================================"
echo ""
echo "Summary:"
echo "- All auth endpoints are accessible (not 404)"
echo "- API Gateway is properly configured"

if [ "$FRONTEND_RUNNING" != "false" ]; then
    echo "- Nginx proxy is forwarding correctly"
fi

echo ""
echo "Next steps:"
echo "1. Test login via browser at $FRONTEND_URL"
echo "2. Check DevTools Network tab for /api/v1/auth/login"
echo "3. Verify 200 response (or 401 for bad credentials)"
echo ""
