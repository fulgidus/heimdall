#!/bin/bash
# Training Dashboard Health Check Script
# Verifies that the Training Dashboard can load successfully

set -e

echo "========================================="
echo "Training Dashboard Health Check"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Backend health
echo "1. Checking backend health..."
BACKEND_STATUS=$(curl -s http://localhost:8001/health | jq -r '.status' 2>/dev/null || echo "error")
if [ "$BACKEND_STATUS" = "healthy" ]; then
    echo -e "${GREEN}✓${NC} Backend is healthy"
else
    echo -e "${RED}✗${NC} Backend is not healthy"
    exit 1
fi
echo ""

# Check 2: Training API endpoint
echo "2. Checking training API endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/api/v1/training/jobs 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓${NC} Training API endpoint accessible (HTTP $HTTP_CODE)"
else
    echo -e "${RED}✗${NC} Training API endpoint returned HTTP $HTTP_CODE"
    exit 1
fi
echo ""

# Check 3: Frontend dev server
echo "3. Checking frontend dev server..."
FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 2>/dev/null)
if [ "$FRONTEND_CODE" = "200" ]; then
    echo -e "${GREEN}✓${NC} Frontend dev server accessible (HTTP $FRONTEND_CODE)"
else
    echo -e "${YELLOW}⚠${NC} Frontend dev server returned HTTP $FRONTEND_CODE"
    echo "   Trying to start frontend..."
    cd "$(dirname "$0")/../frontend" && npm run dev > /tmp/vite-dev.log 2>&1 &
    sleep 5
    FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 2>/dev/null)
    if [ "$FRONTEND_CODE" = "200" ]; then
        echo -e "${GREEN}✓${NC} Frontend started successfully"
    else
        echo -e "${RED}✗${NC} Could not start frontend"
        exit 1
    fi
fi
echo ""

# Check 4: Training page loads
echo "4. Checking Training page loads..."
TRAINING_PAGE=$(curl -s http://localhost:3001/training 2>/dev/null)
if echo "$TRAINING_PAGE" | grep -q "<!doctype html"; then
    echo -e "${GREEN}✓${NC} Training page HTML loads"
else
    echo -e "${RED}✗${NC} Training page failed to load"
    exit 1
fi
echo ""

# Check 5: API proxy works through frontend
echo "5. Checking API proxy through frontend..."
PROXY_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/api/v1/training/jobs 2>/dev/null)
if [ "$PROXY_CODE" = "200" ]; then
    echo -e "${GREEN}✓${NC} API proxy working (HTTP $PROXY_CODE)"
else
    echo -e "${RED}✗${NC} API proxy returned HTTP $PROXY_CODE"
    exit 1
fi
echo ""

# Check 6: Verify training jobs response structure
echo "6. Verifying training API response structure..."
JOBS_RESPONSE=$(curl -s http://localhost:3001/api/v1/training/jobs 2>/dev/null)
if echo "$JOBS_RESPONSE" | jq -e '.jobs' > /dev/null 2>&1; then
    JOB_COUNT=$(echo "$JOBS_RESPONSE" | jq '.jobs | length' 2>/dev/null)
    echo -e "${GREEN}✓${NC} API returns valid JSON with $JOB_COUNT jobs"
else
    echo -e "${RED}✗${NC} API response structure invalid"
    echo "Response: $JOBS_RESPONSE"
    exit 1
fi
echo ""

# Check 7: Test models endpoint
echo "7. Checking models API endpoint..."
MODELS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/api/v1/training/models 2>/dev/null)
if [ "$MODELS_CODE" = "200" ] || [ "$MODELS_CODE" = "404" ]; then
    echo -e "${GREEN}✓${NC} Models API endpoint accessible (HTTP $MODELS_CODE)"
else
    echo -e "${RED}✗${NC} Models API endpoint returned HTTP $MODELS_CODE"
    exit 1
fi
echo ""

# Summary
echo "========================================="
echo -e "${GREEN}All checks passed!${NC}"
echo "========================================="
echo ""
echo "Training Dashboard Status:"
echo "  Frontend:  http://localhost:3001/training"
echo "  API:       http://localhost:3001/api/v1/training/jobs"
echo "  Backend:   http://localhost:8001/api/v1/training/jobs"
echo ""
echo "Next steps:"
echo "  1. Open browser to http://localhost:3001/training"
echo "  2. Open DevTools console (F12)"
echo "  3. Verify no API fetch errors"
echo "  4. Test all 4 tabs (Jobs, Metrics, Models, Synthetic)"
echo ""
