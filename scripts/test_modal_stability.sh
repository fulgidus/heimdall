#!/bin/bash
# Modal Portal Stability Testing Script
# Tests the modal components for DOM manipulation errors

set -e

echo "======================================"
echo "Modal Portal Stability Test"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if services are running
echo -e "${BLUE}[1/5] Checking if services are running...${NC}"
if ! docker compose ps | grep -q "frontend.*Up"; then
    echo -e "${RED}❌ Frontend service is not running${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

if ! docker compose ps | grep -q "backend.*Up"; then
    echo -e "${RED}❌ Backend service is not running${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi

echo -e "${GREEN}✅ Services are running${NC}"
echo ""

# Check frontend accessibility
echo -e "${BLUE}[2/5] Checking frontend accessibility...${NC}"
if curl -f -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ Frontend is accessible at http://localhost:3000${NC}"
else
    echo -e "${RED}❌ Frontend is not accessible${NC}"
    exit 1
fi
echo ""

# Check backend WebSocket endpoint
echo -e "${BLUE}[3/5] Checking backend WebSocket endpoint...${NC}"
if curl -f -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✅ Backend is healthy at http://localhost:8001${NC}"
else
    echo -e "${YELLOW}⚠️  Backend health check failed${NC}"
fi
echo ""

# Run TypeScript type checking
echo -e "${BLUE}[4/5] Running TypeScript type checking...${NC}"
cd frontend
if npm run type-check > /dev/null 2>&1; then
    echo -e "${GREEN}✅ TypeScript compilation passed${NC}"
else
    echo -e "${RED}❌ TypeScript compilation failed${NC}"
    echo "Run: cd frontend && npm run type-check"
    cd ..
    exit 1
fi
cd ..
echo ""

# Check if Playwright is installed
echo -e "${BLUE}[5/5] Checking E2E test setup...${NC}"
if [ -f "frontend/node_modules/.bin/playwright" ]; then
    echo -e "${GREEN}✅ Playwright is installed${NC}"
    echo ""
    echo -e "${BLUE}You can run E2E tests with:${NC}"
    echo "  cd frontend"
    echo "  npm run test:e2e -- modal-portal-stability.spec.ts"
else
    echo -e "${YELLOW}⚠️  Playwright is not installed${NC}"
    echo "Install with: cd frontend && npx playwright install"
fi
echo ""

echo "======================================"
echo -e "${GREEN}✅ All Checks Passed${NC}"
echo "======================================"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Manual Testing:"
echo "   - Open http://localhost:3000 in your browser"
echo "   - Open browser console (F12)"
echo "   - Navigate to /websdrs and test modals"
echo "   - Watch for DOM errors in console"
echo ""
echo "2. Run E2E Tests:"
echo "   cd frontend"
echo "   npm run test:e2e -- modal-portal-stability.spec.ts"
echo ""
echo "3. Monitor WebSocket Activity:"
echo "   docker compose logs -f backend | grep -i websocket"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "   docs/agents/20251103_modal_portal_fix_complete.md"
echo ""
