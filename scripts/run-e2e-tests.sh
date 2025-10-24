#!/bin/bash

###############################################################################
# E2E Test Orchestration Script
# 
# Starts backend services, waits for health, runs E2E tests, collects artifacts
#
# Usage:
#   ./run-e2e-tests.sh [--no-build] [--keep-running]
#
# Environment Variables:
#   BASE_URL - Frontend URL (default: http://localhost:3001)
#   TEST_BACKEND_ORIGIN - Backend URL (default: http://localhost:8000)
###############################################################################

set -e

# Configuration
BASE_URL=${BASE_URL:-http://localhost:3001}
TEST_BACKEND_ORIGIN=${TEST_BACKEND_ORIGIN:-http://localhost:8000}
MAX_WAIT=120  # Maximum seconds to wait for services
POLL_INTERVAL=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Flags
BUILD_SERVICES=true
KEEP_RUNNING=false

# Parse arguments
for arg in "$@"; do
  case $arg in
    --no-build)
      BUILD_SERVICES=false
      shift
      ;;
    --keep-running)
      KEEP_RUNNING=true
      shift
      ;;
  esac
done

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}E2E Test Orchestration${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "BASE_URL: ${YELLOW}${BASE_URL}${NC}"
echo -e "TEST_BACKEND_ORIGIN: ${YELLOW}${TEST_BACKEND_ORIGIN}${NC}"
echo ""

###############################################################################
# Helper Functions
###############################################################################

check_health() {
  local url=$1
  local name=$2
  
  echo -n "Checking $name health at $url... "
  
  if curl -s -f "$url" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
    return 0
  else
    echo -e "${RED}✗ FAILED${NC}"
    return 1
  fi
}

wait_for_service() {
  local url=$1
  local name=$2
  local elapsed=0
  
  echo "Waiting for $name to become healthy (max ${MAX_WAIT}s)..."
  
  while [ $elapsed -lt $MAX_WAIT ]; do
    if check_health "$url" "$name"; then
      return 0
    fi
    
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
    echo "  Elapsed: ${elapsed}s / ${MAX_WAIT}s"
  done
  
  echo -e "${RED}ERROR: $name did not become healthy within ${MAX_WAIT}s${NC}"
  return 1
}

cleanup() {
  if [ "$KEEP_RUNNING" = false ]; then
    echo ""
    echo -e "${YELLOW}Cleaning up services...${NC}"
    docker-compose -f ../docker-compose.yml -f ../docker-compose.services.yml down
    echo -e "${GREEN}Services stopped${NC}"
  else
    echo ""
    echo -e "${YELLOW}Services kept running (--keep-running flag)${NC}"
  fi
}

trap cleanup EXIT

###############################################################################
# Step 1: Start Backend Services
###############################################################################

echo -e "${GREEN}Step 1: Starting backend services${NC}"
echo ""

cd "$(dirname "$0")/.."

if [ "$BUILD_SERVICES" = true ]; then
  echo "Building services..."
  docker-compose -f docker-compose.yml -f docker-compose.services.yml build
fi

echo "Starting services..."
docker-compose -f docker-compose.yml -f docker-compose.services.yml up -d

echo ""
echo -e "${GREEN}Services started, waiting for health checks...${NC}"
echo ""

###############################################################################
# Step 2: Wait for Services to be Healthy
###############################################################################

echo -e "${GREEN}Step 2: Waiting for services to be healthy${NC}"
echo ""

# Wait for API Gateway (main entry point)
wait_for_service "${TEST_BACKEND_ORIGIN}/health" "API Gateway"

# Wait for other key services
wait_for_service "http://localhost:8001/health" "RF Acquisition"
wait_for_service "http://localhost:8004/health" "Data Ingestion"

# Check database
wait_for_service "http://localhost:5432" "PostgreSQL" || echo "PostgreSQL check skipped (no HTTP endpoint)"

echo ""
echo -e "${GREEN}All services are healthy!${NC}"
echo ""

###############################################################################
# Step 3: Run E2E Tests
###############################################################################

echo -e "${GREEN}Step 3: Running Playwright E2E tests${NC}"
echo ""

cd frontend

# Export environment variables for tests
export BASE_URL
export TEST_BACKEND_ORIGIN

# Install Playwright browsers if not already installed
if [ ! -d "$HOME/.cache/ms-playwright" ]; then
  echo "Installing Playwright browsers..."
  npx playwright install chromium
fi

# Run tests
echo "Running E2E tests..."
npx playwright test --reporter=html,json,list

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}✓ All E2E tests passed!${NC}"
else
  echo -e "${RED}✗ Some E2E tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi

###############################################################################
# Step 4: Collect Artifacts
###############################################################################

echo ""
echo -e "${GREEN}Step 4: Collecting test artifacts${NC}"
echo ""

ARTIFACTS_DIR="e2e-artifacts-$(date +%Y%m%d-%H%M%S)"
mkdir -p "../$ARTIFACTS_DIR"

# Copy Playwright report
if [ -d "playwright-report" ]; then
  echo "Copying Playwright report..."
  cp -r playwright-report "../$ARTIFACTS_DIR/"
fi

# Copy HAR files
if [ -f "playwright-report/network.har" ]; then
  echo "Copying network HAR file..."
  cp playwright-report/network.har "../$ARTIFACTS_DIR/"
fi

# Copy screenshots
if [ -d "test-results" ]; then
  echo "Copying test results (screenshots, videos, traces)..."
  cp -r test-results "../$ARTIFACTS_DIR/"
fi

# Collect backend logs
echo "Collecting backend service logs..."
docker-compose -f ../docker-compose.yml -f ../docker-compose.services.yml logs --no-color > "../$ARTIFACTS_DIR/backend-logs.txt"

echo ""
echo -e "${GREEN}Artifacts collected in: ../$ARTIFACTS_DIR${NC}"

###############################################################################
# Step 5: Generate Summary Report
###############################################################################

echo ""
echo -e "${GREEN}Step 5: Generating summary report${NC}"
echo ""

cat > "../$ARTIFACTS_DIR/SUMMARY.md" <<EOF
# E2E Test Execution Summary

**Date**: $(date)
**Base URL**: $BASE_URL
**Backend Origin**: $TEST_BACKEND_ORIGIN
**Exit Code**: $TEST_EXIT_CODE

## Services Status

- API Gateway: ✓ Healthy
- RF Acquisition: ✓ Healthy
- Data Ingestion: ✓ Healthy

## Test Results

See \`playwright-report/index.html\` for detailed results.

## Artifacts

- \`playwright-report/\` - HTML test report
- \`playwright-report/network.har\` - Network traffic log
- \`test-results/\` - Screenshots, videos, traces
- \`backend-logs.txt\` - Backend service logs

## Commands to Reproduce

\`\`\`bash
# Start backend
docker-compose -f docker-compose.yml -f docker-compose.services.yml up -d

# Wait for health
curl http://localhost:8000/health

# Run tests
cd frontend
BASE_URL=$BASE_URL TEST_BACKEND_ORIGIN=$TEST_BACKEND_ORIGIN npx playwright test
\`\`\`

## Exit Code

$TEST_EXIT_CODE (0 = success, non-zero = failure)
EOF

echo -e "${GREEN}Summary report: ../$ARTIFACTS_DIR/SUMMARY.md${NC}"

###############################################################################
# Final Status
###############################################################################

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}E2E Test Execution Complete${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}✓ SUCCESS: All tests passed${NC}"
  echo -e "View report: file://$(pwd)/../$ARTIFACTS_DIR/playwright-report/index.html"
  exit 0
else
  echo -e "${RED}✗ FAILURE: Some tests failed${NC}"
  echo -e "View report: file://$(pwd)/../$ARTIFACTS_DIR/playwright-report/index.html"
  exit $TEST_EXIT_CODE
fi
