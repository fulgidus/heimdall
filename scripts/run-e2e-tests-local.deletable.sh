#!/bin/bash

###############################################################################
# E2E Test Orchestration Script (GitHub Workflow Aligned)
# 
# This script mirrors the GitHub Actions workflow (.github/workflows/e2e-tests.yml)
# for local testing. It starts essential services, builds frontend, and runs
# Playwright E2E tests against a real backend.
#
# Usage:
#   ./run-e2e-tests-local.sh [--keep-running] [--no-build]
#
# Environment Variables:
#   BASE_URL - Frontend URL (default: http://localhost:3001)
#   TEST_BACKEND_ORIGIN - Backend URL (default: http://localhost:8000)
#   NODE_VERSION - Node.js version (default: 20)
#
# Exit Codes:
#   0 - All E2E tests passed
#   1 - Tests failed or setup error
###############################################################################

# Configuration (aligned with .github/workflows/e2e-tests.yml)
BASE_URL=${BASE_URL:-http://localhost:3001}
TEST_BACKEND_ORIGIN=${TEST_BACKEND_ORIGIN:-http://localhost:8000}
MAX_WAIT=60  # seconds for service health checks
POLL_INTERVAL=5

# Flags
BUILD_SERVICES=true
KEEP_RUNNING=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-build)
      BUILD_SERVICES=false
      shift
      ;;
    --keep-running)
      KEEP_RUNNING=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

###############################################################################
# Utility Functions
###############################################################################

log_header() {
  echo ""
  echo -e "${BLUE}════════════════════════════════════════${NC}"
  echo -e "${BLUE}$1${NC}"
  echo -e "${BLUE}════════════════════════════════════════${NC}"
  echo ""
}

log_section() {
  echo ""
  echo -e "${YELLOW}▶ $1${NC}"
}

log_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
  echo -e "${RED}✗ $1${NC}"
}

log_info() {
  echo -e "${BLUE}ℹ $1${NC}"
}

cleanup_on_exit() {
  local exit_code=$?
  
  echo ""
  log_section "Cleanup"
  
  if [ "$KEEP_RUNNING" = false ]; then
    log_info "Stopping backend services..."
    docker compose down -v 2>/dev/null || true
    log_success "Services stopped"
  else
    log_info "Services kept running (use 'docker compose down -v' to stop manually)"
  fi
  
  exit $exit_code
}

trap cleanup_on_exit EXIT

check_command() {
  if ! command -v "$1" &> /dev/null; then
    log_error "Required command not found: $1"
    echo "Please install $1 and try again."
    exit 1
  fi
}

wait_for_health() {
  local url=$1
  local service=$2
  local timeout=${3:-300}
  
  log_info "Waiting for $service at $url (max ${timeout}s)..."
  
  local elapsed=0
  while [ $elapsed -lt $timeout ]; do
    if curl -f -s "$url" > /dev/null 2>&1; then
      log_success "$service is healthy"
      return 0
    fi
    
    elapsed=$((elapsed + POLL_INTERVAL))
    sleep $POLL_INTERVAL
  done
  
  log_error "$service did not become healthy within ${timeout}s"
  return 1
}

###############################################################################
# Main Script
###############################################################################

log_header "E2E Test Orchestration (GitHub Workflow Aligned)"

#!/usr/bin/env bash


# Prevent the script from being sourced. Running it with `. script` causes $0/BASH_SOURCE to differ
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "This script must be executed, not sourced. Run: ./scripts/run-e2e-tests-local.sh"
  # If sourced, return to avoid exiting the parent shell; otherwise exit
  return 1 2>/dev/null || exit 1
fi

echo "Configuration:"
echo "  BASE_URL: $BASE_URL"
echo "  TEST_BACKEND_ORIGIN: $TEST_BACKEND_ORIGIN"
echo "  Build services: $BUILD_SERVICES"
echo "  Keep running: $KEEP_RUNNING"

# Verify required commands
check_command "docker"
check_command "curl"
check_command "npm"
check_command "node"

# Resolve project root from this script location
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

cd "$PROJECT_ROOT"

###############################################################################
# Step 1: Start Backend Services (Only Essential)
###############################################################################

log_section "Step 1: Starting backend services (postgres, redis, 4 microservices)"

if [ "$BUILD_SERVICES" = true ]; then
  log_info "Building services..."
  docker compose up -d postgres redis
  
  # Build only essential microservices (skip training service for disk space)
  docker compose build api-gateway rf-acquisition data-ingestion-web inference 2>&1 | tail -10
fi

log_info "Starting services..."
docker compose up -d api-gateway rf-acquisition data-ingestion-web inference

log_info "Service status:"
docker compose ps --services | grep -E "postgres|redis|api-gateway|rf-acquisition|data-ingestion-web|inference" || true

###############################################################################
# Step 2: Wait for Backend Services to be Healthy
###############################################################################

log_section "Step 2: Waiting for backend services to be healthy"

if ! wait_for_health "${TEST_BACKEND_ORIGIN}/health" "API Gateway"; then
  log_error "API Gateway failed to become healthy"
  log_info "Showing API Gateway logs:"
  docker compose logs api-gateway | tail -20
  exit 1
fi

# Non-critical: these services are optional for E2E tests
wait_for_health "http://localhost:8001/health" "RF Acquisition" 60 || log_info "RF Acquisition not ready (non-critical)"
wait_for_health "http://localhost:8004/health" "Data Ingestion" 60 || log_info "Data Ingestion not ready (non-critical)"

log_success "Backend services are ready"

###############################################################################
# Step 3: Setup Frontend
###############################################################################

log_section "Step 3: Setting up and starting frontend"

cd "$PROJECT_ROOT/frontend"

# Install dependencies (clean install like GitHub)
log_info "Installing frontend dependencies..."
npm ci 2>&1 | tail -5

# Install Playwright browsers
log_info "Installing Playwright browsers..."
npx playwright install --with-deps chromium 2>&1 | tail -5

# Build frontend
log_info "Building frontend..."
npm run build 2>&1 | tail -10

# Start frontend dev server in background
log_info "Starting frontend dev server..."
npm run dev > /tmp/frontend-dev.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be ready
log_info "Waiting for frontend dev server (max 120s)..."
for attempt in $(seq 1 60); do
  if curl -f -s http://localhost:3001 > /dev/null 2>&1; then
    log_success "Frontend dev server is ready"
    break
  fi
  
  if [ $attempt -eq 60 ]; then
    log_error "Frontend dev server failed to start"
    log_info "Frontend logs:"
    cat /tmp/frontend-dev.log
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
  fi
  
  sleep 2
done

###############################################################################
# Step 4: Run Playwright E2E Tests
###############################################################################

log_section "Step 4: Running Playwright E2E tests"

export BASE_URL
export TEST_BACKEND_ORIGIN
export CI=true

log_info "Running tests..."
echo ""

if npx playwright test --reporter=html,json,list; then
  TEST_EXIT_CODE=0
  log_success "All E2E tests passed!"
else
  TEST_EXIT_CODE=$?
  log_error "Some E2E tests failed (exit code: $TEST_EXIT_CODE)"
fi

###############################################################################
# Step 5: Collect Artifacts (Local Filesystem)
###############################################################################

log_section "Step 5: Collecting test artifacts"

ARTIFACTS_DIR="e2e-artifacts-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ARTIFACTS_DIR"

# Copy Playwright report
if [ -d "playwright-report" ]; then
  log_info "Copying Playwright report..."
  cp -r playwright-report "$ARTIFACTS_DIR/" 2>/dev/null || true
fi

# Copy test results (screenshots, videos)
if [ -d "test-results" ]; then
  log_info "Copying test results..."
  cp -r test-results "$ARTIFACTS_DIR/" 2>/dev/null || true
fi

# Collect backend logs on failure
if [ $TEST_EXIT_CODE -ne 0 ]; then
  log_info "Collecting backend service logs (test failed)..."
  mkdir -p "$ARTIFACTS_DIR/logs"
  
  docker compose logs api-gateway > "$ARTIFACTS_DIR/logs/api-gateway.log" 2>&1 || true
  docker compose logs rf-acquisition > "$ARTIFACTS_DIR/logs/rf-acquisition.log" 2>&1 || true
  docker compose logs data-ingestion-web > "$ARTIFACTS_DIR/logs/data-ingestion-web.log" 2>&1 || true
  docker compose logs postgres > "$ARTIFACTS_DIR/logs/postgres.log" 2>&1 || true
  docker compose logs redis > "$ARTIFACTS_DIR/logs/redis.log" 2>&1 || true
fi

# Generate summary report
cat > "$ARTIFACTS_DIR/SUMMARY.md" <<EOF
# E2E Test Execution Summary

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Base URL**: $BASE_URL
**Backend Origin**: $TEST_BACKEND_ORIGIN
**Exit Code**: $TEST_EXIT_CODE

## Test Result

- Status: $([ $TEST_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")
- Exit Code: $TEST_EXIT_CODE

## Environment

- Node.js: $(node --version)
- npm: $(npm --version)
- Docker: $(docker --version)

## Artifacts

- \`playwright-report/\` - HTML test report
- \`test-results/\` - Screenshots and videos
- \`logs/\` - Backend service logs (if test failed)

## To View Report

\`\`\`bash
# Open HTML report in browser
open $ARTIFACTS_DIR/playwright-report/index.html
# or
xdg-open $ARTIFACTS_DIR/playwright-report/index.html  # Linux
\`\`\`

## To Reproduce Locally

\`\`\`bash
# Ensure backend services are running
docker compose up -d postgres redis api-gateway rf-acquisition data-ingestion-web inference

# Wait for health
curl http://localhost:8000/health

# Run frontend
cd frontend
npm ci
npm run build
npm run dev &

# Run tests
BASE_URL=http://localhost:3001 \\
TEST_BACKEND_ORIGIN=http://localhost:8000 \\
npx playwright test

# Cleanup
docker compose down -v
\`\`\`

## Notes

- This report was generated by \`scripts/run-e2e-tests-local.sh\`
- Aligns with GitHub Actions workflow: \`.github/workflows/e2e-tests.yml\`
- Backend services: postgres, redis, api-gateway, rf-acquisition, data-ingestion-web, inference

EOF

log_success "Artifacts collected in: $ARTIFACTS_DIR"
log_info "View summary: cat $ARTIFACTS_DIR/SUMMARY.md"

###############################################################################
# Final Status
###############################################################################

log_header "E2E Test Execution Complete"

if [ $TEST_EXIT_CODE -eq 0 ]; then
  log_success "All tests PASSED"
  echo ""
  echo "Artifacts: $ARTIFACTS_DIR"
  echo "Report: file://$(pwd)/$ARTIFACTS_DIR/playwright-report/index.html"
  exit 0
else
  log_error "Some tests FAILED (exit code: $TEST_EXIT_CODE)"
  echo ""
  echo "Artifacts: $ARTIFACTS_DIR"
  echo "Report: file://$(pwd)/$ARTIFACTS_DIR/playwright-report/index.html"
  echo "Logs: $ARTIFACTS_DIR/logs/"
  exit $TEST_EXIT_CODE
fi
