#!/bin/bash

###############################################################################
# E2E Test Validation Script
# 
# Validates that E2E tests are properly configured and can detect real
# backend calls (NO mocking).
#
# Usage: ./validate-e2e-setup.sh
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}E2E Test Setup Validation${NC}"
echo -e "${BLUE}=====================================${NC}\n"

validation_passed=0
validation_failed=0

check_step() {
  local name="$1"
  local result="$2"
  
  if [ "$result" = "0" ]; then
    echo -e "${GREEN}✓${NC} $name"
    validation_passed=$((validation_passed + 1))
  else
    echo -e "${RED}✗${NC} $name"
    validation_failed=$((validation_failed + 1))
  fi
}

###############################################################################
# Step 1: Check File Structure
###############################################################################

echo -e "${YELLOW}[Step 1] Checking file structure...${NC}\n"

# Check Playwright config
if [ -f "$PROJECT_ROOT/frontend/playwright.config.ts" ]; then
  check_step "Playwright config exists" 0
else
  check_step "Playwright config exists" 1
fi

# Check test helper
if [ -f "$PROJECT_ROOT/frontend/e2e/helpers/test-utils.ts" ]; then
  check_step "Test helpers exist" 0
else
  check_step "Test helpers exist" 1
fi

# Check test files
test_files=(
  "login.spec.ts"
  "dashboard.spec.ts"
  "projects.spec.ts"
  "websdr-management.spec.ts"
  "analytics.spec.ts"
  "localization.spec.ts"
  "settings.spec.ts"
  "profile.spec.ts"
  "system-status.spec.ts"
)

missing_tests=0
for test_file in "${test_files[@]}"; do
  if [ ! -f "$PROJECT_ROOT/frontend/e2e/$test_file" ]; then
    missing_tests=$((missing_tests + 1))
  fi
done

if [ $missing_tests -eq 0 ]; then
  check_step "All 9 test files present" 0
else
  check_step "All 9 test files present (missing: $missing_tests)" 1
fi

# Check orchestration script
if [ -x "$PROJECT_ROOT/scripts/run-e2e-tests.sh" ]; then
  check_step "Orchestration script executable" 0
else
  check_step "Orchestration script executable" 1
fi

echo ""

###############################################################################
# Step 2: Check Dependencies
###############################################################################

echo -e "${YELLOW}[Step 2] Checking dependencies...${NC}\n"

cd "$PROJECT_ROOT/frontend"

# Check package.json
if grep -q "@playwright/test" package.json; then
  check_step "Playwright dependency in package.json" 0
else
  check_step "Playwright dependency in package.json" 1
fi

# Check node_modules
if [ -d "node_modules/@playwright/test" ]; then
  check_step "Playwright installed in node_modules" 0
else
  check_step "Playwright installed in node_modules" 1
fi

# Check test scripts
if grep -q "test:e2e" package.json; then
  check_step "E2E test scripts in package.json" 0
else
  check_step "E2E test scripts in package.json" 1
fi

echo ""

###############################################################################
# Step 3: Verify NO Mocking Configuration
###############################################################################

echo -e "${YELLOW}[Step 3] Verifying NO mocking/stubbing...${NC}\n"

# Check for MSW (Mock Service Worker)
if grep -q "msw" "$PROJECT_ROOT/frontend/package.json" 2>/dev/null; then
  check_step "NO MSW (Mock Service Worker) found" 1
else
  check_step "NO MSW (Mock Service Worker) found" 0
fi

# Check for axios-mock-adapter
if grep -q "axios-mock-adapter" "$PROJECT_ROOT/frontend/package.json" 2>/dev/null; then
  check_step "NO axios-mock-adapter found" 1
else
  check_step "NO axios-mock-adapter found" 0
fi

# Check test files don't use mocking
mock_patterns=("jest.mock" "vi.mock" "mockResolvedValue" "mockImplementation")
mock_found=0

for pattern in "${mock_patterns[@]}"; do
  if grep -r "$pattern" "$PROJECT_ROOT/frontend/e2e/"*.spec.ts 2>/dev/null; then
    echo -e "  ${YELLOW}Warning: Found '$pattern' in E2E tests${NC}"
    mock_found=$((mock_found + 1))
  fi
done

if [ $mock_found -eq 0 ]; then
  check_step "NO mock patterns in E2E tests" 0
else
  check_step "NO mock patterns in E2E tests ($mock_found patterns found)" 1
fi

echo ""

###############################################################################
# Step 4: Check Test Configuration
###############################################################################

echo -e "${YELLOW}[Step 4] Checking test configuration...${NC}\n"

# Check playwright.config.ts has HAR recording
if grep -q "recordHar" "$PROJECT_ROOT/frontend/playwright.config.ts"; then
  check_step "HAR recording enabled" 0
else
  check_step "HAR recording enabled" 1
fi

# Check for screenshot configuration
if grep -q "screenshot.*only-on-failure" "$PROJECT_ROOT/frontend/playwright.config.ts"; then
  check_step "Screenshot on failure configured" 0
else
  check_step "Screenshot on failure configured" 1
fi

# Check for trace configuration
if grep -q "trace.*on-first-retry" "$PROJECT_ROOT/frontend/playwright.config.ts"; then
  check_step "Trace on retry configured" 0
else
  check_step "Trace on retry configured" 1
fi

# Check base URL configuration
if grep -q "baseURL.*process.env.BASE_URL" "$PROJECT_ROOT/frontend/playwright.config.ts"; then
  check_step "Base URL from environment" 0
else
  check_step "Base URL from environment" 1
fi

echo ""

###############################################################################
# Step 5: Verify Test Helper Functions
###############################################################################

echo -e "${YELLOW}[Step 5] Verifying test helper functions...${NC}\n"

helpers_file="$PROJECT_ROOT/frontend/e2e/helpers/test-utils.ts"

# Check for waitForBackendCall
if grep -q "waitForBackendCall" "$helpers_file"; then
  check_step "waitForBackendCall helper exists" 0
else
  check_step "waitForBackendCall helper exists" 1
fi

# Check for setupRequestLogging (NO mocking)
if grep -q "setupRequestLogging" "$helpers_file"; then
  check_step "setupRequestLogging helper exists" 0
else
  check_step "setupRequestLogging helper exists" 1
fi

# Check for login helper
if grep -q "export.*function login" "$helpers_file"; then
  check_step "login helper exists" 0
else
  check_step "login helper exists" 1
fi

# Verify helpers don't mock requests
if grep -q "route.fulfill\|route.abort\|page.route" "$helpers_file"; then
  check_step "Helpers don't mock requests" 1
else
  check_step "Helpers don't mock requests" 0
fi

echo ""

###############################################################################
# Step 6: Check Backend Configuration Files
###############################################################################

echo -e "${YELLOW}[Step 6] Checking backend configuration...${NC}\n"

cd "$PROJECT_ROOT"

# Check docker compose files
if [ -f "docker compose.yml" ]; then
  check_step "docker compose.yml exists" 0
else
  check_step "docker compose.yml exists" 1
fi

# Check for api-gateway service
if grep -q "api-gateway:" docker compose.yml; then
  check_step "API Gateway service configured" 0
else
  check_step "API Gateway service configured" 1
fi

# Check for health check endpoints
if grep -q "health" docker compose.yml; then
  check_step "Health checks configured" 0
else
  check_step "Health checks configured" 1
fi

echo ""

###############################################################################
# Summary
###############################################################################

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}=====================================${NC}\n"

echo -e "Passed: ${GREEN}$validation_passed${NC}"
echo -e "Failed: ${RED}$validation_failed${NC}"
echo -e "Total:  $((validation_passed + validation_failed))\n"

if [ $validation_failed -eq 0 ]; then
  echo -e "${GREEN}✅ ALL VALIDATIONS PASSED${NC}"
  echo -e "\n${YELLOW}Next steps:${NC}"
  echo -e "  1. Start backend: ${BLUE}docker compose up -d${NC}"
  echo -e "  2. Run E2E tests: ${BLUE}cd frontend && npm run test:e2e${NC}"
  echo -e "  3. View report: ${BLUE}npm run test:e2e:report${NC}\n"
  exit 0
else
  echo -e "${RED}❌ VALIDATION FAILED${NC}"
  echo -e "\n${YELLOW}Please fix the issues above before running E2E tests.${NC}\n"
  exit 1
fi
