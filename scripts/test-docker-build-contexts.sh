#!/bin/bash
# Test script to verify Docker build contexts are correct
# This script should pass after the fix for Docker build context issues

set -e

echo "🧪 Testing Docker build contexts fix..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Validate service files exist
echo "Test 1: Validating service files..."
if ./scripts/validate-service-files.sh > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} All service files present"
else
    echo -e "  ${RED}✗${NC} Service validation failed"
    exit 1
fi

# Test 2: Verify docker-compose configuration
echo "Test 2: Verifying docker-compose.yml build contexts..."

# Check rf-acquisition
context=$(docker compose config | grep -A 2 "rf-acquisition:" | grep "context:" | awk '{print $2}')
if [[ "$context" == *"services/rf-acquisition"* ]]; then
    echo -e "  ${GREEN}✓${NC} rf-acquisition has correct context: $context"
else
    echo -e "  ${RED}✗${NC} rf-acquisition has wrong context: $context"
    exit 1
fi

# Check data-ingestion-web
context=$(docker compose config | grep -A 2 "data-ingestion-web:" | grep "context:" | awk '{print $2}')
if [[ "$context" == *"services/data-ingestion-web"* ]]; then
    echo -e "  ${GREEN}✓${NC} data-ingestion-web has correct context: $context"
else
    echo -e "  ${RED}✗${NC} data-ingestion-web has wrong context: $context"
    exit 1
fi

# Check inference
context=$(docker compose config | grep -A 2 "inference:" | grep "context:" | awk '{print $2}')
if [[ "$context" == *"services/inference"* ]]; then
    echo -e "  ${GREEN}✓${NC} inference has correct context: $context"
else
    echo -e "  ${RED}✗${NC} inference has wrong context: $context"
    exit 1
fi

# Check api-gateway (should be ./services for common/auth access)
context=$(docker compose config | grep -A 2 "api-gateway:" | grep "context:" | awk '{print $2}')
if [[ "$context" == *"services"* ]] && [[ "$context" != *"services/api-gateway"* ]]; then
    echo -e "  ${GREEN}✓${NC} api-gateway has correct context: $context (needs common/auth)"
else
    echo -e "  ${RED}✗${NC} api-gateway has unexpected context: $context"
    exit 1
fi

# Check training
context=$(docker compose config | grep -A 2 "training:" | grep "context:" | awk '{print $2}')
if [[ "$context" == *"services/training"* ]]; then
    echo -e "  ${GREEN}✓${NC} training has correct context: $context"
else
    echo -e "  ${RED}✗${NC} training has wrong context: $context"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""
echo "Summary:"
echo "  - rf-acquisition: context set to ./services/rf-acquisition"
echo "  - data-ingestion-web: context set to ./services/data-ingestion-web"
echo "  - inference: context set to ./services/inference"
echo "  - api-gateway: context remains ./services (needs common/auth)"
echo "  - training: context set to ./services/training"
echo ""
echo "These changes fix the Docker build errors in CI where files were not found."
