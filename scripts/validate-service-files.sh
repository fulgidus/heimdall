#!/bin/bash
# Validate that all required service files exist before building Docker images

set -e

echo "🔍 Validating service files for Docker build..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

missing=0

# Services to check
services=("api-gateway" "rf-acquisition" "data-ingestion-web" "inference" "training")

for svc in "${services[@]}"; do
    echo ""
    echo "📦 Checking service: $svc"
    
    # Check if service directory exists
    if [ ! -d "services/$svc" ]; then
        echo -e "  ${RED}✗${NC} Directory services/$svc/ not found"
        missing=1
        continue
    fi
    
    # Check for src directory
    if [ ! -d "services/$svc/src" ]; then
        echo -e "  ${RED}✗${NC} Missing: services/$svc/src/"
        missing=1
    else
        echo -e "  ${GREEN}✓${NC} Found: services/$svc/src/"
    fi
    
    # Check for requirements.txt
    if [ ! -f "services/$svc/requirements.txt" ]; then
        echo -e "  ${RED}✗${NC} Missing: services/$svc/requirements.txt"
        missing=1
    else
        echo -e "  ${GREEN}✓${NC} Found: services/$svc/requirements.txt"
    fi
    
    # Check for Dockerfile
    if [ ! -f "services/$svc/Dockerfile" ]; then
        echo -e "  ${RED}✗${NC} Missing: services/$svc/Dockerfile"
        missing=1
    else
        echo -e "  ${GREEN}✓${NC} Found: services/$svc/Dockerfile"
    fi
    
    # Special check for rf-acquisition entrypoint.py
    if [ "$svc" = "rf-acquisition" ]; then
        if [ ! -f "services/$svc/entrypoint.py" ]; then
            echo -e "  ${RED}✗${NC} Missing: services/$svc/entrypoint.py"
            missing=1
        else
            echo -e "  ${GREEN}✓${NC} Found: services/$svc/entrypoint.py"
        fi
    fi
done

echo ""
if [ "$missing" -ne 0 ]; then
    echo -e "${RED}❌ Validation failed: One or more required files are missing${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All required service files present${NC}"
    exit 0
fi
