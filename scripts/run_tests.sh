#!/bin/bash
# Comprehensive test runner for Heimdall project
# Usage: ./scripts/run_tests.sh [category]
#
# Categories:
#   all         - Run all tests (default)
#   unit        - Run unit tests only
#   integration - Run integration tests
#   performance - Run performance benchmarks
#   coverage    - Run with coverage report
#   training    - Run training service tests
#   backend     - Run backend service tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
CATEGORY="${1:-all}"

echo -e "${GREEN}=== Heimdall Test Runner ===${NC}"
echo "Category: $CATEGORY"
echo ""

# Function to run tests with proper output
run_tests() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${YELLOW}Running: $description${NC}"
    echo "Command: $cmd"
    echo ""
    
    if eval "$cmd"; then
        echo -e "${GREEN}✓ $description passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $description failed${NC}"
        echo ""
        return 1
    fi
}

case "$CATEGORY" in
    all)
        echo "Running all tests..."
        run_tests "pytest -v" "All tests"
        ;;
    
    unit)
        echo "Running unit tests only..."
        run_tests "pytest -m unit -v" "Unit tests"
        ;;
    
    integration)
        echo "Running integration tests..."
        echo -e "${YELLOW}Note: Integration tests require Docker services to be running${NC}"
        run_tests "pytest -m integration -v" "Integration tests"
        ;;
    
    performance)
        echo "Running performance benchmarks..."
        run_tests "pytest -m performance -v" "Performance tests"
        ;;
    
    coverage)
        echo "Running tests with coverage..."
        run_tests "pytest --cov=services --cov-report=html --cov-report=term-missing" "Tests with coverage"
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    
    training)
        echo "Running training service tests..."
        cd services/training
        run_tests "pytest -v" "Training service tests"
        cd "$PROJECT_ROOT"
        ;;
    
    backend)
        echo "Running backend service tests..."
        cd services/backend
        run_tests "pytest tests/unit/ -v" "Backend unit tests"
        cd "$PROJECT_ROOT"
        ;;
    
    quick)
        echo "Running quick tests (unit + performance, no slow)..."
        run_tests "pytest -m 'unit or (performance and not slow)' -v" "Quick tests"
        ;;
    
    *)
        echo -e "${RED}Unknown category: $CATEGORY${NC}"
        echo ""
        echo "Available categories:"
        echo "  all         - Run all tests"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests"
        echo "  performance - Run performance benchmarks"
        echo "  coverage    - Run with coverage report"
        echo "  training    - Run training service tests"
        echo "  backend     - Run backend service tests"
        echo "  quick       - Run quick tests (unit + fast performance)"
        exit 1
        ;;
esac

echo -e "${GREEN}=== Test run complete ===${NC}"
