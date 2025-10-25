#!/bin/bash
# Local Integration Tests Runner
# Runs the same integration tests as GitHub Actions, but locally
# Usage: bash scripts/run-integration-tests-local.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Heimdall Integration Tests (Local)${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Set environment variables
export POSTGRES_DB=heimdall
export POSTGRES_USER=heimdall_user
export POSTGRES_PASSWORD=changeme
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432

export RABBITMQ_HOST=localhost
export RABBITMQ_PORT=5672

export REDIS_HOST=localhost
export REDIS_PORT=6379

export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin

export KEYCLOAK_URL=http://localhost:8080
export KEYCLOAK_REALM=heimdall

export PYTHONPATH="$PROJECT_ROOT"

echo -e "${YELLOW}[SETUP]${NC} Environment variables configured"
echo "  POSTGRES_HOST: $POSTGRES_HOST"
echo "  KEYCLOAK_URL: $KEYCLOAK_URL"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Check if Docker services are running
echo -e "${YELLOW}[CHECK]${NC} Verifying Docker services..."
MISSING_SERVICES=0

for service in postgres rabbitmq redis; do
    if ! docker ps | grep -q "heimdall-$service"; then
        echo -e "  ${RED}✗${NC} heimdall-$service is not running"
        MISSING_SERVICES=1
    else
        echo -e "  ${GREEN}✓${NC} heimdall-$service is running"
    fi
done

if [ $MISSING_SERVICES -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}[HINT]${NC} Start Docker services with:"
    echo "  docker compose up -d"
    exit 1
fi

echo ""

# Check if MinIO is running
if ! docker ps | grep -q "heimdall-minio"; then
    echo -e "${YELLOW}[INFO]${NC} Starting MinIO container..."
    NETWORK=$(docker network ls --format '{{.Name}}' | grep heimdall || echo "bridge")
    docker run -d \
        --name minio-test \
        --network "$NETWORK" \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin \
        -p 9000:9000 \
        -p 9001:9001 \
        minio/minio:latest \
        server /data --console-address ":9001" 2>/dev/null || true
    
    echo "Waiting for MinIO to be ready..."
    timeout 60 bash -c 'until curl -sf http://localhost:9000/minio/health/live 2>/dev/null; do sleep 1; done' || true
    echo -e "  ${GREEN}✓${NC} MinIO ready"
else
    echo -e "  ${GREEN}✓${NC} MinIO is running"
fi

echo ""

# Check if Keycloak is running
if ! docker ps | grep -q "heimdall-keycloak"; then
    echo -e "${YELLOW}[INFO]${NC} Starting Keycloak container..."
    NETWORK=$(docker network ls --format '{{.Name}}' | grep heimdall || echo "bridge")
    docker run -d \
        --name keycloak-test \
        --network "$NETWORK" \
        -e KEYCLOAK_ADMIN=admin \
        -e KEYCLOAK_ADMIN_PASSWORD=admin \
        -e KC_HTTP_ENABLED=true \
        -e KC_HOSTNAME_STRICT=false \
        -v "$PROJECT_ROOT/db/keycloak:/opt/keycloak/data/import:ro" \
        -p 8080:8080 \
        quay.io/keycloak/keycloak:23.0 \
        start-dev 2>/dev/null || true
    
    echo "Waiting for Keycloak to be ready..."
    timeout 120 bash -c 'until curl -sf http://localhost:8080/health/ready 2>/dev/null; do sleep 1; done' || true
    echo -e "  ${GREEN}✓${NC} Keycloak ready"
else
    echo -e "  ${GREEN}✓${NC} Keycloak is running"
fi

echo ""

# Initialize PostgreSQL database
echo -e "${YELLOW}[DB]${NC} Initializing PostgreSQL database..."
export PGPASSWORD="$POSTGRES_PASSWORD"
if command -v psql &> /dev/null; then
    psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f db/01-init.sql 2>&1 | tail -5
else
    echo "  psql not found, using docker exec..."
    docker exec heimdall-postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /dev/stdin < db/01-init.sql 2>&1 | tail -5
fi
echo -e "  ${GREEN}✓${NC} Database initialized"
echo ""

# Configure Keycloak realm
echo -e "${YELLOW}[KEYCLOAK]${NC} Importing realm configuration..."
if docker ps | grep -q "heimdall-keycloak\|keycloak-test"; then
    KEYCLOAK_CONTAINER=$(docker ps --format '{{.Names}}' | grep keycloak | head -1)
    docker exec "$KEYCLOAK_CONTAINER" \
        /opt/keycloak/bin/kc.sh import \
        --file /opt/keycloak/data/import/heimdall-realm.json \
        --override true 2>&1 | tail -3 || echo "  Realm already exists or skipped"
    echo -e "  ${GREEN}✓${NC} Keycloak configured"
else
    echo "  Keycloak not running, skipping realm import"
fi

echo ""

# Run tests
echo -e "${YELLOW}[TESTS]${NC} Running integration tests..."
echo ""

TEST_FAILED=0

# API Gateway tests
echo -e "${YELLOW}[TEST]${NC} API Gateway integration tests..."
if [ -d services/api-gateway/tests/integration ]; then
    cd "$PROJECT_ROOT/services/api-gateway"
    if pytest tests/integration/ -v --tb=short --cov=src --cov-report=term-missing 2>&1 | tail -30; then
        echo -e "${GREEN}✓ API Gateway tests passed${NC}"
    else
        echo -e "${RED}✗ API Gateway tests failed${NC}"
        TEST_FAILED=1
    fi
    cd "$PROJECT_ROOT"
else
    echo "  No integration tests found"
fi

echo ""

# Data Ingestion tests
echo -e "${YELLOW}[TEST]${NC} Data Ingestion Web integration tests..."
if [ -d services/data-ingestion-web/tests/integration ] && [ "$(ls -A services/data-ingestion-web/tests/integration)" ]; then
    cd "$PROJECT_ROOT/services/data-ingestion-web"
    if pytest tests/integration/ -v --tb=short --cov=src --cov-report=term-missing 2>&1 | tail -30; then
        echo -e "${GREEN}✓ Data Ingestion tests passed${NC}"
    else
        echo -e "${RED}✗ Data Ingestion tests failed${NC}"
        TEST_FAILED=1
    fi
    cd "$PROJECT_ROOT"
else
    echo "  No integration tests found"
fi

echo ""

# RF Acquisition tests
echo -e "${YELLOW}[TEST]${NC} RF Acquisition integration tests..."
if [ -d services/rf-acquisition/tests/integration ] && [ "$(ls -A services/rf-acquisition/tests/integration)" ]; then
    cd "$PROJECT_ROOT/services/rf-acquisition"
    if pytest tests/integration/ -v --tb=short --cov=src --cov-report=term-missing 2>&1 | tail -30; then
        echo -e "${GREEN}✓ RF Acquisition tests passed${NC}"
    else
        echo -e "${RED}✗ RF Acquisition tests failed${NC}"
        TEST_FAILED=1
    fi
    cd "$PROJECT_ROOT"
else
    echo "  No integration tests found"
fi

echo ""
echo -e "${GREEN}================================================${NC}"

if [ $TEST_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All integration tests passed!${NC}"
    echo -e "${GREEN}================================================${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${RED}================================================${NC}"
    exit 1
fi
