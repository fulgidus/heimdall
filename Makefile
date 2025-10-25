# Heimdall SDR - Development Makefile
# Cross-platform compatible commands

.PHONY: help dev-up dev-down test test-local test-api get-token lint format build-docker db-migrate clean setup lock-deps audit-deps deps-check

# Default target
help:
	@echo "Heimdall SDR Development Commands:"
	@echo ""
	@echo "  🚀 QUICK START:"
	@echo "    setup                 Setup dev environment (first time only)"
	@echo ""
	@echo "  INFRASTRUCTURE (Phase 1):"
	@echo "    dev-up                Start development environment"
	@echo "    dev-down              Stop development environment"
	@echo "    infra-status          Show docker compose status"
	@echo "    health-check          Run health check script"
	@echo "    postgres-connect      Connect to PostgreSQL CLI"
	@echo "    redis-cli             Connect to Redis CLI"
	@echo ""
	@echo "  UI DASHBOARDS:"
	@echo "    rabbitmq-ui           Open RabbitMQ management UI"
	@echo "    minio-ui              Open MinIO console"
	@echo "    grafana-ui            Open Grafana dashboards"
	@echo "    prometheus-ui         Open Prometheus metrics"
	@echo ""
	@echo "  DEVELOPMENT:"
	@echo "    test                  Run all tests in Docker containers (with auth)"
	@echo "    test-local            Run tests locally (without Docker)"
	@echo "    test-api              Test API Gateway with authentication"
	@echo "    get-token             Get Keycloak authentication token"
	@echo "    lint                  Run code linting"
	@echo "    format                Auto-format code"
	@echo "    db-migrate            Run database migrations"
	@echo ""
	@echo "  MAINTENANCE:"
	@echo "    clean                 Clean all generated files"
	@echo ""
	@echo "  DEPENDENCY MANAGEMENT:"
	@echo "    lock-deps             Generate lock files from requirements"
	@echo "    audit-deps            Audit dependencies for conflicts and vulnerabilities"
	@echo "    deps-check            Run lock-deps and audit-deps"
	@echo ""

# Development Environment
dev-up:
	docker compose up -d
	@echo "Development environment started. Waiting for services to be healthy..."
	@timeout /t 10 /nobreak
	docker compose ps

dev-down:
	docker compose down
	@echo "Development environment stopped."

dev-logs:
	docker compose logs -f

# Infrastructure Operations (Phase 1)
infra-status:
	docker compose ps

infra-logs:
	docker compose logs -f

postgres-connect:
	docker compose exec postgres psql -U heimdall_user -d heimdall

postgres-cli:
	docker compose exec postgres bash

redis-cli:
	docker compose exec redis redis-cli -a changeme

rabbitmq-ui:
	@echo "RabbitMQ UI: http://localhost:15672 (guest/guest)"

minio-ui:
	@echo "MinIO UI: http://localhost:9001 (minioadmin/minioadmin)"

grafana-ui:
	@echo "Grafana UI: http://localhost:3000 (admin/admin)"

prometheus-ui:
	@echo "Prometheus UI: http://localhost:9090"

# Testing
test:
	@echo "Running tests in all services..."
	@echo "Getting authentication token from Keycloak..."
	@if ! TOKEN=$$(./scripts/get-keycloak-token.sh 2>/dev/null); then \
		echo "⚠ WARNING: Failed to get Keycloak token. Tests may fail if authentication is required."; \
		TOKEN=""; \
	else \
		echo "✓ Authentication token obtained successfully"; \
	fi; \
	export KEYCLOAK_TOKEN="$$TOKEN"; \
	for service in services/*/; do \
		if [ -d "$$service/tests" ]; then \
			service_name=$$(basename $$service); \
			echo "Testing $$service_name..."; \
			if docker compose ps $$service_name 2>&1 | grep -q "Up"; then \
				docker compose exec -T -e KEYCLOAK_TOKEN="$$TOKEN" $$service_name sh -c "if command -v pytest >/dev/null 2>&1; then \
					if [ -d tests ]; then \
						if pip list 2>/dev/null | grep -q pytest-cov; then \
							pytest tests/ -v --cov=src/ --cov-report=term-missing || true; \
						else \
							pytest tests/ -v || true; \
						fi; \
					else \
						echo 'No tests directory found in container'; \
					fi; \
				else \
					echo 'pytest not installed in $$service_name'; \
				fi" || echo "Failed to run tests in $$service_name"; \
			else \
				echo "⚠ Service $$service_name is not running - skipping tests"; \
			fi; \
		fi; \
	done
	@echo "Test run completed."

# Run tests locally (without Docker)
test-local:
	@echo "Running tests locally..."
	@if [ -d "venv" ] && [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠ Virtual environment exists but not activated."; \
		echo "  Activate it with: source venv/bin/activate"; \
		echo "  Or run: make setup"; \
		exit 1; \
	fi
	@if command -v pytest >/dev/null 2>&1; then \
		for service in services/*/; do \
			if [ -d "$$service/tests" ]; then \
				service_name=$$(basename $$service); \
				echo ""; \
				echo "=== Testing $$service_name ==="; \
				cd "$$service" && pytest tests/ -v || true; \
				cd ../..; \
			fi; \
		done; \
	else \
		echo "pytest not installed."; \
		echo "Run 'make setup' to install all dependencies."; \
		exit 1; \
	fi

# Get Keycloak token (for manual testing)
get-token:
	@if [ -f .env ]; then \
		./scripts/get-keycloak-token.sh; \
	else \
		echo "ERROR: .env file not found. Copy .env.example to .env first."; \
		exit 1; \
	fi

# Test API Gateway with authentication
test-api:
	@echo "Testing API Gateway with authentication..."
	@TOKEN=$$(./scripts/get-keycloak-token.sh 2>/dev/null) || { echo "Failed to get token"; exit 1; }; \
	API_GATEWAY_URL=$$(grep -v '^#' .env 2>/dev/null | grep API_GATEWAY_URL | cut -d'=' -f2 | tr -d '\r' || echo "http://localhost:8000"); \
	echo "Calling $$API_GATEWAY_URL/health..."; \
	curl -s -H "Authorization: Bearer $$TOKEN" "$$API_GATEWAY_URL/health" | jq . 2>/dev/null || \
	curl -s -H "Authorization: Bearer $$TOKEN" "$$API_GATEWAY_URL/health"

# Code Quality
lint:
	@echo "Running linting on all Python files..."
	@find . -name "*.py" -type f | while read -r file; do \
		echo "Linting $$file..."; \
		black --check "$$file" && \
		ruff check "$$file"; \
	done

format:
	@echo "Formatting all Python files..."
	@find . -name "*.py" -type f | while read -r file; do \
		echo "Formatting $$file..."; \
		black "$$file" && \
		ruff --fix "$$file"; \
	done

# Docker Operations
build-docker:
	@echo "Building all Docker images..."
	@for service in services/*/; do \
		if [ -f "$$service/Dockerfile" ]; then \
			service_name=$$(basename $$service); \
			echo "Building $$service_name..."; \
			docker build -t heimdall:$$service_name "$$service"; \
		fi; \
	done

# Database Operations
db-migrate:
	@if [ -d "db/migrations" ]; then \
		echo "Running database migrations..."; \
		docker compose exec postgres psql -U heimdall_user -d heimdall -c "SELECT 1;" && \
		alembic upgrade head; \
	else \
		echo "No migrations directory found. Run infrastructure setup first."; \
	fi

# Cleanup
clean:
	@echo "Cleaning up..."
	docker compose down -v --remove-orphans
	docker system prune -f
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup completed."

# Health Checks
health-check:
	@echo "Checking service health..."
	python scripts/health-check.py

health-check-postgres:
	docker compose exec postgres pg_isready -U heimdall_user

health-check-rabbitmq:
	docker compose exec rabbitmq rabbitmq-diagnostics -q ping

health-check-redis:
	docker compose exec redis redis-cli -a changeme ping

health-check-minio:
	@curl -f http://localhost:9000/minio/health/live || echo "MinIO health check failed"

# Documentation Audit (cross-platform)
audit-docs:
	@echo "Running documentation audit..."
	python scripts/audit_documentation.py --format=both

validate-doc-links:
	@echo "Validating documentation links..."
	python scripts/generate_doc_index.py --check-anchors

check-docs: audit-docs validate-doc-links
	@echo ""
	@echo "========================================="
	@echo "  Documentation audit complete"
	@echo "========================================="

# Quick setup for new developers
setup:
	@echo "Setting up Heimdall development environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file from .env.example"; \
	else \
		echo "✓ .env file already exists"; \
	fi
	@echo ""
	@echo "Installing Python dependencies..."
	@./scripts/setup-dev-environment.sh
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "  1. Activate virtual environment: source venv/bin/activate"
	@echo "  2. Start services: make dev-up"
	@echo "  3. Run tests: make test-local"

# Dependency Management
lock-deps:
	@echo "Generating lock files from requirements..."
	python scripts/lock_requirements.py --verbose

audit-deps:
	@echo "Auditing dependencies across all services..."
	python scripts/audit_dependencies.py --format=all

deps-check: lock-deps audit-deps
	@echo "✅ Dependency audit complete"
	@echo "Check audit-results/ for detailed reports"
