# Heimdall SDR - Development Makefile
# Windows PowerShell compatible commands

.PHONY: help dev-up dev-down test lint format build-docker db-migrate clean

# Default target
help:
	@echo "Heimdall SDR Development Commands:"
	@echo ""
	@echo "  INFRASTRUCTURE (Phase 1):"
	@echo "    dev-up                Start development environment"
	@echo "    dev-down              Stop development environment"
	@echo "    infra-status          Show docker-compose status"
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
	@echo "    test                  Run all tests"
	@echo "    lint                  Run code linting"
	@echo "    format                Auto-format code"
	@echo "    db-migrate            Run database migrations"
	@echo ""
	@echo "  MAINTENANCE:"
	@echo "    clean                 Clean all generated files"
	@echo "    setup                 Initial setup"
	@echo ""

# Development Environment
dev-up:
	docker-compose up -d
	@echo "Development environment started. Waiting for services to be healthy..."
	@timeout /t 10 /nobreak
	docker-compose ps

dev-down:
	docker-compose down
	@echo "Development environment stopped."

dev-logs:
	docker-compose logs -f

# Infrastructure Operations (Phase 1)
infra-status:
	docker-compose ps

infra-logs:
	docker-compose logs -f

postgres-connect:
	docker-compose exec postgres psql -U heimdall_user -d heimdall

postgres-cli:
	docker-compose exec postgres bash

redis-cli:
	docker-compose exec redis redis-cli -a changeme

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
	@for /d %%i in (services\*) do ( \
		if exist "%%i\tests" ( \
			echo "Testing %%i..." && \
			docker-compose exec %%~ni pytest tests/ -v --cov=src/ --cov-report=term-missing \
		) \
	)

# Code Quality
lint:
	@echo "Running linting on all Python files..."
	@for /r . %%i in (*.py) do ( \
		echo "Linting %%i..." && \
		black --check %%i && \
		ruff check %%i \
	)

format:
	@echo "Formatting all Python files..."
	@for /r . %%i in (*.py) do ( \
		echo "Formatting %%i..." && \
		black %%i && \
		ruff --fix %%i \
	)

# Docker Operations
build-docker:
	@echo "Building all Docker images..."
	@for /d %%i in (services\*) do ( \
		if exist "%%i\Dockerfile" ( \
			echo "Building %%i..." && \
			docker build -t heimdall:%%~ni %%i \
		) \
	)

# Database Operations
db-migrate:
	@if exist "db\migrations" ( \
		echo "Running database migrations..." && \
		docker-compose exec postgres psql -U heimdall_user -d heimdall -c "SELECT 1;" && \
		alembic upgrade head \
	) else ( \
		echo "No migrations directory found. Run infrastructure setup first." \
	)

# Cleanup
clean:
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@for /r . %%i in (__pycache__) do @if exist "%%i" rmdir /s /q "%%i"
	@for /r . %%i in (*.pyc) do @if exist "%%i" del "%%i"
	@for /r . %%i in (.pytest_cache) do @if exist "%%i" rmdir /s /q "%%i"
	@echo "Cleanup completed."

# Health Checks
health-check:
	@echo "Checking service health..."
	python scripts/health-check.py

health-check-postgres:
	docker-compose exec postgres pg_isready -U heimdall_user

health-check-rabbitmq:
	docker-compose exec rabbitmq rabbitmq-diagnostics -q ping

health-check-redis:
	docker-compose exec redis redis-cli -a changeme ping

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
	@if not exist ".env" copy .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make dev-up"
