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

# Quick setup for new developers
setup:
	@echo "Setting up Heimdall development environment..."
	@if not exist ".env" copy .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make dev-up"

# ============================================================================
# CODE QUALITY TARGETS (Cross-platform)
# ============================================================================

# Python Quality Checks
.PHONY: lint-python format-python test-python type-check-python

lint-python:
	@echo "Running Python linting..."
	black services/ scripts/ --check
	ruff check services/ scripts/

format-python:
	@echo "Formatting Python code..."
	black services/ scripts/
	ruff check services/ scripts/ --fix

type-check-python:
	@echo "Running mypy type checking..."
	mypy services/ scripts/ --config-file=pyproject.toml

test-python:
	@echo "Running Python tests with coverage..."
	pytest services/ --cov=services --cov-report=term --cov-report=xml --cov-report=html

# TypeScript/Frontend Quality Checks
.PHONY: lint-typescript format-typescript type-check-typescript test-typescript

lint-typescript:
	@echo "Running TypeScript linting..."
	cd frontend && npm run lint

format-typescript:
	@echo "Formatting TypeScript code..."
	cd frontend && npm run format

type-check-typescript:
	@echo "Running TypeScript type checking..."
	cd frontend && npm run type-check

test-typescript:
	@echo "Running TypeScript tests with coverage..."
	cd frontend && npm run test:coverage

# Security Scans
.PHONY: security-scan security-python security-deps

security-scan: security-python security-deps
	@echo "✅ Security scan complete"

security-python:
	@echo "Running bandit security scan..."
	pip install bandit
	bandit -r services/ scripts/ -f json -o bandit-report.json || true
	bandit -r services/ scripts/

security-deps:
	@echo "Running safety dependency scan..."
	pip install safety
	find services -name "requirements*.txt" -exec cat {} \; > all-requirements.txt
	safety check --file=all-requirements.txt --json > safety-report.json || true
	safety check --file=all-requirements.txt

# Combined Quality Checks
.PHONY: quality-check quality-check-python quality-check-typescript

quality-check-python: lint-python type-check-python test-python
	@echo "✅ All Python quality checks passed"

quality-check-typescript: lint-typescript type-check-typescript test-typescript
	@echo "✅ All TypeScript quality checks passed"

quality-check: quality-check-python quality-check-typescript
	@echo "✅ All quality checks passed"

# Local CI Simulation
.PHONY: ci-local

ci-local: quality-check security-scan
	@echo "✅ Local CI checks passed - ready for push"

# Pre-commit Setup
.PHONY: install-pre-commit

install-pre-commit:
	@echo "Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo "✅ Pre-commit hooks installed. Run 'pre-commit run --all-files' to test."

