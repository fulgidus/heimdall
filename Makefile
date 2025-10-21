# Heimdall SDR - Development Makefile
# Windows PowerShell compatible commands

.PHONY: help dev-up dev-down test lint format build-docker db-migrate clean

# Default target
help:
	@echo "Heimdall SDR Development Commands:"
	@echo ""
	@echo "  dev-up          Start development environment (docker-compose)"
	@echo "  dev-down        Stop development environment"
	@echo "  test            Run all tests with coverage"
	@echo "  lint            Run code linting (Black + Ruff)"
	@echo "  format          Auto-format all Python code"
	@echo "  build-docker    Build all Docker images"
	@echo "  db-migrate      Run database migrations"
	@echo "  clean           Clean up generated files and containers"
	@echo ""

# Development Environment
dev-up:
	docker-compose up -d
	@echo "Development environment started. Check status with: docker-compose ps"

dev-down:
	docker-compose down
	@echo "Development environment stopped."

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
	@python scripts/health-check-all.py

# Quick setup for new developers
setup:
	@echo "Setting up Heimdall development environment..."
	@if not exist ".env" copy .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make dev-up"
