# Development Guide

Comprehensive guide for developers contributing to Heimdall.

## Development Environment Setup

### Prerequisites

- **Docker** 20.10+ with Docker Compose
- **Python** 3.11+
- **Node.js** 18+ with pnpm
- **Git** 2.30+
- **Make** (build automation)
- **8GB RAM** minimum (16GB recommended for ML work)
- **20GB disk space**

### Initial Setup

```bash
# Clone repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Copy environment template
cp .env.example .env

# Start infrastructure
docker-compose up -d

# Verify all services are running
make health-check
```

### Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Frontend Environment

```bash
cd frontend

# Install dependencies (use pnpm, not npm!)
pnpm install

# Start development server
pnpm dev

# Runs on http://localhost:5173
```

## Project Structure

```
heimdall/
├── services/           # Backend microservices
│   ├── rf-acquisition/ # WebSDR data fetching
│   ├── training/       # ML model training
│   ├── inference/      # Real-time inference
│   ├── rf-acquisition/      # RF data acquisition + sessions management
│   └── api-gateway/    # API gateway
├── frontend/           # React + TypeScript UI
├── db/                 # Database scripts and migrations
├── docs/               # Documentation
├── scripts/            # Utility scripts
├── tests/              # Integration tests
└── docker-compose.yml  # Local development infrastructure
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/my-feature-name
```

### 2. Make Changes

Follow these guidelines:
- **Backend**: Python 3.11, FastAPI, type hints required
- **Frontend**: TypeScript strict mode, React functional components
- **Tests**: Maintain >80% coverage
- **Documentation**: Update relevant docs

### 3. Run Tests

```bash
# Backend tests
make test

# Backend tests with coverage
make test-coverage

# Frontend tests
cd frontend && pnpm test

# Frontend tests with coverage
cd frontend && pnpm test:coverage

# E2E tests (requires backend running)
cd frontend && pnpm test:e2e
```

### 4. Lint and Format

```bash
# Backend linting
make lint

# Backend formatting
make format

# Frontend linting
cd frontend && pnpm lint

# Frontend formatting
cd frontend && pnpm format
```

### 5. Commit Changes

Follow conventional commits:

```bash
git add .
git commit -m "feat(acquisition): add support for new WebSDR receiver"

# Commit types: feat, fix, docs, style, refactor, perf, test, chore
```

### 6. Push and Create PR

```bash
git push origin feature/my-feature-name

# Then create PR on GitHub targeting 'develop' branch
```

## Common Development Tasks

### Running Individual Services

```bash
# Start specific service
docker-compose up <service-name>

# View logs
docker-compose logs -f <service-name>

# Restart service
docker-compose restart <service-name>

# Rebuild and restart
docker-compose up -d --build <service-name>
```

### Database Operations

```bash
# Connect to PostgreSQL
psql -h localhost -U heimdall_user -d heimdall

# Run migrations
make db-migrate

# Create new migration
make db-migration-create NAME=add_new_table

# Rollback migration
make db-rollback
```

### Working with MinIO (Object Storage)

```bash
# Access MinIO console
open http://localhost:9001
# Login: minioadmin / minioadmin

# Upload test file
aws s3 cp test.npy s3://heimdall-raw-iq/ --endpoint-url http://localhost:9000

# List buckets
aws s3 ls --endpoint-url http://localhost:9000
```

### Message Queue (RabbitMQ)

```bash
# Access RabbitMQ management UI
open http://localhost:15672
# Login: guest / guest

# List queues
rabbitmqctl list_queues

# Purge queue
rabbitmqctl purge_queue acquisition.websdr-fetch
```

### Monitoring and Debugging

```bash
# View Grafana dashboards
open http://localhost:3000
# Login: admin / admin

# View Prometheus metrics
open http://localhost:9090

# Check service health
curl http://localhost:8001/health  # RF acquisition
curl http://localhost:8002/health  # Training
curl http://localhost:8003/health  # Inference
```

## Testing Best Practices

### Unit Tests

```python
# tests/test_signal_processor.py
import pytest
from services.rf_acquisition.src.utils.iq_processor import IQProcessor

@pytest.fixture
def processor():
    return IQProcessor()

def test_compute_snr(processor, sample_iq_data):
    """Test SNR calculation."""
    snr = processor.compute_snr(sample_iq_data)
    assert snr > 0
    assert snr < 100  # Reasonable range
```

### Integration Tests

```python
# tests/integration/test_rf_acquisition_flow.py
@pytest.mark.integration
async def test_acquisition_flow():
    """Test end-to-end RF acquisition."""
    # Trigger acquisition
    response = await client.post("/acquire", json={
        "frequency_mhz": 145.5,
        "duration_seconds": 5
    })
    assert response.status_code == 202
    
    # Wait for completion
    task_id = response.json()["task_id"]
    # ... poll for completion
```

### E2E Tests (Frontend)

```typescript
// frontend/src/tests/e2e/dashboard.spec.ts
import { test, expect } from '@playwright/test';

test('dashboard shows WebSDR status', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('Heimdall');
  await expect(page.locator('.websdr-status')).toBeVisible();
});
```

## Debugging

### Backend Debugging

```bash
# Run service with debugger
cd services/rf-acquisition
python -m debugpy --listen 5678 --wait-for-client src/main.py

# Attach from VS Code with launch.json configuration
```

### Frontend Debugging

```bash
# Run with source maps
cd frontend
pnpm dev

# Use browser DevTools
# React DevTools extension recommended
```

### Docker Debugging

```bash
# Enter running container
docker exec -it heimdall-rf-acquisition-1 bash

# View real-time logs
docker-compose logs -f --tail=100

# Check resource usage
docker stats
```

## Performance Profiling

### Backend Profiling

```python
# Use cProfile
python -m cProfile -o output.prof src/main.py

# Analyze with snakeviz
snakeviz output.prof
```

### Frontend Profiling

```typescript
// Use React Profiler
import { Profiler } from 'react';

<Profiler id="Dashboard" onRender={callback}>
  <Dashboard />
</Profiler>
```

## CI/CD Pipeline

### GitHub Actions Workflows

- **Test**: Runs on every PR
- **Build**: Builds Docker images on merge to develop
- **Deploy**: Deploys to staging/production

### Running CI Checks Locally

```bash
# Run all checks that CI runs
make ci-check

# Individual checks
make test        # Unit tests
make lint        # Linting
make type-check  # Type checking
make security    # Security scanning
```

## Code Quality Tools

### Backend

- **Black**: Code formatter
- **Ruff**: Fast Python linter
- **mypy**: Static type checker
- **pytest**: Testing framework
- **coverage**: Code coverage

### Frontend

- **Prettier**: Code formatter
- **ESLint**: JavaScript/TypeScript linter
- **TypeScript**: Static type checking
- **Vitest**: Testing framework
- **Playwright**: E2E testing

## Troubleshooting Common Issues

### Port Conflicts

```bash
# Find process using port
lsof -i :5432

# Kill process
kill -9 <PID>

# Or change port in .env
```

### Container Won't Start

```bash
# Check logs
docker-compose logs <service>

# Remove and recreate
docker-compose down -v
docker-compose up -d
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Test connection
psql -h localhost -U heimdall_user -d heimdall -c "SELECT 1;"
```

### Frontend Build Errors

```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules pnpm-lock.yaml
pnpm install

# Clear cache
pnpm store prune
```

## Getting Help

- **Documentation**: Check [docs/index.md](index.md)
- **FAQ**: See [FAQ](FAQ.md)
- **Issues**: Search [GitHub Issues](https://github.com/fulgidus/heimdall/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/fulgidus/heimdall/discussions)
- **Email**: alessio.corsi@gmail.com

## Additional Resources

- [Architecture Guide](ARCHITECTURE.md) - System design deep-dive
- [API Reference](API.md) - REST API documentation
- [Contributing Guidelines](../CONTRIBUTING.md) - How to contribute
