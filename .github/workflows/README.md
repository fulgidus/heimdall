# GitHub Workflows

This directory contains all CI/CD workflows for the Heimdall project.

## Active Workflows

### 🔬 `ci.yml` - Main CI Pipeline
**Triggers**: Push to main/develop/fix/*, Pull Requests
**Purpose**: Comprehensive CI pipeline with all code quality and testing

**Jobs**:
1. **backend-quality** - Python code quality (Black, Ruff, mypy)
2. **frontend-quality** - TypeScript code quality (ESLint, TypeScript, Prettier)
3. **backend-tests** - Unit tests for all backend services (matrix strategy)
4. **frontend-tests** - Vitest unit tests with coverage
5. **integration-tests** - Integration tests with Docker services (PostgreSQL, Redis, RabbitMQ, MinIO)
6. **e2e-tests** - Playwright E2E tests with real backend
7. **ci-summary** - Aggregate results and post PR comment

**Features**:
- Parallel execution of independent jobs
- Matrix strategy for backend services (tests 5 services in parallel)
- Proper job dependencies (quality → tests → integration → e2e)
- Caching for pip and npm dependencies
- Codecov integration for coverage reports
- PR comments with detailed results

### 📊 `coverage.yml` - Coverage Report Generation
**Triggers**: Push to develop, Pull Requests
**Purpose**: Generate and publish coverage reports with badges

**Features**:
- Backend coverage (pytest + coverage.py)
- Frontend coverage (Vitest + v8)
- Coverage badges generated
- Reports uploaded to docs/coverage/
- PR comments with coverage summary

### 🔒 `security-scan.yml` - Security Scanning
**Triggers**: PR, Push, Scheduled (daily at 2 AM UTC), Manual
**Purpose**: Security vulnerability scanning

**Tools**:
- **Bandit** - Python code security issues
- **Safety** - Python dependency vulnerabilities
- **Trivy** - Container security scanning
- **Dependabot** - GitHub dependency alerts

### 📦 `dependency-updates.yml` - Automated Dependency Updates
**Triggers**: Scheduled (Monday 2 AM UTC), Manual
**Purpose**: Automated dependency updates with testing

**Process**:
1. Run `scripts/lock_requirements.py` to update lock files
2. Run test suite to verify no breakage
3. Create PR if tests pass
4. Create issue if tests fail

### 📋 `doc-audit.yml` - Documentation Validation
**Triggers**: PR/Push affecting docs/, Manual
**Purpose**: Ensure documentation integrity

**Checks**:
- Orphaned files (not linked from index)
- Broken links (invalid file references)
- Documentation structure validation

## Disabled Workflows

The following workflows have been consolidated into `ci.yml` and disabled:

- `ci-test.yml.disabled` - Replaced by new ci.yml
- `python-quality.yml.disabled` - Now backend-quality job
- `typescript-quality.yml.disabled` - Now frontend-quality job
- `integration-tests.yml.disabled` - Now integration-tests job
- `e2e-tests.yml.disabled` - Now e2e-tests job
- `test-coverage.yml.disabled` - Coverage in unit test jobs
- `type-safety.yml.disabled` - Now part of backend-quality

These are kept for reference but won't trigger. Delete them when confident the new CI works.

## Workflow Architecture

```
Push/PR Trigger
├── ci.yml (Main CI)
│   ├── backend-quality (parallel)
│   ├── frontend-quality (parallel)
│   ├── backend-tests (depends on backend-quality, matrix: 5 services)
│   ├── frontend-tests (depends on frontend-quality)
│   ├── integration-tests (depends on backend-tests)
│   ├── e2e-tests (depends on frontend-tests + integration-tests)
│   └── ci-summary (depends on all, posts PR comment)
├── coverage.yml (if develop branch)
├── security-scan.yml (if triggered)
└── doc-audit.yml (if docs changed)
```

## Running Tests Locally

### Backend Tests
```bash
# Install dependencies
cd services/<service-name>
pip install -r requirements.txt
pip install -r requirements-test.txt  # if exists

# Run unit tests
pytest tests/ -k "not e2e and not integration"

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Frontend Tests
```bash
cd frontend

# Install dependencies
npm ci

# Run unit tests
npm run test:run

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Integration Tests
```bash
# Start infrastructure
docker compose up -d postgres redis rabbitmq

# Run integration tests
cd services/<service-name>
pytest tests/integration/
```

## CI Configuration

### Environment Variables
- `PYTHON_VERSION`: "3.11"
- `NODE_VERSION`: "20"

### Service Credentials (for tests)
- PostgreSQL: `heimdall_user` / `test_password` / `heimdall_test`
- Redis: localhost:6379
- RabbitMQ: guest/guest
- MinIO: minioadmin/minioadmin

### Timeouts
- backend-quality: 20 minutes
- frontend-quality: 15 minutes
- backend-tests: 30 minutes
- frontend-tests: 15 minutes
- integration-tests: 30 minutes
- e2e-tests: 30 minutes

## Troubleshooting

### CI Fails on Backend Quality
- Check Black formatting: `black --check services/ scripts/`
- Check Ruff linting: `ruff check services/ scripts/`
- Check mypy: `mypy services/ scripts/ --config-file=pyproject.toml`

### CI Fails on Frontend Quality
- Check ESLint: `npm run lint`
- Check TypeScript: `npm run type-check`
- Check Prettier: `npm run format:check`

### CI Fails on Tests
- Review test output in GitHub Actions logs
- Download artifacts for detailed reports
- Run tests locally to reproduce

### CI Fails on E2E
- Check backend service logs (uploaded as artifacts on failure)
- Download Playwright report
- Run `npm run test:e2e:headed` locally to debug

## Best Practices

1. **Always run tests locally before pushing**
2. **Keep CI fast** - avoid unnecessary work
3. **Use caching** - already configured for pip and npm
4. **Fix quality issues first** - tests won't run if quality fails
5. **Monitor CI times** - optimize if jobs take too long
6. **Review artifacts** - coverage and reports are valuable

## Maintenance

### Adding a New Service
1. Add service to backend-tests matrix in `ci.yml`
2. Ensure service has tests/ directory
3. Add service to integration-tests if needed

### Modifying Quality Checks
- Edit `backend-quality` or `frontend-quality` jobs in `ci.yml`
- Update thresholds in `pyproject.toml` or `tsconfig.json`

### Changing CI Triggers
- Edit `on:` section in respective workflow files
- Test with `workflow_dispatch` before committing
