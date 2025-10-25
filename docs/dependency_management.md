# Dependency Management Guide

This document explains the dependency management strategy for the Heimdall SDR project, including how dependencies are organized, pinned, and updated.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Why Pin Dependencies?](#why-pin-dependencies)
- [Centralized Requirements](#centralized-requirements)
- [Adding New Dependencies](#adding-new-dependencies)
- [Updating Dependencies](#updating-dependencies)
- [Version Strategy](#version-strategy)
- [Conflict Resolution](#conflict-resolution)
- [Security Auditing](#security-auditing)
- [Testing Requirements](#testing-requirements)
- [Tools and Scripts](#tools-and-scripts)

## Overview

Heimdall uses a **centralized dependency management system** with version pinning to ensure:

- **Reproducible builds** across all environments
- **Consistency** across microservices
- **Security** through automated vulnerability scanning
- **Stability** by avoiding unexpected breaking changes

## Directory Structure

```
services/
├── requirements/              # Centralized requirement definitions
│   ├── base.txt              # Core dependencies (all services)
│   ├── base.lock             # Locked versions for base
│   ├── api.txt               # FastAPI framework dependencies
│   ├── api.lock              # Locked versions for API
│   ├── data.txt              # Data processing & storage
│   ├── data.lock             # Locked versions for data
│   ├── ml.txt                # Machine learning stack
│   ├── ml.lock               # Locked versions for ML
│   ├── dev.txt               # Development and testing tools
│   ├── dev.lock              # Locked versions for dev
│   ├── audit_report.json     # Automated audit results
│   └── audit_report.md       # Human-readable audit
│
├── rf-acquisition/
│   └── requirements.txt      # Service-specific requirements
├── training/
│   └── requirements.txt      # Service-specific requirements
├── inference/
│   └── requirements.txt      # Service-specific requirements
├── data-ingestion-web/
│   └── requirements.txt      # Service-specific requirements
└── api-gateway/
    └── requirements.txt      # Service-specific requirements
```

## Why Pin Dependencies?

Version pinning provides several critical benefits:

### 1. **Reproducibility**
- Same versions across development, testing, and production
- Builds are identical regardless of when they occur
- Debugging is easier with consistent environments

### 2. **Security**
- Controlled updates allow security review
- Automated scanning detects vulnerabilities
- Updates are tested before deployment

### 3. **Stability**
- Prevents unexpected breaking changes
- No surprise updates during deployment
- Easier to track down regression causes

### 4. **Compliance**
- Audit trail of all dependency versions
- Clear documentation of what's deployed
- License compliance tracking

## Centralized Requirements

Dependencies are organized into **modular requirement files**:

### `base.txt` - Core Dependencies
Shared by all services:
- Configuration (pydantic, pydantic-settings, python-dotenv)
- Logging (structlog)
- HTTP clients (httpx, aiohttp)
- Database (sqlalchemy, psycopg2-binary, alembic)

### `api.txt` - API Framework
For services with REST APIs:
- FastAPI
- Uvicorn
- Python-multipart

### `data.txt` - Data Processing
For services handling data storage:
- SQLAlchemy (explicit reference)
- Redis client
- Celery

### `ml.txt` - Machine Learning
For training and inference services:
- PyTorch and PyTorch Lightning
- Scientific computing (numpy, scipy, scikit-learn)
- MLflow for experiment tracking

### `dev.txt` - Development Tools
Testing and code quality:
- pytest and plugins
- black, ruff, mypy
- ipython

## Adding New Dependencies

Follow this process to add a new dependency:

### Step 1: Choose the Appropriate File

Determine which centralized requirement file should include the dependency:

- **All services need it?** → `base.txt`
- **Only API services?** → `api.txt`
- **Only one service?** → That service's `requirements.txt`
- **ML-related?** → `ml.txt`
- **Dev/testing only?** → `dev.txt`

### Step 2: Add to Requirements File

Edit the appropriate `.txt` file:

```bash
# Example: Adding requests to base.txt
echo "requests==2.31.0" >> services/requirements/base.txt
```

**Guidelines:**
- Use exact version pinning (`==`) for production dependencies
- Use minimum version (`>=`) only when necessary
- Include a comment explaining why the dependency is needed
- Check for conflicts with existing packages

### Step 3: Generate Lock Files

Run the lock script to generate `.lock` files:

```bash
python scripts/lock_requirements.py --verbose
```

This will:
- Resolve all dependencies and sub-dependencies
- Pin exact versions in `.lock` files
- Run security audit
- Generate reports

### Step 4: Review Changes

Check what changed:

```bash
git diff services/requirements/*.lock
```

Review:
- Are new sub-dependencies reasonable?
- Any security vulnerabilities reported?
- Any unexpected version changes?

### Step 5: Test Locally

Test with the new dependency:

```bash
# Install from lock file
pip install -r services/requirements/base.lock

# Run tests
make test

# Test affected services
docker-compose up -d <affected-service>
```

### Step 6: Commit Both Files

Commit both `.txt` and `.lock` files:

```bash
git add services/requirements/base.txt services/requirements/base.lock
git commit -m "feat: add requests library for HTTP calls"
```

## Updating Dependencies

There are three ways to update dependencies:

### 1. Automated Weekly Updates (Recommended)

The automated workflow runs every Monday at 02:00 UTC:

- Generates new lock files with latest versions
- Runs full test suite
- Creates PR if tests pass
- Creates issue if tests fail

**No action needed** unless tests fail.

### 2. Manual Updates

For urgent security updates or specific version bumps:

```bash
# Update version in .txt file
vim services/requirements/base.txt

# Regenerate lock files
python scripts/lock_requirements.py --verbose

# Test changes
make test

# Commit
git add services/requirements/*.txt services/requirements/*.lock
git commit -m "chore: update package-name to version X.Y.Z"
```

### 3. Security Updates (Urgent)

For critical vulnerabilities (CVEs):

```bash
# Check current vulnerabilities
python scripts/audit_dependencies.py --format=all

# Update affected package
vim services/requirements/<file>.txt

# Regenerate locks
python scripts/lock_requirements.py --verbose

# Verify fix
python scripts/audit_dependencies.py --format=all

# Fast-track PR
git checkout -b security/fix-cve-YYYY-XXXXX
git add .
git commit -m "security: fix CVE-YYYY-XXXXX in package-name"
git push
# Create PR with "security" label
```

## Version Strategy

Our version update policy:

### Patch Updates (X.Y.Z → X.Y.Z+1)
- **Automated**: Weekly workflow
- **Testing**: Full suite must pass
- **Review**: Optional
- **Risk**: Low

### Minor Updates (X.Y.Z → X.Y+1.0)
- **Automated**: Weekly workflow
- **Testing**: Full suite + manual verification
- **Review**: Required
- **Risk**: Medium

### Major Updates (X.Y.Z → X+1.0.0)
- **Manual only**: Requires planning
- **Testing**: Extensive testing required
- **Review**: Required + design review
- **Risk**: High

### Development Dependencies
- More flexible versioning allowed
- Use `>=` for dev tools
- Test before committing

## Conflict Resolution

When dependencies conflict:

### 1. Identify the Conflict

Run the audit tool:

```bash
python scripts/audit_dependencies.py --format=all
```

Review `audit-results/audit.md` for conflicts.

### 2. Analyze Impact

Determine:
- Which services are affected?
- Are both versions required?
- Can we standardize on one version?

### 3. Resolve

**Option A: Standardize**
```bash
# Update all services to use the same version
# Edit services/*/requirements.txt
# Regenerate locks
python scripts/lock_requirements.py --verbose
```

**Option B: Isolate**
If services truly need different versions:
- Document why in code comments
- Use version constraints carefully
- Monitor for issues

**Option C: Upgrade**
```bash
# Upgrade to latest compatible version
# Test thoroughly
# Update all dependent services
```

### 4. Verify Resolution

```bash
python scripts/audit_dependencies.py --format=all
# Should show no conflicts
```

## Security Auditing

### Automated Scans

Security scans run:
- **Weekly**: Via automated workflow
- **On every lock file generation**
- **On CI/CD pipeline**

### Manual Audit

Run manual audit anytime:

```bash
python scripts/audit_dependencies.py --format=all
```

Reports generated in `audit-results/`:
- `audit.json` - Machine-readable
- `audit.md` - Human-readable summary
- `versions.csv` - Version matrix across services

### Vulnerability Response

When vulnerabilities are found:

1. **Assess Severity**
   - Critical/High: Fix immediately
   - Medium: Fix within 7 days
   - Low: Fix in next regular update

2. **Update Package**
   ```bash
   # Update to secure version
   vim services/requirements/<file>.txt
   python scripts/lock_requirements.py --verbose
   ```

3. **Test Thoroughly**
   ```bash
   make test
   docker-compose up -d
   # Run integration tests
   ```

4. **Document**
   - Note CVE in commit message
   - Update CHANGELOG.md
   - Tag PR with "security"

## Testing Requirements

All dependency updates must pass:

### 1. Unit Tests
```bash
pytest services/*/tests/unit/ -v
```

### 2. Integration Tests
```bash
pytest services/*/tests/integration/ -v
```

### 3. Type Checking
```bash
mypy services/*/src/
```

### 4. Linting
```bash
black --check services/
ruff check services/
```

### 5. Build Tests
```bash
docker-compose build
```

**Policy**: All tests must pass before merging dependency updates.

## Tools and Scripts

### `scripts/lock_requirements.py`

Generates `.lock` files from `.txt` files.

**Usage:**
```bash
python scripts/lock_requirements.py [--verbose] [--allow-unsafe]
```

**Options:**
- `--verbose` - Show detailed output
- `--allow-unsafe` - Include unsafe packages (setuptools, pip)

**Output:**
- `.lock` files in `services/requirements/`
- `audit_report.json` - Detailed audit results
- `audit_report.md` - Summary report

### `scripts/audit_dependencies.py`

Analyzes dependencies across all services.

**Usage:**
```bash
python scripts/audit_dependencies.py [--format=all] [--output=audit-results/]
```

**Options:**
- `--format` - Output format: all, json, markdown, csv
- `--output` - Output directory (default: audit-results)

**Output:**
- `audit.json` - Full audit data
- `audit.md` - Summary with recommendations
- `versions.csv` - Version matrix

### Makefile Targets

Convenient make targets:

```bash
make lock-deps          # Generate lock files
make audit-deps         # Run dependency audit
make deps-check         # Both lock and audit
```

## Best Practices

1. **Always use lock files in production**
   - Docker builds use `.lock` files
   - Never use `.txt` files directly in containers

2. **Commit both .txt and .lock files**
   - `.txt` is the source of truth
   - `.lock` is the compiled result

3. **Review lock file changes**
   - Check for unexpected sub-dependencies
   - Verify version changes are reasonable

4. **Run audits before major releases**
   - Ensure no known vulnerabilities
   - Check for outdated packages

5. **Keep dependencies minimal**
   - Only add when truly needed
   - Regularly review and remove unused

6. **Document special cases**
   - If pinning to old version, explain why
   - If using pre-release, document reason

7. **Test dependency updates thoroughly**
   - Run full test suite
   - Check for deprecation warnings
   - Test in staging environment

## Troubleshooting

### Lock File Generation Fails

**Problem:** `pip-compile` fails to resolve dependencies

**Solution:**
1. Check for conflicting version constraints
2. Try `--allow-unsafe` flag
3. Update `pip-tools`: `pip install --upgrade pip-tools`
4. Check PyPI availability

### Tests Fail After Update

**Problem:** Tests pass locally but fail in CI

**Solution:**
1. Ensure lock files are committed
2. Clear pip cache in CI
3. Check for environment-specific issues
4. Review deprecation warnings

### Version Conflicts

**Problem:** Multiple services need different versions

**Solution:**
1. Try to standardize if possible
2. Check if newer version works for all
3. Document if isolation needed
4. Monitor for issues

### Security Scan False Positives

**Problem:** Safety reports issues in dev dependencies

**Solution:**
1. Check if vulnerability affects production use
2. Update if possible
3. Document why keeping version if needed
4. Use `# nosafety` comment if false positive

## FAQ

**Q: Should I pin dependencies in development?**  
A: Yes, for consistency, but you can use `>=` for dev tools if needed.

**Q: How often are dependencies updated?**  
A: Automatically weekly, manually as needed for security.

**Q: What if a package doesn't have a new version?**  
A: That's fine, lock files will use existing version.

**Q: Can I use `~=` or `>=` instead of `==`?**  
A: For production deps, use `==`. For dev tools, `>=` is acceptable.

**Q: What about indirect dependencies?**  
A: Lock files handle those automatically with exact versions.

**Q: Do I need to update all services at once?**  
A: No, but it's recommended to keep them synchronized.

## Support

For issues or questions:
- Create an issue using the dependency-update template
- Check existing issues for similar problems
- Review audit reports for guidance
- Ask in project discussions

---

**Last Updated**: 2025-10-25  
**Version**: 1.0  
**Maintainer**: Heimdall Development Team
