# Docker Build Context Fix - Technical Summary

**Date**: 2025-10-25  
**Issue**: CI E2E workflow failing with Docker build errors  
**Commit**: Reference dab7faa3e347d6e230312c1cc6fc03310a356bc5

## Problem Description

The E2E workflow was failing during Docker image builds with BuildKit checksum errors:
- `/requirements.txt`: not found
- `/src`: not found  
- `/entrypoint.py`: not found

## Root Cause Analysis

### The Issue
The build context in `docker-compose.services.yml` was incorrectly configured:

**Before (BROKEN)**:
```yaml
services:
  rf-acquisition:
    build:
      context: ./services              # ❌ WRONG
      dockerfile: rf-acquisition/Dockerfile
```

**Dockerfile expectations**:
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .              # ❌ Looks for ./services/requirements.txt
COPY src/ ./src/                     # ❌ Looks for ./services/src/
COPY entrypoint.py ./entrypoint.py   # ❌ Looks for ./services/entrypoint.py
```

When the build context is `./services`, Docker searches for files relative to that directory:
- `COPY requirements.txt .` → looks for `services/requirements.txt` (doesn't exist)
- `COPY src/ ./src/` → looks for `services/src/` (doesn't exist)
- Actual files are in `services/rf-acquisition/requirements.txt` and `services/rf-acquisition/src/`

### Why It Happens
Docker COPY commands are relative to the **build context**, not the Dockerfile location. The context defines the "root" directory that Docker can see during the build.

## Solution

### Fixed Build Contexts

**After (CORRECT)**:
```yaml
services:
  rf-acquisition:
    build:
      context: ./services/rf-acquisition  # ✅ CORRECT
      dockerfile: Dockerfile
  
  data-ingestion-web:
    build:
      context: ./services/data-ingestion-web  # ✅ CORRECT
      dockerfile: Dockerfile
  
  inference:
    build:
      context: ./services/inference  # ✅ CORRECT
      dockerfile: Dockerfile
```

Now Docker can find the files:
- `COPY requirements.txt .` → finds `services/rf-acquisition/requirements.txt` ✅
- `COPY src/ ./src/` → finds `services/rf-acquisition/src/` ✅
- `COPY entrypoint.py .` → finds `services/rf-acquisition/entrypoint.py` ✅

### Special Case: API Gateway

The `api-gateway` service needs access to the `common/auth` module, which is outside its service directory:

```
services/
├── api-gateway/
│   └── src/
└── common/
    └── auth/
```

**Solution**: Keep the build context as `./services` so it can access both directories:

```yaml
services:
  api-gateway:
    build:
      context: ./services              # ✅ Needs access to common/
      dockerfile: api-gateway/Dockerfile
```

Dockerfile uses prefixed paths:
```dockerfile
COPY api-gateway/requirements.txt .
COPY api-gateway/src/ ./src/
COPY common/auth ./auth/              # Access to common module
```

## Changes Made

### 1. docker-compose.services.yml
- Updated build contexts for 3 services to use per-service directories
- Kept api-gateway with `./services` context for common/auth access

### 2. CI Workflow (.github/workflows/e2e-tests.yml)
- Added debug step to list repository structure
- Added validation step using new script
- Simplified inline validation

### 3. Validation Script (scripts/validate-service-files.sh)
- Checks all required files exist before build
- Validates: src/, requirements.txt, Dockerfile, entrypoint.py
- Color-coded output for easy troubleshooting

### 4. Test Script (scripts/test-docker-build-contexts.sh)
- Verifies all build contexts are correctly configured
- Can be run locally to validate the fix

## Testing

### Local Testing
```bash
# Run validation
./scripts/validate-service-files.sh

# Test build contexts
./scripts/test-docker-build-contexts.sh

# Build services locally
docker compose -f docker-compose.services.yml build api-gateway rf-acquisition data-ingestion-web inference
```

### CI Testing
The E2E workflow now includes:
1. Repository structure debug output
2. Service file validation
3. Docker build with correct contexts
4. Service health checks

## Edge Cases Handled

1. **api-gateway common/auth access**: Kept `./services` context
2. **rf-acquisition entrypoint.py**: Special validation check added
3. **.dockerignore**: No interference (no .dockerignore in service directories)
4. **Git checkout**: Full tree checkout with `fetch-depth: 0`
5. **Build cache**: Validation runs before build to fail fast

## Expected Behavior After Fix

### Before (FAILED)
```
Step 4/17 : COPY requirements.txt .
ERROR: "/requirements.txt": not found
```

### After (SUCCESS)
```
Step 4/17 : COPY requirements.txt .
 ---> Using cache
Step 5/17 : RUN pip install -r requirements.txt
 ---> Running in abc123...
```

## Prevention

To prevent similar issues in the future:

1. **Always validate build contexts** when adding new services
2. **Run validation script** before committing Dockerfile changes
3. **Test builds locally** with `docker compose build`
4. **Document dependencies** if a service needs files outside its directory

## References

- **Docker documentation**: [Build context](https://docs.docker.com/build/building/context/)
- **Problem commit**: dab7faa3e347d6e230312c1cc6fc03310a356bc5
- **Fix commit**: [Current commit]
- **Validation script**: `scripts/validate-service-files.sh`
- **Test script**: `scripts/test-docker-build-contexts.sh`
