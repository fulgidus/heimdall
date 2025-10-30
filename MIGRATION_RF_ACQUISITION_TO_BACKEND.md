# Migration Guide: rf-acquisition → backend

## Overview

The `rf-acquisition` microservice has been refactored into a generalist `backend` service to better reflect its broader responsibilities beyond just RF data acquisition. This service now handles:

- RF data acquisition from WebSDR receivers
- CRUD operations for sessions, WebSDRs, and known sources
- State management and persistence
- General low-load backend operations

## What Changed

### Service Name
- **Old**: `rf-acquisition`
- **New**: `backend`

### Container Name
- **Old**: `heimdall-rf-acquisition`
- **New**: `heimdall-backend`

### Directory Structure
- **Old**: `services/rf-acquisition/`
- **New**: `services/backend/`

### Environment Variables

#### New Primary Variables
```bash
BACKEND_HOST=backend
BACKEND_PORT=8001
BACKEND_URL=http://backend:8001
```

#### Legacy Variables (Maintained for Compatibility)
```bash
RF_ACQUISITION_HOST=backend
RF_ACQUISITION_PORT=8001
RF_ACQUISITION_URL=http://backend:8001
```

### API Endpoints

#### New Primary Endpoints
- `/api/v1/backend/*` - All backend operations
- `/api/v1/backend/health` - Health check

#### Legacy Endpoints (Maintained for Compatibility)
- `/api/v1/rf-acquisition/*` - Proxies to backend
- `/api/v1/acquisition/*` - Proxies to backend

## Migration Steps

### For Local Development

1. **Update your environment file** (`.env`):
   ```bash
   # Add new variables (optional, defaults work)
   BACKEND_HOST=backend
   BACKEND_PORT=8001
   BACKEND_URL=http://backend:8001
   ```

2. **Rebuild the backend service**:
   ```bash
   docker compose build backend
   ```

3. **Restart services**:
   ```bash
   docker compose down
   docker compose up -d
   ```

4. **Verify the service is running**:
   ```bash
   curl http://localhost:8001/health
   # Should return: {"status":"healthy","service":"backend",...}
   ```

### For Production Deployment

1. **Update environment variables**:
   - Set `BACKEND_HOST=backend`
   - Set `BACKEND_PORT=8001`
   - Set `BACKEND_URL=http://backend:8001`
   - Legacy `RF_ACQUISITION_*` variables still work

2. **Update Kubernetes/Helm configurations** (if applicable):
   - Rename service from `rf-acquisition` to `backend`
   - Update service selectors and labels
   - Update ingress rules if needed

3. **Deploy the updated configuration**:
   ```bash
   # Kubernetes example
   kubectl apply -f k8s/backend-service.yaml
   
   # Docker Compose example
   docker compose pull
   docker compose up -d backend
   ```

4. **Verify health**:
   ```bash
   curl http://your-domain/api/v1/backend/health
   ```

### For API Clients

**No changes required!** Existing endpoints continue to work:
- `/api/v1/rf-acquisition/*` → proxies to backend
- `/api/v1/acquisition/*` → proxies to backend

**Optional: Update to new endpoints** for better clarity:
- Use `/api/v1/backend/*` for all backend operations

## Breaking Changes

**None.** This refactoring maintains complete backward compatibility:
- ✅ Legacy API endpoints still work
- ✅ Environment variables support both old and new names
- ✅ Service responds to both service names where appropriate

## Rollback Plan

If you need to rollback:

1. **Revert to previous commit**:
   ```bash
   git checkout <previous-commit-hash>
   ```

2. **Rebuild services**:
   ```bash
   docker compose build rf-acquisition
   docker compose up -d
   ```

## Verification

After migration, verify:

1. **Service is running**:
   ```bash
   docker compose ps backend
   # Should show: heimdall-backend | running
   ```

2. **Health check responds**:
   ```bash
   curl http://localhost:8001/health
   # Should return healthy status
   ```

3. **API Gateway routes work**:
   ```bash
   # New endpoint
   curl http://localhost:8000/api/v1/backend/health
   
   # Legacy endpoint (should still work)
   curl http://localhost:8000/api/v1/rf-acquisition/health
   ```

4. **Frontend loads**:
   - Open http://localhost:3000
   - Check that Dashboard and System Status pages load
   - Verify WebSDR list appears

## Support

For issues or questions:
1. Check GitHub Issues: https://github.com/fulgidus/heimdall/issues
2. Review the PR: [Link to PR]
3. Contact: alessio.corsi@gmail.com

## Timeline

- **Development**: 2025-10-30
- **Testing**: Ongoing
- **Production Rollout**: TBD

---

**Last Updated**: 2025-10-30
**Version**: 1.0
