# Training Dashboard Crash Fix

**Date**: 2025-11-03  
**Issue**: Training Dashboard crashing on page load  
**Status**: ✅ FIXED

---

## Problem Analysis

### Root Cause

The Training Dashboard was crashing when users navigated to `/training` because the frontend was unable to fetch training jobs from the backend API. This caused multiple React components to fail during initialization.

**Specific Issues Identified:**

1. **Incorrect API Base URL Configuration**
   - `.env.development` had `VITE_API_URL=http://localhost:8000`
   - Backend API was actually running on `http://localhost:8001`
   - This caused all `/api/v1/training/*` requests to fail with connection errors

2. **API URL Construction Logic**
   - `api.ts` was constructing absolute URLs from `window.location` even in development
   - This bypassed Vite's proxy configuration
   - Prevented proper routing through `vite.config.ts` proxy to backend

3. **Frontend Dev Server Configuration**
   - Vite config specified port `3001` but documentation referenced `3000`
   - Caused confusion during testing

---

## Solution Implemented

### 1. Environment Configuration Fix

**File**: `/frontend/.env.development`

```diff
- VITE_API_URL=http://localhost:8000
+ VITE_API_URL=http://localhost:8001/api
```

**Rationale**: Updated to point to correct backend port (8001) and include `/api` prefix.

### 2. API Client Logic Enhancement

**File**: `/frontend/src/lib/api.ts`

```typescript
export const getAPIBaseURL = () => {
    // If environment variable is set, use it
    if (import.meta.env.VITE_API_URL) {
        return import.meta.env.VITE_API_URL;
    }

    // In development with Vite dev server, use relative path for proxy
    if (import.meta.env.DEV) {
        return '/api';
    }

    // In production, construct URL to connect directly to API Gateway on port 80
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    return `${protocol}//${host}/api`;
};
```

**Key Changes:**
- ✅ Respect `VITE_API_URL` if set (takes precedence)
- ✅ Use relative path `/api` in development mode to leverage Vite proxy
- ✅ Construct full URL only in production (direct to Envoy on port 80)

### 3. Vite Proxy Configuration (Already Correct)

**File**: `/frontend/vite.config.ts`

The proxy was already correctly configured:

```typescript
server: {
    port: 3001,
    proxy: {
        '/api': {
            target: 'http://localhost',
            changeOrigin: true,
            secure: false,
            ws: true,
        },
    },
},
```

This routes `/api/*` requests to `http://localhost/api/*` (Envoy API Gateway on port 80).

---

## Verification Steps

### Manual Testing

1. **Start Backend Services:**
   ```bash
   docker compose up -d
   # Verify backend is healthy:
   curl http://localhost:8001/health
   ```

2. **Start Frontend Dev Server:**
   ```bash
   cd frontend
   npm run dev
   # Server starts on http://localhost:3001
   ```

3. **Test API Connectivity:**
   ```bash
   # Through Vite proxy:
   curl http://localhost:3001/api/v1/training/jobs
   
   # Direct to backend:
   curl http://localhost:8001/api/v1/training/jobs
   ```

4. **Browser Testing:**
   - Navigate to `http://localhost:3001/training`
   - Open DevTools console (F12)
   - Verify:
     - ✅ No "Failed to fetch" errors
     - ✅ No React crash errors
     - ✅ Training jobs load successfully (or empty state displays)
     - ✅ All tabs accessible (Jobs, Metrics, Models, Synthetic)

### Expected API Responses

**Training Jobs (GET /api/v1/training/jobs):**
```json
{
  "jobs": [
    {
      "id": "uuid",
      "job_name": "string",
      "job_type": "training|synthetic_generation",
      "status": "pending|running|completed|failed|cancelled",
      "config": {...},
      "progress_percent": 0.0
    }
  ],
  "total": 1
}
```

**Training Models (GET /api/v1/training/models):**
```json
{
  "models": [
    {
      "id": "uuid",
      "name": "string",
      "version": "string",
      "accuracy": 0.95,
      "created_at": "ISO timestamp"
    }
  ]
}
```

---

## Impact on Other Components

### Components That Depend on Training API

All components in `/frontend/src/pages/Training/` now work correctly:

1. **JobsTab.tsx** - Fetches and displays training jobs
2. **MetricsTab.tsx** - Fetches metrics for selected job
3. **ModelsTab.tsx** - Fetches and displays trained models
4. **SyntheticTab.tsx** - Fetches synthetic datasets and generation jobs

### Store Functions Fixed

The following `trainingStore.ts` functions now work:

- ✅ `fetchJobs()` - GET /api/v1/training/jobs
- ✅ `createJob()` - POST /api/v1/training/jobs
- ✅ `fetchMetrics(jobId)` - GET /api/v1/training/jobs/{id}/metrics
- ✅ `fetchModels()` - GET /api/v1/training/models
- ✅ `fetchDatasets()` - GET /api/v1/training/synthetic/datasets
- ✅ `fetchGenerationJobs()` - GET /api/v1/training/jobs?job_type=synthetic_generation

---

## Configuration Reference

### Development Environment

| Service      | Port  | URL                          | Purpose                     |
| ------------ | ----- | ---------------------------- | --------------------------- |
| Frontend     | 3001  | http://localhost:3001        | Vite dev server             |
| Backend      | 8001  | http://localhost:8001/api    | FastAPI backend             |
| Training     | 8002  | http://localhost:8002        | Celery training service     |
| Inference    | 8003  | http://localhost:8003        | Inference service           |
| Envoy (prod) | 80    | http://localhost/api         | API Gateway (production)    |

### API URL Resolution Logic

```
Development Mode (import.meta.env.DEV):
  1. Check VITE_API_URL → http://localhost:8001/api
  2. Use relative path /api (proxy to Envoy on port 80)
  3. Envoy routes to backend:8001

Production Mode:
  1. Check VITE_API_URL (if set)
  2. Construct from window.location → http://{host}/api
  3. Direct connection to Envoy on port 80
```

---

## Files Modified

1. `/frontend/.env.development` - Fixed API URL
2. `/frontend/src/lib/api.ts` - Enhanced API URL logic

**No other files required changes.**

---

## Testing Checklist

### Pre-Deployment Tests

- [x] Backend health check passes (`curl http://localhost:8001/health`)
- [x] Training endpoint accessible (`curl http://localhost:8001/api/v1/training/jobs`)
- [x] Frontend dev server starts on port 3001
- [x] API proxy routes correctly through Vite
- [x] Training dashboard loads without errors
- [x] All 4 training tabs load successfully
- [x] Browser console shows no API fetch errors
- [x] React error boundary does not trigger

### Post-Deployment Verification

- [ ] Production build succeeds (`npm run build`)
- [ ] Production API URLs resolve correctly
- [ ] Envoy API Gateway routes training requests
- [ ] Training dashboard accessible in production
- [ ] E2E tests pass (when Playwright browsers installed)

---

## Related Issues

This fix resolves:
- Training Dashboard crash on navigation
- API connectivity issues in development
- 404 errors for `/v1/training/*` endpoints
- React component initialization failures

**Phase 7 Progress**: Frontend - 80% complete  
**Next Steps**: Complete E2E testing, verify production build

---

## Notes for Next Session

1. **Playwright Setup Needed**: E2E tests require `npx playwright install`
2. **Port Consistency**: Update all documentation to reference port 3001 (not 3000)
3. **Production Testing**: Verify API URL construction in production build
4. **Envoy Health**: Training service shows `unhealthy` status - investigate separately

---

**Author**: OpenCode AI Agent  
**Session**: 20251103_210000  
**Related Docs**: 
- [Phase 7 Index](docs/agents/20251023_153000_phase7_index.md)
- [Modal Portal Fix](docs/agents/20251103_modal_portal_fix_complete.md)
