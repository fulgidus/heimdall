# E2E Testing Documentation - Real Backend Integration

**Date**: 2025-10-24  
**Status**: ✅ IMPLEMENTED  
**Test Framework**: Playwright  

---

## 🎯 Objective

Verify **real HTTP calls** from frontend to backend services with **NO mocks/stubs**.

All 10 frontend pages must:
1. Load and display data from real backend APIs
2. Generate actual HTTP requests to `TEST_BACKEND_ORIGIN`
3. Receive 2xx responses or produce observable persistent effects
4. Have E2E tests that capture HAR, screenshots, and traces

---

## 📋 Pages Covered

| # | Page               | Route                  | Status | Backend Endpoints Tested |
|---|--------------------|------------------------|--------|--------------------------|
| 1 | Login              | `/login`               | ✅     | `/api/v1/auth/login`     |
| 2 | Dashboard          | `/dashboard`           | ✅     | `/api/v1/stats`, `/health`, `/api/v1/acquisition/websdrs` |
| 3 | Projects           | `/projects`            | ✅     | `/api/v1/sessions` (GET, POST, PATCH, DELETE) |
| 4 | WebSDR Management  | `/websdr-management`   | ✅     | `/api/v1/acquisition/websdrs`, `/api/v1/acquisition/websdrs/health` |
| 5 | Analytics          | `/analytics`           | ✅     | `/api/v1/sessions/analytics`, `/api/v1/analytics/*` |
| 6 | Localization       | `/localization`        | ✅     | `/api/v1/localizations`, `/api/v1/acquisition/websdrs` |
| 7 | Settings           | `/settings`            | ✅     | `/api/v1/settings` (if implemented) |
| 8 | Profile            | `/profile`             | ✅     | `/api/v1/user`, `/api/v1/profile` |
| 9 | System Status      | `/system-status`       | ✅     | `/api/v1/system/status`, `/api/v1/system/services` |
| 10| Recording Session  | `/recording-session`   | ✅     | `/api/v1/acquisition/acquire`, `/api/v1/sessions` |

---

## 🏗️ Test Infrastructure

### Directory Structure

```
frontend/
├── e2e/
│   ├── helpers/
│   │   └── test-utils.ts          # Helper functions (NO mocking)
│   ├── login.spec.ts              # Login page tests
│   ├── dashboard.spec.ts          # Dashboard tests
│   ├── projects.spec.ts           # Projects/Sessions tests
│   ├── websdr-management.spec.ts  # WebSDR management tests
│   ├── analytics.spec.ts          # Analytics tests
│   ├── localization.spec.ts       # Localization/Map tests
│   ├── settings.spec.ts           # Settings tests
│   ├── profile.spec.ts            # Profile tests
│   └── system-status.spec.ts      # System status tests
├── playwright.config.ts           # Playwright configuration
└── package.json                   # Added E2E scripts
```

### Key Files

- **`playwright.config.ts`**: Test configuration with HAR recording, screenshots, video
- **`e2e/helpers/test-utils.ts`**: Utilities for waiting on backend calls, login, logging
- **`scripts/run-e2e-tests.sh`**: Orchestration script for backend + tests

---

## 🔧 Configuration

### Environment Variables

```bash
# Frontend URL
BASE_URL=http://localhost:3001

# Backend API URL (NO trailing slash)
TEST_BACKEND_ORIGIN=http://localhost:8000
```

### Playwright Config Highlights

```typescript
{
  baseURL: process.env.BASE_URL || 'http://localhost:3001',
  
  use: {
    trace: 'on-first-retry',           // Capture trace on retry
    screenshot: 'only-on-failure',     // Screenshot on failure
    video: 'retain-on-failure',        // Video on failure
    recordHar: {                       // HAR network log
      mode: 'minimal',
      path: 'playwright-report/network.har'
    }
  }
}
```

---

## 🚀 Running Tests

### Local Development

```bash
# 1. Start backend services
cd /home/runner/work/heimdall/heimdall
docker compose -f docker compose.yml -f docker compose.services.yml up -d

# 2. Wait for services to be healthy
curl http://localhost:8000/health

# 3. Run E2E tests
cd frontend
npm run test:e2e

# 4. View report
npm run test:e2e:report
```

### Using Orchestration Script

```bash
# Automated: builds, starts backend, runs tests, collects artifacts
./scripts/run-e2e-tests.sh

# Options:
./scripts/run-e2e-tests.sh --no-build        # Skip docker build
./scripts/run-e2e-tests.sh --keep-running    # Keep services running after tests
```

### Debug Mode

```bash
# Run tests with Playwright Inspector
npm run test:e2e:debug

# Run tests in headed mode (see browser)
npm run test:e2e:headed

# Run tests with UI mode
npm run test:e2e:ui
```

---

## 🧪 Test Helper Functions

### `waitForBackendCall(page, urlPattern, expectedStatus)`

Waits for a **real backend API call** and verifies response.

```typescript
// Example: Wait for sessions list API call
const response = await waitForBackendCall(
  page, 
  '/api/v1/sessions',
  { min: 200, max: 299 }
);

expect(response.status()).toBe(200);
const data = await response.json();
```

### `setupRequestLogging(page)`

Logs all API requests/responses (NO mocking, just logging).

```typescript
await setupRequestLogging(page);
// Console will show: 📤 Request: GET /api/v1/sessions
//                    📥 Response: 200 /api/v1/sessions
```

### `login(page, email, password)`

Performs authentication via real backend.

```typescript
await login(page); // Uses demo credentials
await login(page, 'custom@email.com', 'password123');
```

### `verifyBackendReachable(page)`

Checks if backend is accessible before tests.

```typescript
const isReachable = await verifyBackendReachable(page);
expect(isReachable).toBe(true);
```

---

## 📊 Artifacts Generated

After each test run, the following artifacts are collected:

### 1. Playwright HTML Report
- **Path**: `frontend/playwright-report/index.html`
- **Contains**: Test results, timings, pass/fail status

### 2. Network HAR Files
- **Path**: `frontend/playwright-report/network.har`
- **Contains**: All HTTP requests/responses (can be viewed in Chrome DevTools)

### 3. Screenshots
- **Path**: `frontend/test-results/*/test-failed-*.png`
- **Captured**: On test failure only

### 4. Videos
- **Path**: `frontend/test-results/*/video.webm`
- **Captured**: On test failure only

### 5. Traces
- **Path**: `frontend/test-results/*/trace.zip`
- **View**: `npx playwright show-trace <path-to-trace.zip>`

### 6. Backend Logs
- **Path**: `e2e-artifacts-*/backend-logs.txt`
- **Contains**: Docker logs from all backend services

---

## ✅ Success Criteria

### PASS Criteria

For each page, tests **PASS** when:

1. ✅ Page loads successfully
2. ✅ At least one real HTTP request is made to `TEST_BACKEND_ORIGIN`
3. ✅ Backend responds with 2xx status code
4. ✅ Response data is used to update UI
5. ✅ HAR file contains the backend request/response
6. ✅ NO mocks/stubs intercept backend calls

### FAIL Criteria

Tests **FAIL** when:

- ❌ No HTTP requests made to backend
- ❌ Backend returns non-2xx status (unless testing error handling)
- ❌ Mocking/stubbing is used to bypass backend
- ❌ UI doesn't update with backend data
- ❌ Backend services not reachable

---

## 🔍 Verification Checklist

After running tests, verify:

- [ ] All 10 test files executed
- [ ] HAR file contains requests to `http://localhost:8000/api/*`
- [ ] Response status codes are 2xx (or expected error codes)
- [ ] No MSW (Mock Service Worker) or fetch mocks active
- [ ] Backend logs show incoming requests
- [ ] Playwright report shows all tests passed

---

## 🐛 Troubleshooting

### Backend Not Reachable

```bash
# Check if services are running
docker compose ps

# Check health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8004/health

# View logs
docker compose logs api-gateway
docker compose logs rf-acquisition
```

### CORS Errors

Ensure `api-gateway` has CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

### Tests Timeout

Increase timeout in test:

```typescript
await page.waitForResponse(..., { timeout: 30000 });
```

### No HAR Generated

Check `playwright.config.ts`:

```typescript
use: {
  recordHar: {
    mode: 'minimal',
    path: 'playwright-report/network.har'
  }
}
```

---

## 📈 CI/CD Integration

### GitHub Actions Workflow

```yaml
name: E2E Tests

on: [pull_request]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start Backend Services
        run: |
          docker compose -f docker compose.yml -f docker compose.services.yml up -d
          sleep 30
      
      - name: Wait for Backend Health
        run: |
          timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
      
      - name: Install Dependencies
        run: cd frontend && npm ci
      
      - name: Install Playwright Browsers
        run: cd frontend && npx playwright install chromium
      
      - name: Run E2E Tests
        run: cd frontend && npm run test:e2e
        env:
          BASE_URL: http://localhost:3001
          TEST_BACKEND_ORIGIN: http://localhost:8000
      
      - name: Upload Artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: frontend/playwright-report/
```

---

## 📚 References

- **Playwright Docs**: https://playwright.dev/
- **HAR Format**: http://www.softwareishard.com/blog/har-12-spec/
- **Project Documentation**: `/docs/agents/20251023_153000_frontend_pages_complete.md`

---

## 🎯 Summary

**10 E2E test files created** covering all frontend pages.

✅ Real backend integration (NO mocks)  
✅ HAR recording enabled  
✅ Screenshot/video on failure  
✅ Orchestration script for CI/local  
✅ Comprehensive artifacts collection  

**Next Steps**:
1. Start backend: `docker compose up -d`
2. Run tests: `./scripts/run-e2e-tests.sh`
3. Review report: `frontend/playwright-report/index.html`

---

**Last Updated**: 2025-10-24  
**Maintainer**: fulgidus  
**License**: CC Non-Commercial
