# Login 404 Error - Fix Verification Guide

## Problem Summary
The login endpoint was returning `404 Not Found` when the frontend tried to authenticate users. This was caused by a broken Docker build for the frontend container.

## Root Cause
The frontend Docker build was failing with error:
```
sh: 1: tsc: not found
```

This prevented the frontend container from building, which prevented the entire application from starting via `docker compose up`.

## Solution Applied
Modified `frontend/Dockerfile` to properly invoke TypeScript and Vite during the build process:

```dockerfile
# Set NODE_ENV to development to ensure devDependencies are installed
ENV NODE_ENV=development
RUN npm ci --legacy-peer-deps || npm install --legacy-peer-deps
# Use npx to ensure binaries from node_modules are found
RUN npx tsc -b && npx vite build
```

**Why this works:**
- `NODE_ENV=development` ensures devDependencies (including TypeScript and Vite) are installed
- `npm ci` provides reproducible builds from package-lock.json
- `npx` automatically finds executables in `node_modules/.bin` without requiring them to be in PATH

## Verification Steps

### Step 1: Verify Build Works
```bash
cd /path/to/heimdall
docker compose build frontend
```

**Expected Result:** Build completes successfully without "tsc: not found" error.

### Step 2: Start All Services
```bash
docker compose up -d
```

**Expected Result:** All containers start successfully.

Check status:
```bash
docker compose ps
```

All services should show "healthy" or "running" status.

### Step 3: Wait for Keycloak Initialization
Keycloak needs 2-3 minutes to:
1. Start up
2. Create the Heimdall realm
3. Configure the frontend client
4. Create the admin user

Monitor initialization:
```bash
docker compose logs keycloak-init -f
```

Wait for message: `"Keycloak initialization complete!"`

### Step 4: Test Login Endpoint (Direct to API Gateway)
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@heimdall.local","password":"admin"}' \
  -w "\nHTTP Status: %{http_code}\n"
```

**Expected Result:**
- HTTP Status: `200 OK`
- Response contains `access_token` field
- **NOT** `404 Not Found`

Example successful response:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Step 5: Test Through Frontend Proxy
```bash
curl -X POST http://localhost:3000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@heimdall.local","password":"admin"}' \
  -w "\nHTTP Status: %{http_code}\n"
```

**Expected Result:** Same as Step 4 (200 OK with JWT token)

This tests the full path:
```
Browser/curl -> Frontend Nginx (port 3000) -> API Gateway (port 8000) -> Keycloak (port 8080)
```

### Step 6: Test Web UI Login
1. Open browser to http://localhost:3000
2. Click "Login" or navigate to http://localhost:3000/login
3. Enter credentials:
   - Email: `admin@heimdall.local`
   - Password: `admin`
4. Click "Login"

**Expected Result:**
- No 404 error
- Successful login
- Redirect to dashboard at http://localhost:3000/dashboard

### Step 7: Run Unit Tests
```bash
cd /path/to/heimdall

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pytest pytest-asyncio httpx fastapi[standard] pydantic-settings

# Run auth endpoint tests
python -m pytest services/api-gateway/tests/test_auth_endpoints.py -v
```

**Expected Result:** All 6 tests pass:
```
test_auth_login_endpoint_exists PASSED
test_auth_login_with_json_body PASSED
test_auth_login_with_form_data PASSED
test_auth_refresh_endpoint_exists PASSED
test_auth_check_endpoint_exists PASSED
test_all_auth_endpoints_use_correct_paths PASSED
```

### Step 8: Run E2E Tests (Optional)
```bash
cd frontend

# Install dependencies
npm install --legacy-peer-deps

# Run Playwright E2E tests
npm run test:e2e
```

**Expected Result:** Login E2E tests pass, including:
- Login page loads
- Form submission works
- JWT token received
- Redirect to dashboard
- Token stored in localStorage

## Troubleshooting

### If Build Still Fails
1. Clear Docker build cache:
   ```bash
   docker compose build frontend --no-cache
   ```

2. Check for leftover containers:
   ```bash
   docker compose down -v
   docker compose up -d
   ```

### If Login Returns 404
1. Check API Gateway is running:
   ```bash
   docker compose logs api-gateway
   ```

2. Verify API Gateway health:
   ```bash
   curl http://localhost:8000/health
   ```

3. Check Nginx proxy configuration in frontend:
   ```bash
   docker compose exec frontend cat /etc/nginx/nginx.conf | grep -A 10 "location /api"
   ```

### If Login Returns 500
This means the endpoint exists but Keycloak isn't ready yet. Wait another minute and try again.

### If Login Returns 401
Either:
- Keycloak user isn't created yet (check keycloak-init logs)
- Wrong credentials (default is admin@heimdall.local / admin)

## Success Criteria
- ✅ Frontend Docker image builds successfully
- ✅ All containers start and reach healthy status
- ✅ Login endpoint returns 200 (not 404)
- ✅ JWT token is returned from Keycloak
- ✅ Web UI login works end-to-end
- ✅ Unit tests pass
- ✅ E2E tests pass (if run)

## Files Changed
- `frontend/Dockerfile` - Fixed build process to use npx for TypeScript and Vite

## Related Documentation
- Frontend: `frontend/README.md`
- API Gateway: `services/api-gateway/README.md`
- Keycloak Setup: `docs/KEYCLOAK.md`
- E2E Testing: `frontend/e2e/README.md`
