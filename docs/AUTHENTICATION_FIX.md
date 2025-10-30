# Authentication Login 404 Fix

## Problem Statement

Login was failing with 404 error on `POST /api/v1/auth/login` in the Heimdall frontend application.

## Root Cause

The Vite development proxy configuration was incorrectly rewriting API paths:

**Before Fix:**
```typescript
// frontend/vite.config.ts
proxy: {
    '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => {
            // Auth paths passed through correctly
            if (path.startsWith('/api/v1/auth')) {
                return path;
            }
            // OTHER paths had /api prefix stripped ❌
            return path.replace(/^\/api/, '');
        },
    },
}
```

This caused:
- Auth endpoints: `POST /api/v1/auth/login` → `http://localhost:8000/api/v1/auth/login` ✅
- Other endpoints: `GET /api/v1/backend/websdrs` → `http://localhost:8000/v1/backend/websdrs` ❌

The API Gateway expects **all** paths to include the `/api/v1/...` prefix.

## Solution

Removed the path rewriting logic from Vite proxy configuration:

**After Fix:**
```typescript
// frontend/vite.config.ts
proxy: {
    '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // No rewriting - API Gateway expects /api/v1/* paths
    },
}
```

Now all paths are preserved:
- Auth endpoints: `POST /api/v1/auth/login` → `http://localhost:8000/api/v1/auth/login` ✅
- Other endpoints: `GET /api/v1/backend/websdrs` → `http://localhost:8000/api/v1/backend/websdrs` ✅

## Architecture Overview

### Development Flow (Vite dev server on port 3001)

```
Browser → /api/v1/auth/login
    ↓
Vite proxy (no rewrite)
    ↓
http://localhost:8000/api/v1/auth/login
    ↓
API Gateway → Keycloak
```

### Production Flow (Nginx on port 3000)

```
Browser → /api/v1/auth/login
    ↓
Nginx proxy (location ^~ /api/)
    ↓
http://api-gateway:8000/api/v1/auth/login
    ↓
API Gateway → Keycloak
```

## Configuration Details

### Frontend Configuration

**Base URL:**
```typescript
// frontend/src/lib/api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
```

**Auth Store:**
```typescript
// frontend/src/store/authStore.ts
const tokenUrl = '/api/v1/auth/login';  // Relative path
```

**API Services:**
```typescript
// frontend/src/services/api/*.ts
api.get('/v1/backend/websdrs')  // baseURL adds /api prefix
```

### Backend Configuration

**API Gateway Endpoints:**
```python
# services/api-gateway/src/main.py
@app.post("/api/v1/auth/login")        # Login endpoint
@app.post("/api/v1/auth/refresh")      # Token refresh
@app.get("/api/v1/auth/check")         # Auth status check
@app.get("/api/v1/auth/me")            # User info
```

**Nginx Proxy:**
```nginx
# frontend/nginx.conf
location ^~ /api/ {
    proxy_pass http://api-gateway:8000/api/;  # Preserves /api/ prefix
}
```

**Keycloak:**
- Running on port 8080
- Realm: `heimdall`
- Frontend Client: `heimdall-frontend`
- API Gateway Client: `api-gateway`

## Testing

### Automated Tests

**Frontend Tests:**
```bash
cd frontend
npm test -- authStore.test.ts
```

Expected result: 5/5 tests passing

**Backend Tests:**
```bash
cd services/api-gateway
pytest tests/test_auth_endpoints.py -v
```

Expected result: All endpoints return 200/400/401/500 (NOT 404)

### Manual E2E Testing

1. **Start Services:**
   ```bash
   docker-compose up -d
   ```

2. **Check Keycloak is Ready:**
   ```bash
   docker logs heimdall-keycloak-init
   ```
   Should see: "Keycloak initialization complete!"

3. **Test Login via Browser:**
   - Open http://localhost:3000
   - Open DevTools → Network tab
   - Click Login
   - Enter credentials:
     - Email: `admin@heimdall.local`
     - Password: `admin`
   - Verify request:
     - URL: `/api/v1/auth/login`
     - Method: `POST`
     - Status: `200` (success) or `401` (bad credentials)
     - NOT `404`!

4. **Verify Token Storage:**
   - Open DevTools → Application → Local Storage
   - Check `auth-store` contains:
     - `token`: JWT access token
     - `refreshToken`: Refresh token
     - `user`: User information

5. **Test Authenticated Requests:**
   - Navigate to dashboard
   - Check Network tab
   - Verify API requests include `Authorization: Bearer <token>` header

### Debugging Tips

**If login still returns 404:**

1. Check API Gateway logs:
   ```bash
   docker logs heimdall-api-gateway | grep -i auth
   ```

2. Verify endpoint registration:
   ```bash
   docker exec heimdall-api-gateway curl -X POST http://localhost:8000/api/v1/auth/login
   ```
   Should return 400 (missing credentials), NOT 404

3. Test directly from host:
   ```bash
   curl -v -X POST http://localhost:8000/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","password":"test"}'
   ```

4. Check Nginx proxy:
   ```bash
   docker logs heimdall-frontend | grep -i auth
   ```

**If Keycloak errors:**

1. Check Keycloak health:
   ```bash
   curl http://localhost:8080/health/ready
   ```

2. Verify realm exists:
   ```bash
   docker logs heimdall-keycloak | grep -i heimdall
   ```

3. Check init script:
   ```bash
   docker logs heimdall-keycloak-init
   ```

## Security Considerations

### OAuth2 Flow

1. **Login (Password Grant):**
   ```
   POST /api/v1/auth/login
   Body: { email, password }
   Response: { access_token, refresh_token, expires_in }
   ```

2. **Token Refresh:**
   ```
   POST /api/v1/auth/refresh
   Body: { refresh_token }
   Response: { access_token, refresh_token }
   ```

3. **Authenticated Requests:**
   ```
   GET /api/v1/backend/websdrs
   Headers: { Authorization: Bearer <access_token> }
   ```

### Token Management

- **Storage:** localStorage via Zustand persist middleware
- **Expiry:** Access token expires in 1 hour (configurable in Keycloak)
- **Refresh:** Automatic refresh when receiving 401 response
- **Security:** Tokens validated by Keycloak public keys on backend

### CORS Configuration

API Gateway CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specific domains only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Related Files

- `frontend/vite.config.ts` - Vite proxy configuration (FIXED)
- `frontend/src/lib/api.ts` - Axios instance configuration
- `frontend/src/store/authStore.ts` - Authentication state management
- `services/api-gateway/src/main.py` - Auth endpoint definitions
- `frontend/nginx.conf` - Production proxy configuration
- `scripts/init-keycloak.sh` - Keycloak initialization script
- `docker-compose.yml` - Service orchestration

## Changelog

- **2025-10-30:** Fixed Vite proxy path rewriting issue
- **2025-10-30:** Added auth endpoint tests
- **2025-10-30:** Verified configuration across development and production modes
