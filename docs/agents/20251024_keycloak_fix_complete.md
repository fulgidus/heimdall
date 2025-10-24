# Keycloak Authentication Fix - Complete Implementation

**Date**: 2025-10-24  
**Status**: ✅ COMPLETE  
**Session**: Keycloak Production Setup  

## Problem Statement

The frontend was making API calls that resulted in 403 Forbidden errors. Investigation revealed:
- Keycloak authentication was enabled but realm didn't exist
- Init script had incorrect API paths for Keycloak 23
- Login endpoint wasn't properly formatting requests for Keycloak
- AUTH_ENABLED was set to False in API Gateway

## Root Causes Identified

### 1. Keycloak 23 API Path Changes
**Issue**: Script used `/auth/realms/` paths (legacy format)  
**Fix**: Keycloak 23 uses `/realms/` directly (no `/auth/` prefix)

```bash
# BEFORE (404 Not Found)
curl http://localhost:8080/auth/realms/master

# AFTER (200 OK)
curl http://localhost:8080/realms/master
```

### 2. Initialization Script Issues
**Problems**:
- Used bash shebang (not in Alpine)
- jq dependency for JSON parsing (fragile)
- No idempotent checks for existing resources
- Timeout occurring before Keycloak ready

**Solutions**:
- Changed shebang to `/bin/sh` for Alpine compatibility
- Replaced jq with grep/cut for JSON parsing
- Added existence checks before create operations
- Improved Keycloak readiness detection

### 3. Login Endpoint Format Mismatch
**Issue**: Frontend sends JSON `{"email":"...", "password":"..."}` but Keycloak expects form-urlencoded with `username` parameter

**Before**:
```python
# Forwarded raw body as-is
response = await client.post(token_endpoint, content=body)
```

**After**:
```python
# Parse JSON and convert to form-urlencoded
body = await request.json()
form_data = {
    "client_id": client_id,
    "username": body.get("email") or body.get("username"),
    "password": body.get("password"),
    "grant_type": "password"
}
response = await client.post(token_endpoint, data=form_data)
```

### 4. Authentication Disabled
**Issue**: `AUTH_ENABLED = False` in API Gateway prevented auth checks

## Changes Made

### 1. Fixed `scripts/init-keycloak.sh` (290 lines)

**All API path updates**:
- `KEYCLOAK_URL/auth/realms/master` → `KEYCLOAK_URL/realms/master`
- `KEYCLOAK_URL/auth/admin/realms/` → `KEYCLOAK_URL/admin/realms/`
- `KEYCLOAK_URL/auth/realms/{realm}/protocol/...` → `KEYCLOAK_URL/realms/{realm}/protocol/...`

**Script improvements**:
- ✅ Shebang: `#!/bin/sh` (Alpine compatible)
- ✅ Readiness check: Returns HTTP 200 for `/realms/master`
- ✅ Token parsing: Uses `grep -o '"access_token":"[^"]*' | cut -d'"' -f4`
- ✅ Idempotent realm creation: Checks `"realm":"heimdall"` in response
- ✅ Idempotent client creation: Checks `clientId` existence before creating
- ✅ Idempotent user creation: Checks username existence before creating
- ✅ Better error messages: Shows HTTP status codes and responses

### 2. Enabled Authentication in API Gateway

**File**: `services/api-gateway/src/main.py`

```python
# Before
AUTH_ENABLED = False  # TEMPORARILY DISABLED FOR FRONTEND TESTING

# After
AUTH_ENABLED = True  # ENABLED: Keycloak OAuth2 authentication
```

### 3. Fixed Login Endpoint

**File**: `services/api-gateway/src/main.py` (lines 205-244)

**Key changes**:
- Parses JSON body from request (not raw form data)
- Converts `email` field to Keycloak `username` parameter
- Uses `data=form_data` for form-urlencoded encoding
- Includes `client_id` from environment
- Sets `grant_type=password`

## Test Results

### ✅ Keycloak Initialization
```bash
[OK] Keycloak is ready (HTTP 200)
[OK] Token obtained
[OK] Realm created successfully
[OK] API Gateway client created
[OK] Frontend client created
[OK] User created
[OK] Password set
```

### ✅ Login Test
```bash
POST /auth/login HTTP/1.1
Content-Type: application/json

{"email":"admin@heimdall.local","password":"admin"}

Response: {
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "expires_in": 300,
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "scope": "profile email"
}
```

### ✅ Protected Endpoint Test
```bash
GET /auth/check HTTP/1.1
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...

Response: {
  "authenticated": true,
  "auth_enabled": true,
  "user": {
    "id": "02f98de2-0250-47db-9edf-56730a6df76e",
    "username": "admin@heimdall.local",
    "email": "admin@heimdall.local",
    "roles": ["offline_access", "default-roles-heimdall"],
    "is_admin": false
  }
}
```

## Deployment Instructions

### 1. Clean Start (if needed)
```bash
docker compose down -v
docker compose up -d keycloak postgres
sleep 30  # Wait for services to start
docker compose up -d  # Start all services including keycloak-init
```

### 2. Verify Setup
```bash
# Check Keycloak realm exists
curl http://localhost:8080/realms/heimdall | jq .realm

# Test login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@heimdall.local","password":"admin"}' | jq .access_token
```

### 3. Environment Variables
Required in `.env` (see `.env.example`):
```bash
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=admin
KEYCLOAK_REALM=heimdall
APP_USER_EMAIL=admin@heimdall.local
APP_USER_PASSWORD=admin
KEYCLOAK_API_GATEWAY_CLIENT_ID=api-gateway
KEYCLOAK_API_GATEWAY_CLIENT_SECRET=api-gateway-secret
VITE_KEYCLOAK_CLIENT_ID=heimdall-frontend
```

## Key Learnings

### Keycloak 23 API Changes
- Version 23 removed the `/auth/` path prefix (legacy compatibility layer)
- All endpoints now use direct paths: `/realms/`, `/admin/`, `/protocol/`
- Health check endpoint: `/health/ready` returns JSON with status

### Alpine Linux Container Constraints
- Must use `/bin/sh` not `#!/bin/bash` (bash not installed by default)
- jq may not be available; prefer grep/cut for JSON parsing
- Use `apk add` for package installation in Alpine

### Keycloak OAuth2 Token Flow
- Client must specify `client_id` (public clients use openid-connect)
- Username field parameter is `username` (not `email`)
- Form data must be `application/x-www-form-urlencoded` (not JSON)
- Token endpoint: `/realms/{realm}/protocol/openid-connect/token`

### API Gateway Authentication
- Must explicitly enable `AUTH_ENABLED=True` to activate Keycloak checks
- Middleware intercepts requests and validates Bearer tokens
- Protected endpoints return 401 if token missing or invalid

## Files Modified

| File | Changes |
|------|---------|
| `scripts/init-keycloak.sh` | Updated all API paths, improved idempotency, fixed shebang |
| `services/api-gateway/src/main.py` | Enabled AUTH_ENABLED, fixed login endpoint payload handling |
| `docker-compose.yml` | No changes (keycloak-init service already configured) |
| `.env.example` | No changes (already had all variables) |

## Git Commit
```
fix: complete Keycloak authentication implementation

- Fixed init-keycloak.sh paths: removed /auth/ prefix for Keycloak 23 API endpoints
- Script now properly checks realm exists, creates realm/clients/users idempotently
- Enabled AUTH_ENABLED=True in API Gateway
- Fixed /auth/login endpoint to properly format JSON requests as form-urlencoded for Keycloak
- Login endpoint now accepts both 'email' and 'username' fields (maps to Keycloak username)
- Tested end-to-end: login and protected endpoints working with JWT tokens
- Keycloak realm 'heimdall' fully initialized with admin user (admin@heimdall.local / admin)
- Credentials stored in .env.example for consistency
```

## Next Steps

1. **Frontend Integration**: Update React login form to use `/auth/login` endpoint
2. **Token Storage**: Implement JWT token storage in localStorage/sessionStorage
3. **Protected Endpoints**: Configure protected routes in frontend router
4. **Role-Based Access**: Implement role checks (is_admin, is_operator, etc.)
5. **Token Refresh**: Add refresh token flow before expiration
6. **Logout**: Implement token revocation and session cleanup

## Production Checklist

- [ ] Change `KEYCLOAK_ADMIN_PASSWORD` from default "admin"
- [ ] Change all client secrets to strong values
- [ ] Configure `KEYCLOAK_URL` for production domain
- [ ] Set `KC_HOSTNAME_STRICT=true` for HTTPS
- [ ] Enable HTTPS certificates (cert-manager)
- [ ] Implement token rotation strategy
- [ ] Setup monitoring/alerting for auth failures
- [ ] Create backup strategy for Keycloak database

---

**Status**: Ready for E2E Testing  
**Related PRs**: #27 (E2E tests with OAuth2)  
**Follow-up Issues**: Frontend token handling, logout flow, admin panel
