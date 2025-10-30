# Authentication Fix - Visual Diagrams

## Problem: Incorrect Path Rewriting

### Before Fix (BROKEN)

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (localhost:3001 - Development)                       │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ Request: /api/v1/auth/login
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ Vite Dev Server Proxy                                        │
│                                                               │
│ Rule:                                                         │
│   if (path.startsWith('/api/v1/auth')) {                    │
│       return path;  ← AUTH WORKS ✅                          │
│   }                                                           │
│   return path.replace(/^\/api/, '');  ← OTHERS BREAK ❌     │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ Auth: /api/v1/auth/login ✅
                      │ Others: /v1/backend/websdrs ❌
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ API Gateway (localhost:8000)                                 │
│                                                               │
│ Registered Routes:                                            │
│   ✅ POST /api/v1/auth/login        ← Exists                │
│   ✅ POST /api/v1/auth/refresh      ← Exists                │
│   ✅ GET  /api/v1/backend/websdrs   ← Exists                │
│   ❌ GET  /v1/backend/websdrs       ← Does NOT exist (404)  │
└─────────────────────────────────────────────────────────────┘
```

**Result:** Auth works, but all other endpoints return 404!

---

## Solution: Remove Path Rewriting

### After Fix (WORKING)

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (localhost:3001 - Development)                       │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ Request: /api/v1/auth/login
                      │ Request: /api/v1/backend/websdrs
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ Vite Dev Server Proxy                                        │
│                                                               │
│ Rule:                                                         │
│   proxy: {                                                    │
│     '/api': {                                                 │
│       target: 'http://localhost:8000',                       │
│       changeOrigin: true,                                     │
│       // No rewriting - pass through all paths ✅            │
│     }                                                         │
│   }                                                           │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ /api/v1/auth/login ✅
                      │ /api/v1/backend/websdrs ✅
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ API Gateway (localhost:8000)                                 │
│                                                               │
│ Registered Routes:                                            │
│   ✅ POST /api/v1/auth/login        ← Matches!              │
│   ✅ POST /api/v1/auth/refresh      ← Matches!              │
│   ✅ GET  /api/v1/backend/websdrs   ← Matches!              │
└─────────────────────────────────────────────────────────────┘
```

**Result:** ALL endpoints work correctly! 🎉

---

## Production Flow (Nginx)

### Production Environment

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (external)                                            │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ Request: http://heimdall.local/api/v1/auth/login
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ Nginx (port 3000)                                             │
│                                                               │
│ Frontend Container:                                           │
│   location ^~ /api/ {                                         │
│     proxy_pass http://api-gateway:8000/api/;                 │
│     ← Forwards to API Gateway with /api/ preserved ✅        │
│   }                                                           │
│                                                               │
│   location / {                                                │
│     try_files $uri $uri/ /index.html;                        │
│     ← Serves React SPA                                        │
│   }                                                           │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ /api/v1/auth/login ✅
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ API Gateway Container (port 8000)                            │
│                                                               │
│ Docker Network: heimdall-network                             │
│ Service Name: api-gateway                                     │
│                                                               │
│ Routes:                                                       │
│   ✅ POST /api/v1/auth/login                                 │
│   ✅ POST /api/v1/auth/refresh                               │
│   ✅ GET  /api/v1/auth/check                                 │
└─────────────────────────────────────────────────────────────┘
                      │
                      │ Proxy to Keycloak
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ Keycloak Container (port 8080)                               │
│                                                               │
│ Realm: heimdall                                               │
│ Token Endpoint:                                               │
│   /realms/heimdall/protocol/openid-connect/token            │
│                                                               │
│ Response: JWT access_token + refresh_token                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Authentication Flow

### Complete Login Sequence

```
┌──────────┐       ┌──────────┐       ┌────────────┐       ┌──────────┐
│ Browser  │       │  Nginx   │       │ API Gateway│       │ Keycloak │
│          │       │  Proxy   │       │            │       │          │
└────┬─────┘       └────┬─────┘       └─────┬──────┘       └────┬─────┘
     │                  │                    │                   │
     │ 1. POST /api/v1/auth/login            │                   │
     │    {email, password}                  │                   │
     ├──────────────────>│                   │                   │
     │                  │                    │                   │
     │                  │ 2. Forward request │                   │
     │                  ├────────────────────>│                   │
     │                  │                    │                   │
     │                  │                    │ 3. POST token endpoint
     │                  │                    ├───────────────────>│
     │                  │                    │   OAuth2 password  │
     │                  │                    │   grant            │
     │                  │                    │                   │
     │                  │                    │ 4. Validate credentials
     │                  │                    │    Generate tokens│
     │                  │                    │<───────────────────┤
     │                  │                    │   {access_token,  │
     │                  │                    │    refresh_token} │
     │                  │                    │                   │
     │                  │ 5. Return tokens   │                   │
     │                  │<────────────────────┤                   │
     │                  │                    │                   │
     │ 6. 200 OK        │                    │                   │
     │    {access_token,│                    │                   │
     │     refresh_token}                    │                   │
     │<──────────────────┤                   │                   │
     │                  │                    │                   │
     │ 7. Store in localStorage              │                   │
     │    auth-store = {token, refreshToken} │                   │
     │                  │                    │                   │
     │ 8. Subsequent requests with Bearer token                  │
     │    Authorization: Bearer <token>      │                   │
     ├──────────────────>┼────────────────────>                  │
     │                  │                    │                   │
```

---

## File Structure

### Frontend Configuration Files

```
frontend/
├── vite.config.ts          ← FIXED: Removed path rewriting
├── src/
│   ├── lib/
│   │   └── api.ts          ← Axios instance (baseURL: '/api')
│   ├── store/
│   │   └── authStore.ts    ← Auth state (uses '/api/v1/auth/*')
│   └── services/
│       └── api/
│           ├── websdr.ts   ← Uses '/v1/backend/*' (baseURL adds /api)
│           ├── session.ts
│           └── inference.ts
└── nginx.conf              ← Production proxy config
```

### Backend Configuration Files

```
services/
├── api-gateway/
│   ├── src/
│   │   └── main.py         ← Routes: @app.post("/api/v1/auth/login")
│   └── tests/
│       └── test_auth_endpoints.py  ← NEW: Endpoint tests
└── common/
    └── auth/
        └── keycloak_auth.py  ← Keycloak integration
```

---

## Configuration Matrix

| Environment | Component | Path Format | Target | Result |
|-------------|-----------|-------------|--------|--------|
| **Development** | Browser | `/api/v1/auth/login` | → Vite | ✅ |
| | Vite Proxy | `/api/v1/auth/login` | → localhost:8000 | ✅ |
| | API Gateway | `/api/v1/auth/login` | → Keycloak | ✅ |
| **Production** | Browser | `/api/v1/auth/login` | → Nginx | ✅ |
| | Nginx | `/api/v1/auth/login` | → api-gateway:8000 | ✅ |
| | API Gateway | `/api/v1/auth/login` | → Keycloak | ✅ |

---

## Key Takeaways

### ✅ What Changed
- Removed path rewriting logic from Vite proxy
- Simplified configuration (8 lines → 1 line)
- All paths now handled consistently

### ✅ What Stayed the Same
- Frontend API calls (still use `/api/v1/*`)
- Backend route definitions (still at `/api/v1/*`)
- Nginx configuration (already correct)
- Keycloak configuration (already correct)

### ✅ Result
- Auth works ✅
- All other endpoints work ✅
- Development works ✅
- Production works ✅

---

**Total Lines Changed:** 8 lines removed from vite.config.ts  
**Impact:** Fixed 404 errors for all non-auth API endpoints  
**Breaking Changes:** None  
**Security Issues:** None
