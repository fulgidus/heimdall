# Heimdall SDR - Authentication & Authorization Guide

## Overview

Heimdall SDR uses **Keycloak** as a centralized Identity and Access Management (IAM) solution, providing:

- **Single Sign-On (SSO)** for all web interfaces
- **JWT-based authentication** for microservices
- **Role-Based Access Control (RBAC)**
- **Service-to-service authentication** using OAuth2 client credentials
- **Integration with external identity providers** (optional)

## Table of Contents

- [Architecture](#architecture)
- [User Roles](#user-roles)
- [Getting Started](#getting-started)
- [Frontend Integration](#frontend-integration)
- [Backend Service Integration](#backend-service-integration)
- [API Authentication](#api-authentication)
- [Service-to-Service Authentication](#service-to-service-authentication)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## Architecture

### Components

```
┌─────────────────┐
│   Frontend      │  ◄─── SSO (OIDC/PKCE)
│   (React)       │
└────────┬────────┘
         │ JWT Bearer Token
         ▼
┌─────────────────┐      ┌──────────────┐
│  API Gateway    │ ◄────┤  Keycloak    │
│                 │      │  (Auth)      │
└────────┬────────┘      └──────────────┘
         │ JWT Bearer Token      ▲
         ▼                       │
┌─────────────────┐             │
│  Microservices  │ ────────────┘
│  (RF, Training, │   Client Credentials
│   Inference)    │   (Service Auth)
└─────────────────┘
```

### Authentication Flows

1. **User Login (Frontend)**
   - User accesses frontend
   - Redirected to Keycloak login
   - After successful login, receives JWT token
   - Token sent with all API requests

2. **API Request (Backend)**
   - API Gateway validates JWT token
   - Checks user roles/permissions
   - Proxies request to backend service
   - Backend service can optionally re-validate token

3. **Service-to-Service Communication**
   - Service requests token using client credentials
   - Token cached and reused until expiration
   - Services validate each other's tokens

## User Roles

Heimdall defines three primary roles:

### 1. Admin
**Full system access** - Can manage all resources

**Permissions:**
- ✅ View all data and metrics
- ✅ Create, modify, delete users
- ✅ Trigger RF acquisitions
- ✅ Train and deploy ML models
- ✅ Access all administrative interfaces
- ✅ Modify system configuration

**Use Case:** System administrators, project owners

### 2. Operator
**Read/write access** to signals and models

**Permissions:**
- ✅ View all data and metrics
- ✅ Trigger RF acquisitions
- ✅ Train ML models
- ✅ Submit inference requests
- ✅ Create and manage recording sessions
- ❌ Manage users
- ❌ Access administrative interfaces

**Use Case:** Radio operators, data scientists

### 3. Viewer
**Read-only access** to data and visualizations

**Permissions:**
- ✅ View signals and detections
- ✅ Access dashboards and visualizations
- ✅ View model performance metrics
- ❌ Trigger acquisitions
- ❌ Train models
- ❌ Modify any data

**Use Case:** Observers, analysts, stakeholders

## Getting Started

### 1. Access Keycloak Admin Console

```bash
# Open Keycloak admin console
open http://localhost:8080

# Login with admin credentials
Username: admin
Password: admin
```

### 2. Select Heimdall Realm

1. Click on realm dropdown (top left)
2. Select **"Heimdall SDR"** realm

### 3. Default Users

Three default users are created:

| Username | Password | Role | Description |
|----------|----------|------|-------------|
| `admin` | `admin` | Admin | Full system access |
| `operator` | `operator` | Operator | Can trigger acquisitions and train models |
| `viewer` | `viewer` | Viewer | Read-only access |

⚠️ **Change these passwords immediately in production!**

### 4. Create New User

1. Navigate to **Users** section
2. Click **Add User**
3. Fill in user details:
   - Username (required)
   - Email (required)
   - First Name, Last Name
4. Click **Save**
5. Go to **Credentials** tab
6. Set password (uncheck "Temporary" for permanent password)
7. Go to **Role Mappings** tab
8. Assign appropriate realm role (admin/operator/viewer)

## Frontend Integration

The React frontend uses **react-oidc-context** for authentication.

### Configuration

```typescript
// src/auth/keycloakConfig.ts
export const keycloakConfig = {
  authority: 'http://localhost:8080/realms/heimdall',
  client_id: 'heimdall-frontend',
  redirect_uri: window.location.origin,
  response_type: 'code',
  scope: 'openid profile email',
  post_logout_redirect_uri: window.location.origin,
};
```

### Usage in Components

```typescript
import { useAuth } from 'react-oidc-context';

function MyComponent() {
  const auth = useAuth();
  
  // Check if user is authenticated
  if (!auth.isAuthenticated) {
    return <button onClick={() => auth.signinRedirect()}>Login</button>;
  }
  
  // Access user info
  const user = auth.user?.profile;
  
  // Get access token for API calls
  const token = auth.user?.access_token;
  
  // Make authenticated API call
  const response = await fetch('http://localhost:8000/api/v1/signals', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  return <div>Welcome, {user?.preferred_username}!</div>;
}
```

### Protected Routes

```typescript
import { useAuth } from 'react-oidc-context';
import { Navigate } from 'react-router-dom';

function ProtectedRoute({ children, requiredRole }) {
  const auth = useAuth();
  
  if (!auth.isAuthenticated) {
    return <Navigate to="/login" />;
  }
  
  const roles = auth.user?.profile?.realm_roles || [];
  
  if (requiredRole && !roles.includes(requiredRole)) {
    return <div>Access Denied</div>;
  }
  
  return children;
}

// Usage
<Route path="/admin" element={
  <ProtectedRoute requiredRole="admin">
    <AdminPanel />
  </ProtectedRoute>
} />
```

## Backend Service Integration

### 1. Add Authentication Dependency

```python
# services/your-service/requirements.txt
PyJWT[crypto]==2.8.0
cryptography==41.0.7
python-jose[cryptography]==3.3.0
```

### 2. Import Authentication Module

```python
# services/your-service/src/main.py
import sys
import os

# Add common auth module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../common'))

from auth import get_current_user, require_admin, require_operator, User
```

### 3. Protect Endpoints

```python
from fastapi import FastAPI, Depends
from auth import get_current_user, require_operator, User

app = FastAPI()

# Require any authenticated user
@app.get("/data")
async def get_data(user: User = Depends(get_current_user)):
    return {"message": f"Hello {user.username}"}

# Require operator role
@app.post("/acquisition/trigger")
async def trigger_acquisition(user: User = Depends(require_operator)):
    # Only operators and admins can access
    return {"status": "triggered", "by": user.username}

# Require admin role
@app.delete("/users/{user_id}")
async def delete_user(user_id: str, user: User = Depends(require_admin)):
    # Only admins can access
    return {"status": "deleted"}

# Custom role check
from auth import require_role

@app.post("/custom")
async def custom_endpoint(user: User = Depends(require_role(["operator", "admin"]))):
    return {"access": "granted"}
```

## API Authentication

### Get Access Token

```bash
# Get token for admin user
curl -X POST http://localhost:8080/realms/heimdall/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=heimdall-frontend" \
  -d "username=admin" \
  -d "password=admin" \
  | jq -r '.access_token'
```

### Use Token in API Requests

```bash
# Save token to variable
TOKEN=$(curl -s -X POST http://localhost:8080/realms/heimdall/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=heimdall-frontend" \
  -d "username=admin" \
  -d "password=admin" \
  | jq -r '.access_token')

# Use token to access protected endpoint
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/acquisition/health

# Check authentication status
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/check
```

### Python Example

```python
import requests

# Get token
token_url = "http://localhost:8080/realms/heimdall/protocol/openid-connect/token"
token_data = {
    "grant_type": "password",
    "client_id": "heimdall-frontend",
    "username": "admin",
    "password": "admin"
}
response = requests.post(token_url, data=token_data)
access_token = response.json()["access_token"]

# Use token
api_url = "http://localhost:8000/api/v1/acquisition/health"
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(api_url, headers=headers)
print(response.json())
```

## Service-to-Service Authentication

Services can authenticate with each other using **client credentials flow**.

### 1. Configure Service Client

Services are pre-configured in Keycloak with client credentials:

- `api-gateway` - Main API gateway
- `rf-acquisition` - RF acquisition service
- `training` - Training service
- `inference` - Inference service
- `data-ingestion-web` - Data ingestion service

### 2. Get Service Token

```python
import os
import requests

def get_service_token():
    """Get access token for service-to-service communication."""
    token_url = f"{os.getenv('KEYCLOAK_URL')}/realms/{os.getenv('KEYCLOAK_REALM')}/protocol/openid-connect/token"
    
    data = {
        "grant_type": "client_credentials",
        "client_id": os.getenv("KEYCLOAK_CLIENT_ID"),
        "client_secret": os.getenv("KEYCLOAK_CLIENT_SECRET"),
    }
    
    response = requests.post(token_url, data=data)
    response.raise_for_status()
    
    return response.json()["access_token"]

# Use token
token = get_service_token()
headers = {"Authorization": f"Bearer {token}"}
response = requests.post("http://other-service:8001/api/endpoint", headers=headers)
```

### 3. Token Caching

```python
from datetime import datetime, timedelta
from typing import Optional

class ServiceAuthClient:
    """Client for service-to-service authentication with token caching."""
    
    def __init__(self):
        self.token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
    
    def get_token(self) -> str:
        """Get cached token or fetch new one."""
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.token
        
        # Fetch new token
        token_url = f"{os.getenv('KEYCLOAK_URL')}/realms/{os.getenv('KEYCLOAK_REALM')}/protocol/openid-connect/token"
        
        data = {
            "grant_type": "client_credentials",
            "client_id": os.getenv("KEYCLOAK_CLIENT_ID"),
            "client_secret": os.getenv("KEYCLOAK_CLIENT_SECRET"),
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        
        result = response.json()
        self.token = result["access_token"]
        
        # Set expiry with 5-minute buffer
        expires_in = result.get("expires_in", 300)
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)
        
        return self.token
```

## Security Best Practices

### 1. Token Management

✅ **DO:**
- Store tokens securely (httpOnly cookies for web)
- Implement token refresh logic
- Clear tokens on logout
- Use short token lifespans (1 hour for access tokens)

❌ **DON'T:**
- Store tokens in localStorage (XSS risk)
- Hard-code tokens in source code
- Share tokens between users
- Log tokens in application logs

### 2. Password Security

✅ **DO:**
- Enforce strong password policies
- Enable password reset functionality
- Use temporary passwords for new users
- Implement brute force protection (enabled by default)

❌ **DON'T:**
- Use default passwords in production
- Share passwords via insecure channels
- Disable password complexity requirements
- Store passwords in plain text

### 3. Client Secrets

✅ **DO:**
- Rotate client secrets regularly
- Store secrets in environment variables
- Use secret management tools (Vault, K8s Secrets)
- Limit secret access to authorized personnel

❌ **DON'T:**
- Commit secrets to version control
- Use default secrets in production
- Share secrets via email or chat
- Hard-code secrets in application code

### 4. Role-Based Access Control

✅ **DO:**
- Follow principle of least privilege
- Assign roles based on job function
- Regularly review user permissions
- Use fine-grained permissions when needed

❌ **DON'T:**
- Grant admin access unnecessarily
- Create overly permissive roles
- Skip authorization checks
- Allow privilege escalation

## Troubleshooting

### Issue: "Missing authentication token"

**Cause:** No Authorization header in request

**Solution:**
```bash
# Make sure to include Bearer token
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" http://localhost:8000/api/v1/...
```

### Issue: "Token expired"

**Cause:** Access token has exceeded its lifetime (default: 1 hour)

**Solution:**
- Use refresh token to get new access token
- Re-authenticate if refresh token also expired
- Frontend should handle automatic token refresh

### Issue: "Invalid token"

**Cause:** Token is malformed or signature verification failed

**Possible Causes:**
- Token was modified
- Wrong Keycloak URL configuration
- Keycloak realm keys changed
- Network issues fetching JWKS

**Solution:**
- Verify `KEYCLOAK_URL` environment variable
- Check Keycloak is accessible from service
- Ensure correct realm name
- Get a fresh token

### Issue: "Insufficient permissions"

**Cause:** User doesn't have required role

**Solution:**
1. Check user's current roles in Keycloak admin console
2. Assign appropriate role to user
3. User must log out and log back in for role changes to take effect

### Issue: Cannot connect to Keycloak

**Cause:** Keycloak service not running or misconfigured

**Solution:**
```bash
# Check Keycloak container status
docker-compose ps keycloak

# View Keycloak logs
docker-compose logs keycloak

# Restart Keycloak
docker-compose restart keycloak

# Check health
curl http://localhost:8080/health/ready
```

### Issue: CORS errors in frontend

**Cause:** Keycloak CORS not configured for frontend origin

**Solution:**
1. Open Keycloak admin console
2. Go to Clients → heimdall-frontend
3. Add frontend URL to "Web Origins" (e.g., `http://localhost:5173`)
4. Save changes

## Additional Resources

- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [OAuth 2.0 Specification](https://oauth.net/2/)
- [OpenID Connect Specification](https://openid.net/connect/)
- [JWT.io - JWT Debugger](https://jwt.io/)
- [Heimdall Architecture Guide](./ARCHITECTURE.md)
- [Development Credentials](./dev-credentials.md)

---

*Last updated: 2025-10-24*
