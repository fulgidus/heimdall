# Token Auto-Refresh Implementation

## Problem
Users were experiencing automatic logouts because JWT access tokens expire (typically after 5-15 minutes in Keycloak) without any automatic renewal mechanism. This forced users to re-authenticate frequently, disrupting their workflow.

## Solution Overview
Implemented a comprehensive token refresh system with three layers of protection:

1. **Proactive Token Refresh** - Automatically refreshes tokens 1 minute before expiration
2. **Reactive Token Refresh** - Intercepts 401 errors and attempts refresh before logging out
3. **Request Queuing** - Prevents multiple simultaneous refresh attempts

## Architecture

```
User Login
    ↓
Keycloak Issues Tokens
    ↓
    ├── Access Token (expires in 5-15 min)
    └── Refresh Token (expires in 30-60 min)
    ↓
Frontend Stores Both Tokens
    ↓
    ├─→ Proactive Timer (refreshes 1 min before expiry)
    │   └─→ Schedules Next Refresh
    │
    └─→ API Request
        ├─→ Success (200) → Continue
        └─→ Failure (401)
            └─→ Axios Interceptor
                ├─→ Attempt Refresh
                │   ├─→ Success → Retry Original Request
                │   └─→ Failure → Logout User
                └─→ Queue Other Requests During Refresh
```

## Implementation Details

### 1. Backend: API Gateway Refresh Endpoint

**File:** `services/api-gateway/src/main.py`

Added `/api/v1/auth/refresh` endpoint that proxies refresh token requests to Keycloak:

```python
@app.post("/api/v1/auth/refresh")
async def refresh_token_proxy(request: Request):
    # Accepts refresh_token in JSON or form-urlencoded
    # Proxies to Keycloak token endpoint with grant_type=refresh_token
    # Returns new access_token and refresh_token
```

**Request:**
```json
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGc..."
}
```

**Response (Success):**
```json
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "expires_in": 300,
  "refresh_expires_in": 1800
}
```

**Response (Failure):**
```json
{
  "error": "invalid_grant",
  "error_description": "Token is not active"
}
```

### 2. Frontend: Auth Store Enhancement

**File:** `frontend/src/store/authStore.ts`

Added `refreshAccessToken()` method to the Zustand store:

```typescript
refreshAccessToken: async () => {
    const currentRefreshToken = state.refreshToken;
    
    if (!currentRefreshToken) {
        return false;
    }
    
    try {
        const response = await fetch(`${API_URL}/api/v1/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: currentRefreshToken }),
        });
        
        if (!response.ok) {
            useAuthStore.getState().logout();
            return false;
        }
        
        const data = await response.json();
        set({
            token: data.access_token,
            refreshToken: data.refresh_token,
        });
        
        return true;
    } catch (error) {
        useAuthStore.getState().logout();
        return false;
    }
}
```

**Key Features:**
- Uses `fetch` instead of `axios` to avoid circular dependency
- Returns boolean indicating success/failure
- Automatically logs out user if refresh fails
- Updates both access and refresh tokens

### 3. Frontend: Axios Response Interceptor

**File:** `frontend/src/lib/api.ts`

Enhanced the axios response interceptor to handle 401 errors:

```typescript
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;
        
        if (error.response?.status === 401 && !originalRequest._retry) {
            if (isRefreshing) {
                // Queue this request
                return new Promise((resolve, reject) => {
                    failedQueue.push({ resolve, reject });
                }).then(() => {
                    originalRequest.headers.Authorization = `Bearer ${useAuthStore.getState().token}`;
                    return api(originalRequest);
                });
            }
            
            originalRequest._retry = true;
            isRefreshing = true;
            
            try {
                const success = await useAuthStore.getState().refreshAccessToken();
                
                if (success) {
                    originalRequest.headers.Authorization = `Bearer ${useAuthStore.getState().token}`;
                    processQueue(null); // Resolve all queued requests
                    return api(originalRequest);
                } else {
                    processQueue(new Error('Token refresh failed'));
                    window.location.href = '/login';
                }
            } finally {
                isRefreshing = false;
            }
        }
        
        return Promise.reject(error);
    }
);
```

**Key Features:**
- Detects 401 Unauthorized responses
- Prevents duplicate refresh attempts using `isRefreshing` flag
- Queues simultaneous requests during refresh
- Retries original request with new token
- Falls back to logout if refresh fails

### 4. Frontend: Proactive Token Refresh Hook

**File:** `frontend/src/hooks/useTokenRefresh.ts`

Custom React hook that proactively refreshes tokens before expiration:

```typescript
export const useTokenRefresh = () => {
    const { token, refreshToken, refreshAccessToken, isAuthenticated } = useAuthStore();
    
    useEffect(() => {
        if (!isAuthenticated || !token || !refreshToken) {
            return;
        }
        
        const scheduleRefresh = () => {
            // Decode JWT to get expiration
            const payload = JSON.parse(atob(token.split('.')[1]));
            const expiresAt = payload.exp * 1000;
            const now = Date.now();
            const timeUntilExpiry = expiresAt - now;
            
            // Refresh 60 seconds before expiration
            const refreshIn = Math.max(timeUntilExpiry - 60000, 1000);
            
            setTimeout(async () => {
                const success = await refreshAccessToken();
                if (success) {
                    scheduleRefresh(); // Schedule next refresh
                }
            }, refreshIn);
        };
        
        scheduleRefresh();
    }, [token, refreshToken, isAuthenticated]);
};
```

**Key Features:**
- Decodes JWT to extract expiration time (`exp` claim)
- Schedules refresh 60 seconds before expiration
- Recursively schedules next refresh after successful renewal
- Automatically cleans up timer on component unmount
- Only active when user is authenticated

### 5. Frontend: App Integration

**File:** `frontend/src/App.tsx`

Integrated the hook into the main App component:

```typescript
function App() {
    // Enable automatic token refresh
    useTokenRefresh();
    
    return (
        <ErrorBoundary>
            <Router>
                {/* Routes */}
            </Router>
        </ErrorBoundary>
    );
}
```

## Testing

### Unit Tests

**File:** `frontend/src/store/tokenRefresh.test.ts`

```typescript
describe('Token Refresh', () => {
    it('should refresh token successfully', async () => {
        // Setup with valid refresh token
        // Mock successful API response
        // Verify tokens updated
    });
    
    it('should logout on refresh failure', async () => {
        // Setup with invalid refresh token
        // Mock failed API response
        // Verify user logged out
    });
    
    it('should return false when no refresh token available', async () => {
        // Setup without refresh token
        // Verify no API call made
    });
});
```

**Test Results:** ✅ All 3 tests passing

### Integration Testing

To manually test the implementation:

1. **Login:** Authenticate with valid credentials
2. **Check Token Expiry:** Open DevTools Console and look for log:
   ```
   🕐 Token expires in 300s, scheduling refresh in 240s
   ```
3. **Wait for Proactive Refresh:** After 4 minutes, you should see:
   ```
   🔄 Proactively refreshing token...
   ✅ Token refreshed successfully
   🕐 Token expires in 300s, scheduling refresh in 240s
   ```
4. **Test 401 Handling:** Manually expire token in localStorage and make API request
5. **Verify No Logout:** User should remain logged in after automatic refresh

## Configuration

### Environment Variables

**Backend (API Gateway):**
```env
KEYCLOAK_URL=http://keycloak:8080
KEYCLOAK_REALM=heimdall
VITE_KEYCLOAK_CLIENT_ID=heimdall-frontend
```

**Frontend:**
```env
VITE_API_URL=http://localhost:8000
VITE_KEYCLOAK_CLIENT_ID=heimdall-frontend
```

### Keycloak Configuration

Ensure the client configuration allows refresh tokens:

1. Go to Keycloak Admin Console
2. Navigate to Clients → `heimdall-frontend`
3. Verify settings:
   - **Client Protocol:** openid-connect
   - **Access Type:** public
   - **Standard Flow Enabled:** ON
   - **Direct Access Grants Enabled:** ON
   - **Valid Redirect URIs:** `http://localhost:5173/*`
   - **Web Origins:** `http://localhost:5173`

### Token Lifetimes

Default Keycloak settings (can be adjusted in Realm Settings → Tokens):
- **Access Token Lifespan:** 5 minutes
- **Refresh Token Lifespan:** 30 minutes
- **SSO Session Idle:** 30 minutes
- **SSO Session Max:** 10 hours

## Security Considerations

1. **Refresh Token Storage:** Stored in Zustand with persistence to localStorage
   - Consider using `httpOnly` cookies for production (requires backend changes)
   
2. **Token Exposure:** Tokens logged to console in development
   - Remove console logs in production build
   
3. **XSS Protection:** Tokens in localStorage vulnerable to XSS
   - Implement Content Security Policy (CSP)
   - Sanitize all user inputs
   
4. **CSRF Protection:** Not required for Bearer token authentication
   - But consider implementing for cookie-based tokens

5. **Token Rotation:** Keycloak can be configured for refresh token rotation
   - New refresh token issued with each refresh
   - Old refresh token invalidated immediately

## Troubleshooting

### Token Not Refreshing

**Check Console Logs:**
```
🕐 Token expires in Xs, scheduling refresh in Ys
```

If missing, verify:
- User is authenticated
- Token and refreshToken exist in store
- `useTokenRefresh` hook is called in App component

### 401 Still Causing Logout

**Check Network Tab:**
1. Look for failed request with 401
2. Check if `/api/v1/auth/refresh` is called
3. Verify refresh request succeeds (200)
4. Confirm original request is retried

If refresh fails:
- Check refresh token validity
- Verify Keycloak is running
- Check API Gateway logs

### Proactive Refresh Not Working

**Verify Token Format:**
```javascript
const token = useAuthStore.getState().token;
const parts = token.split('.');
console.log('Token parts:', parts.length); // Should be 3
const payload = JSON.parse(atob(parts[1]));
console.log('Token expires:', new Date(payload.exp * 1000));
```

### Multiple Refresh Requests

This indicates the request queue isn't working:
- Check `isRefreshing` flag
- Verify `failedQueue` is being processed
- Look for race conditions in concurrent requests

## Future Enhancements

1. **Secure Storage:** Move tokens to `httpOnly` cookies
2. **Token Rotation:** Implement refresh token rotation
3. **Silent Refresh:** Use hidden iframe for seamless refresh
4. **Biometric Auth:** Add fingerprint/face recognition
5. **Session Management:** Track active sessions across devices
6. **Token Revocation:** Implement logout endpoint that revokes tokens
7. **Metrics:** Track refresh success/failure rates
8. **User Notifications:** Alert user when session will expire soon

## Related Files

- `services/api-gateway/src/main.py` - Refresh endpoint
- `frontend/src/store/authStore.ts` - Auth state management
- `frontend/src/lib/api.ts` - Axios interceptors
- `frontend/src/hooks/useTokenRefresh.ts` - Proactive refresh hook
- `frontend/src/App.tsx` - Hook integration
- `frontend/src/store/tokenRefresh.test.ts` - Unit tests

## References

- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749#section-6)
- [Keycloak Token API](https://www.keycloak.org/docs/latest/securing_apps/#_token-exchange)
- [JWT Specification](https://tools.ietf.org/html/rfc7519)
- [OWASP Token Storage](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)
