# CORS Configuration Guide

## Overview

Heimdall implements a robust Cross-Origin Resource Sharing (CORS) configuration at multiple levels to ensure secure and flexible cross-origin access in both development and production environments.

## Architecture

CORS is implemented at two levels:

1. **Backend (FastAPI)** - Application-level CORS middleware
2. **Envoy Gateway** - Gateway-level CORS headers and pre-flight handling

This dual-layer approach ensures:
- Proper CORS headers even when backend is accessed directly
- Centralized CORS policy at the gateway level
- Optimized pre-flight request handling
- WebSocket support with CORS

## Configuration

### Environment Variables

All CORS settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173,http://localhost:8000` | Comma-separated list of allowed origins |
| `CORS_ALLOW_CREDENTIALS` | `true` | Allow cookies and authorization headers |
| `CORS_ALLOW_METHODS` | `GET,POST,PUT,DELETE,PATCH,OPTIONS` | Allowed HTTP methods |
| `CORS_ALLOW_HEADERS` | `Authorization,Content-Type,Accept,Origin,X-Requested-With` | Allowed request headers |
| `CORS_EXPOSE_HEADERS` | `*` | Headers exposed to the browser |
| `CORS_MAX_AGE` | `3600` | Pre-flight cache duration (seconds) |

### Development Setup

Default configuration works out-of-the-box for local development:

```bash
# Frontend dev server (Vite)
http://localhost:3000  # Production build
http://localhost:5173  # Vite dev server
http://localhost:8000  # API Gateway
```

No changes needed - just run `docker-compose up`!

### Production Setup

For production, update `.env` with your domain:

```bash
# Single domain
CORS_ORIGINS=https://heimdall.example.com

# Multiple domains
CORS_ORIGINS=https://heimdall.example.com,https://app.heimdall.example.com

# Increase pre-flight cache for performance
CORS_MAX_AGE=7200
```

### Wildcard (NOT Recommended for Production)

For testing only:

```bash
CORS_ORIGINS=*
```

⚠️ **Security Warning**: Wildcard allows ANY origin and should never be used in production!

## Implementation Details

### Backend (FastAPI)

Located in `services/backend/src/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.get_cors_methods_list(),
    allow_headers=settings.get_cors_headers_list(),
    expose_headers=["*"],
    max_age=settings.cors_max_age,
)
```

### Envoy Gateway

Located in `db/envoy/envoy.yaml`:

- **CORS Filter**: Handles pre-flight OPTIONS requests
- **CORS Policy**: Applied to all routes (including WebSocket `/ws`)
- **Headers**: Configured at virtual host level

```yaml
cors:
  allow_origin_string_match:
    - safe_regex:
        regex: ".*"  # Dev: all origins; Prod: specific regex
  allow_methods: "GET, POST, PUT, DELETE, PATCH, OPTIONS"
  allow_headers: "Authorization, Content-Type, Accept, Origin, X-Requested-With"
  expose_headers: "Content-Length, Content-Type, Authorization"
  max_age: "3600"
  allow_credentials: true
```

## Testing

### Manual Testing

Test CORS with curl:

```bash
# Pre-flight request
curl -X OPTIONS http://localhost:8000/api/v1/health \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: GET" \
  -v

# Actual request with credentials
curl http://localhost:8000/api/v1/health \
  -H "Origin: http://localhost:3000" \
  -H "Authorization: Bearer your-token" \
  -v
```

Expected headers in response:
- `access-control-allow-origin: http://localhost:3000`
- `access-control-allow-credentials: true`
- `access-control-allow-methods: GET, POST, ...`

### Automated Tests

Run backend CORS tests:

```bash
cd services/backend
pytest tests/test_cors_config.py -v
```

## Troubleshooting

### CORS Error in Browser

**Symptom**: Console shows "CORS policy: No 'Access-Control-Allow-Origin' header"

**Solutions**:
1. Check `CORS_ORIGINS` includes your frontend URL
2. Verify Envoy is running (`docker-compose ps envoy`)
3. Check backend health (`curl http://localhost:8001/health`)
4. Ensure frontend uses correct API URL (via Envoy gateway)

### Pre-flight Requests Failing

**Symptom**: OPTIONS requests return 404 or no CORS headers

**Solutions**:
1. Verify Envoy CORS filter is enabled (`db/envoy/envoy.yaml`)
2. Check CORS policy in virtual hosts configuration
3. Ensure backend supports OPTIONS method

### Credentials Not Working

**Symptom**: Authorization header not sent or cookies not included

**Solutions**:
1. Set `CORS_ALLOW_CREDENTIALS=true`
2. Frontend must use `credentials: 'include'` in fetch/axios
3. Cannot use wildcard `*` with credentials - use specific origins

### WebSocket CORS Issues

**Symptom**: WebSocket connection fails with CORS error

**Solutions**:
1. WebSocket inherits HTTP CORS policy
2. Ensure `/ws` route has CORS policy in Envoy
3. Use `ws://` or `wss://` with same origin as HTTP

## Security Best Practices

1. **Never use wildcard (`*`) in production**
   - Always specify exact allowed origins

2. **Use HTTPS in production**
   - CORS with credentials requires secure origins

3. **Minimize exposed headers**
   - Only expose headers that frontend needs

4. **Keep pre-flight cache reasonable**
   - 1 hour (3600s) for dev, 2 hours (7200s) for prod

5. **Monitor CORS errors**
   - Log blocked CORS requests for security audits

## Integration with Keycloak

CORS configuration supports Keycloak authentication:

- `Authorization` header allowed for Bearer tokens
- Credentials enabled for cookie-based sessions
- `/auth` route proxied through Envoy with CORS

Frontend can authenticate with:

```typescript
fetch('http://localhost:8000/api/v1/protected', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${keycloakToken}`,
  },
  credentials: 'include',
})
```

## Performance Optimization

Pre-flight requests are cached for 1 hour (3600s) by default:

- Reduces OPTIONS requests by 99%+
- Browser caches CORS policy per origin
- Can be increased to 2 hours (7200s) in production

## References

- [MDN CORS Documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [FastAPI CORS Middleware](https://fastapi.tiangolo.com/tutorial/cors/)
- [Envoy CORS Filter](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/cors_filter)
