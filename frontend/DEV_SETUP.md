# Frontend Development Setup

## Quick Start for Local Development

### Option 1: Docker (Recommended for Production-like Testing)

```bash
# From project root
docker compose up -d frontend

# View logs
docker compose logs -f frontend

# Rebuild after code changes
docker compose build --no-cache frontend
docker compose up -d frontend
```

Access at: http://localhost:3000

### Option 2: Vite Dev Server (Recommended for Active Development)

**Prerequisites:**
- Ensure Envoy API Gateway is running on port 80
- All backend services should be running via Docker

```bash
# Terminal 1: Start backend services (from project root)
docker compose up -d postgres redis rabbitmq minio keycloak backend training inference envoy

# Terminal 2: Start frontend dev server (from frontend directory)
cd frontend
pnpm install
pnpm dev
```

Access at: http://localhost:3001

**Vite Dev Server Features:**
- ✅ Hot Module Replacement (HMR) - instant updates
- ✅ Auto proxy to Envoy (port 80) for all `/api/*` requests
- ✅ WebSocket proxy for `/ws` connections
- ✅ No CORS issues - proxy handles everything
- ✅ Fast TypeScript compilation
- ✅ Detailed console logging for API requests

**Proxy Configuration:**
The Vite dev server automatically proxies these paths to `http://localhost`:
- `/api/*` → API Gateway
- `/ws` → WebSocket
- `/backend/*` → Backend service health
- `/training/*` → Training service health
- `/inference/*` → Inference service health
- `/health/*` → System health checks

## Troubleshooting

### "CORS Error" when using `pnpm dev`

**Cause:** Envoy API Gateway is not running or not accessible on `http://localhost` (port 80)

**Solution:**
```bash
# Check if Envoy is running
docker compose ps envoy

# If not running, start it
docker compose up -d envoy

# Test API endpoint
curl http://localhost/api/v1/acquisition/websdrs-all
```

### "Connection refused" errors

**Cause:** Backend services are not running

**Solution:**
```bash
# Start all backend services
docker compose up -d
```

### Docker build not picking up changes

**Cause:** Docker is using cached layers

**Solution:**
```bash
# Force fresh build with no cache
docker compose build --no-cache frontend

# Remove old image and rebuild
docker image rm heimdall-frontend -f
docker compose build frontend
docker compose up -d frontend
```

### Browser shows old code after rebuild

**Cause:** Browser cache

**Solution:**
- Hard refresh: `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)
- Or: Open DevTools (F12) → Right-click refresh → "Empty Cache and Hard Reload"
- Or: Use Incognito/Private window

## Environment Variables

### Development (Vite Dev Server)

Edit `frontend/.env.development`:
```env
VITE_API_URL=
VITE_ENV=development
VITE_ENABLE_DEBUG=true
```

### Production (Docker Build)

Set in `docker-compose.yml`:
```yaml
frontend:
  build:
    args:
      VITE_API_URL: ""
      VITE_ENV: production
```

## Ports

| Service | Dev Port | Docker Port |
|---------|----------|-------------|
| Vite Dev Server | 3001 | N/A |
| Frontend (nginx) | N/A | 3000 |
| Envoy API Gateway | 80 | 80 |
| Backend API | N/A | 8001 |

## Recommended Development Workflow

1. **Start Backend Services:**
   ```bash
   docker compose up -d
   ```

2. **Start Frontend Dev Server:**
   ```bash
   cd frontend
   pnpm dev
   ```

3. **Edit Code:**
   - Changes auto-reload in browser
   - API calls automatically proxied to Envoy
   - No need to rebuild Docker image

4. **Test Production Build:**
   ```bash
   docker compose build frontend
   docker compose up -d frontend
   # Test at http://localhost:3000
   ```

5. **Commit Changes:**
   ```bash
   git add .
   git commit -m "feat: your changes"
   ```

## Common Commands

```bash
# Install dependencies
pnpm install

# Start dev server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview

# Run tests
pnpm test

# Run linter
pnpm lint

# Format code
pnpm format
```

## API Configuration Explained

The frontend uses a smart API base URL configuration in `src/lib/api.ts`:

```typescript
const getAPIBaseURL = () => {
  // If VITE_API_URL is set and not a relative path, use it
  if (import.meta.env.VITE_API_URL && !import.meta.env.VITE_API_URL.startsWith('/')) {
    return import.meta.env.VITE_API_URL;
  }

  // Otherwise, construct URL from window.location
  const protocol = window.location.protocol; // http: or https:
  const host = window.location.hostname; // localhost, etc.
  return `${protocol}//${host}`; // e.g., "http://localhost"
};
```

**Result:**
- Dev server (port 3001): `http://localhost` → proxied to Envoy on port 80
- Docker (port 3000): `http://localhost` → proxied to Envoy on port 80
- Production: Uses actual hostname (e.g., `https://heimdall.example.com`)

This ensures `/api/v1/...` paths work correctly without double `/api/api/` prefixes!
