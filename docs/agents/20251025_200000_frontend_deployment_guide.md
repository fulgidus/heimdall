# Frontend Deployment Guide

**Date**: 2025-10-25  
**Phase**: Phase 7 - Frontend Build & Deployment  
**Version**: 1.0.0

## Overview

This guide covers the complete build, deployment, and optimization strategies for the Heimdall frontend application.

## Table of Contents

1. [Build Strategies](#build-strategies)
2. [Environment Configuration](#environment-configuration)
3. [Docker Build & Run](#docker-build--run)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)
6. [Monitoring](#monitoring)

---

## Build Strategies

### Development Build

For local development with hot-reload:

```bash
# Using npm
cd frontend
npm run dev

# Using build script
./scripts/serve-frontend.sh development 5173
```

**Features**:
- Hot module replacement (HMR)
- Source maps enabled
- Console logs preserved
- Fast rebuild times

### Staging Build

For pre-production testing:

```bash
# Using build script
./scripts/build-frontend.sh staging

# Manual build
cd frontend
npm ci
npm run build -- --mode staging
```

**Features**:
- Minified assets
- Source maps enabled
- Some console logs preserved
- Production-like performance

### Production Build

For production deployment:

```bash
# Using build script
./scripts/build-frontend.sh production

# Manual build
cd frontend
npm ci
npm run build -- --mode production
```

**Features**:
- Full minification with Terser
- Console logs removed
- Source maps enabled (for debugging)
- Optimized chunk splitting
- Asset compression

**Build Output**:
- `dist/` - Production-ready static files
- `dist/assets/js/` - JavaScript chunks
- `dist/assets/css/` - CSS files
- `dist/manifest.json` - Build metadata

---

## Environment Configuration

### Environment Files

Create environment-specific files:

```bash
frontend/.env.development
frontend/.env.staging
frontend/.env.production
```

### Environment Variables

**Required**:
- `VITE_API_URL` - Backend API endpoint
- `VITE_MAPBOX_TOKEN` - Mapbox GL token

**Optional**:
- `VITE_LOG_LEVEL` - Logging level (debug|info|warn|error)
- `VITE_ANALYTICS` - Enable analytics (true|false)
- `VITE_FEATURE_REALTIME` - Enable real-time updates
- `VITE_FEATURE_AUTH` - Enable authentication
- `VITE_FEATURE_FILTERS` - Enable advanced filters

**Example `.env.production`**:

```env
VITE_API_URL=https://api.heimdall.io
VITE_MAPBOX_TOKEN=pk.your_production_token
VITE_LOG_LEVEL=warn
VITE_ANALYTICS=true
VITE_FEATURE_REALTIME=true
VITE_FEATURE_AUTH=true
VITE_FEATURE_FILTERS=true
```

### Configuration Module

Access configuration in code:

```typescript
import config from '@/config';

console.log(config.apiUrl);        // http://localhost:8000
console.log(config.environment);   // development
console.log(config.enableFeatures.realtimeUpdates); // true
```

---

## Docker Build & Run

### Build Docker Image

```bash
# Build frontend image
cd frontend
docker build -t heimdall-frontend:latest .

# Build with specific tag
docker build -t heimdall-frontend:v1.0.0 .
```

**Build Process**:
1. **Stage 1 (Builder)**: Install dependencies and build app
2. **Stage 2 (Nginx)**: Copy built assets and serve with nginx

**Image Size**: ~50MB (alpine-based)

### Run Docker Container

```bash
# Run container (standalone)
docker run -d \
  --name heimdall-frontend \
  -p 3000:80 \
  heimdall-frontend:latest

# Run with environment variables
docker run -d \
  --name heimdall-frontend \
  -p 3000:80 \
  -e VITE_API_URL=http://localhost:8000 \
  heimdall-frontend:latest
```

### Docker Compose

Start all services including frontend:

```bash
# Start all services
docker-compose up -d

# Start only frontend
docker-compose up -d frontend

# Rebuild and start
docker-compose up -d --build frontend
```

**Frontend accessible at**: http://localhost:3000

### Health Check

```bash
# Check container health
docker ps | grep frontend

# Test health endpoint
curl http://localhost:3000/health
# Output: healthy

# View logs
docker logs heimdall-frontend
```

---

## Performance Optimization

### Build Optimization

**Vite Configuration** (`vite.config.ts`):

- **Terser minification**: Removes console logs, dead code
- **Chunk splitting**: Separates vendor, UI, and app code
- **Asset hashing**: Enables long-term caching
- **Source maps**: Production debugging support

**Bundle Analysis**:

```bash
npm run build:analyze
# Opens bundle visualizer in browser
```

**Target Metrics**:
- Total bundle size: < 500KB (gzipped)
- Initial load: < 200KB
- Largest chunk: < 150KB

### Nginx Optimization

**Compression** (`nginx.conf`):
- Gzip compression enabled
- Compression level: 6
- Compressed types: text/*, application/javascript, application/json

**Caching Strategy** (`default.conf`):
- Static assets (hashed): 1 year cache
- HTML files: No cache (must-revalidate)

**Security Headers**:
- `X-Frame-Options: SAMEORIGIN`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`

### Runtime Optimization

**Code Splitting**:
- React/React-DOM: `vendor` chunk
- Mapbox GL: `mapbox` chunk
- Charts: `charts` chunk
- UI components: `ui` chunk
- Data fetching: `data` chunk

**Lazy Loading**:

```typescript
// Lazy load pages
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analytics = lazy(() => import('./pages/Analytics'));
```

**Asset Optimization**:
- Images: WebP format, < 100KB each
- Icons: SVG sprite or icon font
- Fonts: WOFF2 format, subset if possible

---

## Troubleshooting

### Build Failures

**Issue**: `npm run build` fails with TypeScript errors

```bash
# Check TypeScript errors
npm run type-check

# Fix and rebuild
npm run build
```

**Issue**: Out of memory during build

```bash
# Increase Node memory
NODE_OPTIONS="--max-old-space-size=4096" npm run build
```

### Docker Build Issues

**Issue**: Docker build fails at npm install

```bash
# Clear npm cache
docker build --no-cache -t heimdall-frontend:latest .

# Check package-lock.json is committed
git status package-lock.json
```

**Issue**: Container starts but shows 404

```bash
# Check nginx logs
docker logs heimdall-frontend

# Verify dist directory exists
docker exec heimdall-frontend ls -la /usr/share/nginx/html
```

### Runtime Issues

**Issue**: API calls fail with CORS errors

**Solution**: Check `VITE_API_URL` points to correct backend

```bash
# Verify environment variable
docker exec heimdall-frontend env | grep VITE_API_URL
```

**Issue**: Mapbox map not loading

**Solution**: Verify `VITE_MAPBOX_TOKEN` is set correctly

```typescript
// Check in browser console
import config from '@/config';
console.log(config.mapboxToken);
```

### Performance Issues

**Issue**: Slow initial load

**Check**:
1. Bundle size: `npm run build` (check output sizes)
2. Network tab in DevTools (check resource loading)
3. Lighthouse audit: `npm run build && npm run preview`

**Solutions**:
- Implement lazy loading
- Optimize images
- Review chunk splitting

---

## Monitoring

### Build Metrics

**CI/CD Tracking**:
- Build time: Target < 5 minutes
- Bundle size: Tracked in PR comments
- Test coverage: Uploaded to Codecov

**Local Monitoring**:

```bash
# Build and analyze
npm run build
du -sh dist/

# Check individual chunks
ls -lh dist/assets/js/
```

### Production Metrics

**Nginx Logs**:

```bash
# Access logs
docker exec heimdall-frontend tail -f /var/log/nginx/access.log

# Error logs
docker exec heimdall-frontend tail -f /var/log/nginx/error.log
```

**Performance Monitoring**:
- Response time: < 100ms (95th percentile)
- Cache hit rate: > 80%
- Error rate: < 0.1%

**Health Monitoring**:

```bash
# Health check endpoint
curl http://localhost:3000/health

# Docker health status
docker inspect heimdall-frontend --format='{{.State.Health.Status}}'
```

### Alerts & Notifications

**Recommended Alerts**:
1. Container unhealthy for > 2 minutes
2. Error rate > 1%
3. Response time > 500ms
4. Disk usage > 80%

---

## CI/CD Integration

### GitHub Actions Workflow

**Workflow**: `.github/workflows/frontend-build.yml`

**Triggers**:
- Pull requests touching `frontend/**`
- Pushes to `main` or `develop` branches

**Jobs**:
1. **Build**: Install, lint, test, build
2. **Docker**: Build and tag Docker image (main/develop only)

**Artifacts**:
- Build output: `frontend-dist` (5 day retention)
- Coverage report: Uploaded to Codecov
- Bundle size: Commented on PR

### Deployment Pipeline

**Recommended Flow**:

1. **Development**: Push to feature branch
2. **PR**: Automated build + tests run
3. **Review**: Bundle size checked, tests pass
4. **Merge**: Merge to `develop`
5. **Staging**: Auto-deploy to staging environment
6. **Testing**: Manual QA on staging
7. **Production**: Merge to `main`, deploy to production

---

## Quick Reference

### Common Commands

```bash
# Development
npm run dev                    # Start dev server
npm run build                  # Production build
npm run preview               # Preview production build

# Testing
npm run lint                   # Lint code
npm run type-check            # TypeScript check
npm run test                   # Run tests
npm run test:coverage         # Generate coverage

# Scripts
./scripts/build-frontend.sh production    # Build for production
./scripts/serve-frontend.sh development   # Serve development

# Docker
docker-compose up -d frontend             # Start frontend
docker-compose logs -f frontend           # View logs
docker-compose restart frontend           # Restart service
```

### Environment Checklist

- [ ] `.env.production` created with production values
- [ ] `VITE_API_URL` points to production API
- [ ] `VITE_MAPBOX_TOKEN` is production token
- [ ] Feature flags configured correctly
- [ ] Build completes without errors
- [ ] Docker image builds successfully
- [ ] Health check responds correctly
- [ ] All pages load without errors

---

## Support

**Documentation**: `/docs/`  
**Issues**: https://github.com/fulgidus/heimdall/issues  
**Contact**: alessio.corsi@gmail.com

---

**Last Updated**: 2025-10-25  
**Maintainer**: fulgidus
