# GitHub Actions Service Container Troubleshooting Guide

## Problem: Container Shows Help Text Instead of Starting

### Symptoms
```
Service container <name> failed.
Error: Failed to initialize container <image>
```

Container logs show help/usage text instead of the service running.

### Root Cause
GitHub Actions service containers **cannot specify custom commands** via the `options` field. The service will use the image's default CMD/ENTRYPOINT, which may just display help text.

### Solution Pattern

#### ❌ WRONG - Using Service Container with Custom Command Needed
```yaml
services:
  minio:
    image: minio/minio:latest
    options: >-
      --health-cmd "curl -f http://localhost:9000/minio/health/live"
    # ❌ No way to specify: server /data --console-address ":9001"
    ports:
      - 9000:9000
```

#### ✅ CORRECT - Manual Container Startup in Steps
```yaml
steps:
  - name: Start MinIO container
    run: |
      # Auto-discover GitHub Actions network
      NETWORK=$(docker network ls --format '{{.Name}}' | grep github || echo "bridge")
      
      # Start container with custom command
      docker run -d \
        --name minio \
        --network "$NETWORK" \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin \
        -p 9000:9000 \
        -p 9001:9001 \
        minio/minio:latest \
        server /data --console-address ":9001"  # ✅ Custom command works!
      
      # Wait for service to be ready
      timeout 60 bash -c 'until curl -sf http://localhost:9000/minio/health/live; do sleep 2; done'
```

## Common Services Requiring Custom Commands

| Service | Image | Required Command |
|---------|-------|-----------------|
| MinIO | `minio/minio:latest` | `server /data --console-address ":9001"` |
| Keycloak | `quay.io/keycloak/keycloak:23.0` | `start-dev` |
| Elasticsearch | `elasticsearch:8.x` | May need custom settings via env vars |
| Custom Apps | Various | Often need specific entrypoint args |

## Implementation Checklist

When converting from service container to manual startup:

- [ ] Remove service from `services:` section
- [ ] Add startup step after checkout
- [ ] Find/use correct network (GitHub Actions auto-creates one)
- [ ] Add all environment variables
- [ ] Specify proper command after image name
- [ ] Mount volumes if needed (`-v` flag)
- [ ] Map ports (`-p` flag)
- [ ] Add health check wait loop
- [ ] Add cleanup step (`if: always()`) to stop container

## Network Discovery

GitHub Actions creates a network for service containers. Find it with:
```bash
NETWORK=$(docker network ls --format '{{.Name}}' | grep github || echo "bridge")
```

This ensures manually started containers can communicate with service containers (PostgreSQL, Redis, etc.).

## Health Check Patterns

### HTTP Endpoint Check
```bash
timeout 60 bash -c 'until curl -sf http://localhost:9000/health; do sleep 2; done'
```

### TCP Port Check
```bash
timeout 60 bash -c 'until nc -z localhost 5432; do sleep 2; done'
```

### Custom Command Check
```bash
timeout 60 bash -c 'until docker exec mycontainer mycommand; do sleep 2; done'
```

## Cleanup Pattern

Always cleanup manually started containers:
```yaml
- name: Cleanup containers
  if: always()  # Runs even if previous steps fail
  run: |
    docker stop minio keycloak 2>/dev/null || true
    docker rm minio keycloak 2>/dev/null || true
```

## Volume Mounting

When you need to mount files from the repository:
```bash
-v ${{ github.workspace }}/path/to/config:/container/path:ro
```

Note: Use `:ro` for read-only when appropriate.

## Related Issues

- [GitHub Actions Documentation - Service Containers](https://docs.github.com/en/actions/using-containerized-services/about-service-containers)
- Heimdall Project: [Integration Tests Fix](20251024_235900_integration_tests_fix.md)

## Examples from Heimdall Project

See `.github/workflows/integration-tests.yml` for complete working examples of:
- MinIO with custom command
- Keycloak with custom command and volume mount
- Network discovery
- Health check loops
- Cleanup steps

## Summary

**Key Takeaway**: If a container needs a custom command to start properly, you **cannot use it as a GitHub Actions service container**. Instead, start it manually in a workflow step where you have full control over the `docker run` command.
