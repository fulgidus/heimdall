# Health Check & Readiness Probes Documentation

## Overview

This document describes the comprehensive health check system implemented across all Heimdall microservices. The system provides standardized health monitoring with dependency validation for Kubernetes readiness/liveness probes and Docker healthchecks.

## Architecture

### Health Check Endpoints

Each service exposes three health check endpoints:

1. **`/health`** - Liveness Probe
   - Basic service health without dependency checks
   - Fast response (< 100ms)
   - Returns 200 if service is alive
   - Used by Kubernetes liveness probes

2. **`/health/detailed`** - Detailed Health Status
   - Comprehensive health with dependency status
   - Checks all registered dependencies
   - Returns dependency response times and error messages
   - Used for monitoring and debugging

3. **`/ready`** - Readiness Probe
   - Validates service can handle requests
   - Checks critical dependencies
   - Returns 200 if ready, 503 if not ready
   - Used by Kubernetes readiness probes and Docker healthchecks

4. **`/startup`** - Startup Probe
   - Checks if service has finished starting
   - Currently aliases to `/ready`
   - Used by Kubernetes startup probes

## Health Check Utilities

### Core Components

Located in `services/common/health.py`:

#### HealthStatus Enum
```python
class HealthStatus(str, Enum):
    UP = "up"           # Service/dependency is healthy
    DOWN = "down"       # Service/dependency is unavailable
    DEGRADED = "degraded"  # Service is partially functional
    UNKNOWN = "unknown"    # Status cannot be determined
```

#### DependencyHealth
Represents the health status of a single dependency:
- `name`: Dependency identifier (e.g., "database", "redis")
- `status`: HealthStatus enum value
- `response_time_ms`: Time taken to check the dependency
- `error_message`: Optional error details if check failed

#### HealthCheckResponse
Complete health check result:
- `status`: Overall service health status
- `service_name`: Name of the service
- `version`: Service version
- `timestamp`: Check timestamp (ISO 8601)
- `uptime_seconds`: Service uptime
- `dependencies`: List of DependencyHealth objects
- `ready`: Boolean indicating if service is ready

#### HealthChecker
Main health check orchestrator:
```python
# Initialize
health_checker = HealthChecker("service-name", "1.0.0")

# Register dependencies
health_checker.register_dependency("database", check_db_func)
health_checker.register_dependency("redis", check_redis_func)

# Check all dependencies
result = await health_checker.check_all()
```

## Dependency Checkers

Located in `services/common/dependency_checkers.py`:

### Available Checkers

1. **`check_postgresql(connection_string)`**
   - Validates PostgreSQL database connectivity
   - Executes `SELECT 1` query
   - Supports both asyncpg and psycopg2

2. **`check_redis(redis_url)`**
   - Validates Redis connectivity
   - Executes PING command
   - Supports async and sync Redis clients

3. **`check_rabbitmq(broker_url)`**
   - Validates RabbitMQ broker connectivity
   - Creates and closes a test channel
   - Supports aio_pika and pika

4. **`check_minio(endpoint, access_key, secret_key, secure=False)`**
   - Validates MinIO/S3 connectivity
   - Lists buckets to verify access
   - Handles http:// and https:// prefixes

5. **`check_celery(broker_url, backend_url=None)`**
   - Validates Celery broker and backend
   - Combines RabbitMQ and Redis checks

## Service-Specific Configuration

### RF Acquisition Service

**Dependencies Checked:**
- PostgreSQL database
- Redis cache
- Celery (RabbitMQ + Redis backend)
- MinIO object storage

**Endpoints:**
- `/health` - Liveness (basic)
- `/health/detailed` - Full status with all 4 dependencies
- `/ready` - Readiness (validates all dependencies)
- `/startup` - Startup probe

**Docker Healthcheck:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/ready || exit 1
```

### Inference Service

**Dependencies Checked:**
- PostgreSQL database
- Redis cache

**Endpoints:**
- `/health` - Liveness (basic)
- `/health/detailed` - Full status with both dependencies
- `/ready` - Readiness (validates both dependencies)
- `/startup` - Startup probe

**Docker Healthcheck:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/ready || exit 1
```

### Training Service

**Dependencies Checked:**
- PostgreSQL database
- MinIO object storage

**Endpoints:**
- `/health` - Liveness (basic)
- `/health/detailed` - Full status with both dependencies
- `/ready` - Readiness (validates both dependencies)
- `/startup` - Startup probe

**Docker Healthcheck:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/ready || exit 1
```

### API Gateway

**Dependencies Checked:**
- RF Acquisition service
- Inference service
- Training service
- Data Ingestion service

**Endpoints:**
- `/health` - Liveness (basic)
- `/health/detailed` - Full status with all backend services
- `/ready` - Readiness (always ready, but reports backend status)
- `/startup` - Startup probe

**Note:** API Gateway is always "ready" even if backends are down - it will proxy requests and return appropriate errors.

**Docker Healthcheck:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1
```

### Data Ingestion Service

**Dependencies Checked:**
- PostgreSQL database

**Endpoints:**
- `/health` - Liveness (basic)
- `/health/detailed` - Full status with database dependency
- `/ready` - Readiness (validates database)
- `/startup` - Startup probe

**Docker Healthcheck:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8004/ready || exit 1
```

## Response Examples

### Liveness Probe (`/health`)
```json
{
  "status": "healthy",
  "service": "rf-acquisition",
  "version": "0.1.0",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### Detailed Health (`/health/detailed`)
```json
{
  "status": "up",
  "service_name": "rf-acquisition",
  "version": "0.1.0",
  "timestamp": "2025-01-15T10:30:00.000Z",
  "uptime_seconds": 3600,
  "ready": true,
  "dependencies": [
    {
      "name": "database",
      "status": "up",
      "response_time_ms": "15.23",
      "error_message": null
    },
    {
      "name": "redis",
      "status": "up",
      "response_time_ms": "5.67",
      "error_message": null
    },
    {
      "name": "celery",
      "status": "up",
      "response_time_ms": "45.89",
      "error_message": null
    },
    {
      "name": "minio",
      "status": "down",
      "response_time_ms": "0.00",
      "error_message": "MinIO health check failed: Connection refused"
    }
  ]
}
```

### Readiness Probe (`/ready`)
```json
{
  "ready": true,
  "service": "rf-acquisition",
  "dependencies": [
    {
      "name": "database",
      "status": "up",
      "response_time_ms": "12.34",
      "error_message": null
    }
  ]
}
```

## Kubernetes Integration

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rf-acquisition
spec:
  template:
    spec:
      containers:
      - name: rf-acquisition
        image: heimdall/rf-acquisition:latest
        ports:
        - containerPort: 8001
        
        # Liveness probe - checks if container is alive
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Readiness probe - checks if ready to receive traffic
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Startup probe - checks if application has started
        startupProbe:
          httpGet:
            path: /startup
            port: 8001
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 12  # 2 minutes max startup time
```

## Docker Compose Integration

Health checks are configured in `docker-compose.yml` for all services:

```yaml
services:
  rf-acquisition:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
      minio:
        condition: service_healthy
```

## Development vs Production

### Development Mode
- More lenient readiness checks
- Non-critical dependencies don't block startup
- Warnings logged but service stays ready
- Controlled via `ENVIRONMENT=development`

### Production Mode
- Strict dependency validation
- All dependencies must be healthy
- Service not ready until all checks pass
- Controlled via `ENVIRONMENT=production`

## Testing

### Unit Tests
Located in `services/common/tests/`:
- `test_health.py` - 13 tests for health utilities
- `test_dependency_checkers.py` - 11 tests for dependency checkers

### Integration Tests
Located in `services/*/tests/integration/`:
- `test_health_endpoints.py` - Tests for all health endpoints
- Validates response format and status codes
- Tests dependency health checking

### Running Tests
```bash
# Run all health check tests
pytest services/common/tests/ -v

# Run service-specific integration tests
pytest services/rf-acquisition/tests/integration/test_health_endpoints.py -v
```

## Monitoring

### Prometheus Metrics
Health check endpoints can be scraped by Prometheus:

```yaml
scrape_configs:
  - job_name: 'heimdall-services'
    metrics_path: '/health/detailed'
    scrape_interval: 30s
    static_configs:
      - targets:
        - 'rf-acquisition:8001'
        - 'inference:8003'
        - 'training:8002'
```

### Dashboard Integration
The frontend dashboard uses `/health/detailed` to display:
- Service status indicators
- Dependency health visualization
- Response time metrics
- Error messages and alerts

## Troubleshooting

### Service Reports as Unhealthy

1. **Check detailed health status:**
   ```bash
   curl http://localhost:8001/health/detailed | jq
   ```

2. **Identify failing dependency:**
   - Look for `"status": "down"` in dependencies array
   - Check `error_message` field for details

3. **Common issues:**
   - Database connection refused: Check PostgreSQL is running
   - Redis timeout: Verify Redis is accessible
   - RabbitMQ connection failed: Check broker configuration
   - MinIO unreachable: Verify MinIO URL and credentials

### Container Keeps Restarting

1. **Check Docker logs:**
   ```bash
   docker logs heimdall-rf-acquisition
   ```

2. **Verify dependencies are healthy:**
   ```bash
   docker compose ps
   ```

3. **Test health endpoint manually:**
   ```bash
   docker compose exec rf-acquisition curl -f http://localhost:8001/ready
   ```

### Slow Health Checks

- Check `response_time_ms` in detailed health response
- Network latency to dependencies
- Overloaded database or cache
- Consider increasing timeout values

## Best Practices

1. **Use `/health` for liveness probes**
   - Fast, simple check
   - Doesn't validate dependencies
   - Prevents unnecessary restarts

2. **Use `/ready` for readiness probes**
   - Validates critical dependencies
   - Prevents routing traffic to unhealthy instances
   - Returns 503 when not ready

3. **Set appropriate timeouts**
   - `start_period`: Allow time for slow-starting services
   - `interval`: Balance between responsiveness and overhead
   - `timeout`: Account for network latency

4. **Monitor dependency health**
   - Track response times over time
   - Alert on degraded performance
   - Use metrics for capacity planning

5. **Test in development**
   - Verify health checks work with local infrastructure
   - Test failure scenarios
   - Validate readiness behavior

## Migration Guide

### From Old Health Checks

If migrating from basic `/health` endpoints:

1. Services now expose three endpoints: `/health`, `/health/detailed`, `/ready`
2. Update Docker healthchecks to use `/ready` instead of `/health`
3. Update Kubernetes probes to use appropriate endpoints
4. Dependencies are now automatically validated
5. Response format changed to include dependency status

### Backwards Compatibility

- `/health` endpoint maintains original simple response format
- New `/health/detailed` provides comprehensive status
- Old monitoring can continue using `/health`
- Gradual migration to `/ready` for readiness checks

## Future Enhancements

1. **Custom dependency checkers**
   - Add service-specific health checks
   - Validate business logic constraints

2. **Circuit breakers**
   - Automatically disable failing dependencies
   - Graceful degradation of functionality

3. **Metrics export**
   - Prometheus metrics endpoint
   - Health check timing histograms

4. **Configurable thresholds**
   - Response time warnings
   - Dependency criticality levels

5. **Historical health tracking**
   - Store health check results
   - Trend analysis and reporting
