# API Reference

Comprehensive reference for Heimdall REST API endpoints.

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.heimdall.example.com/api/v1
```

## Authentication

Include API key in Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/api/v1/health
```

## Task Endpoints

### Submit RF Acquisition Task

Submit a new RF acquisition task to the system.

**Endpoint**: `POST /tasks/rf-acquisition`

**Request Body**:
```json
{
  "frequencies": [145.500, 433.025],
  "duration": 60,
  "bandwidth": 2400,
  "name": "Test acquisition",
  "priority": "normal"
}
```

**Parameters**:
- `frequencies` (array): Target frequencies in MHz
- `duration` (integer): Acquisition duration in seconds (30-300)
- `bandwidth` (integer, optional): Bandwidth in Hz (default: 2400)
- `name` (string, optional): Human-readable task name
- `priority` (string, optional): "low", "normal", "high" (default: "normal")

**Response** (201):
```json
{
  "id": "task-abc123",
  "status": "submitted",
  "created_at": "2025-10-22T10:30:00Z",
  "frequencies": [145.500, 433.025],
  "duration": 60,
  "estimated_completion": "2025-10-22T10:32:00Z"
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/tasks/rf-acquisition \
  -H "Content-Type: application/json" \
  -d '{
    "frequencies": [145.500],
    "duration": 60
  }'
```

---

### Get Task Status

Retrieve current status of an RF acquisition task.

**Endpoint**: `GET /tasks/{task_id}`

**Path Parameters**:
- `task_id` (string): Task identifier

**Response** (200):
```json
{
  "id": "task-abc123",
  "status": "processing",
  "progress": 45,
  "started_at": "2025-10-22T10:30:15Z",
  "estimated_completion": "2025-10-22T10:32:00Z",
  "stages": [
    {
      "name": "rf_acquisition",
      "status": "completed",
      "duration_seconds": 60.5
    },
    {
      "name": "signal_processing",
      "status": "in_progress",
      "progress": 45
    }
  ]
}
```

**Possible Status Values**:
- `submitted`: Task submitted, waiting in queue
- `processing`: Task actively processing
- `completed`: Task completed successfully
- `failed`: Task failed with error
- `cancelled`: Task cancelled by user

**Example**:
```bash
curl http://localhost:8000/api/v1/tasks/task-abc123
```

---

### List Tasks

Get list of tasks with filtering.

**Endpoint**: `GET /tasks`

**Query Parameters**:
- `status` (string, optional): Filter by status
- `limit` (integer, optional): Max results (default: 50)
- `offset` (integer, optional): Pagination offset (default: 0)
- `sort` (string, optional): "created_at", "-created_at" (default: "-created_at")

**Response** (200):
```json
{
  "items": [
    {
      "id": "task-abc123",
      "status": "completed",
      "created_at": "2025-10-22T10:30:00Z"
    }
  ],
  "total": 150,
  "limit": 50,
  "offset": 0
}
```

**Example**:
```bash
curl "http://localhost:8000/api/v1/tasks?status=completed&limit=10"
```

---

### Cancel Task

Cancel an in-progress or queued task.

**Endpoint**: `DELETE /tasks/{task_id}`

**Path Parameters**:
- `task_id` (string): Task identifier

**Response** (200):
```json
{
  "id": "task-abc123",
  "status": "cancelled",
  "cancelled_at": "2025-10-22T10:31:00Z"
}
```

**Example**:
```bash
curl -X DELETE http://localhost:8000/api/v1/tasks/task-abc123
```

---

## Result Endpoints

### Get Localization Result

Retrieve localization result for a completed task.

**Endpoint**: `GET /results/{task_id}`

**Path Parameters**:
- `task_id` (string): Task identifier

**Response** (200):
```json
{
  "task_id": "task-abc123",
  "status": "completed",
  "location": {
    "latitude": 45.0736,
    "longitude": 7.5836,
    "altitude_m": null,
    "uncertainty_m": 45.2
  },
  "signal_measurements": [
    {
      "station_name": "Giaveno",
      "frequency": 145.500,
      "signal_strength_dbm": -85.5,
      "bearing_degrees": 120.5,
      "quality": 0.92
    }
  ],
  "confidence": 0.85,
  "processing_time_ms": 245,
  "completed_at": "2025-10-22T10:32:15Z"
}
```

**Example**:
```bash
curl http://localhost:8000/api/v1/results/task-abc123
```

---

### Get Historical Results

Query historical results with filtering.

**Endpoint**: `GET /results`

**Query Parameters**:
- `start_time` (string, ISO 8601): Start time filter
- `end_time` (string, ISO 8601): End time filter
- `frequency` (number, optional): Filter by frequency
- `min_confidence` (number, optional): Minimum confidence (0-1)
- `limit` (integer, optional): Max results (default: 100)
- `offset` (integer, optional): Pagination offset

**Response** (200):
```json
{
  "items": [...],
  "total": 250,
  "limit": 100,
  "offset": 0
}
```

**Example**:
```bash
curl "http://localhost:8000/api/v1/results?start_time=2025-10-22T00:00:00Z&end_time=2025-10-22T23:59:59Z&frequency=145.500"
```

---

## Health & Status Endpoints

### API Health

Check API and dependent services health.

**Endpoint**: `GET /health`

**Response** (200):
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T10:35:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "rabbitmq": "healthy",
    "minio": "healthy"
  },
  "version": "1.0.0"
}
```

---

### WebSDR Network Status

Check status of WebSDR receiver network.

**Endpoint**: `GET /health/websdrs`

**Response** (200):
```json
{
  "stations": [
    {
      "name": "Giaveno",
      "status": "online",
      "last_check": "2025-10-22T10:34:50Z",
      "signal_strength": -85,
      "uptime_percentage": 99.8
    }
  ],
  "network_health": "excellent",
  "active_stations": 7,
  "total_stations": 7
}
```

---

### Database Health

Check database connectivity and performance.

**Endpoint**: `GET /health/database`

**Response** (200):
```json
{
  "status": "healthy",
  "connection_time_ms": 2.5,
  "query_time_ms": 15.3,
  "tables": {
    "signal_measurements": { "rows": 1250000 },
    "task_results": { "rows": 8500 }
  }
}
```

---

## Configuration Endpoints

### Get System Configuration

Retrieve current system configuration.

**Endpoint**: `GET /config`

**Response** (200):
```json
{
  "acquisition_settings": {
    "default_duration": 60,
    "default_bandwidth": 2400,
    "max_concurrent_tasks": 10
  },
  "processing_settings": {
    "enable_gpu": true,
    "batch_size": 32
  },
  "storage_settings": {
    "retention_days": 90
  }
}
```

---

### Update Configuration

Update system configuration (admin only).

**Endpoint**: `PUT /config`

**Request Body**:
```json
{
  "acquisition_settings": {
    "max_concurrent_tasks": 15
  }
}
```

**Response** (200):
```json
{
  "status": "updated",
  "message": "Configuration updated successfully"
}
```

---

## Error Responses

### 400 Bad Request

Invalid request parameters.

```json
{
  "error": "invalid_parameters",
  "message": "Frequency must be between 0 and 6000 MHz",
  "details": {
    "field": "frequencies[0]",
    "value": 8000
  }
}
```

---

### 404 Not Found

Resource not found.

```json
{
  "error": "task_not_found",
  "message": "Task with ID 'task-abc123' not found"
}
```

---

### 429 Too Many Requests

Rate limit exceeded.

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Max 100 per minute.",
  "retry_after": 30
}
```

---

### 500 Internal Server Error

Server error.

```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred",
  "request_id": "req-abc123"
}
```

---

## Rate Limiting

API requests are rate-limited:

- **Default**: 100 requests/minute
- **Burst**: 200 requests/minute for up to 10 seconds

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1698001500
```

---

## Pagination

List endpoints support cursor-based pagination:

```bash
# Get first 50 items
curl "http://localhost:8000/api/v1/tasks?limit=50"

# Get next page
curl "http://localhost:8000/api/v1/tasks?limit=50&offset=50"
```

---

## WebSocket Connection

For real-time task updates:

**Endpoint**: `WS /ws/tasks/{task_id}`

**Example**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/tasks/task-abc123');

ws.onopen = () => {
  console.log('Connected to task updates');
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Progress:', update.progress);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

---

**Last Updated**: October 2025

**See Also**: [API Documentation](./api_documentation.md) | [Usage Guide](./usage.md)
