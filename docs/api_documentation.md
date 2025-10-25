# API Documentation

## Overview

Heimdall provides a comprehensive REST API for RF acquisition, signal processing, and localization tasks.

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.heimdall.example.com/api/v1
```

## Authentication

Currently, the API uses environment-based authentication. For production:

```bash
# All requests must include API key
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Tasks API

#### Submit RF Acquisition Task

```http
POST /tasks/rf-acquisition
Content-Type: application/json

{
  "frequencies": [145.500, 433.025],
  "duration": 60,
  "bandwidth": 2400,
  "name": "Test acquisition"
}
```

**Response (201 Created)**:
```json
{
  "id": "task-12345",
  "status": "submitted",
  "created_at": "2025-10-22T10:30:00Z",
  "frequencies": [145.500, 433.025],
  "duration": 60,
  "estimated_completion": "2025-10-22T10:32:00Z"
}
```

#### Get Task Status

```http
GET /tasks/{task_id}
```

**Response (200 OK)**:
```json
{
  "id": "task-12345",
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
    },
    {
      "name": "ml_inference",
      "status": "pending"
    }
  ]
}
```

#### Cancel Task

```http
DELETE /tasks/{task_id}
```

**Response (200 OK)**:
```json
{
  "id": "task-12345",
  "status": "cancelled",
  "cancelled_at": "2025-10-22T10:31:00Z"
}
```

### Results API

#### Get Localization Results

```http
GET /results/{task_id}
```

**Response (200 OK)**:
```json
{
  "task_id": "task-12345",
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
  "processing_time_ms": 245
}
```

#### Get Historical Results

```http
GET /results?
  start_time=2025-10-22T00:00:00Z&
  end_time=2025-10-22T23:59:59Z&
  frequency=145.500&
  limit=100
```

### Health & Status

#### API Health

```http
GET /health
```

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T10:35:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "rabbitmq": "healthy",
    "minio": "healthy"
  }
}
```

#### WebSDR Status

```http
GET /health/websdrs
```

**Response (200 OK)**:
```json
{
  "stations": [
    {
      "name": "Giaveno",
      "status": "online",
      "last_check": "2025-10-22T10:34:50Z",
      "signal_strength": -85,
      "uptime_percentage": 99.8
    },
    {
      "name": "Torino",
      "status": "online",
      "last_check": "2025-10-22T10:34:45Z",
      "signal_strength": -88,
      "uptime_percentage": 99.5
    }
  ],
  "network_health": "excellent"
}
```

### Configuration API

#### Get WebSDR Configuration

```http
GET /config/websdrs
```

**Response (200 OK)**:
```json
{
  "stations": [
    {
      "id": "giaveno",
      "name": "Aquila di Giaveno",
      "url": "http://sdr1.ik1jns.it:8076/",
      "location": {
        "latitude": 45.02,
        "longitude": 7.29
      },
      "active": true
    }
  ]
}
```

#### Update Processing Parameters

```http
PUT /config/processing
Content-Type: application/json

{
  "bandwidth": 2400,
  "sample_rate": 12000,
  "timeout_seconds": 120,
  "max_concurrent_tasks": 10
}
```

## Error Responses

### 400 Bad Request

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

### 404 Not Found

```json
{
  "error": "task_not_found",
  "message": "Task with ID 'task-12345' not found"
}
```

### 500 Internal Server Error

```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred during processing",
  "request_id": "req-abc123"
}
```

## Rate Limiting

- **Default limit**: 100 requests/minute
- **Headers**:
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1698001500
  ```

## Pagination

List endpoints support pagination:

```http
GET /results?limit=50&offset=100
```

## WebSocket Connection

For real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/tasks/task-12345');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Task progress:', update.progress);
};
```

## Rate Limit Handling

When rate limited (429 Too Many Requests):

```python
import time
import requests

def make_request_with_retry(url):
    while True:
        response = requests.get(url)
        if response.status_code == 429:
            wait_time = int(response.headers['Retry-After'])
            time.sleep(wait_time)
            continue
        return response
```

---

**See Also**: [Usage Guide](./usage.md) | [Python Client Examples](./api_reference.md)
