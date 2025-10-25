# Heimdall SDR - API Documentation

## Overview

The Heimdall SDR API provides comprehensive access to radio signal data, anomaly detection results, and system management capabilities. The API follows RESTful principles and includes real-time WebSocket endpoints for live data streaming.

## Base URL

```
Production: https://api.heimdall.example.com
Development: http://localhost:8000
```

## Authentication

### JWT Bearer Token

All API endpoints require authentication via JWT Bearer tokens, except for the health check endpoint.

```http
Authorization: Bearer <jwt_token>
```

### Obtaining Access Token

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "permissions": ["signals:read", "signals:write"],
  "user_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Refresh Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "refresh_token_string"
}
```

## API Versioning

The API uses URL path versioning:
- Current version: `v1`
- Backward compatibility: Maintained for at least 2 major versions
- Deprecation notice: 6 months minimum before removal

## Rate Limiting

- **Standard users**: 100 requests per minute
- **Premium users**: 1000 requests per minute
- **WebSocket connections**: 5 concurrent connections per user

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Data Models

### SignalDetection

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "websdr_station_id": "twente-nl",
  "frequency_hz": 14235000,
  "signal_strength_db": -73.5,
  "bandwidth_hz": 2400,
  "modulation_type": "USB",
  "signal_features": {
    "spectral_centroid": 1500.0,
    "bandwidth": 2400,
    "snr_estimate": 15.3,
    "peak_frequency": 14235000
  },
  "anomaly_score": 0.15,
  "is_anomaly": false,
  "detection_timestamp": "2024-01-15T14:30:00Z",
  "processing_metadata": {
    "processing_time_ms": 125,
    "model_version": "1.2.0",
    "quality_score": 0.89
  },
  "created_at": "2024-01-15T14:30:05Z",
  "updated_at": "2024-01-15T14:30:05Z"
}
```

### WebSDRStation

```json
{
  "id": "twente-nl",
  "name": "University of Twente WebSDR",
  "url": "http://websdr.ewi.utwente.nl:8901/",
  "location": "Enschede, Netherlands",
  "latitude": 52.2387,
  "longitude": 6.8509,
  "frequency_min": 0,
  "frequency_max": 29000000,
  "status": "active",
  "api_config": {
    "api_type": "http_streaming",
    "poll_interval": 1.0,
    "timeout": 30
  },
  "last_seen": "2024-01-15T14:30:00Z",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T14:30:00Z"
}
```

### AnomalyEvent

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "detection_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "unusual_signal_pattern",
  "severity": "medium",
  "description": "Detected unusual modulation pattern on 14.235 MHz",
  "metadata": {
    "confidence": 0.87,
    "affected_frequencies": [14235000, 14236000],
    "duration_seconds": 15.6,
    "geographic_correlation": ["twente-nl", "hackgreen-uk"]
  },
  "acknowledged": false,
  "acknowledged_by": null,
  "acknowledged_at": null,
  "created_at": "2024-01-15T14:30:10Z"
}
```

## REST API Endpoints

### Health Check

#### GET /health

Check API service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "rabbitmq": "healthy",
    "websdr_collectors": "healthy"
  }
}
```

### Signal Detections

#### GET /api/v1/signals/detections

Retrieve signal detections with filtering and pagination.

**Parameters:**
- `frequency_min` (integer, required): Minimum frequency in Hz
- `frequency_max` (integer, required): Maximum frequency in Hz
- `time_start` (string, required): Start time in ISO 8601 format
- `time_end` (string, required): End time in ISO 8601 format
- `websdr_station` (string, optional): Filter by station ID
- `anomaly_threshold` (float, optional): Minimum anomaly score (0.0-1.0)
- `modulation_type` (string, optional): Filter by modulation type
- `page` (integer, optional, default=1): Page number
- `page_size` (integer, optional, default=100): Items per page (max 1000)
- `sort_by` (string, optional, default="detection_timestamp"): Sort field
- `sort_order` (string, optional, default="desc"): Sort order (asc/desc)

**Example Request:**
```http
GET /api/v1/signals/detections?frequency_min=14000000&frequency_max=15000000&time_start=2024-01-15T12:00:00Z&time_end=2024-01-15T18:00:00Z&websdr_station=twente-nl&page=1&page_size=50
```

**Response:**
```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "websdr_station_id": "twente-nl",
      "frequency_hz": 14235000,
      "signal_strength_db": -73.5,
      "bandwidth_hz": 2400,
      "modulation_type": "USB",
      "anomaly_score": 0.15,
      "is_anomaly": false,
      "detection_timestamp": "2024-01-15T14:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_items": 1247,
    "total_pages": 25,
    "has_next": true,
    "has_previous": false
  },
  "filters_applied": {
    "frequency_range": [14000000, 15000000],
    "time_range": ["2024-01-15T12:00:00Z", "2024-01-15T18:00:00Z"],
    "station": "twente-nl"
  }
}
```

#### GET /api/v1/signals/detections/{detection_id}

Retrieve a specific signal detection by ID.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "websdr_station_id": "twente-nl",
  "frequency_hz": 14235000,
  "signal_strength_db": -73.5,
  "bandwidth_hz": 2400,
  "modulation_type": "USB",
  "signal_features": {
    "spectral_centroid": 1500.0,
    "bandwidth": 2400,
    "snr_estimate": 15.3,
    "peak_frequency": 14235000
  },
  "anomaly_score": 0.15,
  "is_anomaly": false,
  "detection_timestamp": "2024-01-15T14:30:00Z",
  "processing_metadata": {
    "processing_time_ms": 125,
    "model_version": "1.2.0",
    "quality_score": 0.89
  },
  "created_at": "2024-01-15T14:30:05Z",
  "updated_at": "2024-01-15T14:30:05Z"
}
```

#### POST /api/v1/signals/analyze

Analyze uploaded signal data for anomalies.

**Request Body:**
```json
{
  "signal_data": {
    "samples": [0.1, 0.2, -0.1, 0.3],
    "sample_rate": 48000,
    "frequency": 14235000,
    "duration": 1.0
  },
  "analysis_config": {
    "fft_size": 1024,
    "overlap": 0.5,
    "window_function": "hamming"
  }
}
```

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440002",
  "signal_features": {
    "spectral_centroid": 1500.0,
    "bandwidth": 2400,
    "snr_estimate": 15.3,
    "peak_frequency": 14235000
  },
  "anomaly_score": 0.15,
  "is_anomaly": false,
  "confidence": 0.92,
  "processing_time_ms": 125,
  "model_version": "1.2.0"
}
```

### WebSDR Stations

#### GET /api/v1/stations

Retrieve list of WebSDR stations.

**Parameters:**
- `status` (string, optional): Filter by status (active/inactive/maintenance)
- `location` (string, optional): Filter by location/country
- `frequency_range` (string, optional): Filter by frequency coverage

**Response:**
```json
{
  "data": [
    {
      "id": "twente-nl",
      "name": "University of Twente WebSDR",
      "url": "http://websdr.ewi.utwente.nl:8901/",
      "location": "Enschede, Netherlands",
      "latitude": 52.2387,
      "longitude": 6.8509,
      "frequency_min": 0,
      "frequency_max": 29000000,
      "status": "active",
      "last_seen": "2024-01-15T14:30:00Z"
    }
  ],
  "total_count": 7,
  "active_count": 6,
  "inactive_count": 1
}
```

#### GET /api/v1/stations/{station_id}

Retrieve detailed information about a specific WebSDR station.

**Response:**
```json
{
  "id": "twente-nl",
  "name": "University of Twente WebSDR",
  "url": "http://websdr.ewi.utwente.nl:8901/",
  "location": "Enschede, Netherlands",
  "latitude": 52.2387,
  "longitude": 6.8509,
  "frequency_min": 0,
  "frequency_max": 29000000,
  "status": "active",
  "api_config": {
    "api_type": "http_streaming",
    "poll_interval": 1.0,
    "timeout": 30
  },
  "statistics": {
    "uptime_percentage": 98.5,
    "average_response_time_ms": 150,
    "total_detections_24h": 15423,
    "anomalies_detected_24h": 12
  },
  "last_seen": "2024-01-15T14:30:00Z",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T14:30:00Z"
}
```

#### GET /api/v1/stations/{station_id}/status

Get real-time status of a WebSDR station.

**Response:**
```json
{
  "station_id": "twente-nl",
  "status": "active",
  "connection_status": "connected",
  "last_data_received": "2024-01-15T14:30:00Z",
  "response_time_ms": 145,
  "current_listeners": 23,
  "signal_quality": 0.92,
  "frequency_coverage": {
    "current_frequency": 14235000,
    "active_bands": ["20m", "40m", "80m"]
  }
}
```

### Anomaly Events

#### GET /api/v1/anomalies

Retrieve anomaly events with filtering.

**Parameters:**
- `severity` (string, optional): Filter by severity (low/medium/high/critical)
- `event_type` (string, optional): Filter by event type
- `time_start` (string, required): Start time in ISO 8601 format
- `time_end` (string, required): End time in ISO 8601 format
- `acknowledged` (boolean, optional): Filter by acknowledgment status
- `station_id` (string, optional): Filter by WebSDR station
- `page` (integer, optional, default=1): Page number
- `page_size` (integer, optional, default=50): Items per page

**Response:**
```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "detection_id": "550e8400-e29b-41d4-a716-446655440000",
      "event_type": "unusual_signal_pattern",
      "severity": "medium",
      "description": "Detected unusual modulation pattern on 14.235 MHz",
      "acknowledged": false,
      "created_at": "2024-01-15T14:30:10Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_items": 234,
    "total_pages": 5
  },
  "summary": {
    "total_anomalies": 234,
    "by_severity": {
      "critical": 2,
      "high": 15,
      "medium": 67,
      "low": 150
    },
    "acknowledged_count": 189,
    "unacknowledged_count": 45
  }
}
```

#### POST /api/v1/anomalies/{anomaly_id}/acknowledge

Acknowledge an anomaly event.

**Request Body:**
```json
{
  "notes": "Investigated - confirmed false positive due to equipment test"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "acknowledged": true,
  "acknowledged_by": "550e8400-e29b-41d4-a716-446655440003",
  "acknowledged_at": "2024-01-15T14:35:00Z",
  "notes": "Investigated - confirmed false positive due to equipment test"
}
```

### Statistics and Analytics

#### GET /api/v1/statistics/summary

Get overall system statistics.

**Parameters:**
- `time_period` (string, optional, default="24h"): Time period (1h/6h/24h/7d/30d)

**Response:**
```json
{
  "time_period": "24h",
  "generated_at": "2024-01-15T14:30:00Z",
  "signal_detections": {
    "total": 125430,
    "by_station": {
      "twente-nl": 18456,
      "hackgreen-uk": 15234,
      "bratislava-sk": 12987
    },
    "by_frequency_band": {
      "hf": 89234,
      "vhf": 23456,
      "uhf": 12740
    }
  },
  "anomalies": {
    "total": 156,
    "by_severity": {
      "critical": 2,
      "high": 15,
      "medium": 67,
      "low": 72
    },
    "detection_rate": 0.0012
  },
  "system_health": {
    "uptime_percentage": 99.8,
    "average_processing_time_ms": 125,
    "active_stations": 6,
    "total_stations": 7
  }
}
```

#### GET /api/v1/statistics/frequency-usage

Get frequency usage statistics.

**Parameters:**
- `frequency_min` (integer, optional): Minimum frequency in Hz
- `frequency_max` (integer, optional): Maximum frequency in Hz
- `time_period` (string, optional, default="24h"): Time period
- `granularity` (string, optional, default="1h"): Data granularity (5m/15m/1h/6h)

**Response:**
```json
{
  "frequency_range": [14000000, 15000000],
  "time_period": "24h",
  "granularity": "1h",
  "data": [
    {
      "timestamp": "2024-01-15T13:00:00Z",
      "frequency_bins": [
        {
          "frequency_start": 14000000,
          "frequency_end": 14100000,
          "signal_count": 234,
          "average_strength": -75.2,
          "anomaly_count": 2
        }
      ]
    }
  ],
  "popular_frequencies": [
    {
      "frequency": 14235000,
      "signal_count": 1456,
      "percentage": 12.5
    }
  ]
}
```

### Machine Learning Models

#### GET /api/v1/models

Retrieve information about ML models.

**Response:**
```json
{
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440004",
      "name": "isolation-forest-v1",
      "version": "1.2.0",
      "model_type": "anomaly_detection",
      "status": "deployed",
      "metrics": {
        "accuracy": 0.94,
        "precision": 0.89,
        "recall": 0.92,
        "f1_score": 0.90
      },
      "deployment_date": "2024-01-10T10:00:00Z",
      "last_training_date": "2024-01-08T15:30:00Z"
    }
  ]
}
```

#### POST /api/v1/models/{model_id}/predict

Make predictions using a specific model.

**Request Body:**
```json
{
  "features": {
    "spectral_centroid": 1500.0,
    "bandwidth": 2400,
    "snr_estimate": 15.3,
    "signal_power": -65.2
  }
}
```

**Response:**
```json
{
  "model_id": "550e8400-e29b-41d4-a716-446655440004",
  "prediction": {
    "anomaly_score": 0.15,
    "is_anomaly": false,
    "confidence": 0.92
  },
  "processing_time_ms": 45,
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## WebSocket API

### Connection

Connect to the WebSocket endpoint for real-time data streaming:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/signals/live?token=jwt_token');
```

### Live Signal Data

Subscribe to real-time signal detections:

```javascript
// Send subscription message
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['signal_detections'],
  filters: {
    station_ids: ['twente-nl', 'hackgreen-uk'],
    frequency_range: [14000000, 15000000],
    anomaly_threshold: 0.5
  }
}));

// Receive real-time data
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Signal detection:', data);
};
```

**Message Format:**
```json
{
  "type": "signal_detection",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "websdr_station_id": "twente-nl",
    "frequency_hz": 14235000,
    "signal_strength_db": -73.5,
    "anomaly_score": 0.15,
    "detection_timestamp": "2024-01-15T14:30:00Z"
  },
  "timestamp": "2024-01-15T14:30:05Z"
}
```

### Live Anomaly Alerts

Subscribe to real-time anomaly notifications:

```javascript
// Subscribe to anomaly alerts
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['anomaly_alerts'],
  filters: {
    severity: ['medium', 'high', 'critical'],
    stations: ['twente-nl']
  }
}));
```

**Message Format:**
```json
{
  "type": "anomaly_alert",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "event_type": "unusual_signal_pattern",
    "severity": "high",
    "frequency_hz": 14235000,
    "station_id": "twente-nl",
    "description": "Detected unusual modulation pattern",
    "anomaly_score": 0.87,
    "created_at": "2024-01-15T14:30:10Z"
  },
  "timestamp": "2024-01-15T14:30:10Z"
}
```

### System Status Updates

Subscribe to system health updates:

```javascript
// Subscribe to system status
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['system_status']
}));
```

**Message Format:**
```json
{
  "type": "system_status",
  "data": {
    "overall_status": "healthy",
    "services": {
      "websdr_collectors": "healthy",
      "signal_processor": "degraded",
      "ml_detector": "healthy"
    },
    "station_status": {
      "twente-nl": "active",
      "hackgreen-uk": "active",
      "bratislava-sk": "maintenance"
    }
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid frequency range specified",
    "details": {
      "field": "frequency_min",
      "value": -1000000,
      "constraint": "must be positive"
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440005",
    "timestamp": "2024-01-15T14:30:00Z"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Codes

- `AUTHENTICATION_REQUIRED`: Valid JWT token required
- `INVALID_TOKEN`: JWT token is invalid or expired
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `VALIDATION_ERROR`: Request validation failed
- `RESOURCE_NOT_FOUND`: Requested resource does not exist
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `SERVICE_UNAVAILABLE`: External service unavailable
- `PROCESSING_ERROR`: Signal processing failed
- `MODEL_ERROR`: ML model prediction failed

## SDK and Client Libraries

### Python SDK

```python
from heimdall_sdk import HeimdallClient

# Initialize client
client = HeimdallClient(
    base_url="https://api.heimdall.example.com",
    api_key="your_api_key"
)

# Get signal detections
detections = await client.signals.get_detections(
    frequency_range=(14000000, 15000000),
    time_range=("2024-01-15T12:00:00Z", "2024-01-15T18:00:00Z"),
    station_id="twente-nl"
)

# Analyze signal data
result = await client.signals.analyze(
    signal_data=signal_samples,
    sample_rate=48000,
    frequency=14235000
)

# Subscribe to live data
async with client.websocket.connect() as ws:
    await ws.subscribe_signals(
        stations=["twente-nl"],
        frequency_range=(14000000, 15000000)
    )
    
    async for signal in ws.stream():
        print(f"Signal detected: {signal.frequency_hz} Hz")
```

### JavaScript SDK

```javascript
import { HeimdallClient } from '@heimdall/sdk';

// Initialize client
const client = new HeimdallClient({
  baseURL: 'https://api.heimdall.example.com',
  apiKey: 'your_api_key'
});

// Get signal detections
const detections = await client.signals.getDetections({
  frequencyRange: [14000000, 15000000],
  timeRange: ['2024-01-15T12:00:00Z', '2024-01-15T18:00:00Z'],
  stationId: 'twente-nl'
});

// Real-time WebSocket connection
const ws = client.websocket.connect();
ws.subscribeSignals({
  stations: ['twente-nl'],
  frequencyRange: [14000000, 15000000]
});

ws.on('signal_detection', (signal) => {
  console.log('Signal detected:', signal);
});
```

## API Changelog

### Version 1.0.0 (Current)
- Initial API release
- Core signal detection endpoints
- WebSDR station management
- Anomaly detection and alerts
- Real-time WebSocket streaming
- ML model integration

### Upcoming Features (v1.1.0)
- Enhanced filtering options
- Bulk operations for signal analysis
- Advanced statistics endpoints
- Custom alert rules
- Export functionality

---

For additional support or feature requests, please visit our [GitHub repository](https://github.com/fulgidus/heimdall) or contact the development team.
