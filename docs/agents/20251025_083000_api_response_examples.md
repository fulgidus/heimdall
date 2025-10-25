# API Response Examples

## Dashboard Metrics Endpoint

**Endpoint**: `GET /api/v1/analytics/dashboard/metrics`

**Response Example**:
```json
{
  "signalDetections": 384,
  "systemUptime": 12847,
  "modelAccuracy": 0.94,
  "predictionsTotal": 1289,
  "predictionsSuccessful": 1225,
  "predictionsFailed": 64,
  "lastUpdate": "2025-10-25T08:26:42.123456"
}
```

**Dashboard Display**:
- Signal Detections Card: Shows "384" (24h count)
- System Uptime Card: Shows "3.5h" (12847 seconds = 3.57 hours)
- Model Accuracy Card: Shows "94.0%"

---

## Enhanced Model Info Endpoint

**Endpoint**: `GET /api/v1/analytics/model/info`

**Response Example**:
```json
{
  "active_version": "v1.0.0",
  "stage": "Production",
  "model_name": "heimdall-inference",
  "accuracy": 0.94,
  "latency_p95_ms": 245.0,
  "cache_hit_rate": 0.82,
  "loaded_at": "2025-10-25T04:52:15.000000",
  "uptime_seconds": 12847,
  "last_prediction_at": "2025-10-25T08:12:34.567890",
  "predictions_total": 1289,
  "predictions_successful": 1225,
  "predictions_failed": 64,
  "is_ready": true,
  "health_status": "healthy",
  "model_id": "heimdall-v1.0.0",
  "version": "1.0.0",
  "description": "Heimdall SDR Localization Neural Network",
  "architecture": "CNN-based (ResNet-18)",
  "input_shape": [1, 128, 256],
  "output_shape": [4],
  "parameters": 11689472,
  "training_date": "2025-09-15T14:30:00Z",
  "status": "active",
  "framework": "PyTorch",
  "backend": "ONNX Runtime"
}
```

**Dashboard Usage**:
- System Activity Table: Shows version, health status, last prediction time
- Model Accuracy Card: Uses `accuracy` field (0.94 = 94%)
- Uptime calculations: Uses `uptime_seconds` field

---

## WebSDR Health Check Endpoint

**Endpoint**: `GET /api/v1/acquisition/websdrs/health`

**Response Example**:
```json
{
  "1": {
    "websdr_id": 1,
    "name": "Piedmont North",
    "status": "online",
    "response_time_ms": 145,
    "last_check": "2025-10-25T08:26:42.123456",
    "uptime": 99.8,
    "avg_snr": 18.5
  },
  "2": {
    "websdr_id": 2,
    "name": "Piedmont South",
    "status": "online",
    "response_time_ms": 162,
    "last_check": "2025-10-25T08:26:42.123456",
    "uptime": 98.2,
    "avg_snr": 16.3
  },
  "3": {
    "websdr_id": 3,
    "name": "Liguria West",
    "status": "offline",
    "last_check": "2025-10-25T08:26:42.123456",
    "error_message": "Connection timeout"
  }
  // ... 4 more WebSDRs
}
```

**Dashboard Display**:
- Active WebSDR Card: "2/7" (66%)
- WebSDR Network Status Grid: Shows each WebSDR with online/offline status
- Signal strength bars: Based on `avg_snr` values

---

## Services Health Check Endpoints

**Endpoint Pattern**: `GET /api/v1/{service}/health`

**Example for rf-acquisition**:
```json
{
  "status": "healthy",
  "service": "rf-acquisition",
  "version": "0.1.0",
  "timestamp": "2025-10-25T08:26:42.123456"
}
```

**Dashboard Display**:
- Services Status List: Shows all 5 services with health badges
  - ✅ api-gateway: healthy
  - ✅ rf-acquisition: healthy
  - ✅ training: healthy
  - ✅ inference: healthy
  - ✅ data-ingestion-web: healthy

---

## Data Flow Timeline

```
T=0s: Dashboard Component Mounts
  ↓
T=0.1s: fetchDashboardData() called
  ↓
T=0.2s: Parallel API calls initiated:
  ├─ GET /api/v1/analytics/dashboard/metrics
  ├─ GET /api/v1/acquisition/websdrs
  ├─ GET /api/v1/acquisition/websdrs/health
  ├─ GET /api/v1/analytics/model/info
  ├─ GET /api/v1/api-gateway/health
  ├─ GET /api/v1/rf-acquisition/health
  ├─ GET /api/v1/training/health
  ├─ GET /api/v1/inference/health
  └─ GET /api/v1/data-ingestion-web/health
  ↓
T=0.5s: All responses received
  ↓
T=0.6s: Store state updated with real data
  ↓
T=0.7s: Dashboard re-renders with live metrics
  ↓
T=30s: Auto-refresh triggered
  ↓
(Repeat from T=0.1s)
```

---

## Before vs After Comparison

### Signal Detections Card

**Before (Mocked)**:
```javascript
metrics: {
  signalDetections: 0  // Hardcoded
}
```
Display: "0 detections"

**After (Real)**:
```javascript
metrics: {
  signalDetections: 384  // From backend API
}
```
Display: "384 detections" (increments over time)

---

### System Uptime Card

**Before (Mocked)**:
```javascript
metrics: {
  systemUptime: 0  // Hardcoded
}
```
Display: "0h"

**After (Real)**:
```javascript
metrics: {
  systemUptime: 12847  // Real seconds from service start
}
```
Display: "3.5h" (calculated as 12847 / 3600)

---

### Model Accuracy Card

**Before (Mocked)**:
```javascript
data: {
  modelInfo: null  // No data
}
```
Display: "N/A"

**After (Real)**:
```javascript
data: {
  modelInfo: {
    accuracy: 0.94  // From backend
  }
}
```
Display: "94.0%"

---

## Testing the APIs

### Using curl

```bash
# Test dashboard metrics
curl http://localhost:8000/api/v1/analytics/dashboard/metrics

# Test enhanced model info
curl http://localhost:8000/api/v1/analytics/model/info

# Test WebSDR health
curl http://localhost:8000/api/v1/acquisition/websdrs/health

# Test service health (example: inference)
curl http://localhost:8000/api/v1/inference/health
```

### Using browser DevTools

1. Open Dashboard page: http://localhost:3000/dashboard
2. Open DevTools (F12)
3. Go to Network tab
4. Filter by "Fetch/XHR"
5. Refresh page
6. Observe real API calls:
   - `/api/v1/analytics/dashboard/metrics` → 200 OK
   - `/api/v1/analytics/model/info` → 200 OK
   - `/api/v1/acquisition/websdrs` → 200 OK
   - `/api/v1/acquisition/websdrs/health` → 200 OK
   - 5x `/api/v1/{service}/health` → 200 OK

### Verification Checklist

- [ ] Dashboard shows non-zero signal detections
- [ ] System uptime increments over time
- [ ] Model accuracy shows 94.0%
- [ ] Active WebSDR count shows real ratio (e.g., 5/7)
- [ ] Services status shows all 5 services
- [ ] No console errors
- [ ] Auto-refresh updates metrics every 30s
- [ ] Manual refresh button works

---

## Production Considerations

When connected to real database and production services:

1. **Signal Detections**: Query `measurements` table for 24h count
2. **Prediction Counts**: Query `predictions` table for totals
3. **Uptime**: Store service start timestamp in Redis/database
4. **WebSDR Stats**: Calculate from historical health check data
5. **Cache**: Add Redis caching for frequently accessed metrics

### Database Queries (Future)

```sql
-- Signal detections in last 24 hours
SELECT COUNT(*) 
FROM measurements 
WHERE timestamp >= NOW() - INTERVAL '24 hours';

-- Prediction statistics
SELECT 
  COUNT(*) as total,
  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
  SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
FROM predictions;

-- WebSDR uptime percentage
SELECT 
  websdr_id,
  AVG(CASE WHEN status = 'online' THEN 1.0 ELSE 0.0 END) * 100 as uptime_pct
FROM websdr_health_checks
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY websdr_id;
```
