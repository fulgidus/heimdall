# System Status Real-time Updates - Testing Guide

## Overview
This document provides comprehensive testing instructions for the System Status page real-time update functionality.

## Changes Made

### Backend (Python)
1. **Event Publisher** (`services/backend/src/events/publisher.py`)
   - Added `publish_comprehensive_health()` method to publish aggregated health data for all components
   - Event type: `system:comprehensive_health`
   - Routing key: `system.health.comprehensive`

2. **Health Monitor** (`services/backend/src/tasks/comprehensive_health_monitor.py`)
   - Enhanced to publish both comprehensive (aggregated) and individual health updates
   - Includes infrastructure components: PostgreSQL, Redis, RabbitMQ, MinIO, Celery workers

3. **Celery Beat Schedule** (`services/backend/src/main.py`)
   - Updated `monitor-comprehensive-health` schedule from 30s to **1 second**
   - Ensures near real-time updates as required

### Frontend (TypeScript/React)
1. **Types** (`frontend/src/services/api/schemas.ts`)
   - Enhanced `ServiceHealthSchema` to include infrastructure-specific fields
   - Added support for `worker_count`, `online_count`, `total_count`, `model_info`, etc.

2. **Store** (`frontend/src/store/systemStore.ts`)
   - Added `infrastructureHealth` state (separate from `servicesHealth`)
   - Added `updateComprehensiveHealthFromWebSocket()` method
   - Automatically separates microservices from infrastructure components

3. **WebSocket Hook** (`frontend/src/hooks/useSystemWebSocket.ts`)
   - Subscribes to `system:comprehensive_health` events
   - Backwards compatible with legacy `services:health` events

4. **UI** (`frontend/src/pages/SystemStatus.tsx`)
   - Added **Infrastructure Components** card showing:
     - PostgreSQL (database)
     - Redis (cache)
     - RabbitMQ (queue)
     - MinIO (storage)
     - Celery workers
   - Updated System Overview section with infrastructure health count
   - Real-time status updates every 1 second

## Testing Instructions

### Prerequisites
- Docker and Docker Compose installed
- Node.js 18+ and npm installed
- Python 3.11+ installed

### Step 1: Start Infrastructure
```bash
cd /home/runner/work/heimdall/heimdall
docker compose up -d
```

Wait for all containers to be healthy:
```bash
docker compose ps
```

Expected output: 13 containers running (PostgreSQL, Redis, RabbitMQ, MinIO, etc.)

### Step 2: Start Backend Service
```bash
cd services/backend
python -m uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### Step 3: Start Celery Worker
In a new terminal:
```bash
cd services/backend
celery -A src.main.celery_app worker --loglevel=info
```

### Step 4: Start Celery Beat (Scheduler)
In a new terminal:
```bash
cd services/backend
celery -A src.main.celery_app beat --loglevel=info
```

You should see output like:
```
Scheduler: Sending due task monitor-comprehensive-health
```

This will trigger every 1 second.

### Step 5: Start Frontend Dev Server
In a new terminal:
```bash
cd frontend
npm run dev
```

### Step 6: Open Browser and Test

1. Navigate to: `http://localhost:5173/system-status`

2. **Verify System Overview Section**:
   - Should show 4 cards:
     - Microservices (3/3 Healthy)
     - Infrastructure (5/5 Healthy)
     - WebSDR Receivers (X/Y Online)
     - System Health (Good)

3. **Verify Infrastructure Components Card**:
   Should display a table with these components:
   - **PostgreSQL** (database) - Status: healthy, Message: "Database connection OK"
   - **Redis** (cache) - Status: healthy, Message: "Cache connection OK"
   - **RabbitMQ** (queue) - Status: healthy, Message: "Message queue connection OK"
   - **MinIO** (storage) - Status: healthy, Message: "Object storage OK..."
   - **Celery** (worker) - Status: healthy, Message: "X worker(s) active"

4. **Verify Real-time Updates**:
   - Open browser DevTools → Console
   - Look for WebSocket messages every 1 second:
     ```
     [useSystemWebSocket] Received system:comprehensive_health: {...}
     ```
   - Stop a service (e.g., `docker compose stop minio`)
   - Within 1-2 seconds, MinIO should show as "unhealthy" with red badge
   - Restart service (`docker compose start minio`)
   - Within 1-2 seconds, MinIO should return to "healthy" with green badge

### Step 7: Verify Event Flow

#### Backend Logs
Check Celery worker logs for:
```
Published comprehensive health update for 8 components
```

#### Frontend Console
Check browser console for:
```javascript
{
  event: "system:comprehensive_health",
  timestamp: "2025-11-04T15:49:27.802923",
  data: {
    components: {
      backend: { status: "healthy", ... },
      postgresql: { status: "healthy", type: "database", ... },
      redis: { status: "healthy", type: "cache", ... },
      // ... more components
    }
  }
}
```

## Expected Results

### ✅ Success Criteria
1. **All infrastructure components visible** on System Status page
2. **Real-time updates every 1 second** (check browser console for WebSocket events)
3. **No fake/mock data** - all values come from actual health checks
4. **Status badges update live** when services start/stop
5. **Response times displayed** for microservices
6. **Worker counts displayed** for Celery
7. **Error messages shown** when components are unhealthy

### ❌ Common Issues

**Issue**: Infrastructure components not showing
- **Fix**: Check that Celery beat is running and triggering `monitor-comprehensive-health` task
- **Verify**: Check Celery beat logs for "Sending due task monitor-comprehensive-health"

**Issue**: WebSocket not receiving events
- **Fix**: Check that RabbitMQ events consumer is running in backend
- **Verify**: Check backend logs for "RabbitMQ events consumer started successfully"

**Issue**: Components showing "Loading..." forever
- **Fix**: Check that comprehensive_health_monitor task is publishing events
- **Verify**: Check Celery worker logs for "Published comprehensive health update"

**Issue**: Health checks failing
- **Fix**: Ensure all Docker containers are running and healthy
- **Verify**: Run `docker compose ps` and check STATUS column

## Validation Script

Run the included test script to verify event structure:
```bash
python test_system_health_websocket.py
```

Expected output:
```
✅ Event structure validation PASSED
✅ Frontend store separation PASSED
ALL TESTS PASSED ✅
```

## Performance Notes

- **Update Frequency**: 1 second (configurable in `services/backend/src/main.py`)
- **Network Overhead**: ~2-5KB per update (JSON payload)
- **Browser Performance**: Minimal impact (React batches updates)
- **Backend Load**: Lightweight checks (<100ms total per cycle)

## Troubleshooting

### Enable Debug Logging

**Backend**:
```python
# In services/backend/src/main.py
logging.basicConfig(level=logging.DEBUG)
```

**Frontend**:
```typescript
// In frontend/src/hooks/useSystemWebSocket.ts
console.debug('[useSystemWebSocket] Received event:', data);
```

### Check WebSocket Connection
```javascript
// In browser console
// Should show "connected" state
window.wsManager?.getState()
```

### Manually Trigger Health Check
```bash
# From another terminal
cd services/backend
celery -A src.main.celery_app call monitor_comprehensive_health
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Celery Beat (Scheduler)                   │
│                    Every 1 second triggers                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         Celery Worker: monitor_comprehensive_health          │
│  - Check PostgreSQL, Redis, RabbitMQ, MinIO, Celery         │
│  - Check backend, training, inference services               │
│  - Aggregate all health data                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            EventPublisher.publish_comprehensive_health()     │
│  - Event: system:comprehensive_health                        │
│  - Routing key: system.health.comprehensive                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    RabbitMQ (heimdall.events)                │
│                    Topic Exchange                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            RabbitMQEventConsumer (Backend Thread)            │
│  - Consumes from heimdall.events exchange                    │
│  - Routes to WebSocket manager                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                WebSocket Connection Manager                  │
│  - Broadcasts to all connected clients                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Frontend: useSystemWebSocket Hook               │
│  - Subscribe to system:comprehensive_health                  │
│  - Call updateComprehensiveHealthFromWebSocket()             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     SystemStore (Zustand)                     │
│  - Separate servicesHealth and infrastructureHealth          │
│  - Update state triggers React re-render                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               SystemStatus.tsx Component                     │
│  - Display Infrastructure Components card                    │
│  - Real-time updates every 1 second                          │
└─────────────────────────────────────────────────────────────┘
```

## Code References

- Backend Event Publisher: `services/backend/src/events/publisher.py` (line 130-148)
- Health Monitor Task: `services/backend/src/tasks/comprehensive_health_monitor.py` (line 340-370)
- Celery Beat Schedule: `services/backend/src/main.py` (line 85-88)
- Frontend Store: `frontend/src/store/systemStore.ts` (line 92-114)
- WebSocket Hook: `frontend/src/hooks/useSystemWebSocket.ts` (line 18-28)
- UI Component: `frontend/src/pages/SystemStatus.tsx` (line 377-441)

## Support

For issues or questions, please check:
1. Docker containers are healthy: `docker compose ps`
2. Backend logs: Check FastAPI console output
3. Celery worker logs: Check Celery worker console
4. Celery beat logs: Check Celery beat console
5. Frontend console: Open browser DevTools → Console
6. WebSocket connection: Check Network tab → WS filter
