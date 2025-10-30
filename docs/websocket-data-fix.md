# WebSocket and Backend Data Format Fix

## Problem Statement
The WebSocket and backend were returning incorrect/inconsistent data formats to the frontend, particularly in the WebSDR Management page. Data types were mismatched, causing display issues.

## Root Causes Identified

### 1. Type Mismatch in Frontend Store
**Issue**: `dashboardStore.ts` declared `websdrsHealth` as `Record<number, WebSDRHealthStatus>` but backend sends UUID strings as keys.

**Fix**: Changed to `Record<string, WebSDRHealthStatus>` to match backend UUID keys.

### 2. Missing `response_time_ms` Field
**Issue**: Backend health check endpoint didn't include `response_time_ms` field that frontend expected.

**Fix**: 
- Modified `OpenWebRXFetcher.health_check()` to measure and return response time
- Updated acquisition router to extract and include `response_time_ms` in health response

### 3. Inconsistent UUID Usage in Error Cases
**Issue**: Backend error handler used integer ID instead of UUID string, breaking frontend expectations.

**Fix**: Updated error case in `acquisition.py` to consistently use UUID strings (`ws_config['uuid']`).

### 4. No Real-time WebSocket Updates
**Issue**: Health status changes weren't pushed to frontend via WebSocket, only available via polling.

**Fix**:
- Added WebSocket broadcasting in `check_websdrs_health()` endpoint
- Broadcast `websdrs_update` event with complete health status
- Frontend now receives real-time updates without polling

### 5. Event Name Inconsistencies
**Issue**: DashboardStore subscribed to `websdrs:status` but backend sent `websdrs_update`.

**Fix**: Aligned all frontend components to subscribe to `websdrs_update` event.

## Changes Made

### Backend Files Modified

#### `services/backend/src/fetchers/openwebrx_fetcher.py`
```python
async def health_check(self) -> Dict[int, dict]:
    """Returns dict with 'online' (bool) and 'response_time_ms' (float)"""
    # Measures connection time and returns structured response
```

#### `services/backend/src/routers/acquisition.py`
- Extract `online` and `response_time_ms` from health check results
- Use UUID keys consistently in both success and error cases
- Broadcast health updates via WebSocket to all connected clients

### Frontend Files Modified

#### `frontend/src/store/dashboardStore.ts`
- Changed `websdrsHealth` type from `Record<number, ...>` to `Record<string, ...>`
- Updated WebSocket subscription from `websdrs:status` to `websdrs_update`

#### `frontend/src/pages/WebSDRManagement.tsx`
- Added WebSocket connection on component mount
- Subscribed to `websdrs_update` events
- Real-time health updates bypass polling when WebSocket connected
- Fallback to 30-second polling if WebSocket disconnected

## Data Flow

### Before Fix
```
Backend Health Check → Returns {id: {status, ...}}  ❌ Missing response_time_ms
                    → Error case uses integer ID   ❌ Inconsistent
Frontend Store      → Record<number, ...>           ❌ Wrong type
WebSocket           → No health broadcasts          ❌ No real-time
Event Names         → Mismatched subscriptions      ❌ Events not received
```

### After Fix
```
Backend Health Check → Returns {uuid: {status, response_time_ms, ...}} ✅
                    → Error case uses UUID                              ✅
                    → Broadcasts via WebSocket                          ✅
Frontend Store      → Record<string, ...>                               ✅
WebSocket           → Real-time health updates                          ✅
Event Names         → Aligned 'websdrs_update'                         ✅
```

## Expected Behavior

### WebSDR Management Page
1. On load: Fetches WebSDR list and health status
2. WebSocket connects and subscribes to `websdrs_update`
3. Every health check (manual or automatic) broadcasts update
4. Frontend receives real-time status changes without polling
5. If WebSocket fails, falls back to 30-second polling

### Health Status Response Format
```json
{
  "550e8400-e29b-41d4-a716-446655440000": {
    "websdr_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "IW2MXM Milano",
    "status": "online",
    "response_time_ms": 245.3,
    "last_check": "2025-10-30T14:30:00.000Z",
    "uptime": 98.5,
    "avg_snr": 15.2
  }
}
```

## Testing Checklist

- [x] TypeScript compilation passes
- [x] Frontend builds successfully
- [ ] Backend health endpoint returns correct structure
- [ ] WebSocket broadcasts health updates
- [ ] WebSDR Management page displays data correctly
- [ ] Real-time updates work without page refresh
- [ ] Response time displays correctly
- [ ] Fallback polling works when WebSocket disconnected

## Related Files

### Backend
- `services/backend/src/fetchers/openwebrx_fetcher.py`
- `services/backend/src/routers/acquisition.py`
- `services/backend/src/routers/websocket.py`

### Frontend
- `frontend/src/pages/WebSDRManagement.tsx`
- `frontend/src/store/dashboardStore.ts`
- `frontend/src/store/websdrStore.ts`
- `frontend/src/services/api/websdr.ts`
- `frontend/src/services/api/schemas.ts`
- `frontend/src/lib/websocket.ts`
