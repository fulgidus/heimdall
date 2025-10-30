# Recording Session WebSocket Implementation

## Overview

This document describes the WebSocket-based recording session workflow implemented for the Heimdall platform. The new implementation replaces the polling-based acquisition with real-time WebSocket communication and splits recordings into 1-second samples for better training data granularity.

## Key Changes

### 1. Workflow Changes

**Old Workflow:**
- Single step: Configure and start acquisition immediately
- Poll for status updates
- Duration: 10-300 seconds with 10-second steps

**New Workflow:**
1. **Start Recording Session** - Configure and initiate session
2. **Assign Source** - Select known source or "Unknown"
3. **Start Acquisition** - Trigger data collection with real-time updates
4. Duration: 1-30 seconds with 1-second steps
5. Each second recorded as separate sample (15s = 15 individual samples)

### 2. Backend Implementation

#### WebSocket Events

**Client → Server:**
- `session:start` - Start a new recording session
  ```json
  {
    "session_name": "Test Recording",
    "frequency_mhz": 145.5,
    "duration_seconds": 15,
    "notes": "Optional notes"
  }
  ```

- `session:assign_source` - Assign source to session
  ```json
  {
    "session_id": "uuid",
    "source_id": "uuid or 'unknown'"
  }
  ```

- `session:complete` - Trigger acquisition
  ```json
  {
    "session_id": "uuid",
    "frequency_hz": 145500000,
    "duration_seconds": 15
  }
  ```

**Server → Client:**
- `session:started` - Session created successfully
- `session:source_assigned` - Source assigned successfully
- `session:completed` - Acquisition started
- `session:progress` - Real-time progress updates (per chunk)
- `session:error` - Error occurred
- `session:status_update` - Broadcast to all clients

#### New Celery Task: `acquire_iq_chunked`

Located in: `services/backend/src/tasks/acquire_iq.py`

**Purpose:** Split long recordings into 1-second chunks for better training granularity

**Process:**
1. Loop through total duration (e.g., 15 seconds)
2. For each second:
   - Fetch 1-second IQ sample from all WebSDRs simultaneously
   - Process signal metrics
   - Save to MinIO with chunk index
   - Write to TimescaleDB as separate measurement
   - Broadcast progress via WebSocket
3. Update session status on completion

**Benefits:**
- More training examples (15s → 15 samples instead of 1)
- Better time resolution for training
- Enables near real-time inference
- Easier to manage and process smaller chunks

#### Session Status States

New states added to `recording_sessions` table:
- `pending` - Initial state
- `recording` - Session started, awaiting source assignment
- `source_assigned` - Source selected, ready for acquisition
- `in_progress` - Acquisition running
- `completed` - Successfully finished
- `failed` - Error occurred

#### Database Migration

File: `db/migrations/04-add-recording-session-states.sql`

Changes:
- Update status constraint to include new states
- Create "Unknown" source placeholder
- Add explanatory comments

### 3. Frontend Implementation

#### Component: `RecordingSession.tsx`

**State Machine:**
```
idle → recording → assigning_source → acquiring → complete
  ↓                                                    ↓
  └────────────────── reset ────────────────────────→
```

**Real-time Updates:**
- Connection to WebSocket on mount
- Subscribe to session events
- Display progress per chunk
- Show measurements count
- Update UI based on session state

**UI Elements:**

1. **Step 1: Configure Session**
   - Session name (required)
   - Frequency in MHz (required)
   - Duration slider: 1-30 seconds with 1s steps
   - Notes (optional)
   - Button: "Start Recording Session"

2. **Step 2: Assign Source**
   - Dropdown with known sources + "Unknown" option
   - Shows automatically after step 1
   - Button: "Assign Source"

3. **Step 3: Start Acquisition**
   - Confirmation step
   - Button: "Start Acquisition"

4. **Progress Display**
   - Real-time chunk counter (e.g., "Chunk 5/15")
   - Progress bar (animated during acquisition)
   - Total measurements count
   - Status messages

**Duration Slider:**
```tsx
<input
  type="range"
  min="1"
  max="30"
  step="1"
  value={formData.duration}
/>
```

Each second equals one 1-second sample for training.

## Testing Guide

### Manual Testing

#### 1. Start Backend Services

```bash
cd /home/runner/work/heimdall/heimdall
docker-compose up -d
```

Verify services:
- PostgreSQL + TimescaleDB
- RabbitMQ (Celery broker)
- Redis (cache)
- MinIO (object storage)
- Backend service (port 8001)

#### 2. Run Database Migration

```bash
docker exec -it heimdall-postgres psql -U postgres -d heimdall -f /docker-entrypoint-initdb.d/migrations/04-add-recording-session-states.sql
```

Verify:
```sql
-- Check new status constraint
SELECT constraint_name, check_clause 
FROM information_schema.check_constraints 
WHERE table_name = 'recording_sessions' 
AND constraint_name = 'valid_status';

-- Check Unknown source exists
SELECT * FROM heimdall.known_sources WHERE name = 'Unknown';
```

#### 3. Test WebSocket Connection

```bash
# Install wscat if not available
npm install -g wscat

# Connect to WebSocket endpoint
wscat -c ws://localhost:80/ws

# Send ping
> {"event": "ping"}

# Should receive pong
< {"event": "pong", "timestamp": "..."}
```

#### 4. Test Recording Workflow

**Start Frontend:**
```bash
cd frontend
npm run dev
```

Navigate to: `http://localhost:5173/recording-session`

**Test Steps:**
1. Fill in session name: "Test Recording 001"
2. Set frequency: 145.500 MHz
3. Set duration: 5 seconds (slider)
4. Add notes: "Test of 1-second chunking"
5. Click "Start Recording Session"
   - Should see "Recording" state
   - WebSocket should receive `session:started`
   
6. Select source: "Unknown" (or any known source)
7. Click "Assign Source"
   - Should see "Source Assigned" state
   - WebSocket should receive `session:source_assigned`

8. Click "Start Acquisition"
   - Should see progress bar
   - Should see "Chunk 1/5", "Chunk 2/5", etc.
   - WebSocket should receive `session:progress` events
   - After 5 chunks, should see "Complete" state

#### 5. Verify Data Storage

**Check Database:**
```sql
-- View session
SELECT * FROM heimdall.recording_sessions 
WHERE session_name = 'Test Recording 001';

-- Should see status = 'completed'
-- Should see duration_seconds = 5

-- View measurements (should be ~5 chunks * 7 WebSDRs = ~35 measurements)
SELECT websdr_station_id, COUNT(*) 
FROM heimdall.measurements 
WHERE timestamp >= (SELECT session_start FROM heimdall.recording_sessions WHERE session_name = 'Test Recording 001')
GROUP BY websdr_station_id;
```

**Check MinIO:**
```bash
# List objects in session folder
docker exec heimdall-minio mc ls local/heimdall-raw-iq/sessions/<session_id>/

# Should see files like:
# chunk_000_websdr_1.npy
# chunk_000_websdr_2.npy
# ...
# chunk_004_websdr_7.npy
```

### Automated Testing

#### Backend Unit Tests

```bash
cd services/backend
pytest tests/unit/test_websocket_sessions.py -v
```

Expected tests:
- Test session start via WebSocket
- Test source assignment
- Test acquisition trigger
- Test chunked acquisition task
- Test progress broadcasting

#### Frontend Unit Tests

```bash
cd frontend
npm run test src/pages/RecordingSession.test.tsx
```

Expected tests:
- Test WebSocket connection
- Test session workflow state machine
- Test event handlers
- Test UI updates on progress

#### Integration Tests

```bash
cd services/backend
pytest tests/integration/test_recording_session_e2e.py -v
```

Expected test:
- Complete workflow: start → assign → acquire → complete
- Verify all 1-second chunks are created
- Verify measurements count matches expectations
- Verify session status updates correctly

## Performance Considerations

### 1-Second Chunking Overhead

**Pros:**
- More training examples (N seconds = N samples)
- Better time resolution for training
- Enables near real-time inference
- Easier error recovery (failed chunk doesn't lose entire recording)

**Cons:**
- More API calls (N chunks vs 1)
- More database writes (N×7 measurements vs 7)
- More MinIO uploads (N×7 files vs 7)

**Mitigation:**
- Celery task manages chunking efficiently
- Bulk database operations where possible
- MinIO handles many small files well
- WebSocket reduces HTTP overhead vs polling

### Expected Timings

For a 15-second recording with 7 WebSDRs:

- Chunk 1-15: ~1-2 seconds each (network dependent)
- Total time: 15-30 seconds (vs 15s for single fetch)
- Measurements created: ~105 (15 chunks × 7 WebSDRs)
- MinIO files: ~105 .npy files

## Troubleshooting

### WebSocket Connection Issues

**Problem:** Frontend can't connect to WebSocket

**Solutions:**
1. Check backend is running: `curl http://localhost:8001/health`
2. Check WebSocket endpoint: `wscat -c ws://localhost:80/ws`
3. Check Envoy proxy is routing correctly
4. Check browser console for CORS errors
5. Verify `VITE_SOCKET_URL` environment variable

### Chunked Acquisition Not Starting

**Problem:** Session stuck in "source_assigned" state

**Solutions:**
1. Check Celery worker is running: `docker-compose logs -f backend-celery`
2. Check RabbitMQ connection: `docker-compose logs -f rabbitmq`
3. Verify `acquire_iq_chunked` task is registered
4. Check session exists: `SELECT * FROM heimdall.recording_sessions WHERE id = '<session_id>'`

### Progress Not Updating

**Problem:** Frontend not receiving progress events

**Solutions:**
1. Check WebSocket connection is still open
2. Check backend logs for broadcast errors
3. Verify frontend is subscribed to `session:progress`
4. Check Celery task is updating state correctly

### Missing Measurements

**Problem:** Not all chunks saved to database

**Solutions:**
1. Check Celery task completion: Look at result in Redis
2. Check MinIO connectivity
3. Check TimescaleDB hypertable health
4. Review task logs for errors
5. Verify WebSDR health (some may be offline)

## Future Enhancements

1. **Pause/Resume**: Allow pausing acquisition mid-session
2. **Retry Failed Chunks**: Automatically retry failed 1-second chunks
3. **Adaptive Chunking**: Adjust chunk size based on signal quality
4. **Real-time Inference**: Run inference on chunks as they arrive
5. **Multi-frequency**: Support recording multiple frequencies simultaneously
6. **Scheduled Sessions**: Schedule recordings for specific times
7. **Quality Metrics**: Display SNR, signal strength per chunk
8. **Export**: Export session data in various formats

## API Reference

### WebSocket Commands

#### Start Session
```typescript
ws.send('session:start', {
  session_name: string,
  frequency_mhz: number,
  duration_seconds: number,
  notes?: string
});
```

#### Assign Source
```typescript
ws.send('session:assign_source', {
  session_id: string,
  source_id: string | 'unknown'
});
```

#### Complete Session
```typescript
ws.send('session:complete', {
  session_id: string,
  frequency_hz: number,
  duration_seconds: number
});
```

### WebSocket Events

#### Session Started
```typescript
{
  event: 'session:started',
  timestamp: string,
  data: {
    session_id: string,
    session_name: string,
    status: 'recording',
    frequency_mhz: number,
    duration_seconds: number
  }
}
```

#### Progress Update
```typescript
{
  event: 'session:progress',
  timestamp: string,
  data: {
    session_id: string,
    chunk: number,
    total_chunks: number,
    progress: number, // 0-100
    measurements_count: number
  }
}
```

## References

- [WebSocket Implementation](../frontend/src/lib/websocket.ts)
- [Recording Session Component](../frontend/src/pages/RecordingSession.tsx)
- [Sessions Router](../services/backend/src/routers/sessions.py)
- [WebSocket Router](../services/backend/src/routers/websocket.py)
- [Chunked Acquisition Task](../services/backend/src/tasks/acquire_iq.py)
- [Database Migration](../db/migrations/04-add-recording-session-states.sql)
