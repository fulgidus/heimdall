# Prompt 03: Phase 5 - Build Training API Endpoints

## Context
GPU configured (Prompt 01 ✓), pipeline tested (Prompt 02 ✓). Now build REST API endpoints to expose training functionality to frontend.

## Current State
- Training works via direct script execution
- Celery task `start_training_job` exists in `training_task.py`
- Database schema already complete (`training_jobs`, `training_metrics`)
- API documentation exists at `docs/TRAINING_API.md`
- Frontend expects endpoints at `/api/v1/training/*`

## Architectural Decisions

### API Location Decision
**Decision**: Training API endpoints go in **backend service**, NOT training service

**Rationale**:
- Backend service is already the main API gateway (port 8001)
- Frontend communicates with backend for all operations
- Training service (port 8002) is internal, focuses on ML pipeline
- Consistent with existing pattern (backend triggers Celery tasks)

**Files**:
- API endpoints: `services/backend/src/routers/training.py` (already exists!)
- Models: `services/backend/src/models/training.py` (already exists!)

### Existing Implementation Check
**Important**: `services/backend/src/routers/training.py` already has training endpoints implemented!

**Verify what exists**:
```bash
grep -E "^@router\.(get|post|delete|put)" services/backend/src/routers/training.py
```

**Expected routes** (from docs/TRAINING_API.md):
- `POST /api/v1/training/jobs` - Create training job
- `GET /api/v1/training/jobs` - List jobs
- `GET /api/v1/training/jobs/{job_id}` - Get job details
- `DELETE /api/v1/training/jobs/{job_id}` - Cancel job
- `GET /api/v1/training/jobs/{job_id}/metrics` - Get metrics
- `POST /api/v1/training/pause/{job_id}` - Pause job
- `POST /api/v1/training/resume/{job_id}` - Resume job

### WebSocket Broadcasting Decision
**Decision**: Use existing RabbitMQ event pattern for real-time updates

**Pattern** (already used for WebSDR health):
```
Celery Task (training_task.py)
    ↓
EventPublisher.publish_training_status() → RabbitMQ
    ↓
RabbitMQEventConsumer (backend service)
    ↓
WebSocket Manager → Frontend clients
```

**Why**: Decouples worker processes from FastAPI event loop (prevents RuntimeError)

**Implementation**:
1. Add training event types to `backend/src/events/publisher.py`
2. Add training event handlers to `backend/src/events/consumer.py`
3. Broadcast from Celery task at key moments:
   - Training started
   - Every epoch completed
   - Training paused/resumed
   - Training completed/failed

## Tasks

### Task 1: Verify Existing API Endpoints
**Goal**: Check if training API already complete

**Files to check**:
- `services/backend/src/routers/training.py`
- `services/backend/src/models/training.py`

**If endpoints exist**: Test them, document any gaps
**If incomplete**: Implement missing endpoints per docs/TRAINING_API.md

### Task 2: Add Training Events to Publisher
**File**: `services/backend/src/events/publisher.py`

**Add methods**:
```python
def publish_training_started(self, job_id: str, job_name: str, config: dict):
    """Publish training job started event."""
    self.publish_event('training.started', {
        'job_id': job_id,
        'job_name': job_name,
        'config': config,
        'timestamp': datetime.utcnow().isoformat()
    })

def publish_training_progress(self, job_id: str, epoch: int, total_epochs: int, metrics: dict):
    """Publish training progress event."""
    self.publish_event('training.progress', {
        'job_id': job_id,
        'epoch': epoch,
        'total_epochs': total_epochs,
        'progress_percent': (epoch / total_epochs) * 100,
        'metrics': metrics,
        'timestamp': datetime.utcnow().isoformat()
    })

def publish_training_completed(self, job_id: str, final_metrics: dict):
    """Publish training job completed event."""
    self.publish_event('training.completed', {
        'job_id': job_id,
        'final_metrics': final_metrics,
        'timestamp': datetime.utcnow().isoformat()
    })

def publish_training_failed(self, job_id: str, error: str):
    """Publish training job failed event."""
    self.publish_event('training.failed', {
        'job_id': job_id,
        'error': error,
        'timestamp': datetime.utcnow().isoformat()
    })
```

### Task 3: Add Training Event Consumer
**File**: `services/backend/src/events/consumer.py`

**Add handler** (in `RabbitMQEventConsumer` class):
```python
def _handle_training_event(self, routing_key: str, data: dict):
    """Handle training events and broadcast to WebSocket."""
    event_type = routing_key.split('.')[-1]  # 'started', 'progress', 'completed', 'failed'
    
    message = {
        'event': f'training_{event_type}',
        'data': data
    }
    
    # Broadcast to WebSocket clients
    asyncio.run_coroutine_threadsafe(
        self.websocket_manager.broadcast(message),
        self.loop
    )
```

**Add routing** (in queue binding):
```python
# In setup_queues() method
queue.bind(exchange, routing_key='training.*')
```

### Task 4: Integrate Events in Training Task
**File**: `services/training/src/tasks/training_task.py`

**Find** `start_training_job` task and **add event publishing**:

```python
from ..events.publisher import get_event_publisher

@shared_task(bind=True, base=TrainingTask)
def start_training_job(self, job_id: str):
    publisher = get_event_publisher()
    
    try:
        # Load job config
        job = load_job_from_db(job_id)
        publisher.publish_training_started(job_id, job['name'], job['config'])
        
        # Training loop
        for epoch in range(total_epochs):
            # ... training code ...
            
            # Publish progress after each epoch
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'learning_rate': current_lr
            }
            publisher.publish_training_progress(job_id, epoch + 1, total_epochs, metrics)
        
        # Training completed
        publisher.publish_training_completed(job_id, final_metrics)
        
    except Exception as e:
        publisher.publish_training_failed(job_id, str(e))
        raise
```

### Task 5: Add WebSocket Endpoint for Training
**File**: `services/backend/src/main.py`

**Add WebSocket route**:
```python
@app.websocket("/ws/training/{job_id}")
async def websocket_training(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training updates."""
    await websocket_manager.connect(websocket, channel=f"training:{job_id}")
    try:
        # Send initial connection message
        await websocket.send_json({
            "event": "connected",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_json()
            
            if data.get("event") == "ping":
                await websocket.send_json({
                    "event": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, channel=f"training:{job_id}")
```

### Task 6: Test API Endpoints
**Test sequence**:

1. **Create training job**:
```bash
curl -X POST http://localhost:8001/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "API Test Training",
    "config": {
      "epochs": 5,
      "batch_size": 16,
      "learning_rate": 0.001,
      "model_architecture": "convnext_large"
    }
  }'
```

2. **Get job status**:
```bash
curl http://localhost:8001/api/v1/training/jobs/{job_id}
```

3. **Monitor via WebSocket** (use websocat or browser):
```bash
websocat ws://localhost:8001/ws/training/{job_id}
```

4. **Get metrics**:
```bash
curl http://localhost:8001/api/v1/training/jobs/{job_id}/metrics
```

5. **List jobs**:
```bash
curl http://localhost:8001/api/v1/training/jobs
```

## Validation Criteria

### Must Pass
- [ ] All API endpoints respond correctly (200/201 status)
- [ ] Creating job triggers Celery task
- [ ] Job status updates in database during training
- [ ] WebSocket receives real-time progress updates
- [ ] Metrics endpoint returns per-epoch data
- [ ] List endpoint shows all jobs with correct status
- [ ] Error handling returns proper HTTP codes (400/404/500)
- [ ] Cancel job stops training gracefully
- [ ] Pause/resume endpoints work (if implemented)

### Non-Breaking Requirement
- Backend service remains operational
- Existing endpoints (WebSDR, recordings, etc.) unaffected
- Training service continues to work standalone

## Common Issues & Solutions

### Issue: WebSocket not receiving events
**Check**: 
- RabbitMQ exchange binding correct
- Event consumer running in background thread
- Publisher using correct routing keys

### Issue: Training task not starting
**Check**:
- Celery worker running
- Task registered in Celery app
- Database connection works from worker

### Issue: CORS errors from frontend
**Solution**: Add WebSocket to CORS allowed origins

## Success Criteria
When all validation passes, training API is complete. Proceed to **Prompt 04: .heimdall Export Format**.

## Deliverables
Document:
- API endpoint test results (curl commands + responses)
- WebSocket message samples
- Any gaps in existing implementation
- Performance notes (response times, concurrent connections)
