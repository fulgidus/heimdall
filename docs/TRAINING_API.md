# Training API Documentation

## Overview

Heimdall Training API provides REST endpoints and WebSocket connections for managing AI model training jobs with real-time progress monitoring.

## Architecture

```
Frontend → REST API → Celery Task → Training Pipeline
                ↓
            Database (training_jobs, training_metrics)
                ↓
         WebSocket ← Frontend (real-time updates)
```

## REST Endpoints

### Create Training Job

```http
POST /api/v1/training/jobs
Content-Type: application/json

{
  "job_name": "My Training Job",
  "config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "model_architecture": "convnext_large"
  }
}
```

**Response:**
```json
{
  "id": "99db769e-1dbb-4f26-a78e-07f0aeda3e8f",
  "job_name": "My Training Job",
  "status": "queued",
  "created_at": "2025-11-01T23:51:16Z",
  "celery_task_id": "2826a3a2-ba9e-47f4-ac94-f47ca1d429cd",
  "config": {...},
  "total_epochs": 100
}
```

### List Training Jobs

```http
GET /api/v1/training/jobs?status=running&limit=50&offset=0
```

**Response:**
```json
{
  "jobs": [
    {
      "id": "...",
      "job_name": "...",
      "status": "running",
      "current_epoch": 45,
      "total_epochs": 100,
      "progress_percent": 45.0,
      "train_loss": 0.234,
      "val_loss": 0.267
    }
  ],
  "total": 1
}
```

### Get Training Job Details

```http
GET /api/v1/training/jobs/{job_id}
```

**Response:**
```json
{
  "job": {
    "id": "...",
    "job_name": "...",
    "status": "running",
    "current_epoch": 45,
    "total_epochs": 100,
    "metrics": {...}
  },
  "recent_metrics": [
    {
      "epoch": 45,
      "train_loss": 0.234,
      "val_loss": 0.267,
      "train_accuracy": 0.87,
      "val_accuracy": 0.84,
      "learning_rate": 0.0005
    }
  ],
  "websocket_url": "ws://localhost:8001/ws/training/{job_id}"
}
```

### Delete Training Job

```http
DELETE /api/v1/training/jobs/{job_id}
```

Cancels running job and deletes all data.

### Get Training Metrics

```http
GET /api/v1/training/jobs/{job_id}/metrics?limit=100
```

Returns detailed per-epoch metrics for charting.

## WebSocket Protocol

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/training/{job_id}');
```

### Message Types

#### 1. Connected (Server → Client)

```json
{
  "event": "connected",
  "job_id": "99db769e-1dbb-4f26-a78e-07f0aeda3e8f",
  "timestamp": "2025-11-01T23:51:20Z",
  "message": "Connected to training job"
}
```

#### 2. Training Status (Server → Client, every 5s)

```json
{
  "event": "training_status",
  "job_id": "99db769e-1dbb-4f26-a78e-07f0aeda3e8f",
  "status": "running",
  "current_epoch": 45,
  "total_epochs": 100,
  "progress_percent": 45.0,
  "metrics": {
    "train_loss": 0.234,
    "val_loss": 0.267,
    "train_accuracy": 0.87,
    "val_accuracy": 0.84,
    "learning_rate": 0.0005
  },
  "error_message": null,
  "timestamp": "2025-11-01T23:52:00Z"
}
```

#### 3. Ping/Pong (Client ↔ Server)

**Client sends:**
```json
{
  "event": "ping"
}
```

**Server responds:**
```json
{
  "event": "pong",
  "timestamp": "2025-11-01T23:52:05Z"
}
```

## Training Configuration

### TrainingConfig Schema

```typescript
interface TrainingConfig {
  // Model
  model_architecture: "convnext_large" | "convnext_tiny" | "convnext_small";
  pretrained: boolean; // default: true
  freeze_backbone: boolean; // default: false

  // Data
  batch_size: number; // 1-512, default: 32
  num_workers: number; // 0-16, default: 4
  validation_split: number; // 0.0-0.5, default: 0.2

  // Features
  n_mels: number; // default: 128
  n_fft: number; // default: 2048
  hop_length: number; // default: 512

  // Training
  epochs: number; // 1-1000, default: 100
  learning_rate: number; // default: 0.001
  weight_decay: number; // default: 0.0001
  dropout_rate: number; // 0.0-0.9, default: 0.2

  // Scheduler
  lr_scheduler: "cosine" | "step" | "plateau";
  warmup_epochs: number; // default: 5
  early_stop_patience: number; // default: 10

  // Hardware
  accelerator: "gpu" | "cpu";
  devices: number; // default: 1

  // Filters
  min_snr_db: number; // default: 10.0
  only_approved: boolean; // default: true
}
```

## Training Job Status States

| Status      | Description                                 |
| ----------- | ------------------------------------------- |
| `pending`   | Job created, waiting for resources          |
| `queued`    | Queued in Celery, will start soon           |
| `running`   | Training in progress                        |
| `completed` | Training finished successfully              |
| `failed`    | Training failed (see error_message)         |
| `cancelled` | Job cancelled by user                       |

## Database Schema

### training_jobs Table

Stores high-level job status and current metrics.

```sql
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY,
    job_name VARCHAR(255) NOT NULL,
    celery_task_id VARCHAR(255) UNIQUE,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    config JSONB NOT NULL,
    
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,
    progress_percent FLOAT DEFAULT 0.0,
    
    train_loss FLOAT,
    val_loss FLOAT,
    train_accuracy FLOAT,
    val_accuracy FLOAT,
    learning_rate FLOAT,
    
    best_epoch INTEGER,
    best_val_loss FLOAT,
    
    checkpoint_path VARCHAR(512),
    onnx_model_path VARCHAR(512),
    mlflow_run_id VARCHAR(255),
    
    error_message TEXT,
    
    dataset_size INTEGER,
    train_samples INTEGER,
    val_samples INTEGER,
    model_architecture VARCHAR(100)
);
```

### training_metrics Table

TimescaleDB hypertable for detailed per-epoch metrics.

```sql
CREATE TABLE training_metrics (
    timestamp TIMESTAMP NOT NULL,
    id UUID NOT NULL,
    training_job_id UUID REFERENCES training_jobs(id),
    epoch INTEGER NOT NULL,
    batch INTEGER,
    
    train_loss FLOAT,
    val_loss FLOAT,
    train_accuracy FLOAT,
    val_accuracy FLOAT,
    learning_rate FLOAT,
    gradient_norm FLOAT,
    
    phase VARCHAR(20), -- 'train', 'val', 'test'
    
    PRIMARY KEY (timestamp, id)
);
```

## Frontend Integration

### React Hook Example

```typescript
import { useEffect, useState } from 'react';

interface TrainingStatus {
  job_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  progress_percent: number;
  metrics: {
    train_loss: number;
    val_loss: number;
    train_accuracy: number;
    val_accuracy: number;
  };
}

function useTrainingWebSocket(jobId: string) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8001/ws/training/${jobId}`);
    
    ws.onopen = () => {
      console.log('Training WebSocket connected');
      setConnected(true);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.event === 'training_status') {
        setStatus(data);
      }
    };
    
    ws.onclose = () => {
      console.log('Training WebSocket disconnected');
      setConnected(false);
    };
    
    ws.onerror = (error) => {
      console.error('Training WebSocket error:', error);
    };
    
    // Ping every 30 seconds
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ event: 'ping' }));
      }
    }, 30000);
    
    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [jobId]);
  
  return { status, connected };
}
```

### Usage in Component

```typescript
function TrainingMonitor({ jobId }: { jobId: string }) {
  const { status, connected } = useTrainingWebSocket(jobId);
  
  if (!connected) {
    return <div>Connecting...</div>;
  }
  
  if (!status) {
    return <div>Waiting for data...</div>;
  }
  
  return (
    <div>
      <h2>{status.status}</h2>
      <progress 
        value={status.current_epoch} 
        max={status.total_epochs}
      />
      <p>Epoch {status.current_epoch}/{status.total_epochs}</p>
      <p>Train Loss: {status.metrics.train_loss?.toFixed(4)}</p>
      <p>Val Loss: {status.metrics.val_loss?.toFixed(4)}</p>
      <p>Accuracy: {(status.metrics.val_accuracy * 100).toFixed(2)}%</p>
    </div>
  );
}
```

## Testing

### Create Job

```bash
curl -X POST http://localhost:8001/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "Test Training",
    "config": {
      "epochs": 10,
      "batch_size": 16
    }
  }'
```

### Monitor via WebSocket (using websocat)

```bash
# Install: cargo install websocat
websocat ws://localhost:8001/ws/training/{job_id}
```

### Check Job Status

```bash
curl http://localhost:8001/api/v1/training/jobs/{job_id} | jq .
```

### List All Jobs

```bash
curl http://localhost:8001/api/v1/training/jobs | jq .
```

## Performance Considerations

- **Database Polling**: WebSocket polls database every 5 seconds
- **Future Enhancement**: Use Redis pub/sub for true push notifications
- **Metrics Storage**: TimescaleDB efficiently stores per-epoch metrics
- **Connection Limit**: No hard limit, but monitor backend memory

## Error Handling

### Job Failures

If training fails, the job status becomes `failed` and `error_message` contains details:

```json
{
  "status": "failed",
  "error_message": "CUDA out of memory. Tried to allocate 2.00 GB..."
}
```

### WebSocket Disconnection

- Client should auto-reconnect with exponential backoff
- Server closes connection when job completes/fails
- Ping/pong keeps connection alive

## Next Steps

1. **Phase 7 (Current)**: Frontend UI with Recharts visualization
2. **Phase 8**: Redis pub/sub for real push notifications
3. **Phase 9**: TensorBoard integration
4. **Phase 10**: Multi-GPU training support

## See Also

- [TRAINING.md](TRAINING.md) - Full training pipeline documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [API.md](API.md) - General API documentation
