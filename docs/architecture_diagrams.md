# Architecture Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│                    http://localhost:3000                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    API Gateway (FastAPI)                        │
│                    http://localhost:8000                        │
├────────────────────────────┬────────────────────────────────────┤
│  Task Submission  │  Result Retrieval  │  Configuration API      │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   RabbitMQ             Redis Cache          PostgreSQL
   (Queue)              (Session/Cache)       (TimescaleDB)
        │                    │                    │
        ▼                    │                    │
   Celery Workers            │                    │
        │                    │                    │
        ├─► RF Acquisition   │                    │
        ├─► Signal Processor │                    │
        └─► ML Inference     │                    │
             │               │                    │
             └───────────────┼────────────────────┘
                             │
                             ▼
                         MinIO (S3)
                     Object Storage
```

## RF Acquisition Pipeline

```
WebSDR Stations (7x)
        │
        ├─► WebSDR Collector Service
        │
        ├─► Signal Preprocessing
        │   - IQ demodulation
        │   - Bandpass filtering
        │   - Automatic gain control
        │
        ├─► Feature Extraction
        │   - Mel-spectrogram
        │   - Power spectral density
        │   - Signal-to-noise ratio
        │
        ├─► Database Storage (TimescaleDB)
        │   - Signal measurements
        │   - Metadata
        │   - Quality metrics
        │
        └─► Localization Pipeline
            - Multilateration
            - Least-squares estimation
            - Uncertainty quantification
```

## Machine Learning Pipeline

```
Training Data
     │
     ▼
PyTorch Lightning
     │
     ├─► GPU Training
     ├─► Validation
     └─► Testing
           │
           ▼
      MLflow Tracking
        │
        ├─► Experiment Registry
        ├─► Model Versioning
        └─► Metrics Dashboard
              │
              ▼
         Best Model
              │
              ├─► PyTorch Format
              ├─► ONNX Export
              └─► Model Registry
```

## Infrastructure Components

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐       │
│  │ PostgreSQL  │  │  Redis   │  │   RabbitMQ   │       │
│  │ + TimescaleDB  │          │  │              │       │
│  └─────────────┘  └──────────┘  └──────────────┘       │
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐       │
│  │    MinIO    │  │  MLflow  │  │   Prometheus │       │
│  │   (S3)      │  │          │  │              │       │
│  └─────────────┘  └──────────┘  └──────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Application Services                   │  │
│  │                                                  │  │
│  │  • API Gateway (FastAPI)                        │  │
│  │  • WebSDR Collector                             │  │
│  │  • Signal Processor                             │  │
│  │  • ML Detector / Inference                      │  │
│  │  • Celery Workers                               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
User Request
     │
     ▼
API Gateway
     │
     ├─► Validate Input
     ├─► Create Task
     └─► Submit to Queue
              │
              ▼
        RabbitMQ
              │
              ▼
     Celery Worker
              │
              ├─► RF Acquisition (from WebSDRs)
              │   └─► Store in MinIO
              │
              ├─► Signal Processing
              │   └─► Extract Features
              │
              ├─► ML Inference
              │   └─► Generate Predictions
              │
              └─► Store Results
                  ├─► PostgreSQL
                  └─► Redis Cache
                       │
                       ▼
              Return to User
```

## Deployment Architecture

### Development
- Single Docker Compose stack
- All services on localhost
- SQLite or local PostgreSQL
- Local MinIO

### Production
- Kubernetes cluster
- Microservices separation
- Cloud PostgreSQL
- Cloud object storage (S3/GCS)
- Multiple replicas for scalability

```
┌────────────────────────────────────────────────────────┐
│              Kubernetes Cluster                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Frontend    │  │ API Gateway  │  │  Workers   │  │
│  │  Pods (3x)   │  │  Pods (3x)   │  │  Pods (5x) │  │
│  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │         Persistent Storage                       │ │
│  │  • PostgreSQL StatefulSet                        │ │
│  │  • S3 Object Storage                             │ │
│  │  • Redis Cache                                   │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │         Monitoring & Logging                     │ │
│  │  • Prometheus                                    │ │
│  │  • Grafana                                       │ │
│  │  • ELK Stack                                     │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Communication Patterns

### Synchronous (REST API)
- Client → API Gateway (HTTP/HTTPS)
- Real-time responses
- Used for: Status queries, configuration

### Asynchronous (Message Queue)
- API Gateway → RabbitMQ → Celery Workers
- Fire-and-forget
- Used for: RF acquisition, processing

### Pub/Sub (WebSocket)
- API Gateway → Frontend (WebSocket)
- Real-time updates
- Used for: Live dashboard updates

---

**Related**: [Architecture Guide](./ARCHITECTURE.md) | [API Reference](./api_reference.md)
