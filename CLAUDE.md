# Heimdall SDR - Codebase Architecture Guide for Claude Code

This document provides a high-level architectural overview of the Heimdall SDR platform to help Claude Code (and other AI assistants) understand the project structure, design patterns, and component interactions.

**Last Updated**: November 2024  
**Project Phase**: 7/11 (Frontend Development)

---

## 1. Project Overview

**Heimdall** is a distributed Software-Defined Radio (SDR) monitoring and analysis platform that:
- Aggregates real-time data from 7 WebSDR receivers across Europe
- Detects radio frequency anomalies using machine learning
- Provides localization of radio transmissions with ±30m accuracy
- Offers real-time visualization through a React web interface

**Key Metrics**:
- Processing latency: <500ms average (52ms API response time)
- Accuracy: ±30m (68% confidence)
- Concurrent capacity: 50+ simultaneous tasks
- Test coverage: >80% across all services

---

## 2. Architecture Patterns

### 2.1 Microservices Architecture

The system is organized as **loosely-coupled microservices** communicating through well-defined interfaces:

| Service | Port | Language | Purpose |
|---------|------|----------|---------|
| **Backend** | 8001 | Python/FastAPI | Core API, RF acquisition, state management |
| **Training** | 8002 | Python/FastAPI | ML model training pipeline |
| **Inference** | 8003 | Python/FastAPI | Real-time anomaly detection |
| **Data Ingestion Web** | 8004 | Python/FastAPI | Data collection UI (disabled) |
| **Frontend** | 3000 | React/TypeScript | Web UI with Mapbox integration |
| **Keycloak** | 8080 | Java | Identity & Access Management |
| **API Gateway (Envoy)** | 80/443 | C++ | Reverse proxy, rate limiting, routing |

**Communication Patterns**:
- **Synchronous**: REST APIs (HTTP/JSON)
- **Asynchronous**: RabbitMQ message queues (AMQP)
- **Real-time**: WebSocket connections (ws://)
- **Caching**: Redis (session, temporary data)

### 2.2 Event-Driven Architecture

Data flows through a **publish-subscribe pipeline** using RabbitMQ:

```
WebSDR Sources
    ↓
[Backend Collector] → RabbitMQ (signal.collected.*)
    ↓
[Signal Processor] → RabbitMQ (signal.processed.*)
    ↓
[ML Detector] → RabbitMQ (signal.anomaly.*)
    ↓
[Alert Manager, Frontend WebSocket]
```

**Exchange & Queue Topology**:
- `heimdall.signals` (topic exchange): Signal data flow
- `heimdall.ml` (direct exchange): Training/inference tasks
- `heimdall.events` (topic exchange): Real-time event broadcasting for WebSocket updates
- Routing keys include station_id and severity for fine-grained filtering

#### Real-Time Event Broadcasting Pattern

**Problem**: Celery tasks running in worker processes cannot directly call async WebSocket broadcast methods in FastAPI due to different event loops, causing `RuntimeError: This event loop is already running` or silent failures.

**Solution**: Use RabbitMQ as an event bus between Celery workers and FastAPI WebSocket manager:

```
Celery Task (Sync)
    ↓
EventPublisher → RabbitMQ (heimdall.events exchange)
    ↓
RabbitMQEventConsumer (FastAPI startup)
    ↓
WebSocket Manager → Connected Clients
```

**Implementation**:
- **Publisher** (`services/backend/src/events/publisher.py`): Singleton `EventPublisher` class with methods for different event types (WebSDR health, service health, signal detection, etc.)
- **Consumer** (`services/backend/src/events/consumer.py`): `RabbitMQEventConsumer` using `ConsumerMixin` for robust connection handling
- **Integration**: Consumer runs in background thread, started in FastAPI `@app.on_event("startup")`
- **Event Loop Bridging**: Uses `asyncio.run_coroutine_threadsafe()` to safely schedule WebSocket broadcasts in FastAPI's event loop

**Key Benefits**:
- Decouples Celery workers from FastAPI WebSocket layer
- Automatic reconnection on connection failures (ConsumerMixin)
- Fire-and-forget publishing (non-blocking, failures don't crash tasks)
- Scalable pattern for all real-time updates (health, signals, training progress, etc.)

**Usage Example**:
```python
# In Celery task
from ..events.publisher import get_event_publisher

publisher = get_event_publisher()
publisher.publish_websdr_health(health_data)
# Event automatically flows to WebSocket clients via RabbitMQ
```

**Routing Keys**:
- `websdr.health.update`: WebSDR station status
- `service.health.*`: Microservice health updates
- `signal.detected`: Signal detection events
- `training.progress.*`: Training job updates
- `localization.complete`: Localization results

### 2.3 Data Architecture

**Three-tier storage strategy**:

1. **PostgreSQL + TimescaleDB** (Primary)
   - Transactional data (configurations, sessions)
   - Time-series measurements (with automatic compression)
   - ML model metadata
   - Search path: `SET search_path TO heimdall, public`

2. **Redis** (Cache & Session)
   - Real-time signal data (TTL: 60s)
   - Aggregated statistics (TTL: 300s)
   - Active ML models (TTL: 3600s)
   - User sessions (TTL: 86400s)

3. **MinIO** (Object Storage - S3-compatible)
   - Raw IQ recordings: `heimdall-raw-iq/year/month/day/station_id/frequency_hz/`
   - ML artifacts: `heimdall-models/models/name/version/`
   - Datasets: `heimdall-datasets/`
   - MLflow artifacts: `heimdall-mlflow/`

---

## 3. Backend Services Architecture

### 3.1 Backend Service (FastAPI)

**File Structure**:
```
services/backend/src/
├── main.py              # App initialization, health checks, Celery setup
├── config.py            # Settings (DATABASE_URL, REDIS_URL, etc.)
├── db.py                # Connection pooling
├── models/              # Pydantic data models
│   └── health.py       # HealthResponse schema
├── routers/             # API route handlers
│   ├── acquisition.py   # WebSDR acquisition endpoints
│   ├── sessions.py      # Recording session management
│   └── websocket.py     # Real-time WebSocket streams
├── fetchers/            # WebSDR data collection
├── processors/          # Data preprocessing
├── storage/             # S3/MinIO operations
├── tasks/               # Celery async tasks
└── utils/               # Helper utilities
```

**Key Components**:

1. **Health System**
   - `/health` - Liveness probe (always succeeds)
   - `/ready` - Readiness probe (checks dependencies)
   - `/health/detailed` - Full dependency status
   - Registered checks: PostgreSQL, Redis, RabbitMQ, MinIO

2. **Celery Configuration**
   - Broker: RabbitMQ (`amqp://guest:guest@localhost:5672/`)
   - Result backend: Redis (DB 1)
   - Beat schedule: Monitor WebSDR uptime every 60s
   - Task TTL: 30 minutes (soft: 25 minutes)

3. **API Endpoints**
   - `GET /api/v1/signals/*` - Signal queries
   - `POST /acquisition/trigger` - Start recording sessions
   - `WS /ws/signals/live` - Real-time signal streaming
   - Authentication: JWT from Keycloak

### 3.2 Training Service

**Purpose**: ML model training pipeline

**File Structure**:
```
services/training/src/
├── main.py              # FastAPI app, health checks
├── config/
│   ├── settings.py      # Environment configuration
│   └── model_config.py   # Model hyperparameters
├── data/
│   ├── dataset.py       # Data loading from DB/MinIO
│   └── features.py      # Feature engineering
├── models/
│   ├── lightning_module.py # PyTorch Lightning wrapper
│   └── localization_net.py  # Neural network architecture
├── mlflow_setup.py      # MLflow experiment tracking
├── train.py             # Training pipeline
└── onnx_export.py       # Model export for inference
```

**ML Pipeline**:
1. Load measurements from PostgreSQL
2. Feature extraction (spectral, temporal, statistical)
3. Train ensemble models (Isolation Forest, LSTM Autoencoder, VAE)
4. Export to ONNX format for inference service
5. Register with MLflow model registry
6. Store artifacts in MinIO

### 3.3 Inference Service

**Purpose**: Real-time anomaly detection and model serving

**File Structure**:
```
services/inference/src/
├── main.py              # FastAPI app
├── models/
│   ├── onnx_loader.py   # Load ONNX models
│   └── schemas.py       # Request/response models
├── routers/
│   ├── predict.py       # Inference endpoints
│   └── analytics.py     # Analytics queries
└── utils/
    ├── batch_predictor.py # Batch inference
    ├── preprocessing.py    # Feature normalization
    ├── uncertainty.py      # Confidence scoring
    ├── model_versioning.py # Model switching
    └── cache.py           # Result caching
```

**Key Features**:
- Load multiple ONNX models in-memory
- Batch prediction for performance
- Uncertainty quantification
- Result caching (Redis)
- Model metadata endpoints

### 3.4 Common Module (Shared Code)

**Purpose**: Shared utilities across all Python services

**File Structure**:
```
services/common/
├── auth/
│   ├── keycloak_auth.py   # JWT validation
│   └── models.py          # TokenData, User schemas
├── health.py              # HealthChecker class
├── dependency_checkers.py # check_postgresql(), check_redis(), etc.
├── schemas.py             # Common Pydantic models
└── test_fixtures.py       # Pytest fixtures
```

**Import Pattern**:
```python
# In any service main.py:
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from common.health import HealthChecker
from common.auth import KeycloakAuth
```

---

## 4. Frontend Architecture

### 4.1 React App Structure

**File Structure**:
```
frontend/src/
├── main.tsx             # Entry point (Vite)
├── App.tsx              # Root component, routing
├── pages/               # Page components (lazy-loaded)
│   ├── Dashboard.tsx
│   ├── Analytics.tsx
│   ├── RecordingSession.tsx
│   └── ...
├── components/
│   ├── layout/          # Layout components
│   ├── Map/             # Mapbox integration
│   ├── Navigation/      # Navigation components
│   ├── ui/              # Reusable UI components
│   └── widgets/         # Dashboard widgets
├── hooks/               # Custom React hooks
│   ├── useTokenRefresh.ts # Auto token renewal
│   └── useWebSocket.ts    # WebSocket management
├── store/               # Zustand state stores
│   ├── authStore.ts     # Authentication state
│   ├── dashboardStore.ts # Dashboard state
│   ├── acquisitionStore.ts # Recording sessions
│   └── ...
├── services/api/        # API client functions
│   ├── index.ts
│   ├── acquisition.ts
│   ├── analytics.ts
│   └── schemas.ts       # TypeScript interfaces
├── contexts/            # React Context (WebSocket)
├── lib/                 # Utility libraries
├── types/               # TypeScript type definitions
└── utils/               # Helper functions
```

### 4.2 State Management (Zustand)

**Pattern**: Multiple focused stores rather than single Redux store

```typescript
// stores follow this pattern:
import { create } from 'zustand';

interface StoreState {
  data: Type[];
  loading: boolean;
  actions: {
    fetchData: () => Promise<void>;
    updateData: (item: Type) => void;
  };
}

export const useStore = create<StoreState>((set) => ({
  data: [],
  loading: false,
  actions: { /* ... */ }
}));

// Usage in components:
const { data, loading, actions } = useStore();
```

**Store Inventory**:
- `authStore` - User auth, token management
- `dashboardStore` - Dashboard UI state
- `acquisitionStore` - Recording session state
- `analyticsStore` - Analytics filters/results
- `websdrStore` - WebSDR station management
- `localizationStore` - Localization results
- `systemStore` - System status
- `sessionStore` - User sessions

### 4.3 API Communication

**Pattern**: Axios-based API client with typed responses

```typescript
// services/api/index.ts
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '10000'),
});

// Add JWT token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

**Service Modules** (under `services/api/`):
- `acquisition.ts` - Recording/session APIs
- `analytics.ts` - Analytics queries
- `inference.ts` - Model prediction APIs
- `websdr.ts` - Station management
- `system.ts` - System status
- `session.ts` - User sessions

### 4.4 Real-Time Communication

**WebSocket Provider Pattern**:
```typescript
// contexts/WebSocketContext.tsx
interface WebSocketContextType {
  isConnected: boolean;
  subscribe: (channel: string, handler: MessageHandler) => void;
  unsubscribe: (channel: string) => void;
}

// Usage:
const { isConnected, subscribe } = useContext(WebSocketContext);

useEffect(() => {
  subscribe('signals', (data) => {
    // Handle real-time signal updates
  });
}, []);
```

**Data Flow**:
1. Frontend establishes WebSocket connection to `/ws/signals/live`
2. Backend publishes updates to Redis pub/sub
3. WebSocket router forwards to connected clients
4. Zustand stores update automatically
5. Components re-render with new data

### 4.5 Key Technologies

| Layer | Technology | Usage |
|-------|-----------|-------|
| **Framework** | React 19 + TypeScript | Component-based UI |
| **Build** | Vite + Rolldown | Fast dev/build |
| **Routing** | React Router v7 | Client-side navigation |
| **State** | Zustand 5 | Global state management |
| **Forms** | React Hook Form | Form state & validation |
| **API** | Axios + TanStack Query | HTTP client & caching |
| **Charts** | Chart.js + react-chartjs-2 | Data visualization |
| **Maps** | Mapbox GL | Geographic visualization |
| **UI Components** | Radix UI | Accessible UI primitives |
| **Styling** | Tailwind CSS | Utility-first CSS |
| **Testing** | Vitest + Playwright | Unit & E2E tests |

---

## 5. Database Architecture

### 5.1 Schema Organization

**Schema**: `heimdall` (set in search_path)

**Core Tables**:

1. **websdr_stations**
   - WebSDR receiver configurations
   - Columns: id, name, url, country, latitude/longitude, frequency ranges
   - Indexes: station_id for fast lookups

2. **measurements** (TimescaleDB hypertable)
   - Time-series signal data
   - Partitioned by timestamp (automatic compression after 30 days)
   - Columns: timestamp, frequency_hz, signal_strength_db, snr_db, iq_data_location
   - Indexes: (websdr_station_id, timestamp DESC), (frequency_hz, timestamp DESC)

3. **known_sources**
   - Reference transmitters (beacons, broadcasts)
   - Columns: name, frequency_hz, latitude, longitude, power_dbm, source_type

4. **models**
   - ML model metadata
   - Columns: model_name, model_type, mlflow_run_id, onnx_model_location, accuracy_meters

5. **inference_requests** (TimescaleDB hypertable)
   - Prediction history for audit trail
   - Columns: timestamp, model_id, input_features, output_prediction

6. **recording_sessions**
   - User-initiated recording sessions
   - Status: pending, running, completed, failed
   - Approval status: pending, approved, rejected

### 5.2 Migrations

**Location**: `db/migrations/`

**Pattern**:
```sql
-- 001-feature-name.sql
-- Clear, descriptive filenames
-- Idempotent (use IF NOT EXISTS, etc.)
-- Single logical change per file
```

**Running Migrations**:
```bash
# Manual execution via psql
psql -h localhost -U heimdall_user -d heimdall -f db/migrations/001-*.sql

# Via Docker
docker exec heimdall-postgres psql -U heimdall_user -d heimdall -f /migrations/001-*.sql
```

### 5.3 Connection Management

**Connection Pooling** (in backend):
```python
# db.py
async def init_pool():
    """Initialize connection pool on startup."""
    global pool
    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=10,
        max_size=20,
    )

async def close_pool():
    """Close pool on shutdown."""
    await pool.close()
```

**Environment Variables**:
```
DATABASE_URL=postgresql://user:password@host:port/db
```

---

## 6. Configuration Management

### 6.1 Environment Variables

**Backend Services**:
```env
# Infrastructure
DATABASE_URL          # PostgreSQL connection string
REDIS_URL            # Redis connection
CELERY_BROKER_URL    # RabbitMQ AMQP URL
CELERY_RESULT_BACKEND_URL # Redis for Celery results
MINIO_URL            # MinIO S3 API endpoint
MINIO_ACCESS_KEY     # MinIO credentials
MINIO_SECRET_KEY

# Service Configuration
SERVICE_NAME         # Service identifier
SERVICE_PORT         # HTTP port
LOG_LEVEL           # logging level (INFO, DEBUG, etc.)
DEBUG               # Development mode

# CORS Configuration
CORS_ORIGINS        # Comma-separated allowed origins
CORS_ALLOW_CREDENTIALS
CORS_ALLOW_METHODS
CORS_ALLOW_HEADERS
CORS_MAX_AGE

# Authentication
KEYCLOAK_URL        # Keycloak server
KEYCLOAK_REALM      # Realm name
KEYCLOAK_CLIENT_ID  # Service client ID
```

**Frontend (Vite env vars)**:
```env
VITE_API_URL              # Backend API base URL (/api)
VITE_API_TIMEOUT          # Request timeout (ms)
VITE_ENV                  # Environment (development/production)
VITE_ENABLE_DEBUG         # Debug mode (auth bypass)
VITE_KEYCLOAK_URL         # Keycloak server
VITE_KEYCLOAK_REALM       # Realm name
VITE_KEYCLOAK_CLIENT_ID   # Frontend client ID
VITE_MAPBOX_TOKEN         # Mapbox GL token
VITE_SESSION_TIMEOUT      # Session TTL (ms)
```

### 6.2 Service Configuration Pattern

**Backend Example**:
```python
# services/backend/src/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://localhost/heimdall"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # CORS (comma-separated, converted to list)
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    cors_allow_credentials: bool = True
    
    @property
    def get_cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 7. Authentication & Authorization

### 7.1 Keycloak Integration

**Architecture**:
```
Frontend (OIDC/PKCE)
    ↓
Keycloak (OAuth2 Provider)
    ↓
API Gateway (JWT verification)
    ↓
Microservices (Token validation)
```

**Flow**:
1. Frontend redirects to Keycloak login
2. User authenticates
3. Keycloak returns JWT token
4. Frontend stores token in localStorage
5. Frontend includes token in `Authorization: Bearer {token}` header
6. Services verify token against Keycloak's JWKS endpoint

### 7.2 JWT Validation

**Common Module**:
```python
# services/common/auth/keycloak_auth.py
class KeycloakAuth:
    def __init__(self, keycloak_url, realm, client_id):
        self.jwks_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
        self.jwk_client = PyJWKClient(self.jwks_url)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify JWT and extract claims."""
        signing_key = self.jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            # audience, issuer verification
        )
        return TokenData(**payload)
```

### 7.3 Role-Based Access Control (RBAC)

**Roles**:
- `admin` - Full system access
- `operator` - Read/write operations
- `viewer` - Read-only access

**Enforcement**:
```python
# FastAPI dependencies
@app.post("/admin-only")
async def admin_endpoint(user: User = Depends(require_admin)):
    return {"access": "granted"}

# Defined in auth module
def require_admin(current_user: User = Depends(get_current_user)):
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
```

---

## 8. Deployment Architecture

### 8.1 Docker Compose Services

**Infrastructure Containers**:
- `postgres` - PostgreSQL 15 + TimescaleDB
- `rabbitmq` - RabbitMQ 3.12 (port 15672 for UI)
- `redis` - Redis 7-alpine
- `minio` - MinIO S3-compatible storage
- `minio-init` - Bucket initialization

**Application Containers**:
- `backend` - FastAPI service (port 8001)
- `training` - Training pipeline (port 8002)
- `inference` - Inference service (port 8003)
- `frontend` - React app via nginx (port 3000)
- `envoy` - API Gateway (ports 80/443)
- `keycloak` - Identity provider (port 8080)

**Health Checks**:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/ready"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### 8.2 Network Architecture

**Network**: `heimdall-network` (bridge driver)

**Service Discovery**: Container names are DNS resolvable within network
- `postgres:5432`
- `rabbitmq:5672`
- `redis:6379`
- `minio:9000`

### 8.3 Volume Management

**Persistent Volumes**:
- `postgres_data` - Database files
- `rabbitmq_data` - Queue persistence
- `redis_data` - Cache snapshots
- `minio_data` - Object storage
- `keycloak_data` - Identity data

---

## 9. Testing Architecture

### 9.1 Test Levels

**Unit Tests**:
- Location: `*/tests/test_*.py`
- Framework: pytest
- Mocking: pytest-mock, unittest.mock

**Integration Tests**:
- Test service interactions
- Real database/Redis connections
- Docker containers running

**E2E Tests**:
- Location: `frontend/e2e/` (Playwright)
- Test full user workflows
- Real backend API calls

### 9.2 Running Tests

```bash
# All tests in Docker
make test

# Tests locally (requires dependencies)
make test-local

# Unit tests only
make test-unit

# With coverage
make test-coverage

# Watch mode
make test-watch

# E2E tests
cd frontend && pnpm test:e2e
```

### 9.3 Test Fixtures

**Common Fixtures** (in `services/common/test_fixtures.py`):
```python
@pytest.fixture
def test_db_connection():
    """Provide test database connection."""
    # Setup
    yield connection
    # Cleanup

@pytest.fixture
def test_redis_client():
    """Provide test Redis client."""
    client = redis.Redis(host='localhost')
    yield client
    client.flushall()
```

---

## 10. Development Workflow

### 10.1 Getting Started

```bash
# 1. Clone and navigate
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# 2. Setup environment
cp .env.example .env

# 3. Start services
make dev-up

# 4. Run tests
make test

# 5. Start developing
# Edit code, changes are auto-detected
```

### 10.2 Common Commands

```bash
# Infrastructure
make dev-up           # Start all services
make dev-down         # Stop services
make infra-status     # Show container status
make health-check     # Verify all services ready

# Testing
make test             # Run all tests
make test-unit        # Unit tests only
make test-e2e         # E2E tests

# Code Quality
make lint             # Check code style
make format           # Auto-format code
make type-check       # TypeScript type checking
make type-check-strict # mypy strict mode

# Database
make db-migrate       # Run migrations
make postgres-connect # Connect to psql CLI

# UI Dashboards
make rabbitmq-ui      # Open RabbitMQ console (port 15672)
make minio-ui         # Open MinIO console (port 9001)
make grafana-ui       # Open Grafana (port 3001)
```

### 10.3 Adding a New Feature

1. **Backend API Endpoint**:
   ```python
   # services/backend/src/routers/feature.py
   from fastapi import APIRouter, Depends
   from common.auth import get_current_user
   
   router = APIRouter(prefix="/api/v1/feature", tags=["feature"])
   
   @router.get("/")
   async def get_feature(user: User = Depends(get_current_user)):
       """Get feature data."""
       return {"data": []}
   
   # In main.py:
   app.include_router(feature.router)
   ```

2. **Frontend Store**:
   ```typescript
   // frontend/src/store/featureStore.ts
   import { create } from 'zustand';
   
   interface FeatureStore {
     data: Feature[];
     loading: boolean;
     fetchData: () => Promise<void>;
   }
   
   export const useFeatureStore = create<FeatureStore>((set) => ({
     data: [],
     loading: false,
     fetchData: async () => {
       set({ loading: true });
       const data = await api.get('/api/v1/feature');
       set({ data, loading: false });
     },
   }));
   ```

3. **Frontend Page/Component**:
   ```typescript
   // frontend/src/pages/Feature.tsx
   import { useFeatureStore } from '../store/featureStore';
   
   export default function FeaturePage() {
     const { data, loading, fetchData } = useFeatureStore();
     
     useEffect(() => {
       fetchData();
     }, []);
     
     if (loading) return <LoadingSpinner />;
     return <div>{/* render data */}</div>;
   }
   ```

4. **Add Tests**:
   ```python
   # services/backend/tests/test_feature.py
   import pytest
   from httpx import AsyncClient
   
   @pytest.mark.asyncio
   async def test_get_feature(client: AsyncClient):
       response = await client.get("/api/v1/feature")
       assert response.status_code == 200
   ```

5. **Update Routes** (if new file):
   ```python
   # services/backend/src/main.py
   from .routers.feature import router as feature_router
   
   app.include_router(feature_router)
   ```

---

## 11. Key Design Decisions

### 11.1 Why This Architecture?

| Decision | Reasoning |
|----------|-----------|
| **Microservices** | Independent scaling, fault isolation, technology flexibility |
| **RabbitMQ for async** | Decouples services, enables complex pipelines, durable queues |
| **Redis for cache** | Sub-millisecond latency, pub/sub for real-time updates |
| **PostgreSQL + TimescaleDB** | ACID compliance, time-series optimizations, complex queries |
| **MinIO for object storage** | S3-compatible, self-hosted, scalable IQ data storage |
| **React + Zustand** | Lightweight, flexible state management, great TypeScript support |
| **Keycloak for auth** | Centralized IAM, OpenID Connect standard, flexible RBAC |
| **Docker Compose** | Multi-container orchestration, volume management, networking |

### 11.2 Important Patterns

**Graceful Degradation**: Services don't fail if non-critical dependencies are down
- Backend works without RabbitMQ (queues local)
- Frontend works in debug mode without Keycloak
- Inference runs with cached models if database is down

**Health Checks**: Three-tier health probing
- `/health` - Always responds (process alive)
- `/ready` - Checks dependencies (ready for requests)
- `/health/detailed` - Full dependency breakdown

**Configuration as Code**: Environment variables override defaults
- No hardcoded credentials
- Easy Docker/K8s deployment
- Development/production parity

**Error Handling**: Consistent error responses
```python
# Standard error format
{
    "detail": "error message",
    "status_code": 400,
    "timestamp": "2024-11-01T12:00:00Z"
}
```

---

## 12. Common Troubleshooting

### Service Won't Start

1. **Check health**: `make health-check`
2. **View logs**: `docker compose logs service_name`
3. **Check ports**: `netstat -tlnp | grep :8001`
4. **Reset**: `make dev-down && make dev-up`

### Database Connection Issues

```bash
# Connect directly to test
make postgres-connect
SELECT * FROM heimdall.websdr_stations;

# Check current schema
SET search_path TO heimdall, public;
\dt
```

### Frontend Not Loading

1. Check VITE_API_URL matches backend address
2. Verify CORS settings in backend
3. Check browser console for errors
4. Verify Keycloak is accessible

### Redis/RabbitMQ Not Responsive

```bash
# Test Redis
docker exec heimdall-redis redis-cli -a changeme ping

# Test RabbitMQ
docker exec heimdall-rabbitmq rabbitmq-diagnostics -q ping

# Check RabbitMQ UI
open http://localhost:15672
# Login: guest/guest
```

---

## 13. Directory Index

```
heimdall/
├── .github/                    # GitHub Actions, agents
├── db/                         # Database init, migrations
│   ├── 01-init.sql
│   ├── migrations/
│   ├── keycloak/              # Keycloak realm config
│   ├── envoy/                 # API Gateway config
│   └── rabbitmq.conf
├── services/
│   ├── backend/               # Main backend service
│   ├── training/              # ML training pipeline
│   ├── inference/             # Inference service
│   ├── data-ingestion-web/    # Data collection (disabled)
│   ├── common/                # Shared code
│   │   ├── auth/             # Keycloak integration
│   │   ├── health.py         # Health checking
│   │   └── dependency_checkers.py
│   ├── requirements/          # Centralized deps
│   └── tests/                # Cross-service tests
├── frontend/                  # React application
│   ├── src/
│   │   ├── pages/            # Page components
│   │   ├── components/       # Reusable components
│   │   ├── store/            # Zustand stores
│   │   ├── services/api/     # API clients
│   │   ├── hooks/            # Custom hooks
│   │   └── lib/              # Utilities
│   └── e2e/                  # Playwright tests
├── docs/                      # User documentation
├── scripts/                   # Utility scripts
├── docker-compose.yml         # Development setup
├── docker-compose.prod.yml    # Production setup
├── Makefile                   # Development commands
├── pyproject.toml             # Python project config
├── package.json               # Frontend package config
└── CLAUDE.md                  # This file
```

---

## 14. Quick Reference: Component Interactions

### Data Flow: WebSDR → Storage

```
WebSDR Station
    ↓
[Backend Collector via WebSocket/HTTP]
    ↓
PostgreSQL: measurements table (TimescaleDB)
    ↓
MinIO: Raw IQ data (S3 bucket)
    ↓
Redis: Cached results
    ↓
[Frontend] displays via Charts + Map
```

### ML Pipeline: Training → Inference

```
Measurements in PostgreSQL
    ↓
[Training Service] loads dataset
    ↓
Feature extraction
    ↓
PyTorch Lightning trains models
    ↓
ONNX export
    ↓
MinIO: Store models
    ↓
[Inference Service] loads ONNX
    ↓
Real-time anomaly detection
    ↓
Results to PostgreSQL + Redis
    ↓
[Frontend] displays alerts
```

### Real-Time Updates: Backend → Frontend

```
PostgreSQL: Signal detection
    ↓
RabbitMQ: Publish anomaly event
    ↓
[Backend] subscribes, sends to Redis pub/sub
    ↓
[Frontend] WebSocket listener
    ↓
Zustand store update
    ↓
React component re-render
```

---

## 15. Performance Characteristics

| Component | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| API Response | 20ms | 52ms | 100ms |
| Database Query | 5ms | 15ms | 50ms |
| Signal Processing | 50ms | 200ms | 500ms |
| Model Inference | 10ms | 30ms | 100ms |
| WebSocket Latency | <10ms | <50ms | <100ms |

**Concurrency Limits**:
- Backend API: 1000+ concurrent connections
- Signal processing: 50 concurrent signals
- Model inference: 100 concurrent predictions
- Database: 20 connection pool

---

## 16. Security Considerations

**Authentication**: 
- All APIs require JWT from Keycloak
- Tokens stored in localStorage (consider HttpOnly cookies)
- Token refresh every hour

**Authorization**:
- Role-based access control (admin/operator/viewer)
- Endpoint-level permission checks
- Resource ownership validation

**Data Protection**:
- Database credentials in environment variables only
- MinIO access keys rotated regularly
- Redis password-protected
- HTTPS/TLS in production (Envoy handles)

**Rate Limiting**:
- Envoy provides global rate limiting
- Per-endpoint limits in routers
- Database connection pooling prevents exhaustion

---

## 17. Performance Optimization Tips

1. **Frontend**:
   - Lazy load pages (already implemented)
   - Use React Query for caching (ready to add)
   - Monitor bundle size with `pnpm build:analyze`

2. **Backend**:
   - Use connection pooling (configured: min 10, max 20)
   - Cache with Redis (TTL-based expiration)
   - Batch database operations

3. **Database**:
   - Indexes on frequent queries (already added)
   - TimescaleDB compression for old data
   - Partitioning by time (automatic)

4. **Network**:
   - WebSocket for real-time (not REST polling)
   - Gzip compression (Envoy handles)
   - CDN for static assets

---

## 18. Resources & Links

- **Project Repository**: https://github.com/fulgidus/heimdall
- **Documentation**: https://fulgidus.github.io/heimdall/
- **Issues**: https://github.com/fulgidus/heimdall/issues
- **Contributing**: CONTRIBUTING.md

---

## 19. Version Information

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11+ | FastAPI, PyTorch, MLflow |
| Node.js | 18+ | React, Vite |
| PostgreSQL | 15+ | TimescaleDB extension |
| Docker | 20.10+ | Multi-stage builds |
| Kubernetes | 1.24+ | Helm charts available |

---

**Last Updated**: November 1, 2024  
**Maintained By**: Heimdall Team  
**Questions?** Open an issue or discussion on GitHub
