# 🤖 AGENTS.md - Heimdall SDR Project Phase Management Guide

**Project**: Heimdall SDR Radio Source Localization  
**Owner**: fulgidus  
**Contributors**: fulgidus + 1  
**Last Updated**: 2025-10-21 09:53:48 UTC  
**License**: CC Non-Commercial

---

## 📑 Table of Contents

- [🤖 AGENTS.md - Heimdall SDR Project Phase Management Guide](#-agentsmd---heimdall-sdr-project-phase-management-guide)
    - [📑 Table of Contents](#-table-of-contents)
    - [🎯 Project Overview](#-project-overview)
        - [Mission](#mission)
        - [Key Deliverables](#key-deliverables)
        - [Success Metrics](#success-metrics)
    - [🏗️ Phase Structure](#️-phase-structure)
    - [🚀 PHASE 0: Repository Setup](#-phase-0-repository-setup)
        - [Objective](#objective)
        - [Tasks](#tasks)
        - [Checkpoints](#checkpoints)
        - [Knowledge Base](#knowledge-base)
    - [🗄️ PHASE 1: Infrastructure \& Database](#️-phase-1-infrastructure--database)
        - [Objective](#objective-1)
        - [Tasks](#tasks-1)
        - [Checkpoints](#checkpoints-1)
        - [Knowledge Base](#knowledge-base-1)
    - [🔧 PHASE 2: Core Services Scaffolding](#-phase-2-core-services-scaffolding)
        - [Objective](#objective-2)
        - [Tasks (high level)](#tasks-high-level)
        - [Checkpoints](#checkpoints-2)
    - [📡 PHASE 3: RF Acquisition Service](#-phase-3-rf-acquisition-service)
        - [Objective](#objective-3)
        - [Key Deliverables](#key-deliverables-1)
        - [Tasks (high level)](#tasks-high-level-1)
        - [Checkpoints](#checkpoints-3)
    - [🖥️ PHASE 4: Data Ingestion Web Interface](#️-phase-4-data-ingestion-web-interface)
        - [Objective](#objective-4)
        - [Tasks (high level)](#tasks-high-level-2)
        - [Checkpoints](#checkpoints-4)
    - [🧠 PHASE 5: Training Pipeline](#-phase-5-training-pipeline)
        - [Objective](#objective-5)
        - [Tasks (high level)](#tasks-high-level-3)
        - [Checkpoints](#checkpoints-5)
    - [⚡ PHASE 6: Inference Service](#-phase-6-inference-service)
        - [Objective](#objective-6)
        - [Tasks (high level)](#tasks-high-level-4)
        - [Checkpoints](#checkpoints-6)
        - [Knowledge Base](#knowledge-base-2)
    - [🎨 PHASE 7: Frontend](#-phase-7-frontend)
        - [Objective](#objective-7)
        - [Key Deliverables](#key-deliverables-2)
        - [Tasks (high level)](#tasks-high-level-5)
        - [Checkpoints](#checkpoints-7)
    - [☸️ PHASE 8: Kubernetes \& Deployment](#️-phase-8-kubernetes--deployment)
        - [Objective](#objective-8)
        - [Key Deliverables](#key-deliverables-3)
        - [Tasks (high level)](#tasks-high-level-6)
        - [Checkpoints](#checkpoints-8)
    - [🧪 PHASE 9: Testing \& QA](#-phase-9-testing--qa)
        - [Objective](#objective-9)
        - [Key Deliverables](#key-deliverables-4)
        - [Tasks (high level)](#tasks-high-level-7)
        - [Checkpoints](#checkpoints-9)
    - [📚 PHASE 10: Documentation \& Release](#-phase-10-documentation--release)
        - [Objective](#objective-10)
        - [Key Deliverables](#key-deliverables-5)
        - [Tasks (high level)](#tasks-high-level-8)
        - [Checkpoints](#checkpoints-10)
    - [👥 Agent Roles](#-agent-roles)
        - [Agent-Infrastructure (fulgidus)](#agent-infrastructure-fulgidus)
        - [Agent-Backend (fulgidus)](#agent-backend-fulgidus)
        - [Agent-ML (fulgidus)](#agent-ml-fulgidus)
        - [Agent-Frontend (contributor)](#agent-frontend-contributor)
        - [Agent-DevOps (fulgidus)](#agent-devops-fulgidus)
        - [Agent-QA (fulgidus + contributor)](#agent-qa-fulgidus--contributor)
        - [Agent-Documentation (fulgidus + contributor)](#agent-documentation-fulgidus--contributor)
    - [🧠 Knowledge Base \& Continuity](#-knowledge-base--continuity)
        - [Context Preservation Strategy](#context-preservation-strategy)
        - [Critical Knowledge Areas](#critical-knowledge-areas)
        - [Handoff Protocols](#handoff-protocols)
    - [✅ Checkpoint System](#-checkpoint-system)
        - [Validation Levels](#validation-levels)
        - [Checkpoint Dependencies](#checkpoint-dependencies)
        - [Rollback Triggers](#rollback-triggers)
    - [🔗 Dependency Matrix](#-dependency-matrix)
        - [Parallelization Opportunities](#parallelization-opportunities)
        - [Critical Path Analysis](#critical-path-analysis)
    - [🎯 Project Success Criteria](#-project-success-criteria)
        - [Technical Metrics](#technical-metrics)
        - [Operational Metrics](#operational-metrics)
        - [Community Metrics](#community-metrics)

---

## 🎯 Project Overview

### Mission

Develop an AI-driven platform for **real-time localization of radio sources** on amateur bands (2m/70cm), using triangulation from geographically distributed WebSDR receivers.

### Key Deliverables

- ✅ Microservices architecture (Python 3.11)
- ✅ ML training pipeline (PyTorch Lightning)
- ✅ Real-time inference engine
- ✅ React + Mapbox frontend
- ✅ Kubernetes deployment
- ✅ CI/CD pipeline

### Success Metrics

- Localization accuracy: ±30m (sigma from neural network)
- Real-time inference latency: <500ms
- Data ingestion: simultaneous from 7 WebSDR
- Uptime: 99.5% (k8s deployment)
- Test coverage: ≥80%

---

## 🏗️ Phase Structure

Each phase follows this pattern:

```
PHASE N: {Name}
Duration: X days
Dependencies: Previous phases
Assignee: Agent(s)
Deliverables: List
TASKS (checkboxes for progress)
CHECKPOINTS (validation gates)
KNOWLEDGE BASE (key learnings)
ROLLBACK PLAN (if needed)
NEXT PHASE ENTRY POINT
```

---

## 🚀 PHASE 0: Repository Setup

**Duration**: 1 day  
**Assignee**: fulgidus (owner) + 1 contributor  
**Status**: COMPLETED  
**Critical Path**: YES

### Objective

Initialize GitHub repository with complete scaffolding, documentation, and CI/CD foundation.

### Tasks

- [x] **T0.1**: Create GitHub repo `fulgidus/heimdall`
  - Visibility: Public
  - License: CC Non-Commercial
  - Initial branch: `develop`
  - Protected branches: `main`, `develop`

- [x] **T0.2**: Setup repository structure

Project root layout:

```
heimdall/
├── .github/
├── services/
├── frontend/
├── db/
├── docs/
├── helm/
├── .gitignore
├── .env.example
├── Makefile
├── README.md
├── AGENTS.md (this file)
├── CONTINUATION_PROMPT.md
├── WEBSDRS.md
├── SETUP.md
├── .copilot-instructions
└── LICENSE
```

- [x] **T0.3**: Create `.copilot-instructions` (350+ lines)
  - Sections: Vision, Architecture, Phases, Conventions, Troubleshooting

- [x] **T0.4**: Create foundational documentation
  - `README.md` - Project overview + quick start
  - `WEBSDRS.md` - 7 WebSDR receiver configuration
  - `SETUP.md` - Dev environment setup
  - `docs/ARCHITECTURE.md` - System design
  - `docs/API.md` - API specifications

- [x] **T0.5**: Setup CI/CD foundation
  - Create `.github/workflows/ci-test.yml` (pytest)
  - Create `.github/workflows/build-docker.yml` (Docker image build)
  - Create `.github/workflows/deploy-k8s.yml` (k8s deployment)

- [x] **T0.6**: Create `.env.example` template

Example contents:

```bash
# PostgreSQL
POSTGRES_DB=heimdall
POSTGRES_USER=heimdall_user
POSTGRES_PASSWORD=changeme

# RabbitMQ
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Redis
REDIS_PASSWORD=changeme

# MLflow
MLFLOW_TRACKING_URI=postgresql://...
MLFLOW_ARTIFACT_URI=s3://minio/mlflow
```

- [x] **T0.7**: Setup Makefile with common tasks

```makefile
make dev-up              # Start docker-compose
make dev-down            # Stop environment
make test                # Run all tests
make lint                # Black + Ruff
make db-migrate          # Alembic migrations
make build-docker        # Build all images
make format              # Auto-format code
```

- [x] **T0.8**: Invite contributors
  - Add 1 additional contributor to repo
  - Share `.copilot-instructions` and `CONTINUATION_PROMPT.md`
  - Setup branch protection rules

### Checkpoints

✅ CP0.1: Repository accessible and all files present

- Repo created with all structure
- GitHub Actions workflows visible
- Contributors can push to `develop`

✅ CP0.2: Documentation complete and clear

- `README.md` is comprehensive
- `SETUP.md` allows new dev to start locally in <1 hour
- WebSDRs verified as accessible

✅ CP0.3: CI/CD pipeline active

- Tests run on PR
- Docker builds complete
- No secrets in repo

### Knowledge Base

Key Decisions:

- Repository is public (CC Non-Commercial encourages open source)
- `develop` as main integration branch (gitflow)
- All services in `services/` directory for clarity
- Frontend separate for independent scaling

Commands Reference:

```bash
# Clone and setup
git clone https://github.com/fulgidus/heimdall.git
cd heimdall
cp .env.example .env
git checkout develop

# First time contributors
git flow init
git flow feature start myfeature
# ... work ...
git flow feature publish myfeature
# Create PR on GitHub
```

Rollback Plan

- Delete repo and recreate if needed (Phase 0 has no data)
- Revert to template if structure wrong

Next Phase Entry Point

When all tasks in Phase 0 complete:

```bash
git checkout develop
git pull origin develop
docker-compose up -d  # Ready for Phase 1
```

---

## 🗄️ PHASE 1: Infrastructure & Database

**Duration**: 2 days  
**Assignee**: Agent-Infrastructure (typically fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 0 ✅  
**Critical Path**: YES

### Objective

Setup all infrastructure components (databases, message queue, caching, object storage) as docker-compose services for local development.

### Tasks

- **T1.1**: Create `docker-compose.yml` with services:
  - PostgreSQL 15 + TimescaleDB extension
  - RabbitMQ 3.12 (with management UI)
  - Redis 7 (caching layer)
  - MinIO (S3-compatible object storage)
  - pgAdmin (database UI)
  - Prometheus + Grafana (monitoring)
  - Network: `heimdall-network`

- **T1.2**: Create `docker-compose.prod.yml` with persistent volumes, resource limits, health checks and logging configuration.

- **T1.3**: Setup PostgreSQL
  - Create `db/init-postgres.sql` (schema initialization)
  - Configure TimescaleDB extension
  - Create `heimdall` database and `heimdall_user`

- **T1.4**: Create Alembic migration framework

Project structure:

```
db/
├── alembic.ini
└── migrations/
    ├── versions/
    │   ├── 001_init_schema.py
    │   └── 002_timescaledb_hypertables.py
    └── env.py
```

- **T1.5**: Design and create database schema (tables: `known_sources`, `measurements`, `training_datasets`, `models`).

Schema examples (SQL omitted for brevity).

- **T1.6**: Setup MinIO (S3-compatible storage)
  - Create buckets: `heimdall-raw-iq`, `heimdall-models`, `heimdall-mlflow`
  - Configure access keys and test upload/download

- **T1.7**: Configure RabbitMQ
  - Create vhosts: `/`, `/production`
  - Create users: `guest` (dev), `app-user` (prod)
  - Configure exchanges and queues for acquisition and training

- **T1.8**: Setup Redis for caching and Celery result backend; add Redis Commander for debug

- **T1.9**: Create health check scripts: `scripts/health-check-postgres.py`, `scripts/health-check-rabbitmq.py`, etc.

- **T1.10**: Setup Prometheus monitoring with exporters for PostgreSQL, RabbitMQ, Redis

### Checkpoints

✅ CP1.1: All services running

```bash
docker-compose up -d
docker-compose ps  # All services "healthy"
```

✅ CP1.2: Database schema initialized

```bash
make db-migrate
psql -h localhost -U heimdall_user -d heimdall -c "\dt"
# Should show: known_sources, measurements, training_datasets, models
```

✅ CP1.3: Object storage functional

```bash
aws s3 ls s3://heimdall-raw-iq --endpoint-url http://localhost:9000
# Should show empty buckets or files
```

✅ CP1.4: Message queue functional

```bash
rabbitmqctl list_queues
# Should show: acquisition.websdr-fetch, training.job-queue
```

✅ CP1.5: All services connected and healthy

```bash
make health-check
# All services report 200 OK
```

### Knowledge Base

Key Learnings:

- TimescaleDB hypertables are essential for time-series query performance
- PostGIS `geography` type allows queries like "sources within X km"
- MinIO buckets should be created upfront (application doesn't auto-create)
- RabbitMQ vhosts separate dev from production concerns
- Health checks prevent "silent" failures when docker-compose starts

Credentials Matrix (dev defaults):

```
Service       | Default User  | Default Password | Port
------------------------------------------------------
PostgreSQL    | heimdall_user | changeme         | 5432
RabbitMQ      | guest         | guest            | 5672
MinIO         | minioadmin    | minioadmin       | 9000
Redis         | (no auth)     | changeme         | 6379
pgAdmin       | admin@pg.com  | admin            | 5050
```

Common Commands:

```bash
# Connect to PostgreSQL
psql -h localhost -U heimdall_user -d heimdall

# RabbitMQ UI
open http://localhost:15672 # guest/guest

# MinIO UI
open http://localhost:9001  # minioadmin/minioadmin

# Redis
redis-cli -p 6379 PING

# Prometheus
open http://localhost:9090
```

Rollback Plan

```bash
docker-compose down -v  # Remove volumes
docker-compose up -d    # Start fresh
make db-migrate         # Re-initialize schema
```

Next Phase Entry Point

When all checkpoints pass:

```bash
docker-compose ps
make db-migrate
# Ready for Phase 2
```

---

## 🔧 PHASE 2: Core Services Scaffolding

**Duration**: 1.5 days  
**Assignee**: Agent-Backend (fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 1 ✅  
**Critical Path**: YES

### Objective

Create base FastAPI applications for all microservices with proper structure, logging, and health checks.

### Tasks (high level)

- **T2.1**: Create service scaffold generator (`scripts/create_service.py`) to generate Dockerfile, requirements.txt, `src/main.py`, tests/

Example usage:

```bash
python scripts/create_service.py rf-acquisition
python scripts/create_service.py training
python scripts/create_service.py inference
python scripts/create_service.py data-ingestion-web
python scripts/create_service.py api-gateway
```

- **T2.2**: Create `services/rf-acquisition/` scaffold with FastAPI, Celery, aiohttp, structured logging.

- **T2.3**: Create `services/training/` scaffold with PyTorch Lightning, MLflow integration.

- **T2.4**: Create `services/inference/` scaffold with FastAPI and ONNX runtime support.

- **T2.5**: Create `services/data-ingestion-web/` scaffold with FastAPI + SQLAlchemy.

- **T2.6**: Create `services/api-gateway/` scaffold with middleware (auth, logging, rate limiting).

- **T2.7**: Create common `requirements-base.txt` and per-service requirements.

Example `services/common/requirements-base.txt`:

```
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
structlog==24.1.0
```

- **T2.8**: Create Dockerfile template for all services (multi-stage, healthcheck).

Example Dockerfile:

```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY src/ src/
HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8000/health')"
CMD ["python", "src/main.py"]
```

- **T2.9**: Create health check endpoint for all services (`/health`).

Example:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "rf-acquisition",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }
```

- **T2.10**: Setup structured logging via `structlog`.

- **T2.11**: Create `docker-compose.services.yml` to run all service containers in dev, with proper `depends_on`.

### Checkpoints

✅ CP2.1: All service scaffolds created (`services/*/src/main.py` exist)

✅ CP2.2: All services build successfully (`docker build services/<service>`)

✅ CP2.3: All services run with health checks (example ports 8001-8004)

✅ CP2.4: Services can connect to infrastructure (logs show successful connections)

Knowledge Base

- `src/main.py` - Entry point with FastAPI/Celery
- `src/config.py` - Configuration from `.env`
- `src/models/` - Data models (Pydantic, SQLAlchemy)
- `src/routers/` - API endpoints (FastAPI) or tasks (Celery)
- `src/utils/` - Helper functions
- `tests/` - Pytest test files (mirroring src structure)

Dockerfile Best Practices:

- Multi-stage build for smaller images
- Non-root user for security
- Health check endpoint required
- Logs to stdout (for docker-compose capture)

Rollback Plan

```bash
docker-compose down
rm -rf services/*/src/__pycache__
docker-compose up -d
```

Next Phase Entry Point

When all checkpoints pass, ready for Phase 3: Implement RF Acquisition.

---

## 📡 PHASE 3: RF Acquisition Service

**Duration**: 3 days  
**Assignee**: Agent-Backend (fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 2 ✅  
**Critical Path**: YES (blocks Phase 4 and 5)

### Objective

Implement simultaneous WebSDR data fetching, signal processing, and Celery task integration for human-assisted recording sessions.

### Key Deliverables

- Simultaneous fetch from 7 WebSDR URLs
- IQ data recording to MinIO
- Metadata (SNR, frequency offset) to PostgreSQL/TimescaleDB
- Celery task coordination
- REST API for triggering acquisitions

### Tasks (high level)

- **T3.1**: Implement `websdr_fetcher.py`

Example class outline:

```python
class WebSDRFetcher:
    def __init__(self, websdrs: List[WebSDRConfig]):
        ...
    async def fetch_iq_simultaneous(self, frequency_mhz: float, duration_seconds: float, start_time: datetime) -> Dict[str, IQData]:
        ...
```

- **T3.2**: Implement `iq_processor.py` to compute SNR, PSD, frequency offset and save to MinIO (HDF5 or .npy).

- **T3.3**: Create WebSDR configuration table `WEBSDRS_CONFIG` in `src/models/websdrs.py`.

- **T3.4**: Create Celery task `acquire_iq` in `src/tasks/acquire_iq.py` that:
  1. Fetches IQ from 7 WebSDR simultaneously
  2. Processes signal metrics
  3. Saves to MinIO
  4. Writes metadata to TimescaleDB
  5. Returns status and updates task progress

- **T3.5**: Create FastAPI endpoints to trigger acquisition and check status (`/acquire`, `/status/{task_id}`).

- **T3.6**: Add error handling and retry logic (exponential backoff, partial results policy).

- **T3.7**: Create DB migration for `measurements` hypertable (TimescaleDB).

- **T3.8**: Create test fixtures to mock WebSDR endpoints for unit and integration tests.

- **T3.9**: Create integration test verifying end-to-end acquisition (mocked) and storage in MinIO/DB.

- **T3.10**: Performance test for aiohttp concurrency.

### Checkpoints

✅ CP3.1: WebSDR fetcher works with all 7 receivers (example usage)

✅ CP3.2: IQ data saved to MinIO successfully

✅ CP3.3: Measurements stored in TimescaleDB

✅ CP3.4: Celery task runs end-to-end (trigger via `/acquire`)

✅ CP3.5: All tests pass (coverage >80%)

Knowledge Base

- WebSDR API quirks: rate limiting, connection drops, IQ format variations, tuning offsets
- aiohttp concurrency: reuse sessions, use `asyncio.gather`, set timeouts, use `TCPConnector`
- HDF5 storage: recommended for large IQ arrays with metadata
- Celery Task Monitoring: use `update_state` for progress and Flower UI for monitoring

Rollback Plan

```bash
celery -A rf_acquisition purge
docker-compose up -d rf-acquisition --build
```

Next Phase Entry Point

After CP3.* pass, merge feature branch into `develop` and proceed to Phase 4.

---

## 🖥️ PHASE 4: Data Ingestion Web Interface

**Duration**: 2 days  
**Assignee**: Agent-Backend (fulgidus) + Agent-Frontend (contributor)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 3 ✅  
**Critical Path**: NO

### Objective

Create web UI and backend for human-assisted data collection with known sources management and session validation.

### Tasks (high level)

- **T4.1**: Create Known Sources Pydantic models (`KnownSourceCreate`, `KnownSourceResponse`).

- **T4.2**: Implement Known Sources CRUD endpoints in `services/data-ingestion-web`.

- **T4.3**: Implement Recording Session model and endpoints to create, approve, reject sessions.

- **T4.4**: Implement Session coordinator that triggers RF acquisition and stores `celery_task_id`.

- **T4.5**: Implement spectrogram preview endpoint that returns base64 PNG for validation.

- **T4.6**: Create DB migrations for recording sessions and related tables.

- **T4.7**: Document REST API (OpenAPI / FastAPI autogenerated docs).

- **T4.8**: Integration test for full session workflow (create source, create session, approve).

### Checkpoints

✅ CP4.1: Known sources CRUD works

✅ CP4.2: Recording sessions work end-to-end

✅ CP4.3: Spectrogram preview available

✅ CP4.4: Database has approved sessions

Knowledge Base

- Session workflow: create source → create session → trigger acquisition → human validation → approved → training pool
- Frontend integration: dropdowns for known sources, progress bar, spectrogram visualization

Rollback Plan

```bash
docker-compose restart data-ingestion-web
```

---

## 🧠 PHASE 5: Training Pipeline

**Duration**: 3 days  
**Assignee**: Agent-ML (fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 1 ✅, Phase 3 ✅  
**Critical Path**: YES (blocks Phase 6)

### Objective

Implement PyTorch Lightning training pipeline for neural network localization model with uncertainty estimation.

### Tasks (high level)

- **T5.1**: Design neural network architecture (`LocalizationNet`) that predicts position and uncertainty parameters.

- **T5.2**: Implement feature extraction utilities (`iq_to_mel_spectrogram`, `compute_mfcc`).

- **T5.3**: Create `HeimdallDataset` to load approved recordings from MinIO.

- **T5.4**: Implement Gaussian negative log-likelihood loss for uncertainty-aware regression.

- **T5.5**: Implement PyTorch Lightning module and trainer integration.

- **T5.6**: Setup MLflow tracking (tracking URI via env / Postgres) and log runs, params, artifacts.

- **T5.7**: Implement ONNX export and upload to MinIO.

- **T5.8**: Training entry point script to load sessions, create data loaders, train and register model.

- **T5.9**: Create comprehensive tests for feature extraction, dataset, model forward, loss, MLflow logging, ONNX export.

- **T5.10**: Create `docs/TRAINING.md` describing architecture and hyperparameters.

### Checkpoints

✅ CP5.1: Model forward pass works (output shapes verified)

✅ CP5.2: Dataset loader works (loads features and ground truth)

✅ CP5.3: Training loop runs without errors (small run)

✅ CP5.4: ONNX export successful and present in `heimdall-models` bucket

✅ CP5.5: Model registered in MLflow

Knowledge Base

- Architecture decision: start with CNN (ResNet-18), consider Transformer later
- Loss justification: Gaussian NLL preferred over MSE to penalize overconfidence
- Feature extraction: mel-spectrogram (128 bins) and optional MFCC
- Training hyperparameters: LR 1e-3, batch size 32, epochs 100, early stopping

Rollback Plan

```bash
mlflow.delete_run(run_id)
```

---

## ⚡ PHASE 6: Inference Service

**Duration**: 2 days  
**Assignee**: Agent-Backend (fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 5 ✅, Phase 2 ✅  
**Critical Path**: YES

### Objective

Deploy trained model for real-time inference with caching and uncertainty visualization.

### Tasks (high level)

- **T6.1**: Implement ONNX model loader from MLflow registry.

- **T6.2**: Create prediction endpoint that runs preprocessing, checks Redis cache, performs ONNX inference, returns position + uncertainty.

- **T6.3**: Implement uncertainty ellipse calculation for visualization.

Example prediction flow (simplified):

```python
# Preprocess
features = preprocess_iq(request.iq_data)

# Cache key
cache_key = hash(features)
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached)

# Inference
session = onnxruntime.InferenceSession('model.onnx')
outputs = session.run(None, {'mel_spectrogram': features})

# Format result and cache
result = {...}
redis_client.setex(cache_key, 3600, json.dumps(result))
return result
```

- **T6.4**: Implement batch prediction endpoint for multiple IQ samples.

- **T6.5**: Create model versioning and A/B testing framework.

- **T6.6**: Implement performance monitoring (latency, throughput, cache hit rate).

- **T6.7**: Create load testing to verify <500ms latency requirement.

- **T6.8**: Add endpoint for model metadata and performance metrics.

- **T6.9**: Implement graceful model reloading without downtime.

- **T6.10**: Create comprehensive tests for all inference endpoints.

### Checkpoints

✅ CP6.1: ONNX model loads successfully from MLflow

✅ CP6.2: Prediction endpoint works (<500ms latency)

✅ CP6.3: Redis caching functional (cache hit >80%)

✅ CP6.4: Uncertainty ellipse calculation accurate

✅ CP6.5: Load test passes (100 concurrent requests)

### Knowledge Base

ONNX Runtime optimizations:
- Use `CPUExecutionProvider` for CPU inference
- Enable graph optimizations (`sess_options.graph_optimization_level`)
- Pre-allocate output tensors for better performance

Redis caching strategy:
- Cache key: hash of preprocessed features (not raw IQ to avoid collisions)
- TTL: 1 hour (balance between accuracy and performance)
- Cache hit rate should be >80% in production

Rollback Plan

```bash
# If inference service fails
docker-compose restart inference
# Reset Redis cache if corrupted
redis-cli FLUSHALL
```

Next Phase Entry Point

When inference service is validated, proceed to Phase 7: Frontend

---

## 🎨 PHASE 7: Frontend

**Duration**: 3 days  
**Assignee**: Agent-Frontend (contributor) + Agent-Backend (fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 6 ✅, Phase 4 ✅  
**Critical Path**: YES

### Objective

Create React + Mapbox frontend for real-time RF source localization with uncertainty visualization.

### Key Deliverables

- Interactive map with WebSDR locations
- Real-time localization display with uncertainty ellipses
- Recording session management UI
- Data collection workflow for operators

### Tasks (high level)

- **T7.1**: Setup React + TypeScript + Vite project structure
- **T7.2**: Integrate Mapbox GL JS for interactive mapping
- **T7.3**: Create WebSDR status dashboard (online/offline indicators)
- **T7.4**: Implement real-time localization display with uncertainty ellipses
- **T7.5**: Create recording session management interface
- **T7.6**: Add spectrogram visualization for signal validation
- **T7.7**: Implement user authentication and role management
- **T7.8**: Create responsive design for mobile/tablet operators
- **T7.9**: Add WebSocket integration for real-time updates
- **T7.10**: Create comprehensive E2E tests with Playwright

### Checkpoints

✅ CP7.1: Map displays 7 WebSDR locations correctly
✅ CP7.2: Recording session workflow complete
✅ CP7.3: Real-time localization updates functional
✅ CP7.4: Mobile responsive design verified
✅ CP7.5: All E2E tests pass

---

## ☸️ PHASE 8: Kubernetes & Deployment

**Duration**: 2 days  
**Assignee**: Agent-DevOps (fulgidus)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 7 ✅, All services ✅  
**Critical Path**: YES

### Objective

Deploy entire platform to Kubernetes with monitoring, logging, and auto-scaling.

### Key Deliverables

- Helm charts for all services
- Production-ready PostgreSQL + TimescaleDB
- Monitoring stack (Prometheus + Grafana)
- Centralized logging (ELK stack)
- Auto-scaling policies
- CI/CD pipeline integration

### Tasks (high level)

- **T8.1**: Create Helm charts for all microservices
- **T8.2**: Setup production PostgreSQL + TimescaleDB operator
- **T8.3**: Deploy production-grade RabbitMQ cluster
- **T8.4**: Configure MinIO with persistent storage
- **T8.5**: Setup monitoring (Prometheus + Grafana + AlertManager)
- **T8.6**: Configure centralized logging (Elasticsearch + Logstash + Kibana)
- **T8.7**: Implement HorizontalPodAutoscaler for services
- **T8.8**: Setup ingress with SSL/TLS (cert-manager)
- **T8.9**: Create backup/restore procedures
- **T8.10**: Integrate CI/CD with GitHub Actions

### Checkpoints

✅ CP8.1: All services deployed via Helm
✅ CP8.2: Monitoring dashboards functional
✅ CP8.3: Auto-scaling responds to load
✅ CP8.4: SSL certificates auto-renewed
✅ CP8.5: Backup/restore procedures verified

---

## 🧪 PHASE 9: Testing & QA

**Duration**: 2 days  
**Assignee**: Agent-QA (fulgidus + contributor)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 8 ✅  
**Critical Path**: YES

### Objective

Comprehensive testing suite ensuring system reliability and performance.

### Key Deliverables

- Unit tests (>80% coverage)
- Integration tests for all services
- End-to-end tests for user workflows
- Performance tests (load, stress, endurance)
- Security tests (penetration, vulnerability)

### Tasks (high level)

- **T9.1**: Achieve >80% unit test coverage for all services
- **T9.2**: Create integration tests for service-to-service communication
- **T9.3**: Implement E2E tests for complete user workflows
- **T9.4**: Run performance tests (localization latency <500ms)
- **T9.5**: Conduct security assessment and penetration testing
- **T9.6**: Create test data management strategy
- **T9.7**: Setup continuous testing in CI/CD pipeline
- **T9.8**: Document test procedures and results
- **T9.9**: Create load testing scenarios for production readiness
- **T9.10**: Validate disaster recovery procedures

### Checkpoints

✅ CP9.1: All unit tests pass with >80% coverage
✅ CP9.2: Integration tests verify service communication
✅ CP9.3: E2E tests validate user workflows
✅ CP9.4: Performance tests meet <500ms latency requirement
✅ CP9.5: Security assessment passes without critical issues

---

## 📚 PHASE 10: Documentation & Release

**Duration**: 1 day  
**Assignee**: Agent-Documentation (fulgidus + contributor)  
**Status**: 🔴 NOT STARTED  
**Depends On**: Phase 9 ✅  
**Critical Path**: YES

### Objective

Complete documentation and prepare for public release.

### Key Deliverables

- User manual and operator guide
- API documentation (OpenAPI specs)
- Deployment guide for system administrators
- Developer contribution guide
- Release notes and version tags

### Tasks (high level)

- **T10.1**: Create comprehensive user manual
- **T10.2**: Generate API documentation from OpenAPI specs
- **T10.3**: Write deployment guide for system administrators
- **T10.4**: Create developer contribution guide
- **T10.5**: Prepare release notes and changelog
- **T10.6**: Create video demonstrations and tutorials
- **T10.7**: Setup GitHub release process with semantic versioning
- **T10.8**: Create Docker Hub automated builds
- **T10.9**: Publish to relevant communities (amateur radio, SDR)
- **T10.10**: Plan future roadmap and feature requests

### Checkpoints

✅ CP10.1: All documentation complete and reviewed
✅ CP10.2: Release artifacts prepared (Docker images, Helm charts)
✅ CP10.3: GitHub release published with proper tags
✅ CP10.4: Community announcement prepared
✅ CP10.5: Future roadmap documented

---

## 👥 Agent Roles

### Agent-Infrastructure (fulgidus)
**Responsibilities**: Database, message queues, caching, monitoring  
**Phases**: 0, 1, 8  
**Skills**: PostgreSQL, Docker, Kubernetes, monitoring

### Agent-Backend (fulgidus)
**Responsibilities**: Microservices, APIs, business logic  
**Phases**: 2, 3, 4, 6  
**Skills**: Python, FastAPI, Celery, system design

### Agent-ML (fulgidus)
**Responsibilities**: Training pipeline, neural networks, MLflow  
**Phases**: 5  
**Skills**: PyTorch, signal processing, ML operations

### Agent-Frontend (contributor)
**Responsibilities**: React UI, user experience, visualization  
**Phases**: 7  
**Skills**: React, TypeScript, Mapbox, WebSockets

### Agent-DevOps (fulgidus)
**Responsibilities**: CI/CD, deployment, production operations  
**Phases**: 8  
**Skills**: Kubernetes, Helm, monitoring, security

### Agent-QA (fulgidus + contributor)
**Responsibilities**: Testing, quality assurance, performance validation  
**Phases**: 9  
**Skills**: Testing frameworks, performance testing, security

### Agent-Documentation (fulgidus + contributor)
**Responsibilities**: Documentation, user guides, community engagement  
**Phases**: 10  
**Skills**: Technical writing, video production, community management

---

## 🧠 Knowledge Base & Continuity

### Context Preservation Strategy

Each phase must maintain knowledge through:

1. **Checkpoint Documentation**: Every checkpoint includes lessons learned and decisions made
2. **Code Comments**: Critical decisions documented in code
3. **Architecture Decision Records (ADRs)**: Formal documentation of technical choices
4. **Handoff Notes**: Detailed transfer of knowledge between agents/phases

### Critical Knowledge Areas

**WebSDR Integration**:
- API quirks and limitations for each of the 7 receivers
- Frequency offset corrections and timing synchronization
- Connection reliability patterns and retry strategies

**ML Model Architecture**:
- Neural network design decisions (CNN vs Transformer)
- Loss function choice (Gaussian NLL for uncertainty)
- Feature extraction pipeline (mel-spectrogram parameters)
- Training hyperparameters and convergence patterns

**Deployment Architecture**:
- Microservices communication patterns
- Database schema evolution strategy
- Caching layers and performance optimizations
- Monitoring and alerting configurations

### Handoff Protocols

When transitioning between phases:
1. Complete all checkpoints in current phase
2. Document any deviations from original plan
3. Update `.copilot-instructions` with new learnings
4. Brief next agent on critical decisions and constraints
5. Ensure all artifacts are properly versioned and accessible

---

## ✅ Checkpoint System

### Validation Levels

**Level 1 - Functional**: Basic functionality working  
**Level 2 - Integration**: Components work together  
**Level 3 - Performance**: Meets latency and throughput requirements  
**Level 4 - Production**: Ready for live deployment  

### Checkpoint Dependencies

```mermaid
graph TD
    CP0[Phase 0: Repository] --> CP1[Phase 1: Infrastructure]
    CP1 --> CP2[Phase 2: Scaffolding]
    CP2 --> CP3[Phase 3: RF Acquisition]
    CP2 --> CP5[Phase 5: Training]
    CP3 --> CP4[Phase 4: Data Ingestion]
    CP3 --> CP5
    CP5 --> CP6[Phase 6: Inference]
    CP4 --> CP7[Phase 7: Frontend]
    CP6 --> CP7
    CP7 --> CP8[Phase 8: Kubernetes]
    CP8 --> CP9[Phase 9: Testing]
    CP9 --> CP10[Phase 10: Documentation]
```

### Rollback Triggers

**Automatic Rollback**: If any checkpoint fails validation 3 times  
**Manual Rollback**: When fundamental architectural issue discovered  
**Escalation Path**: Owner (fulgidus) makes final decisions on major changes

---

## 🔗 Dependency Matrix

| Phase             | Dependencies | Blocks | Critical Path | Parallel Capable |
| ----------------- | ------------ | ------ | ------------- | ---------------- |
| 0: Repository     | None         | All    | YES           | NO               |
| 1: Infrastructure | Phase 0      | 2,3,5  | YES           | NO               |
| 2: Scaffolding    | Phase 1      | 3,4,6  | YES           | NO               |
| 3: RF Acquisition | Phase 2      | 4,5,6  | YES           | NO               |
| 4: Data Ingestion | Phase 3      | 7      | NO            | YES (with 5)     |
| 5: Training       | Phase 1,3    | 6      | YES           | YES (with 4)     |
| 6: Inference      | Phase 2,5    | 7      | YES           | NO               |
| 7: Frontend       | Phase 4,6    | 8      | YES           | NO               |
| 8: Kubernetes     | Phase 7      | 9      | YES           | NO               |
| 9: Testing        | Phase 8      | 10     | YES           | NO               |
| 10: Documentation | Phase 9      | None   | YES           | NO               |

### Parallelization Opportunities

- **Phase 4 + 5**: Data Ingestion and Training can run simultaneously after Phase 3
- **Testing**: Unit tests can be developed during each phase
- **Documentation**: Can be incrementally updated during development

### Critical Path Analysis

**Total Duration**: 21 days (3 weeks)  
**Critical Path**: 0→1→2→3→5→6→7→8→9→10  
**Parallel Savings**: 2 days (Phase 4 parallel with Phase 5)  
**Risk Buffer**: 20% additional time recommended for unforeseen issues

---

## 🎯 Project Success Criteria

### Technical Metrics
- ✅ Localization accuracy: ±30m (68% confidence interval)
- ✅ Real-time inference latency: <500ms (95th percentile)
- ✅ System uptime: >99.5% (measured over 30 days)
- ✅ Test coverage: ≥80% across all services
- ✅ Documentation completeness: 100% of public APIs documented

### Operational Metrics
- ✅ Simultaneous data ingestion from 7 WebSDR receivers
- ✅ Model training pipeline automation (MLflow integration)
- ✅ Kubernetes deployment with auto-scaling
- ✅ Monitoring and alerting functional
- ✅ Backup and disaster recovery procedures tested

### Community Metrics
- ✅ Open source repository with proper licensing
- ✅ Comprehensive documentation for new contributors
- ✅ Amateur radio community engagement
- ✅ Academic paper or technical presentation prepared
- ✅ Roadmap for future enhancements defined
