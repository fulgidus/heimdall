# ✅ PHASE 6: Prerequisites Check

**Last Updated**: 2025-10-22  
**Status**: 🟢 ALL CLEAR - Ready to start Phase 6

---

## 📋 Dependency Verification

### Phase 5 Status: ✅ COMPLETE

- [x] LocalizationNet neural network trained
- [x] PyTorch Lightning training pipeline complete
- [x] ONNX model exported
- [x] Model uploaded to MLflow registry (Production stage)
- [x] Model artifact stored in MinIO `heimdall-models` bucket
- [x] MLflow tracking URI configured
- [x] Training documentation complete (`docs/TRAINING.md`)

**Reference**: `00_PHASE5_COMPLETE_FINAL.md`, `PHASE5_T5.7_ONNX_COMPLETE.md`

---

### Phase 2 Status: ✅ COMPLETE

- [x] FastAPI scaffolding template available
- [x] Service creation script: `scripts/create_service.py`
- [x] Docker multi-stage build template
- [x] Health check endpoint pattern
- [x] Structured logging setup
- [x] Configuration management (pydantic-settings)

**Reference**: `PHASE2_COMPLETE.md`

---

### Phase 1 Infrastructure: ✅ FULLY OPERATIONAL

#### Database
- [x] PostgreSQL 15 + TimescaleDB running
- [x] Schema initialized with measurements hypertable
- [x] Connection string: `postgresql://heimdall_user:changeme@db:5432/heimdall`

#### Message Queue
- [x] RabbitMQ 3.12 running
- [x] Vhosts configured
- [x] Connection string: `amqp://guest:guest@rabbitmq:5672/`

#### Caching Layer ⭐ REQUIRED FOR PHASE 6
- [x] Redis 7 running on port 6379
- [x] Default DB: 0 (general)
- [x] DB 1: Reserved for Celery results
- [x] No authentication configured
- [x] `redis-cli` available for debugging

#### Object Storage
- [x] MinIO running on port 9000
- [x] Bucket `heimdall-models` exists (contains ONNX model from Phase 5)
- [x] Bucket `heimdall-raw-iq` exists (contains training data)
- [x] Connection string: `http://minio:9000`

#### Monitoring
- [x] Prometheus running (port 9090)
- [x] Grafana running (port 3000)
- [x] Redis Commander running (port 8081)

**Command to verify all services**:
```bash
docker-compose ps
# Expected: All 13 containers with status "healthy" or "Up"
```

---

## 🔗 Connection Strings (For Inference Service)

```python
# .env configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_MODEL_NAME=localization_model
MLFLOW_MODEL_STAGE=Production

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # (empty for dev)

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET_MODELS=heimdall-models
MINIO_BUCKET_RAW_IQ=heimdall-raw-iq

DATABASE_URL=postgresql://heimdall_user:changeme@db:5432/heimdall

ONNXRUNTIME_PROVIDERS=CPUExecutionProvider
```

---

## 🧠 MLflow Registry Status

### Verify Model Available

```bash
# Inside docker container or with MLflow client
mlflow models list

# Expected output:
# Name: localization_model
# Stages: ['Production', 'Staging', 'Archived']
# Latest: version X in Production stage
```

### Model Details

```bash
mlflow models get-model-version-details \
  --name localization_model \
  --version 1  # or latest

# Expected output includes:
# - source: s3://minio/mlflow/artifacts/...
# - onnx_model: model.onnx
# - metadata: input shapes, output shapes, hyperparameters
```

---

## 📦 Service Scaffolding Availability

### Option A: Service Already Created (Phase 2)

If `services/inference/` already exists:

```bash
ls -la services/inference/
# Should contain: src/, tests/, Dockerfile, requirements.txt
```

### Option B: Need to Create

If not yet created, run:

```bash
cd heimdall
python scripts/create_service.py inference
# Creates complete scaffold
```

---

## 🎯 Ready-to-Use Components from Previous Phases

### From Phase 3: RF Acquisition

- Example Celery task structure (can be referenced for patterns)
- WebSDR fetching with error handling
- IQ data serialization (HDF5, NPY, JSON metadata)

### From Phase 4: Data Ingestion

- REST API patterns (Pydantic models, FastAPI routers)
- Database integration examples
- Error handling and response formatting
- Load testing framework

### From Phase 5: Training

- Feature extraction: mel-spectrogram, MFCC
- Model output format: (latitude, longitude, sigma_x, sigma_y)
- MLflow integration patterns
- ONNX export utilities

---

## 🚀 Quick Readiness Checklist

- [ ] All 13 Docker containers running (`docker-compose ps`)
- [ ] Redis responding to PING (`redis-cli PING`)
- [ ] MLflow model registry accessible
- [ ] ONNX model present in MinIO
- [ ] `services/inference/` directory exists (or create with script)
- [ ] `.env` file configured with connection strings
- [ ] Read PHASE6_START_HERE.md completely
- [ ] Reviewed previous phase documentation

---

## 🔧 Commands to Run Now

### 1. Verify Docker
```powershell
docker-compose ps
```

### 2. Verify Redis
```powershell
docker-compose exec redis redis-cli PING
# Expected: PONG
```

### 3. Verify MLflow
```powershell
# Option A: Inside Python
docker-compose exec training python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); models = mlflow.search_registered_models(); print(models)"

# Option B: Via MLflow UI
open http://localhost:5000
# Check "Models" tab
```

### 4. Verify ONNX in MinIO
```powershell
docker-compose exec minio mc ls minio/heimdall-models
# Expected: model.onnx file present
```

### 5. Verify Inference Service Scaffold
```powershell
ls services/inference
# Expected: src/, tests/, Dockerfile, requirements.txt
```

---

## 🎯 Success Criteria for Prerequisites

✅ All prerequisites verified when:

1. **Infrastructure**: All 13 Docker containers healthy
2. **Database**: PostgreSQL accessible with schema initialized
3. **Cache**: Redis responding and accessible
4. **Model Registry**: MLflow shows localization_model in Production
5. **ONNX Model**: Present in MinIO bucket with metadata
6. **Service Scaffold**: `services/inference/` structure ready
7. **Configuration**: `.env` file configured for all services

---

## 💡 If Something is Missing

### Missing Redis?
```bash
docker-compose up -d redis
docker-compose exec redis redis-cli PING
```

### Missing ONNX Model?
- Go back to Phase 5 and run T5.7 export task
- Check Phase 5 documentation: `PHASE5_T5.7_ONNX_COMPLETE.md`

### Missing MLflow Model Registry?
- Verify Phase 5 T5.5-T5.6 completed
- Check MLflow UI: http://localhost:5000/models

### Missing Inference Service Scaffold?
```bash
cd services
python ../scripts/create_service.py inference
cd ..
```

---

## 🔄 Next: Phase 6 Implementation

Once all prerequisites verified:

1. Read: `PHASE6_START_HERE.md`
2. Begin: T6.1 (ONNX Model Loader)
3. Update: `manage_todo_list` to track progress
4. Document: Each checkpoint in `PHASE6_PROGRESS.md` (new file)

---

**Status**: 🟢 ALL SYSTEMS GO FOR PHASE 6

**Start Time**: [Your current time]  
**Estimated Completion**: +2 days (by 2025-10-24)

