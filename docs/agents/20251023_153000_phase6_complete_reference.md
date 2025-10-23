# 📖 PHASE 6: Complete Reference Guide

**Date**: 2025-10-22  
**Status**: 🟡 READY TO START  
**Duration**: 2 days  
**Assignee**: Agent-Backend (fulgidus)

---

## 🎯 Phase 6 at a Glance

**Objective**: Build real-time inference service with <500ms latency SLA

**Input**: 
- ONNX model from Phase 5 ✅
- MLflow registry ✅
- Training data patterns ✅

**Output**:
- Inference microservice with REST API
- Model versioning framework
- Performance monitoring
- Production-ready deployment
- Ready for Phase 7 Frontend integration

---

## 📚 Documentation Structure

### Core Documentation
1. **PHASE6_START_HERE.md** ⭐ START HERE
   - High-level overview
   - Task breakdown with pseudocode
   - Getting started steps
   - Success criteria

2. **PHASE6_PREREQUISITES_CHECK.md**
   - Dependency verification checklist
   - Connection strings
   - Service status verification
   - Troubleshooting

3. **PHASE6_PROGRESS_DASHBOARD.md**
   - Real-time progress tracking
   - Task status monitor
   - Daily progress log
   - Checkpoint tracking

4. **PHASE6_CODE_TEMPLATE.md**
   - Complete file structure
   - Copy-paste code snippets
   - Implementation checklist
   - Reference implementations

5. **PHASE6_COMPLETE_REFERENCE.md** (this file)
   - Master index
   - Quick navigation
   - Key concepts
   - Common patterns

---

## 🗺️ Navigation Guide

### If You Want To...

**Understand Phase 6 Overview**
→ Read: PHASE6_START_HERE.md (5 min)

**Verify Your System is Ready**
→ Run: PHASE6_PREREQUISITES_CHECK.md commands (5 min)

**See Real-Time Progress**
→ Monitor: PHASE6_PROGRESS_DASHBOARD.md (continuous)

**Get Started Coding**
→ Use: PHASE6_CODE_TEMPLATE.md (copy-paste snippets)

**Understand the Big Picture**
→ Read: AGENTS.md - Phase 6 Section (10 min)

**Troubleshoot Issues**
→ See: PHASE6_TROUBLESHOOTING.md (create if needed)

**Understand MLflow Integration**
→ See: Phase 5 documentation + MLflow docs

**Understand ONNX Runtime**
→ See: ONNX Runtime official docs + code comments

---

## 🔑 Key Concepts

### 1. ONNX Model Loading (T6.1)

**What**: Load trained neural network from MLflow registry

**Why**: 
- MLflow provides model versioning and staging
- ONNX format ensures portability across frameworks
- Decouples training (Phase 5) from inference (Phase 6)

**How**:
```python
loader = ONNXModelLoader(
    mlflow_uri="http://mlflow:5000",
    model_name="localization_model",
    stage="Production"
)
result = loader.predict(features)
```

**Key Files**:
- MLflow model registry (http://localhost:5000/models)
- ONNX artifact: MinIO `heimdall-models/model.onnx`
- Implementation: `services/inference/src/models/onnx_loader.py`

---

### 2. Prediction Endpoint (T6.2)

**What**: REST API for real-time predictions

**Why**: 
- Enables frontend (Phase 7) and other services to get predictions
- RESTful interface matches architecture pattern
- Allows horizontal scaling

**How**:
```
POST /predict
Content-Type: application/json

{
  "iq_data": [[1.5, 0.3], [1.2, 0.5], ...],
  "cache_enabled": true
}

Response:
{
  "position": {"latitude": 45.12, "longitude": 7.45},
  "uncertainty": {"sigma_x": 28.5, "sigma_y": 31.2, "theta": 45.3},
  "confidence": 0.92,
  "model_version": "1.2.0",
  "inference_time_ms": 245.3
}
```

**SLA**: <500ms (P95 latency)

**Key Files**:
- FastAPI routers: `services/inference/src/routers/predict.py`
- Pydantic schemas: `services/inference/src/models/schemas.py`

---

### 3. Redis Caching (T6.2+)

**What**: Cache predictions to improve latency

**Why**:
- Repeated IQ data → Same prediction (deterministic)
- Cache hit avoids expensive ONNX inference
- Dramatically improves throughput

**How**:
```python
# Input preprocessing
features = preprocess_iq(request.iq_data)
cache_key = hash(features)

# Check cache
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached)

# Inference
result = model.predict(features)
redis_client.setex(cache_key, 3600, json.dumps(result))
return result
```

**Cache Strategy**:
- Key: Hash of preprocessed features (not raw IQ)
- TTL: 1 hour (balance accuracy vs performance)
- Target hit rate: >80%
- Validation: Check if cache hit >80% in production

**Key Service**: Redis running on port 6379 (verified from Phase 1)

---

### 4. Uncertainty Ellipse (T6.3)

**What**: Visualize prediction uncertainty for frontend

**Why**:
- Shows confidence in prediction spatially
- Enables map visualization (Mapbox integration)
- Helps operators understand accuracy

**How**:
```python
# From model outputs (sigma_x, sigma_y)
ellipse = compute_uncertainty_ellipse(
    sigma_x=28.5,  # meters
    sigma_y=31.2,  # meters
    covariance_xy=0.0
)
# Output: semi_major_axis, semi_minor_axis, rotation_angle
# Convert to GeoJSON for Mapbox
geojson = ellipse_to_geojson(lat, lon, major, minor, angle)
```

**Key File**: `services/inference/src/utils/uncertainty.py`

---

### 5. Batch Predictions (T6.4)

**What**: Predict multiple IQ samples in one request

**Why**:
- Optimized ONNX batching (vectorized operations)
- Better throughput than individual requests
- Useful for backfilling/backtesting

**How**:
```
POST /predict/batch
{
  "iq_samples": [
    [[1.5, 0.3], [1.2, 0.5], ...],
    [[2.1, 0.1], [1.9, 0.4], ...],
    ...
  ]
}

Response:
{
  "predictions": [prediction1, prediction2, ...],
  "total_time_ms": 450.5,
  "samples_per_second": 6667
}
```

---

### 6. Model Versioning (T6.5)

**What**: Support multiple model versions with traffic split

**Why**:
- Enables A/B testing (90% v1, 10% v2-beta)
- Zero-downtime model updates
- Canary deployments

**How**:
```python
manager = ModelVersionManager()
manager.set_model("1.2.0", weight=0.9)
manager.set_model("2.0.0-alpha", weight=0.1)
model = manager.select_model()  # Random weighted selection
```

**Use Case**:
- Deploy new model to 10% of traffic
- Monitor performance
- Gradually increase traffic
- Switch fully when confident

---

### 7. Performance Monitoring (T6.6)

**What**: Track latency, cache hit rate, errors

**Why**:
- Production observability (SLA validation)
- Alerts on performance degradation
- Grafana dashboards for operators

**Metrics**:
```
inference_latency_ms          # Histogram (distribution)
cache_hit_rate                # Gauge (%)
cache_hits_total              # Counter
inference_requests_total      # Counter
inference_errors_total        # Counter
model_reloads_total           # Counter
inference_active_requests     # Gauge
```

**Endpoint**: `GET /metrics` (Prometheus format)

---

### 8. Load Testing (T6.7)

**What**: Validate <500ms SLA under concurrent load

**Why**:
- Ensures production readiness
- Prevents SLA violations
- Identifies bottlenecks

**Expected Results**:
```
100 concurrent requests:
- Mean latency: <300ms
- P95 latency: <500ms ✅ (SLA)
- P99 latency: <700ms
- Success rate: 100%
```

**Key Command**:
```bash
pytest tests/load_test_inference.py -v
```

---

### 9. Graceful Reloading (T6.9)

**What**: Reload model without dropping connections

**Why**:
- Zero-downtime deployments
- Update model without service restart
- Continuous availability

**How**:
```python
# Signal handler on SIGHUP
@signal.signal(signal.SIGHUP, handle_reload)
def handle_reload(signum, frame):
    # 1. Load new model
    # 2. Wait for in-flight requests
    # 3. Atomic swap
    # 4. Drain old instance
```

---

## 🧪 Testing Strategy

### Unit Tests
```bash
pytest tests/test_onnx_loader.py              # ONNX loading
pytest tests/test_uncertainty.py              # Ellipse math
pytest tests/test_model_versioning.py         # Traffic split
```

### Integration Tests
```bash
pytest tests/test_predict_endpoints.py        # API endpoints
pytest tests/integration_test_mlflow.py       # MLflow integration
```

### Performance Tests
```bash
pytest tests/load_test_inference.py           # <500ms latency
pytest tests/stress_test_inference.py         # 1000+ concurrent
```

### Coverage Target
```bash
pytest tests/ --cov=src --cov-report=html
# Target: >80% coverage
```

---

## 🔧 Common Tasks

### Verify Prerequisites
```bash
# 1. Docker containers
docker-compose ps
# Expected: 13 containers healthy

# 2. Redis
redis-cli PING
# Expected: PONG

# 3. MLflow registry
# Open http://localhost:5000/models
# Expected: localization_model in Production stage

# 4. ONNX model
# Open http://localhost:9001 (MinIO)
# Expected: heimdall-models/model.onnx exists
```

### Create Service Structure
```bash
python scripts/create_service.py inference
# Creates: services/inference/ with all scaffolding
```

### Build Docker Image
```bash
cd services/inference
docker build -t heimdall-inference:latest .
# or
docker-compose build inference
```

### Run Service Locally
```bash
cd services/inference
pip install -r requirements.txt
python src/main.py
# Open http://localhost:8006/health
```

### Test Prediction Endpoint
```bash
curl -X POST http://localhost:8006/predict \
  -H "Content-Type: application/json" \
  -d '{
    "iq_data": [[1.5, 0.3], [1.2, 0.5]],
    "cache_enabled": true
  }'
```

---

## 📊 Success Criteria Summary

| Checkpoint | Criteria                         | Status    |
| ---------- | -------------------------------- | --------- |
| **CP6.1**  | ONNX model loads from MLflow     | ⏳ Pending |
| **CP6.2**  | /predict endpoint <500ms latency | ⏳ Pending |
| **CP6.3**  | Redis caching >80% hit rate      | ⏳ Pending |
| **CP6.4**  | Uncertainty ellipse correct      | ⏳ Pending |
| **CP6.5**  | Load test: 100 concurrent ✅      | ⏳ Pending |

---

## 📚 External References

### ONNX Runtime
- Official Docs: https://onnxruntime.ai/docs/
- Python API: https://onnxruntime.ai/docs/api/python/
- Performance Tips: https://onnxruntime.ai/docs/performance/

### MLflow Model Registry
- Official Docs: https://mlflow.org/docs/latest/model-registry.html
- Python API: https://mlflow.org/docs/latest/python_api/mlflow.tracking.html
- Model Versioning: https://mlflow.org/docs/latest/model-registry/index.html

### FastAPI
- Official Docs: https://fastapi.tiangolo.com/
- Deployment: https://fastapi.tiangolo.com/deployment/
- Performance: https://fastapi.tiangolo.com/deployment/concepts/

### Redis
- Official Docs: https://redis.io/docs/
- Python Client: https://redis-py.readthedocs.io/
- Caching Patterns: https://redis.io/docs/manual/client-side-caching/

### Prometheus
- Official Docs: https://prometheus.io/docs/
- Python Client: https://github.com/prometheus/client_python
- Metrics Types: https://prometheus.io/docs/concepts/metric_types/

---

## 🎯 Recommended Reading Order

1. **PHASE6_START_HERE.md** (5 min) - Overview and task structure
2. **PHASE6_PREREQUISITES_CHECK.md** (5 min) - Verify system ready
3. **PHASE6_CODE_TEMPLATE.md** (15 min) - Code structure and snippets
4. **Start Implementation**: T6.1 → T6.2 → T6.3 → ... → T6.10
5. **Monitor Progress**: Update PHASE6_PROGRESS_DASHBOARD.md daily

---

## 💡 Pro Tips

1. **Cache Strategy**: Hash preprocessed features (not raw IQ) to maximize hits
2. **ONNX Optimization**: Enable graph optimization for 10-20% speed improvement
3. **Error Handling**: Always fallback gracefully if model unavailable
4. **Testing**: Write tests alongside code, not after
5. **Monitoring**: Add metrics from day 1, not at the end
6. **Documentation**: Update PHASE6_PROGRESS_DASHBOARD.md after each task
7. **Checkpoint Validation**: Run full test suite after each checkpoint

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Verify prerequisites
docker-compose ps                # Should show 13 containers
redis-cli PING                   # Should show PONG

# 2. Create service structure
python scripts/create_service.py inference

# 3. Start reading
code PHASE6_START_HERE.md

# 4. Begin implementation with T6.1
cd services/inference
# Create src/models/onnx_loader.py from PHASE6_CODE_TEMPLATE.md
```

---

## 📞 Getting Help

**Question**: Where is the ONNX model?  
**Answer**: MinIO `heimdall-models/model.onnx` (or check PHASE6_PREREQUISITES_CHECK.md)

**Question**: How do I connect to MLflow?  
**Answer**: Use `mlflow.set_tracking_uri("http://mlflow:5000")` or check .env

**Question**: What's the ONNX model output format?  
**Answer**: Check Phase 5 documentation or inference Phase 5 code

**Question**: SLA latency <500ms - why?  
**Answer**: Real-time inference requirement from product spec (AGENTS.md)

**Question**: How to troubleshoot cache hits?  
**Answer**: Check Redis with `redis-cli` and verify preprocessing consistency

---

## ✅ Pre-Start Checklist

Before coding, verify:

- [ ] Read PHASE6_START_HERE.md
- [ ] Ran PHASE6_PREREQUISITES_CHECK.md successfully
- [ ] All 13 Docker containers healthy
- [ ] Redis responding to PING
- [ ] MLflow shows localization_model
- [ ] ONNX file in MinIO
- [ ] Understand Phase 5 model output format
- [ ] Understand cache strategy
- [ ] Have PHASE6_CODE_TEMPLATE.md open for reference

---

## 🎊 You're Ready!

All documentation is prepared. Phase 6 is ready to start. 

**Next Step**: Read PHASE6_START_HERE.md and begin T6.1 implementation! 🚀

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-22  
**Status**: 🟡 READY TO START PHASE 6

