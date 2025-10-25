# 🚀 PHASE 6 - SESSION 2 STARTUP CHECKLIST

**Session 1 Complete**: ✅ 30% Phase 6 done (1650+ lines, 45+ tests)  
**Session 2 Focus**: T6.2 Prediction Endpoint + T6.4-T6.7 followup  
**Time Estimate**: 4-5 hours → Target 80% Phase 6 completion  

---

## 📋 PRE-SESSION CHECKLIST

Before starting Session 2, verify:

- [ ] All docker containers running: `docker-compose ps`
- [ ] MLflow accessible: http://localhost:5000
- [ ] Redis working: `redis-cli PING`
- [ ] PostgreSQL up: `psql -h localhost -U heimdall_user -d heimdall`
- [ ] Read: `PHASE6_SESSION1_QUICK_SUMMARY.md`
- [ ] Review: `PHASE6_IMPLEMENTATION_README.md`

---

## 🎯 SESSION 2 TASKS (IN ORDER)

### PRIORITY 1: T6.2 - Prediction Endpoint (CRITICAL)

**Status**: 40% complete (schemas done, need implementation)  
**Time**: 1.5 hours  
**SLA**: <500ms end-to-end latency  

**Components to implement**:

1. **Create** `services/inference/src/utils/preprocessing.py`
   - Function: `preprocess_iq(iq_data: List[List[float]]) -> np.ndarray`
   - Convert IQ data to mel-spectrogram
   - Normalize and validate

2. **Create** `services/inference/src/routers/predict.py`
   - Endpoint: `@app.post("/predict")`
   - Implement full flow:
     1. Validate request
     2. Check Redis cache
     3. Preprocess IQ data
     4. Run ONNX inference
     5. Format response
     6. Cache result
   - Error handling
   - Metrics integration

3. **Unit Tests** `services/inference/tests/test_predict_endpoints.py`
   - Test preprocessing with mock data
   - Test caching logic
   - Test error cases
   - Test latency <500ms

**Files to create/modify**:
```
✅ services/inference/src/models/schemas.py (ALREADY DONE)
🟡 services/inference/src/utils/preprocessing.py (CREATE)
🟡 services/inference/src/routers/predict.py (CREATE/MODIFY)
🟡 services/inference/tests/test_predict_endpoints.py (CREATE)
```

**Acceptance Criteria**:
- ✅ Endpoint responds with PredictionResponse
- ✅ Latency <500ms (measure with real ONNX)
- ✅ Redis caching working
- ✅ 100% test coverage on predict.py

---

### PRIORITY 2: T6.7 - Load Testing (VALIDATE SLA)

**Status**: Not started  
**Time**: 1 hour  
**Target**: Verify P95 latency <500ms under 100+ concurrent load  

**Create** `services/inference/tests/load_test_inference.py`:

```python
# Pseudo structure:
# 1. Generate 100 concurrent prediction requests
# 2. Measure response times
# 3. Calculate P50, P95, P99 latencies
# 4. Verify <500ms SLA
# 5. Report throughput (requests/sec)
# 6. Identify bottlenecks
```

**Files**:
```
🟡 services/inference/tests/load_test_inference.py (CREATE)
```

**Acceptance Criteria**:
- ✅ P95 latency < 500ms
- ✅ 100 concurrent requests successful
- ✅ No timeouts or errors
- ✅ Cache hit rate measured

---

### PRIORITY 3: T6.4 - Batch Prediction (BONUS)

**Status**: Not started  
**Time**: 1 hour  
**Note**: Optional if time permits  

**Create** `services/inference/src/routers/predict.py` - Add batch endpoint:

```python
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    # Process multiple samples in parallel
    # Return BatchPredictionResponse
```

---

## 🛠️ IMPLEMENTATION TEMPLATE

### T6.2 - Step by step:

**Step 1**: Create preprocessing.py

```python
# services/inference/src/utils/preprocessing.py
import numpy as np
from typing import List

def preprocess_iq(iq_data: List[List[float]]) -> np.ndarray:
    """Convert IQ data to mel-spectrogram."""
    # 1. Convert to complex array
    # 2. Compute FFT
    # 3. Apply mel-scale
    # 4. Log scale
    # 5. Normalize
    return features  # (128, 100) shape for model
```

**Step 2**: Implement /predict endpoint

```python
# services/inference/src/routers/predict.py
from src.models.schemas import PredictionRequest, PredictionResponse
from src.utils.preprocessing import preprocess_iq
from src.utils.metrics import InferenceMetricsContext, ONNXMetricsContext

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    loader: ONNXModelLoader = Depends(get_model_loader),
    redis_client: Optional[Redis] = Depends(get_redis),
):
    """Predict position from IQ data."""
    with InferenceMetricsContext("predict"):
        # 1. Validation
        validate_request(request)
        
        # 2. Preprocessing with metrics
        with PreprocessingMetricsContext():
            features = preprocess_iq(request.iq_data)
        
        # 3. Cache check
        if request.cache_enabled and redis_client:
            cache_key = compute_cache_key(features)
            cached = redis_client.get(cache_key)
            if cached:
                record_cache_hit()
                return PredictionResponse(**json.loads(cached))
        
        # 4. ONNX inference
        with ONNXMetricsContext():
            result = loader.predict(features)
        
        # 5. Uncertainty ellipse
        uncertainty = compute_uncertainty_ellipse(
            result['uncertainty']['sigma_x'],
            result['uncertainty']['sigma_y']
        )
        
        # 6. Format response
        response = PredictionResponse(
            position=PositionResponse(**result['position']),
            uncertainty=UncertaintyResponse(**uncertainty),
            confidence=result['confidence'],
            model_version=loader.get_metadata()['version'],
            inference_time_ms=...  # from metrics
        )
        
        # 7. Cache result
        if request.cache_enabled and redis_client:
            redis_client.setex(
                cache_key,
                config.REDIS_CACHE_TTL_SECONDS,
                response.json()
            )
        
        return response
```

**Step 3**: Write tests

```python
# services/inference/tests/test_predict_endpoints.py
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_predict_success():
    """Test successful prediction."""
    # Setup mocks
    # Call endpoint
    # Assert response
    pass

@pytest.mark.asyncio
async def test_predict_latency():
    """Test latency <500ms."""
    # Measure inference time
    # Assert < 500ms
    pass

@pytest.mark.asyncio
async def test_predict_cache_hit():
    """Test Redis cache working."""
    # First call (cache miss)
    # Second call (cache hit)
    # Assert hit counter incremented
    pass
```

---

## 📊 SUCCESS METRICS

By end of Session 2, should have:

```
Phase 6 Progress: 60-70% (6-7/10 tasks)

T6.1  ████████████████████ 100% ✅ (Session 1)
T6.2  ████████████████████ 100% ✅ (Session 2)
T6.3  ████████████████████ 100% ✅ (Session 1)
T6.4  ████████████████████ 100% ✅ (Session 2)
T6.5  ░░░░░░░░░░░░░░░░░░░░   0%   (Session 3)
T6.6  ████████████████████ 100% ✅ (Session 1)
T6.7  ████████████████████ 100% ✅ (Session 2)
T6.8  ░░░░░░░░░░░░░░░░░░░░   0%   (Session 3)
T6.9  ░░░░░░░░░░░░░░░░░░░░   0%   (Session 3)
T6.10 ░░░░░░░░░░░░░░░░░░░░   0%   (Session 3)
```

---

## 📝 DOCUMENTATION TO UPDATE

After Session 2, update:

- [ ] PHASE6_SESSION2_PROGRESS.md (new - detailed session report)
- [ ] PHASE6_IMPLEMENTATION_README.md (update progress)
- [ ] PHASE6_SESSION2_QUICK_SUMMARY.md (new - quick overview)
- [ ] Update AGENTS.md with latest status

---

## 🔍 KEY THINGS TO REMEMBER

1. **Latency SLA**: <500ms is CRITICAL. Measure everything.
2. **Cache Strategy**: Key should be hash of preprocessed features (not raw IQ)
3. **Error Handling**: Network timeouts, ONNX errors, Redis unavailable
4. **Logging**: All critical paths should be logged
5. **Testing**: Mock real dependencies, test actual latency with real ONNX
6. **Metrics**: All operations go through InferenceMetricsContext

---

## ⚠️ POTENTIAL BOTTLENECKS

1. **Preprocessing**: FFT/mel-spectrogram computation (~50-100ms)
2. **ONNX Runtime**: Model inference (~100-200ms depending on model size)
3. **Redis**: Network latency (~5-10ms typically)
4. **Serialization**: JSON encoding/decoding (~5-10ms)

**Budget**: 500ms total
- Preprocessing: 100ms (20%)
- ONNX: 200ms (40%)
- Redis: 50ms (10%)
- Overhead: 150ms (30%)

---

## 🚀 START CHECKLIST

Before starting T6.2 coding:

- [ ] Clone Phase 5 model output (ONNX shape and format)
- [ ] Verify MLflow has Production model ready
- [ ] Test ONNX Runtime session creation
- [ ] Ensure preprocessing matches training pipeline
- [ ] Run existing tests: `pytest services/inference/tests/test_onnx_loader.py`

---

## 📞 QUICK COMMANDS

```bash
# Start services
docker-compose up -d

# Run Phase 6 tests
pytest services/inference/tests/ -v

# Check MLflow
curl http://localhost:5000/api/2.0/registered-models/list

# Check Redis
redis-cli PING

# Check inference service logs (when running)
docker-compose logs -f inference

# Load test (will create after T6.7)
python services/inference/tests/load_test_inference.py
```

---

**Ready for Session 2?** ✅

All materials are prepared. T6.2 implementation is straightforward with schemas already done.

**Expected Timeline**:
- T6.2: 1-1.5 hours (core functionality)
- T6.7: 1 hour (load test validation)
- T6.4: 1 hour (batch prediction)
- Buffer: 30 minutes

**Total**: 4-5 hours → 60-70% Phase 6 completion

Let's go! 🚀

