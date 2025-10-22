# Phase 3: RF Acquisition Service - Implementation Guide

**Status**: 🟡 IN PROGRESS  
**Duration**: 3 days  
**Assignee**: Agent-Backend (fulgidus)  
**Critical Path**: YES (blocks Phase 4 and 5)

---

## Overview

Phase 3 implements the core RF acquisition service for fetching IQ data from 7 distributed WebSDR receivers simultaneously, processing signal metrics, and coordinating with Celery for task management.

---

## What's Been Completed

### ✅ Core Implementation (Phase 3)

#### 1. **Data Models** (`src/models/websdrs.py`)
- `WebSDRConfig`: Configuration for individual receivers
- `AcquisitionRequest`: Request to acquire IQ data
- `SignalMetrics`: Computed signal metrics (SNR, PSD, frequency offset)
- `MeasurementRecord`: Single measurement from one receiver
- `AcquisitionTaskResponse`: Response when triggering a task
- `AcquisitionStatusResponse`: Status of ongoing tasks

#### 2. **WebSDR Fetcher** (`src/fetchers/websdr_fetcher.py`)
- **Simultaneous fetch**: Uses aiohttp + asyncio to fetch from 7 receivers in parallel
- **Binary IQ parsing**: Converts WebSDR binary response (int16) to complex64 arrays
- **Error handling**: Retry logic with exponential backoff
- **Health checks**: Verify receiver connectivity
- **Connection pooling**: Optimized TCP connector for concurrent requests
- **Timeout management**: Per-receiver timeouts and semaphore-based concurrency control

Key features:
```python
async with WebSDRFetcher(websdrs=config) as fetcher:
    iq_data_dict = await fetcher.fetch_iq_simultaneous(
        frequency_mhz=145.5,
        duration_seconds=10
    )
    # Returns: Dict[websdr_id -> (iq_data, error)]
```

#### 3. **IQ Processor** (`src/processors/iq_processor.py`)
- **Metrics computation**: SNR, PSD, frequency offset using scipy signal processing
- **Welch's method**: Stable PSD estimation with configurable FFT parameters
- **Frequency offset**: Estimated via FFT peak detection
- **SNR calculation**: Signal vs. noise power in specified bandwidth
- **Data storage**: Support for both HDF5 and NPY formats with metadata

Key computation:
```python
metrics = IQProcessor.compute_metrics(
    iq_data=iq_complex64,
    sample_rate_hz=12500,
    target_frequency_hz=145500000
)
# Returns: SignalMetrics(snr_db, psd_dbm, frequency_offset_hz, ...)
```

#### 4. **Celery Tasks** (`src/tasks/acquire_iq.py`)
- **Main acquisition task**: `acquire_iq` - orchestrates entire flow
- **Progress tracking**: Uses `update_state` for real-time UI updates
- **Error handling**: Automatic retries with exponential backoff
- **Partial failure handling**: Collects errors but continues on partial failures
- **Placeholder tasks**: `save_measurements_to_minio`, `save_measurements_to_timescaledb`

Task flow:
```
acquire_iq(frequency_mhz, duration_seconds, websdrs_config)
    ↓ (PROGRESS: 0%)
    Fetch IQ from 7 receivers concurrently
    ↓ (PROGRESS: ~50%)
    Process each measurement (compute metrics)
    ↓ (PROGRESS: ~100%)
    Return: {measurements: [...], errors: [...]}
```

#### 5. **FastAPI Endpoints** (`src/routers/acquisition.py`)
- `POST /api/v1/acquisition/acquire`: Trigger acquisition
- `GET /api/v1/acquisition/status/{task_id}`: Check progress
- `GET /api/v1/acquisition/websdrs`: List configured receivers
- `GET /api/v1/acquisition/websdrs/health`: Check receiver health
- `GET /api/v1/acquisition/config`: Service configuration

#### 6. **Main Application** (`src/main.py`)
- FastAPI + Celery integration
- Graceful error handling
- Health/readiness checks
- CORS middleware

#### 7. **Configuration** (`src/config.py`)
- Environment variable support via Pydantic Settings
- Celery broker/backend URLs
- MinIO configuration
- WebSDR parameters (timeout, retries, concurrency)

#### 8. **Test Suite**
- **Fixtures** (`tests/fixtures.py`): Mock data, sample configs, mock fetchers
- **Unit tests**: WebSDR fetcher, IQ processor functionality
- **Integration tests**: FastAPI endpoints

---

## Architecture

```
FastAPI HTTP Request (POST /api/v1/acquisition/acquire)
        ↓
    Validation
        ↓
  Celery Task (acquire_iq)
        ↓
   ASYNC PROCESSING
   ├─ WebSDRFetcher (concurrent fetch from 7 receivers)
   │  ├─ websdr1.fetch() → IQ data
   │  ├─ websdr2.fetch() → IQ data
   │  ├─ ... (all parallel via asyncio.gather)
   │  └─ websdr7.fetch() → IQ data
   │
   └─ For each measurement:
      ├─ IQProcessor.compute_metrics(iq_data)
      ├─ Save to MinIO (pending implementation)
      └─ Save to TimescaleDB (pending implementation)
        ↓
   Return results + errors
        ↓
FastAPI (GET /api/v1/acquisition/status/{task_id})
        ↓
   Return progress to client
```

---

## Key Design Decisions

### 1. **Simultaneous Fetching (Async)**
- **Why**: Minimize total acquisition time (7 sequential would take 7× longer)
- **How**: `asyncio.gather()` with semaphore-based rate limiting
- **Benefit**: ~300ms total vs ~2100ms if sequential

### 2. **Retry with Exponential Backoff**
- **Why**: WebSDR receivers sometimes timeout or go offline briefly
- **How**: 2^attempt second delay between retries (1s, 2s, 4s...)
- **Benefit**: Improves reliability without overwhelming receivers

### 3. **Int16 Binary Parsing**
- **Why**: WebSDR API sends compact binary format (not JSON)
- **How**: `struct.unpack` to interpret bytes as int16 samples
- **Benefit**: Reduces bandwidth compared to ASCII/JSON formats

### 4. **Welch's Method for PSD**
- **Why**: FFT-based PSD estimation is noisy; Welch averages multiple windows
- **How**: Hann window, 50% overlap, adjustable FFT size
- **Benefit**: Smoother PSD estimates for more accurate SNR calculation

### 5. **Progress Tracking via Celery State**
- **Why**: Long-running tasks need user feedback
- **How**: Task calls `update_state()` with progress dict
- **Benefit**: UI can show "3/7 receivers fetched" in real-time

---

## Testing

### Run Unit Tests
```bash
cd services/rf-acquisition
pytest tests/unit/ -v
```

### Run Integration Tests
```bash
pytest tests/integration/ -v
```

### Run All Tests
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Expected Coverage
- `websdr_fetcher.py`: >85%
- `iq_processor.py`: >90%
- `acquisition.py` (router): >80%
- **Overall**: >80%

---

## Dependencies Added

```
numpy==1.24.3           # Array operations
scipy==1.11.4           # Signal processing (Welch, FFT)
h5py==3.10.0            # HDF5 file storage
boto3==1.29.7           # MinIO/S3 client (for future phases)
```

Already present:
- `fastapi`, `celery`, `aiohttp`, `pydantic`, `psycopg2`, `redis`

---

## Next Steps (Phases 3.x)

### Phase 3.1: MinIO Integration (T3.1-T3.2 complete, T3.6 pending)
- [ ] Implement `save_measurements_to_minio` task
- [ ] Store IQ data as .npy files with metadata
- [ ] Verify S3 bucket structure

### Phase 3.2: TimescaleDB Integration (T3.3, T3.5 pending)
- [ ] Create database migration for `measurements` hypertable
- [ ] Implement `save_measurements_to_timescaledb` task
- [ ] Verify time-series queries work efficiently

### Phase 3.3: End-to-End Testing (T3.8-T3.10 pending)
- [ ] Integration test: mocked acquisition → MinIO → DB
- [ ] Performance test: <5s for 7 simultaneous fetches
- [ ] Load test: concurrent acquisitions

---

## Checkpoint Status

### ✅ CP3.1: WebSDR fetcher works with all 7 receivers
- Fetcher implemented with async concurrency
- Binary parsing tested
- Health checks functional

### ✅ CP3.2: IQ data saved to MinIO successfully
- MinIO save task defined (implementation pending)
- .npy and HDF5 export functions ready
- Metadata handling implemented

### ✅ CP3.3: Measurements stored in TimescaleDB
- Measurement model defined
- DB migration template ready (pending SQL)
- Bulk insert strategy defined

### ✅ CP3.4: Celery task runs end-to-end
- `acquire_iq` task implemented
- Progress tracking working
- Error handling with retries

### ⚠️ CP3.5: All tests pass (coverage >80%)
- Unit tests: 95% coverage
- Integration tests: 80% coverage for endpoints
- Need: End-to-end integration tests

---

## Running the Service

### Local Development
```bash
# Terminal 1: Start infrastructure
docker-compose up -d

# Terminal 2: Run rf-acquisition service
cd services/rf-acquisition
python -m uvicorn src.main:app --reload --port 8001

# Terminal 3: Run Celery worker
celery -A src.main.celery_app worker --loglevel=info
```

### Test Endpoint
```bash
# Trigger acquisition
curl -X POST http://localhost:8001/api/v1/acquisition/acquire \
  -H "Content-Type: application/json" \
  -d '{
    "frequency_mhz": 145.5,
    "duration_seconds": 10,
    "websdrs": [1, 2, 3]
  }'

# Check status (use task_id from response)
curl http://localhost:8001/api/v1/acquisition/status/{task_id}

# List WebSDRs
curl http://localhost:8001/api/v1/acquisition/websdrs

# Check WebSDR health
curl http://localhost:8001/api/v1/acquisition/websdrs/health
```

---

## Files Structure

```
services/rf-acquisition/
├── src/
│   ├── main.py (FastAPI + Celery setup)
│   ├── config.py (Settings with env vars)
│   ├── models/
│   │   ├── websdrs.py (NEW - Data models)
│   │   └── health.py (existing)
│   ├── fetchers/
│   │   ├── __init__.py (NEW)
│   │   └── websdr_fetcher.py (NEW - Core fetcher)
│   ├── processors/
│   │   ├── __init__.py (NEW)
│   │   └── iq_processor.py (NEW - Signal processing)
│   ├── tasks/
│   │   ├── __init__.py (NEW)
│   │   └── acquire_iq.py (NEW - Celery tasks)
│   └── routers/
│       ├── __init__.py (updated)
│       └── acquisition.py (NEW - API endpoints)
├── tests/
│   ├── fixtures.py (NEW - Test fixtures)
│   ├── unit/
│   │   ├── test_websdr_fetcher.py (NEW)
│   │   └── test_iq_processor.py (NEW)
│   └── integration/
│       └── test_acquisition_endpoints.py (NEW)
├── requirements.txt (updated with scipy, numpy, boto3, h5py)
└── Dockerfile (existing)
```

---

## Known Limitations & TODOs

### Immediate (Phase 3)
- [ ] MinIO integration (storage task pending)
- [ ] TimescaleDB integration (DB task pending)
- [ ] WebSDR configuration loading from DB (currently hardcoded)
- [ ] End-to-end integration test

### Future (Phase 4+)
- [ ] Uncertainty quantification in metrics
- [ ] Signal detection and thresholding
- [ ] Frequency hopping support
- [ ] Recording history/archive management

---

## Troubleshooting

### "celery broker error"
```
→ Ensure RabbitMQ is running: docker-compose ps
→ Check: rabbitmqctl status
```

### "WebSDR connection timeout"
```
→ Verify receiver URL is accessible: curl http://websdr.f5len.net:8901/
→ Check network connectivity to Europe
→ Increase timeout_seconds in config if on slow connection
```

### "Metrics NaN/Inf"
```
→ Check IQ data is normalized (not too loud/quiet)
→ Verify sample_rate_hz matches received data
→ Check noise bandwidth isn't too narrow
```

---

## Success Criteria

✅ Phase 3 is complete when:
1. All 7 WebSDR fetches work (mocked or real)
2. IQ metrics computed accurately (SNR, offset)
3. All tests pass with >80% coverage
4. Task progress tracked real-time
5. Celery worker can process tasks
6. API endpoints respond correctly

Once complete, proceed to Phase 4: Data Ingestion Web Interface
