# Phase 4 - Task A1: E2E Test Suite - EXECUTION GUIDE

**Date**: October 22, 2025  
**Task**: Run End-to-End integration tests  
**Estimated Time**: 15-20 minutes (first run may take longer)  

---

## Prerequisites

✅ All prerequisites met:
- Docker Compose services running (verified)
- RF Acquisition service deployed in Docker
- PostgreSQL, Redis, RabbitMQ accessible
- MinIO available for storage
- virtualenv activated (Python 3.11)

---

## Quick Start

### Step 1: Install E2E Test Dependencies

```powershell
# Activate virtualenv
.venv\Scripts\Activate.ps1

# Install additional test dependencies
pip install pytest-asyncio httpx minio
```

### Step 2: Run E2E Tests

```powershell
cd services/rf-acquisition

# Run all E2E tests
pytest tests/e2e/ -v --tb=short

# Run specific test
pytest tests/e2e/test_complete_workflow.py::test_acquisition_complete_workflow -v

# Run with output capturing disabled (see print statements)
pytest tests/e2e/ -v -s

# Run only quick tests (skip @slow marked)
pytest tests/e2e/ -v -m "not slow"

# Run with detailed timing
pytest tests/e2e/ -v --durations=10
```

### Step 3: Verify Docker Services

Before running tests, verify all services are healthy:

```powershell
# Check container status
docker compose ps

# Should see (all UP or healthy):
# - heimdall-postgres (healthy)
# - heimdall-redis (healthy)
# - heimdall-rabbitmq (healthy)
# - heimdall-minio (healthy)
# - heimdall-rf-acquisition (running or healthy)

# If any are unhealthy, restart:
docker compose restart heimdall-rf-acquisition
```

---

## Test Suite Overview

### Smoke Tests (Quick validation, < 10 seconds each)

```bash
# 1. Health endpoint responding
pytest tests/e2e/test_complete_workflow.py::test_health_endpoint -v

# 2. OpenAPI docs available
pytest tests/e2e/test_complete_workflow.py::test_api_docs_available -v
```

**Expected Output**:
```
✓ Health endpoint responding
✓ OpenAPI documentation available
```

---

### Core E2E Tests (10-30 seconds each)

#### Test 1: Complete Workflow (Priority: **HIGH**)
```bash
pytest tests/e2e/test_complete_workflow.py::test_acquisition_complete_workflow -v -s
```

**What it does**:
1. Triggers acquisition at 145.50 MHz for 2 seconds
2. Waits for completion (polling, max 60 seconds)
3. Verifies 7 measurements in database
4. Validates data quality (SNR, frequency, WebSDR IDs)
5. Checks MinIO files (if available)

**Expected output**:
```
✓ Acquisition triggered with task_id: 550e8400-e29b-41d4-a716-446655440000
✓ Acquisition completed successfully
✓ Database has 7 measurements
✓ All measurements have valid data
✓ MinIO has 21 files (or ⚠ MinIO files not accessible)
```

**Success criteria**:
- [ ] Task completes in < 30 seconds
- [ ] State = SUCCESS
- [ ] 7 measurements in DB
- [ ] SNR values are positive
- [ ] All WebSDR IDs populated

---

#### Test 2: Partial Failure Handling
```bash
pytest tests/e2e/test_complete_workflow.py::test_websdr_partial_failure -v -s
```

**What it does**:
- Simulates one WebSDR failing
- Verifies acquisition still succeeds with 6/7 data
- Checks error logging

**Expected output**:
```
✓ Acquisition succeeded despite WebSDR failure
✓ Database has 6 measurements from working receivers
```

---

#### Test 3: Concurrent Acquisitions
```bash
pytest tests/e2e/test_complete_workflow.py::test_concurrent_acquisitions -v -s
```

**What it does**:
- Triggers 5 acquisitions simultaneously at different frequencies
- Waits for all to complete
- Verifies no cross-contamination (7 measurements each)

**Expected output**:
```
✓ Triggered 5 concurrent acquisitions
✓ All 5 acquisitions completed successfully
✓ Each acquisition has independent data (7 measurements each)
```

---

#### Test 4: Status Polling
```bash
pytest tests/e2e/test_complete_workflow.py::test_acquisition_status_polling -v -s
```

**What it does**:
- Monitors status transitions (PENDING → RUNNING → SUCCESS)
- Verifies progress increases monotonically
- Checks polling reliability

**Expected output**:
```
✓ Initial status: PENDING (0%)
✓ Final status: SUCCESS (100%)
```

---

#### Test 5: API Error Handling
```bash
pytest tests/e2e/test_complete_workflow.py::test_api_error_handling -v -s
```

**What it does**:
- Tests invalid parameters (negative frequency, excessive duration)
- Verifies proper HTTP error codes (400, 422)

**Expected output**:
```
✓ Rejected invalid frequency
✓ Rejected excessive duration
✓ Rejected missing parameters
```

---

#### Test 6: Measurement Retrieval
```bash
pytest tests/e2e/test_complete_workflow.py::test_measurement_retrieval -v -s
```

**What it does**:
- Retrieves measurements after acquisition
- Verifies endpoint returns all 7 measurements
- Validates data structure (frequency, SNR, timestamps)

**Expected output**:
```
✓ Retrieved 7 measurements via API
✓ All measurements have correct structure and frequency
```

---

### Advanced Tests (Slow, 30-120 seconds)

#### Test 7: Long Acquisition (10 seconds)
```bash
pytest tests/e2e/test_complete_workflow.py::test_long_acquisition -v -s -m slow
```

**What it does**:
- Runs 10-second acquisition (vs. normal 2 seconds)
- Tracks progress updates
- Verifies completion within timeout

**Expected output**:
```
✓ Long acquisition completed: SUCCESS
✓ Long acquisition has 7 measurements
```

---

## Run All Tests (Complete Suite)

```powershell
# Run all E2E tests with summary
pytest tests/e2e/ -v --tb=short

# Or with timing info
pytest tests/e2e/ -v --durations=5

# Or for CI/CD (minimal output)
pytest tests/e2e/ -q
```

**Expected output**:
```
test_complete_workflow.py::test_health_endpoint PASSED                    [  8%]
test_complete_workflow.py::test_api_docs_available PASSED                 [ 16%]
test_complete_workflow.py::test_acquisition_complete_workflow PASSED      [ 25%]
test_complete_workflow.py::test_websdr_partial_failure PASSED             [ 33%]
test_complete_workflow.py::test_concurrent_acquisitions PASSED            [ 41%]
test_complete_workflow.py::test_acquisition_status_polling PASSED         [ 50%]
test_complete_workflow.py::test_api_error_handling PASSED                 [ 58%]
test_complete_workflow.py::test_measurement_retrieval PASSED              [ 66%]
test_complete_workflow.py::test_long_acquisition PASSED                   [ 100%]

======================== 9 passed in 2m 34s ========================
```

---

## Troubleshooting

### Error: "Connection refused" (localhost:8001)

**Cause**: RF Acquisition service not running or not accessible

**Fix**:
```powershell
# Check service status
docker compose ps heimdall-rf-acquisition

# If unhealthy or exited, restart
docker compose restart heimdall-rf-acquisition

# Wait 10 seconds for startup
Start-Sleep -Seconds 10

# Check logs
docker compose logs heimdall-rf-acquisition | tail -20

# Retry tests
pytest tests/e2e/ -v
```

---

### Error: "psycopg2.OperationalError: could not connect to server"

**Cause**: Database not accessible

**Fix**:
```powershell
# Check Postgres status
docker compose ps heimdall-postgres

# If unhealthy, restart
docker compose restart heimdall-postgres

# Wait for database to be ready
Start-Sleep -Seconds 10

# Test connection
docker compose exec heimdall-postgres psql -U heimdall_user -d heimdall -c "SELECT 1"

# Retry tests
pytest tests/e2e/ -v
```

---

### Error: "FAILED test_... - AssertionError: Expected 7 measurements, got X"

**Cause**: WebSDR fetch failed or incomplete

**Possible reasons**:
1. Mock WebSDRs not configured correctly
2. Network timeouts
3. Database transaction issues

**Fix**:
```powershell
# Check service logs for errors
docker compose logs heimdall-rf-acquisition | grep -i error

# Run single test with full output
pytest tests/e2e/test_complete_workflow.py::test_acquisition_complete_workflow -vvs

# Check database directly
docker compose exec heimdall-postgres psql -U heimdall_user -d heimdall -c \
  "SELECT COUNT(*) FROM measurements"
```

---

### Error: "Timeout waiting for acquisition to complete"

**Cause**: Acquisition taking too long (possibly mock WebSDR latency)

**Fix**:
```powershell
# Check if service is responsive
curl http://localhost:8001/health

# Increase timeout in test (edit conftest.py)
# Change: await api_helper.wait_for_completion(task_id, timeout_seconds=60)
# To:     await api_helper.wait_for_completion(task_id, timeout_seconds=120)

# Retry with longer timeout
pytest tests/e2e/ -v
```

---

### Error: "minio import not found"

**Cause**: MinIO Python client not installed

**Fix**:
```powershell
pip install minio

# Retry tests
pytest tests/e2e/ -v
```

---

## Performance Baselines

**Expected timings** (on standard development machine):

| Test                 | Duration    | Notes                             |
| -------------------- | ----------- | --------------------------------- |
| Health check         | < 1s        | Smoke test                        |
| Single acquisition   | 10-15s      | 2s acquisition + polling overhead |
| 5 concurrent         | 15-20s      | All parallel                      |
| Long acquisition     | 20-30s      | 10s acquisition + overhead        |
| Full suite (9 tests) | 2-3 minutes | Total including setup             |

**If tests run significantly slower**, check:
- CPU usage (pytest may be slow if system busy)
- Network latency (Docker network performance)
- Database performance (check Postgres logs)

---

## CI/CD Integration

For automated testing in pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run E2E Tests
  run: |
    cd services/rf-acquisition
    pip install pytest pytest-asyncio httpx
    pytest tests/e2e/ -v --tb=short --junit-xml=test-results.xml
    
- name: Upload Test Results
  uses: actions/upload-artifact@v2
  with:
    name: test-results
    path: services/rf-acquisition/test-results.xml
```

---

## Next Steps

After all E2E tests pass ✅:

1. **Task A2**: Docker Integration Validation (health checks, logs)
2. **Task A3**: Database Schema Verification (table structure)
3. **Task B1**: Performance Benchmarking (latency baselines)
4. **Task C1**: API Gateway Integration (proxy endpoints)

---

## Test Statistics

- **Total tests**: 9
- **Smoke tests**: 2 (< 10s each)
- **Core tests**: 6 (10-30s each)
- **Advanced tests**: 1 (slow, > 30s)
- **Total coverage**: Complete acquisition workflow
- **Success criteria**: All must PASS for Phase 4a completion

---

**Ready to test?** Run:
```powershell
cd services/rf-acquisition
pytest tests/e2e/ -v -s
```

**Questions?** Check logs:
```powershell
docker compose logs -f heimdall-rf-acquisition
```

---

**Phase 4a Status**: 🟡 READY (E2E tests created, waiting to run)  
**Next checkpoint**: Oct 22, 2025 end-of-day
