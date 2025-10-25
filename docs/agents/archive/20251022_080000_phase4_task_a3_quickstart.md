# 🚀 PHASE 4 TASK A3: Performance Benchmarking - QUICK START GUIDE

**Status**: 🟡 IN PROGRESS  
**Duration**: 1-2 hours  
**Objective**: Measure API latency, Celery task execution time, and concurrent capacity

---

## 📋 Quick Checklist

### Prerequisites
- [ ] Docker containers running: `docker compose ps` (13/13 healthy)
- [ ] Services accessible:
  - [ ] API Gateway: `curl http://localhost:8000/health`
  - [ ] RF Acquisition: `curl http://localhost:8001/health`
- [ ] Python 3.11+ available: `python --version`

### Execution Steps

#### Step 1: Verify System Status (2 min)
```powershell
# Check all containers are running
docker compose ps

# Verify key services
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8003/health  # Inference
```

#### Step 2: Run Performance Benchmark (10-15 min)
```powershell
# Navigate to project root
cd c:\Users\aless\Documents\Projects\heimdall

# Run performance benchmarking script
python scripts/performance_benchmark.py --output PHASE4_TASK_A3_PERFORMANCE_REPORT.md
```

**What it measures**:
- ✅ API endpoint latency (5 requests per endpoint)
- ✅ Celery task execution time (submission → completion)
- ✅ Concurrent capacity (10 simultaneous tasks)
- ✅ Inference latency verification (<500ms requirement)

**Expected output**:
```
========================================================================
🚀 PHASE 4 TASK A3: Performance Benchmarking
========================================================================

📋 Running Health Checks...
  ✅ API Gateway: OK
  ✅ RF Acquisition: OK

⏱️  Benchmarking API Endpoints...
  ✅ Health Check: 23.45ms (±1.23)
  ✅ Get WebSDR Config: 45.67ms (±2.34)
  ✅ Get Task Status: 34.56ms (±1.89)

⏱️  Benchmarking Celery Tasks...
  Submitting: RF Acquisition Task...
    Task ID: abc123def456
    Completed in 65.23s (State: PARTIAL_FAILURE)

⏱️  Benchmarking Concurrent Capacity (10 requests)...
  ✅ Task 1/10 submitted: task-001
  ✅ Task 2/10 submitted: task-002
  ...
  Total submission time: 2.34s
  Waiting for all 10 tasks to complete...
  Task task-001... completed: 62.45s
  ...

⏱️  Verifying Inference Latency Requirement...
  ✅ PASS Inference latency: 45.67ms (target: <500ms)

📊 Generating Performance Report...
  ✅ Report saved to: PHASE4_TASK_A3_PERFORMANCE_REPORT.md
  ✅ JSON data saved to: PHASE4_TASK_A3_PERFORMANCE_REPORT.json

========================================================================
✅ Benchmarking Complete
========================================================================
```

#### Step 3: Review Performance Report (5 min)
```powershell
# Open the generated report
notepad PHASE4_TASK_A3_PERFORMANCE_REPORT.md

# Or view JSON data
python -m json.tool PHASE4_TASK_A3_PERFORMANCE_REPORT.json | less
```

**Key metrics to verify**:
| Metric                | Target | Status            |
| --------------------- | ------ | ----------------- |
| API Latency (mean)    | <100ms | ✅                 |
| API Latency (max)     | <200ms | ✅                 |
| Celery Task Time      | 60-70s | ✅ (network-bound) |
| Concurrent Completion | >80%   | ✅                 |
| Inference Latency     | <500ms | ✅                 |

#### Step 4: Run Load Test (Optional - if time permits)
```powershell
# Run production-scale load test
python scripts/load_test.py --concurrent 50 --duration 300 `
  --output PHASE4_TASK_B1_LOAD_TEST_REPORT.md
```

**Expected time**: 5-10 minutes  
**What it tests**: 50 concurrent tasks over 5 minutes

---

## 📊 Expected Results

### API Endpoint Benchmarks
```
Health Check:           ~25ms
Get WebSDR Config:      ~45ms
Get Task Status:        ~35ms
```

### Celery Task Execution
```
RF Acquisition Task:    ~65s (network-bound waiting for WebSDRs)
```

### Concurrent Capacity
```
10 concurrent tasks:    100% completion rate
Submission time:        <3s
Mean completion:        ~65s
```

### Inference Service
```
Inference latency:      <50ms (well below 500ms requirement)
```

---

## 🔍 Troubleshooting

### Issue: "Connection refused" error
```powershell
# Verify containers are running
docker compose ps

# If not, start them
docker compose up -d

# Wait 30 seconds for services to initialize
Start-Sleep -Seconds 30

# Retry the benchmark
python scripts/performance_benchmark.py
```

### Issue: "Timeout waiting for task completion"
```
This is normal if:
- WebSDRs are offline (timeout handling working correctly)
- System is under heavy load
- Network latency to external WebSDRs

Expected behavior: Task completes with PARTIAL_FAILURE status
```

### Issue: Inference service error
```
If inference service not responding:
1. It's not required for A3 completion
2. Skip that section and proceed
3. Will be completed in Phase 6

Status will show: "Service not available" (which is OK for Phase 4)
```

---

## 📈 Report Contents

Generated `PHASE4_TASK_A3_PERFORMANCE_REPORT.md` includes:

1. **Executive Summary**
   - Key metrics table
   - All targets vs. actual

2. **Detailed Results**
   - Per-endpoint latency breakdown
   - Celery task execution times
   - Concurrent capacity results
   - Inference latency verification

3. **Checkpoint Validation**
   - CP4.A3: All checkpoints checked

4. **Key Findings**
   - Strengths identified
   - Observations and bottleneck analysis
   - Production recommendations

5. **Next Steps**
   - Task B1 (Load Testing) planning
   - Phase 5 entry point

---

## ✅ Success Criteria (CP4.A3)

**All must pass** ✅:
- [ ] API latency <100ms (mean)
- [ ] Celery task execution baseline established
- [ ] Concurrent handling verified (10+ requests)
- [ ] Inference <500ms latency (if service available)
- [ ] Performance report generated
- [ ] No crashes or memory leaks observed

---

## 🎯 After Task A3

### Option 1: Proceed to Task B1 (Load Testing) - Recommended
```powershell
# If you have another 10-15 minutes
python scripts/load_test.py --concurrent 50 --duration 300
```

### Option 2: Continue with Phase 5 (Training Pipeline)
- A3 + B1 can proceed in parallel with Phase 5
- Phase 5 (ML model training) doesn't depend on load test results
- Can start concurrent work on training pipeline

---

## 📚 References

- **AGENTS.md** (lines 1061-1148): Phase 4 Task A3 specifications
- **PHASE4_PROGRESS_DASHBOARD.md**: Current phase status (48% complete)
- **performance_benchmark.py**: Script implementation
- **docker compose.yml**: Service configuration

---

## ⏱️ Time Estimation

| Step                         | Time       |
| ---------------------------- | ---------- |
| Prerequisites check          | 2 min      |
| Run performance_benchmark.py | 12 min     |
| Review report                | 5 min      |
| Optional: load_test.py       | 10 min     |
| **Total**                    | **29 min** |

**Estimated start time**: 2 hours  
**Estimated completion**: 2 hours + execution time

---

## 🚀 Status Indicators

After completion, Phase 4 will be:
- ✅ Task A1: E2E Tests (87.5% - 7/8 passing)
- ✅ Task A2: Docker Integration (100% - all containers)
- ✅ Task A3: Performance Benchmarking (IN PROGRESS → COMPLETE)
- ⏳ Task B1: Load Testing (Ready to start)

**Overall Phase 4**: 50% → 75% upon A3 completion

---

## 🔗 Quick Links

- Run benchmark: `python scripts/performance_benchmark.py`
- View report: `notepad PHASE4_TASK_A3_PERFORMANCE_REPORT.md`
- Check containers: `docker compose ps`
- View services: `open http://localhost:8001/api/v1/acquisition/config`

**Ready? Run this command:**
```powershell
python scripts/performance_benchmark.py --output PHASE4_TASK_A3_PERFORMANCE_REPORT.md
```

---

**Last Updated**: 2025-10-22  
**Next Session**: Task B1 Load Testing or Phase 5 Setup
