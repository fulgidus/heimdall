# 🔧 PHASE 6: Makefile Additions

Add these commands to your Makefile for Phase 6 development:

```makefile
# PHASE 6: INFERENCE SERVICE DEVELOPMENT TARGETS

# Phase 6 Quick Start
phase6-start:
	@echo "🚀 Starting Phase 6: Inference Service"
	@echo ""
	@code PHASE6_START_HERE.md
	@echo "✅ Read PHASE6_START_HERE.md in your editor"

# Phase 6 Setup
phase6-setup:
	@echo "🔧 Setting up Phase 6 environment..."
	python scripts/create_service.py inference
	@echo "✅ Created services/inference/"
	@echo ""
	@echo "Next steps:"
	@echo "  1. cd services/inference"
	@echo "  2. pip install -r requirements.txt"
	@echo "  3. python src/main.py"

# Phase 6 Prerequisites Check
phase6-check:
	@echo "✅ Checking Phase 6 prerequisites..."
	@echo ""
	@echo "1️⃣  Docker containers:"
	docker-compose ps
	@echo ""
	@echo "2️⃣  Redis connectivity:"
	docker-compose exec redis redis-cli PING
	@echo ""
	@echo "3️⃣  MLflow registry:"
	@echo "    Open http://localhost:5000/models"
	@echo ""
	@echo "All checks passed! Ready for Phase 6 ✅"

# Build inference service
build-inference:
	@echo "Building inference service..."
	docker build -t heimdall-inference:latest services/inference/
	@echo "✅ Build complete: heimdall-inference:latest"

# Run inference service
run-inference:
	@echo "Starting inference service..."
	docker-compose up -d inference
	@echo "✅ Service started at http://localhost:8006"
	@echo ""
	@echo "Check health: curl http://localhost:8006/health"

# Stop inference service
stop-inference:
	docker-compose stop inference
	@echo "✅ Inference service stopped"

# Inference service logs
logs-inference:
	docker-compose logs -f inference

# Test inference service
test-inference:
	@echo "Running inference tests..."
	pytest services/inference/tests/ -v --cov=services/inference/src --cov-report=html
	@echo ""
	@echo "✅ Tests complete. Coverage report: htmlcov/index.html"

# Predict endpoint test
predict-test:
	@echo "Testing /predict endpoint..."
	@powershell -Command "$$IQData = @(@(1.5, 0.3), @(1.2, 0.5)); $$body = @{iq_data = $$IQData; cache_enabled = $$true} | ConvertTo-Json; Invoke-WebRequest -Uri 'http://localhost:8006/predict' -Method Post -Body $$body -ContentType 'application/json' | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 10"

# Load test inference
load-test-inference:
	@echo "Running load test (100 concurrent requests)..."
	pytest services/inference/tests/load_test_inference.py -v
	@echo ""
	@echo "✅ Load test complete"

# Monitor inference metrics
metrics-inference:
	@echo "Opening Prometheus metrics..."
	@echo "http://localhost:9090/graph"
	@echo ""
	@echo "Popular metrics to query:"
	@echo "  - inference_latency_ms (histogram)"
	@echo "  - cache_hit_rate (gauge)"
	@echo "  - inference_requests_total (counter)"
	@echo "  - inference_errors_total (counter)"

# Phase 6 Dashboard
phase6-dashboard:
	@echo "📊 PHASE 6 Status Dashboard"
	@echo "================================"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  - PHASE6_START_HERE.md"
	@echo "  - PHASE6_PREREQUISITES_CHECK.md"
	@echo "  - PHASE6_PROGRESS_DASHBOARD.md"
	@echo "  - PHASE6_CODE_TEMPLATE.md"
	@echo "  - PHASE6_COMPLETE_REFERENCE.md"
	@echo ""
	@echo "📋 Tasks (10):"
	@echo "  T6.1 ONNX Loader .......................... [IN-PROGRESS]"
	@echo "  T6.2 Predict Endpoint ..................... [NOT-STARTED]"
	@echo "  T6.3 Uncertainty Ellipse .................. [NOT-STARTED]"
	@echo "  T6.4 Batch Prediction ..................... [NOT-STARTED]"
	@echo "  T6.5 Model Versioning ..................... [NOT-STARTED]"
	@echo "  T6.6 Performance Monitoring ............... [NOT-STARTED]"
	@echo "  T6.7 Load Testing ......................... [NOT-STARTED]"
	@echo "  T6.8 Model Info Endpoint .................. [NOT-STARTED]"
	@echo "  T6.9 Graceful Reloading ................... [NOT-STARTED]"
	@echo "  T6.10 Comprehensive Tests ................. [NOT-STARTED]"
	@echo ""
	@echo "✅ Checkpoints (5):"
	@echo "  CP6.1: ONNX Model Loader .................. [PENDING]"
	@echo "  CP6.2: Prediction Endpoint ................ [PENDING]"
	@echo "  CP6.3: Redis Caching ...................... [PENDING]"
	@echo "  CP6.4: Uncertainty Visualization .......... [PENDING]"
	@echo "  CP6.5: Load Test Validation ............... [PENDING]"
	@echo ""
	@echo "🎯 SLA Requirements:"
	@echo "  Inference Latency (P95): <500ms"
	@echo "  Cache Hit Rate: >80%"
	@echo "  Code Coverage: >80%"
	@echo "  Load Test: 100 concurrent ✅"
	@echo ""
	@echo "📅 Timeline:"
	@echo "  Start: 2025-10-22 (TODAY)"
	@echo "  Target: 2025-10-24"
	@echo "  Duration: 2 days"

# Complete Phase 6 workflow
phase6-all: phase6-check phase6-setup build-inference run-inference
	@echo ""
	@echo "✅ Phase 6 environment ready!"
	@echo "   Next: code PHASE6_START_HERE.md"

# Phase 6 cleanup
phase6-clean:
	@echo "Cleaning Phase 6 artifacts..."
	rm -rf services/inference/__pycache__
	rm -rf services/inference/.pytest_cache
	rm -rf services/inference/htmlcov
	rm -rf services/inference/.coverage
	@echo "✅ Cleanup complete"

# Update progress
phase6-progress:
	@echo "📊 Updating Phase 6 progress..."
	@code PHASE6_PROGRESS_DASHBOARD.md

# Phase 6 reference
phase6-help:
	@echo "📚 PHASE 6 Quick Reference"
	@echo ""
	@echo "Getting Started:"
	@echo "  make phase6-start       Open PHASE6_START_HERE.md"
	@echo "  make phase6-check       Verify prerequisites"
	@echo "  make phase6-setup       Create service structure"
	@echo ""
	@echo "Development:"
	@echo "  make build-inference    Build Docker image"
	@echo "  make run-inference      Start service"
	@echo "  make logs-inference     View service logs"
	@echo "  make test-inference     Run test suite"
	@echo ""
	@echo "Testing:"
	@echo "  make predict-test       Test /predict endpoint"
	@echo "  make load-test-inference   Run load test (100 concurrent)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make metrics-inference  View Prometheus metrics"
	@echo ""
	@echo "Progress:"
	@echo "  make phase6-dashboard   Show status dashboard"
	@echo "  make phase6-progress    Update progress tracker"
	@echo ""
	@echo "Cleanup:"
	@echo "  make phase6-clean       Clean artifacts"
	@echo "  make stop-inference     Stop service"
```

---

## 📝 How to Use

Add these targets to your `Makefile` (around line 150+):

```bash
# Copy the commands above into Makefile
# Then use:

make phase6-help          # View all Phase 6 commands
make phase6-start         # Read overview (5 min)
make phase6-check         # Verify system ready (5 min)
make phase6-setup         # Create service (2 min)
make build-inference      # Build Docker image
make run-inference        # Start service
make test-inference       # Run tests
make phase6-dashboard     # View status
```

---

## 🎯 Key Make Targets

| Command                    | Purpose                   | Time   |
| -------------------------- | ------------------------- | ------ |
| `make phase6-help`         | Show all Phase 6 commands | 1 sec  |
| `make phase6-start`        | Read documentation        | 5 min  |
| `make phase6-check`        | Verify prerequisites      | 5 min  |
| `make phase6-setup`        | Create service scaffold   | 2 min  |
| `make build-inference`     | Build Docker image        | 1 min  |
| `make run-inference`       | Start service             | 10 sec |
| `make test-inference`      | Run full test suite       | 1 min  |
| `make load-test-inference` | Validate <500ms SLA       | 2 min  |
| `make phase6-dashboard`    | View status               | 1 sec  |

---

## 💡 Quick Workflow

```bash
# 1. Start Phase 6 (5 min)
make phase6-start

# 2. Verify system (5 min)
make phase6-check

# 3. Setup (2 min)
make phase6-setup

# 4. Build (1 min)
make build-inference

# 5. Run (10 sec)
make run-inference

# 6. Test (1 min)
make test-inference

# Total: ~14 minutes to first working service!
```

---

**Add these to your Makefile for convenient Phase 6 development!**

