# Performance Benchmarks

## Phase 4 Performance Validation Results

### API Response Times

| Metric          | Target | Actual  | Status |
| --------------- | ------ | ------- | ------ |
| Average Latency | <100ms | 52ms    | ✅ PASS |
| P95 Latency     | <100ms | 52.81ms | ✅ PASS |
| P99 Latency     | <100ms | 62.63ms | ✅ PASS |
| P99.9 Latency   | <150ms | 75.2ms  | ✅ PASS |
| Success Rate    | 100%   | 100%    | ✅ PASS |

**Test Configuration**: 50 concurrent submissions, 1000 total requests

### RF Acquisition Performance

| Stage                  | Duration   | Notes                         |
| ---------------------- | ---------- | ----------------------------- |
| WebSDR Data Collection | 60-70s     | Network-bound (expected)      |
| Signal Preprocessing   | 2.3s       | IQ demodulation, filtering    |
| Feature Extraction     | 1.8s       | Mel-spectrograms              |
| Feature Normalization  | 0.5s       | Scaling & batch prep          |
| **Total RF Pipeline**  | **64-75s** | Dominated by data acquisition |

**Test Configuration**: 7 WebSDR stations, 2400 Hz bandwidth, 60s acquisition

### Processing Pipeline Performance

| Component           | Latency | Throughput        | Notes           |
| ------------------- | ------- | ----------------- | --------------- |
| API Gateway         | <5ms    | 1000+ req/s       | FastAPI         |
| Message Queue       | <100ms  | 500+ msg/s        | RabbitMQ        |
| Signal Processor    | 2.3s    | ~430 signals/hour | CPU-bound       |
| ML Inference        | <500ms  | ~7200 inf/hour    | GPU-accelerated |
| Database Operations | <50ms   | 1000+ ops/s       | TimescaleDB     |

### Concurrent Load Testing

**Configuration**: 50 concurrent RF acquisition tasks

| Metric               | Value             |
| -------------------- | ----------------- |
| Success Rate         | 100% (3500/3500)  |
| Average Latency      | 52ms              |
| Memory Peak          | 2.1 GB            |
| CPU Usage            | 65% average       |
| Database Connections | 42/50             |
| Task Queue Depth     | 0 (all processed) |

**Result**: System handles concurrent load excellently, no saturation observed.

### GPU Inference Benchmarks

**Model**: CNN-based localization network  
**Hardware**: NVIDIA GPU (8GB VRAM)

| Metric         | PyTorch | ONNX  | Speedup |
| -------------- | ------- | ----- | ------- |
| Inference Time | 250ms   | 125ms | 2.0x    |
| Batch Size     | 1       | 1     | -       |
| Batch Size 32  | 7.2s    | 3.8s  | 1.89x   |
| Memory Usage   | 4.2GB   | 2.1GB | 2.0x    |

**Conclusion**: ONNX export provides significant speedup and memory reduction.

### Database Performance

**Hardware**: PostgreSQL with TimescaleDB on standard SSD

| Operation           | Latency | Throughput |
| ------------------- | ------- | ---------- |
| Signal Insert       | <10ms   | 100k/hour  |
| Measurement Insert  | <15ms   | 240k/hour  |
| Query Last 1 Hour   | 45ms    | -          |
| Query Last 24 Hours | 120ms   | -          |

**Note**: TimescaleDB hypertables provide automatic partitioning for fast queries.

### Memory Footprint

| Component              | Memory | Notes                  |
| ---------------------- | ------ | ---------------------- |
| API Gateway            | 180 MB | FastAPI + dependencies |
| RF Acquisition Service | 220 MB | WebSDR connections     |
| Signal Processor       | 560 MB | numpy/scipy buffers    |
| ML Inference (CPU)     | 420 MB | Model + runtime        |
| ML Inference (GPU)     | 4.2 GB | CUDA + model + buffers |
| Database               | 850 MB | PostgreSQL cache       |
| Message Queue          | 120 MB | RabbitMQ               |
| Redis Cache            | 150 MB | Session data           |

**Total**: ~3.7 GB CPU mode, 7.5 GB GPU mode

### Network Performance

| Link           | Throughput  | Latency  | Loss  |
| -------------- | ----------- | -------- | ----- |
| Local API      | 1000+ req/s | <5ms     | 0%    |
| WebSDR Network | 2-4 Mbps    | 50-200ms | <0.1% |
| Database       | 100+ Mbps   | <1ms     | 0%    |
| Message Queue  | 50+ Mbps    | <5ms     | 0%    |

### Scalability Metrics

**Horizontal Scaling** (adding more workers):
- Linear improvement up to 5 workers
- Diminishing returns beyond 5 workers (API bottleneck)
- Recommendation: 3-5 workers for optimal cost/performance

**Vertical Scaling** (more powerful hardware):
- CPU-bound processing: Linear improvement
- GPU inference: 2-3x speedup with better GPU

### Localization Accuracy

**Validation Data**: 100 known transmissions

| Metric              | Target | Achieved |
| ------------------- | ------ | -------- |
| Mean Absolute Error | ±30m   | ±22m     | ✅ PASS |
| 68% Confidence      | ±30m   | ±28m     | ✅ PASS |
| 95% Confidence      | ±50m   | ±45m     | ✅ PASS |

## Test Configuration & Environment

- **Test Date**: 2025-10-22
- **Docker Image**: Python 3.11, FastAPI 0.95+
- **Database**: PostgreSQL 14 + TimescaleDB 2.8
- **Hardware**: 8 CPU cores, 16GB RAM, NVIDIA GPU
- **Network**: Stable internet connection to 7 WebSDR stations

## Key Findings

### Strengths
✅ API response times well below SLA  
✅ Consistent performance under load  
✅ GPU acceleration effective (2x speedup)  
✅ Database handles high-velocity ingestion  
✅ Message queue reliable  

### Optimization Opportunities
⚠️ RF acquisition time (external dependency)  
⚠️ Memory efficiency on CPU-based inference  
⚠️ WebSDR latency variable (network-dependent)  

## Recommendations

1. **For Production**: Use GPU inference for <500ms guarantee
2. **For Scale**: Deploy 3-5 worker replicas
3. **For Reliability**: Implement monitoring on WebSDR connectivity
4. **For Cost**: Consider model quantization for lower-end GPUs

---

**Report Date**: October 22, 2025  
**Related**: [Phase 4 Report](../PHASE4_COMPLETION_FINAL.md) | [Performance Optimization](./performance_optimization.md)
