# GPU Infrastructure Overview

**Last Updated**: 2024-11-04  
**Maintainer**: DevOps Team

## Quick Links

- 🔧 [Docker Image Policy](./DOCKER_IMAGE_POLICY.md) - Maintenance procedures
- 🚀 [CUDA Migration Guide](./CUDA_MIGRATION_GUIDE.md) - Step-by-step migration
- 📖 [GPU Training Quickstart](../GPU_TRAINING_QUICKSTART.md) - Get started
- 🐛 [Troubleshooting Guide](./troubleshooting_guide.md) - Common issues

## Current Stack (Nov 2024)

### Container Images

| Service | Base Image | CUDA | PyTorch | Status |
|---------|-----------|------|---------|--------|
| Training | nvidia/cuda:12.6.0-runtime-ubuntu22.04 | 12.6 | 2.5.0 | ✅ Active |
| Inference | python:3.11-slim | N/A | N/A | ✅ CPU-only |

### GPU Requirements

**Minimum**:
- NVIDIA GPU with Compute Capability 6.0+ (Pascal generation or newer)
- NVIDIA Driver 535.86.10 or newer
- 4GB VRAM minimum (8GB recommended for training)

**Recommended**:
- RTX 3060 or better (12GB VRAM)
- NVIDIA Driver 550+ (latest stable)
- 16GB system RAM

**Tested On**:
- RTX 3090 (24GB VRAM) - Excellent
- RTX 3080 (10GB VRAM) - Good
- RTX 3060 Ti (8GB VRAM) - Acceptable
- GTX 1080 Ti (11GB VRAM) - Works but older CUDA cores

### Software Stack

```
Host System
  └─ NVIDIA Driver 535+
      └─ Docker 20.10+ with NVIDIA Container Toolkit
          └─ Training Container
              ├─ CUDA 12.6 Runtime
              ├─ Python 3.11
              ├─ PyTorch 2.5.0+cu126
              ├─ TorchVision 0.20.0+cu126
              ├─ PyTorch Lightning 2.1.0+
              ├─ CuPy 13.0.0+
              ├─ MLflow 2.9.0+
              └─ FastAPI + Celery
```

## Architecture Decisions

### Why CUDA 12.6?

**Chosen**: `nvidia/cuda:12.6.0-runtime-ubuntu22.04`

**Reasons**:
1. ✅ Long-term support (through 2026+)
2. ✅ PyTorch 2.5+ compatible
3. ✅ Active maintenance and security updates
4. ✅ Broad GPU driver compatibility (535+)
5. ✅ Not deprecated (avoids future migration)

**Rejected Alternatives**:
- ❌ CUDA 12.1 - Deprecated, scheduled for deletion
- ❌ CUDA 11.8 - Older, limited PyTorch support
- ⚠️ CUDA 13.0 - Too new, limited library support (considered for future)

### Why PyTorch 2.5?

**Reasons**:
1. ✅ Native CUDA 12.6 support
2. ✅ Performance improvements (5-15% faster)
3. ✅ Better memory efficiency
4. ✅ Improved `torch.compile` support
5. ✅ Backward compatible with 2.2 models

**Upgrade Path**: 2.2.0 (cu121) → 2.5.0 (cu126)

### Why Not NGC Containers?

**NGC Container** (`nvcr.io/nvidia/pytorch:25.04-py3`):
- ✅ Pros: Official support, pre-optimized, production-ready
- ❌ Cons: Larger size (~8GB vs ~4GB), less version control
- 📋 Status: Available as `Dockerfile.ngc`, evaluate for production in Q2 2025

## Service Configuration

### Training Service

**Resources** (docker-compose.yml):
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # Use all available GPUs
          capabilities: [gpu]
```

**Environment**:
```bash
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=0  # Optional: restrict to specific GPU
```

**Typical Memory Usage**:
- Base container: ~500MB
- PyTorch loaded: ~1.5GB
- Training task (small): ~3-4GB VRAM
- Training task (large): ~6-8GB VRAM

### Celery Worker Configuration

**Pool**: `solo` (single-threaded)  
**Reason**: Avoids daemon process issues with PyTorch DataLoader workers

**Limits**:
- Time limit: 6 hours (21,600s)
- Soft time limit: 5.5 hours (19,800s)
- Concurrency: 1 task at a time per worker

## Monitoring

### Health Checks

**Container**:
```bash
docker exec heimdall-training curl -f http://localhost:8002/health
```

**GPU Utilization**:
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

**CUDA Availability** (inside container):
```bash
docker exec heimdall-training python -c "import torch; print(torch.cuda.is_available())"
```

### Metrics to Monitor

1. **GPU Utilization**: Should be >80% during training
2. **VRAM Usage**: Should not exceed 90% (OOM risk)
3. **Task Success Rate**: >95% expected
4. **Training Throughput**: Samples/second
5. **Model Export**: ONNX conversion success

## Upgrade Procedures

### Quarterly Review (Every 3 Months)

1. Check NVIDIA deprecation notices
2. Review new PyTorch releases
3. Test new CUDA versions in dev
4. Update documentation

See [DOCKER_IMAGE_POLICY.md](./DOCKER_IMAGE_POLICY.md) for full procedure.

### Emergency Migration

If deprecation notice appears:

1. **Week 1**: Research alternatives, test in dev
2. **Week 2**: Update documentation, prepare migration
3. **Week 3**: Deploy to staging, validate
4. **Week 4**: Deploy to production, monitor

## Troubleshooting

### No GPU Detected

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Check**:
1. NVIDIA driver installed on host: `nvidia-smi`
2. Container toolkit installed: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
3. Correct deployment config in docker-compose.yml

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in training config
2. Enable gradient checkpointing
3. Use mixed precision training (FP16)
4. Reduce model size
5. Clear cache: `torch.cuda.empty_cache()`

### Version Mismatch

**Symptoms**: `version 'CUDA_X.Y' not found`

**Solution**: Rebuild container with matching CUDA version:
```bash
docker-compose build training --no-cache
docker-compose up -d training
```

### Slow Training

**Check**:
1. GPU utilization: Should be >80%
2. CPU bottleneck: Monitor with `htop`
3. I/O bottleneck: Check disk speed
4. Batch size: Try increasing if VRAM available
5. DataLoader workers: Adjust `num_workers`

## Performance Tuning

### Optimal Batch Sizes (by VRAM)

| VRAM | Batch Size | Notes |
|------|-----------|-------|
| 4GB | 8-16 | Minimal, may struggle |
| 8GB | 16-32 | Good for development |
| 12GB | 32-64 | Recommended |
| 24GB | 64-128 | Excellent, max throughput |

### DataLoader Optimization

```python
# In training config
num_workers = min(4, os.cpu_count())  # 4 workers typical
pin_memory = True  # Faster GPU transfer
persistent_workers = True  # Reuse workers
```

### Mixed Precision Training

Automatically enabled in PyTorch Lightning:
```python
trainer = Trainer(precision="16-mixed")  # FP16 training
```

Benefits:
- 2x faster training
- 50% less VRAM usage
- Minimal accuracy loss

## Future Roadmap

### Q1 2025
- [ ] Evaluate CUDA 12.8 when PyTorch 2.7+ available
- [ ] Test NGC containers in production
- [ ] Add automated GPU health monitoring

### Q2 2025
- [ ] Consider multi-GPU training support
- [ ] Evaluate ARM64/Jetson support
- [ ] Benchmark CUDA 13.0 with PyTorch 2.9+

### Q3 2025
- [ ] Implement automatic model quantization
- [ ] Add distributed training support
- [ ] GPU fleet management for multiple nodes

## References

- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---

**Questions?** See [FAQ](./FAQ.md) or contact DevOps team.
