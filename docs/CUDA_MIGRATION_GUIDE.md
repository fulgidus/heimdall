# CUDA 12.6 Migration Guide

**Date**: November 4, 2024  
**Affected Component**: Training Service  
**Migration Type**: Infrastructure Update (Non-Breaking)

## Overview

This guide helps you migrate your Heimdall training service from the deprecated CUDA 12.1.0 base image to the actively maintained CUDA 12.6.0 image.

## Why This Migration?

NVIDIA announced deprecation of the `nvidia/cuda:12.1.0-runtime-ubuntu22.04` Docker image with scheduled deletion. To prevent service disruption, we've migrated to CUDA 12.6.0, which provides:

- ✅ Long-term support through 2026+
- ✅ Latest security updates and bug fixes
- ✅ Better performance with PyTorch 2.5+
- ✅ Compatibility with newer NVIDIA drivers

## What's Changed

| Component | Old Version | New Version |
|-----------|------------|-------------|
| CUDA Base Image | 12.1.0 | 12.6.0 |
| PyTorch | 2.2.0 (cu121) | 2.5.0 (cu126) |
| TorchVision | 0.17.0 | 0.20.0 |
| CuPy | 12.x | 13.0.0+ |

## Prerequisites

- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- NVIDIA Driver 535.86.10 or newer
- Docker with NVIDIA Container Toolkit
- Sufficient disk space for new Docker image (~4GB)

## Migration Steps

### For Development Environments

1. **Pull latest code**
   ```bash
   git pull origin main
   ```

2. **Stop running containers**
   ```bash
   docker-compose down
   ```

3. **Remove old training image** (optional, saves disk space)
   ```bash
   docker rmi heimdall-training:latest
   ```

4. **Rebuild training service**
   ```bash
   docker-compose build training
   ```

5. **Start services**
   ```bash
   docker-compose up -d
   ```

6. **Verify health**
   ```bash
   curl http://localhost:8002/health
   # Expected: {"status":"healthy"}
   ```

7. **Check GPU detection**
   ```bash
   docker logs heimdall-training 2>&1 | grep -i cuda
   # Should show CUDA 12.6 initialization
   ```

### For Production Environments

1. **Schedule maintenance window** (recommended: 30 minutes)

2. **Backup current configuration**
   ```bash
   docker-compose config > docker-compose.backup.yml
   docker image save heimdall-training:latest -o training-image-backup.tar
   ```

3. **Pull and build new image** (before downtime)
   ```bash
   git fetch origin main
   git checkout main
   docker-compose build training --no-cache
   ```

4. **During maintenance window**:
   ```bash
   # Stop training service
   docker-compose stop training
   
   # Backup any in-progress training data from MinIO if needed
   
   # Start new training service
   docker-compose up -d training
   
   # Monitor logs
   docker-compose logs -f training
   ```

5. **Verify functionality**:
   ```bash
   # Health check
   curl http://localhost:8002/health
   
   # Submit test training job
   curl -X POST http://localhost:8002/training/jobs \
     -H "Content-Type: application/json" \
     -d @test_training_config.json
   ```

6. **Monitor for 24 hours**:
   - Check training job success rate
   - Monitor GPU utilization
   - Watch for CUDA out-of-memory errors
   - Verify model export (ONNX) still works

### Rollback Procedure

If issues occur:

1. **Immediate rollback**:
   ```bash
   docker-compose down
   git checkout <previous-commit>
   docker load -i training-image-backup.tar  # If saved
   # OR
   docker pull heimdall-training:<previous-tag>
   docker-compose up -d training
   ```

2. **Report issue**: Open GitHub issue with:
   - Error logs from `docker logs heimdall-training`
   - GPU model and driver version
   - Steps to reproduce

## Verification Tests

### 1. Basic Health Check
```bash
curl http://localhost:8002/health
```
Expected response: `{"status":"healthy"}`

### 2. GPU Detection
```bash
docker exec heimdall-training python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
```
Expected output:
```
CUDA available: True
CUDA version: 12.6
Device count: 1 (or more)
```

### 3. CuPy Verification
```bash
docker exec heimdall-training python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); arr = cp.array([1,2,3]); print(f'CuPy array: {arr}')"
```
Expected: No errors, array printed successfully

### 4. Training Task
Submit a minimal training job via API or CLI:
```bash
# Example using the training API
curl -X POST http://localhost:8002/api/training/generate-synthetic \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 100, "frequency": 145000000}'
```

Check task completion:
```bash
curl http://localhost:8002/api/training/tasks/<task_id>
```

## Common Issues

### Issue: "Could not load dynamic library 'libcudart.so.12.1'"

**Cause**: Application still looking for CUDA 12.1 libraries  
**Solution**: 
```bash
docker-compose down
docker-compose build training --no-cache
docker-compose up -d
```

### Issue: "CUDA out of memory" errors

**Cause**: PyTorch 2.5 may have different memory management  
**Solution**: Reduce batch size in training configuration:
```python
# In training config
batch_size = 16  # Reduce from 32
```

### Issue: SSL certificate errors during build

**Cause**: Proxy or corporate firewall  
**Solution**: Already handled in Dockerfile with `--trusted-host` flags. If still occurring:
```bash
export DOCKER_BUILDKIT=0
docker-compose build training
```

### Issue: Model export (ONNX) fails

**Cause**: ONNX version mismatch  
**Solution**: 
```bash
# Verify ONNX version
docker exec heimdall-training pip list | grep onnx
# Should show onnx>=1.14.0 and onnxruntime>=1.16.3
```

## Performance Comparison

Expected performance improvements with CUDA 12.6 + PyTorch 2.5:

| Metric | CUDA 12.1 | CUDA 12.6 | Change |
|--------|-----------|-----------|--------|
| Training throughput | Baseline | +5-15% | ⬆️ Better |
| Inference latency | Baseline | Similar | ➡️ Same |
| Memory efficiency | Baseline | +10% | ⬆️ Better |
| Build time | ~8 min | ~8 min | ➡️ Same |

*Performance varies by GPU model and workload*

## FAQ

**Q: Do I need to retrain existing models?**  
A: No. Models trained with CUDA 12.1/PyTorch 2.2 are compatible with CUDA 12.6/PyTorch 2.5.

**Q: Will my inference service be affected?**  
A: No. The inference service uses ONNX Runtime which is CUDA-version agnostic.

**Q: Can I still use the old CUDA 12.1 image?**  
A: Yes, but not recommended. It's deprecated and may be deleted without notice.

**Q: What if I'm using a different CUDA version on my host?**  
A: The container includes its own CUDA runtime. Your host just needs NVIDIA drivers 535+.

**Q: Will this work on Jetson/ARM platforms?**  
A: Not yet. Current migration is x86_64 only. ARM support planned for future release.

**Q: Can I use CUDA 12.4 instead of 12.6?**  
A: Yes, but 12.6 is recommended for longer support. See `Dockerfile.cuda126` for customization.

## Alternative: NGC PyTorch Container

For production environments requiring maximum stability, consider migrating to NVIDIA NGC PyTorch containers:

**Benefits**:
- Official NVIDIA support
- Pre-optimized for performance
- Regular security updates
- Production hardening available

**How to use**:
```bash
# Use the NGC-based Dockerfile
docker-compose build training -f services/training/Dockerfile.ngc
```

See `docs/DOCKER_IMAGE_POLICY.md` for details.

## Support

- **Documentation**: `docs/DOCKER_IMAGE_POLICY.md`
- **Issues**: https://github.com/fulgidus/heimdall/issues
- **Discussions**: https://github.com/fulgidus/heimdall/discussions

## References

- [NVIDIA CUDA Container Images](https://hub.docker.com/r/nvidia/cuda)
- [PyTorch 2.5 Release Notes](https://pytorch.org/blog/pytorch2-5/)
- [NVIDIA Driver Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [Docker Image Policy](./DOCKER_IMAGE_POLICY.md)

---

**Last Updated**: 2024-11-04  
**Status**: Active Migration  
**Next Review**: 2025-02-04
