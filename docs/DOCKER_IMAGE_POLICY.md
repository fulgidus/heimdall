# Docker Base Image Policy & Maintenance

**Last Updated**: 2024-11-04  
**Status**: Active

## Overview

This document outlines Heimdall's policy for managing Docker base images, particularly NVIDIA CUDA containers used by the training service. It ensures long-term maintainability and helps prevent disruptions from deprecated images.

## Background

In November 2024, we discovered the training service was using `nvidia/cuda:12.1.0-runtime-ubuntu22.04`, which displayed NVIDIA's deprecation notice indicating scheduled deletion. This prompted a migration strategy to prevent future service disruptions.

## Current Configuration

### Training Service

**Base Image**: `nvidia/cuda:12.6.0-runtime-ubuntu22.04`

- **CUDA Version**: 12.6.0
- **Ubuntu Version**: 22.04 LTS
- **Python Version**: 3.11
- **PyTorch Version**: 2.5.0+ (cu126)
- **Expected Support**: Through 2026+

**Rationale**: CUDA 12.6 is actively maintained by NVIDIA and provides:
- Long-term support commitment
- Compatibility with PyTorch 2.5+ ecosystem
- Security updates and bug fixes
- Broad GPU driver compatibility (535+)

### Alternative Options

We maintain two additional Dockerfile variants for flexibility:

1. **Dockerfile.ngc** - NVIDIA NGC PyTorch Container
   - Base: `nvcr.io/nvidia/pytorch:25.04-py3`
   - Includes: PyTorch 2.7+, CUDA 12.6, optimized ML libraries
   - Pros: Production-ready, pre-optimized, enterprise support
   - Cons: Larger image size, less control over versions
   - Use case: Production deployments requiring stability

2. **Dockerfile.cuda126** - CUDA 12.6 with manual PyTorch
   - Base: `nvidia/cuda:12.6.0-runtime-ubuntu22.04`
   - Identical to main Dockerfile, kept for reference
   - Use case: Fallback if main approach has issues

## Version Compatibility Matrix

| Component | Version | CUDA Support | Notes |
|-----------|---------|--------------|-------|
| CUDA Base Image | 12.6.0 | - | Current |
| PyTorch | 2.5.0+ | 12.6 | Minimum version |
| TorchVision | 0.20.0+ | 12.6 | Matches PyTorch |
| CuPy | 13.0.0+ | 12.x | GPU-accelerated NumPy |
| NVIDIA Driver | 535+ | - | Minimum host requirement |

## Maintenance Schedule

### Quarterly Review (Every 3 Months)

**Check**:
1. NVIDIA deprecation notices on used images
2. New CUDA releases and their support lifecycle
3. PyTorch compatibility with newer CUDA versions
4. Security advisories for current images

**Actions**:
- Update this document with findings
- Plan migration if deprecation notice appears
- Test new versions in development environment

### Annual Update (Every 12 Months)

**Evaluate**:
1. Upgrade to next LTS Ubuntu version (if available)
2. Upgrade to newer CUDA major/minor version
3. Consider NGC containers for production stability
4. Update Python version to latest stable

## Migration Procedure

When a base image needs updating:

### 1. Research Phase
- [ ] Identify target CUDA version
- [ ] Verify PyTorch compatibility
- [ ] Check CuPy and other GPU library support
- [ ] Review NVIDIA's support timeline
- [ ] Document compatibility matrix

### 2. Update Phase
- [ ] Update Dockerfile base image tags
- [ ] Update PyTorch index URL in requirements
- [ ] Update ml.txt with new PyTorch version
- [ ] Update CuPy version if needed
- [ ] Update this documentation

### 3. Testing Phase
- [ ] Build Docker image locally
- [ ] Verify GPU detection: `docker run --gpus all <image> nvidia-smi`
- [ ] Start training service and check health endpoint
- [ ] Run sample training task end-to-end
- [ ] Verify model export (ONNX) works correctly
- [ ] Check Celery worker GPU utilization

### 4. Deployment Phase
- [ ] Merge changes to main branch
- [ ] Deploy to staging environment
- [ ] Monitor for 24-48 hours
- [ ] Deploy to production
- [ ] Update CHANGELOG.md

## Rollback Procedure

If issues occur after migration:

1. **Immediate**: Revert to previous Dockerfile
   ```bash
   cp services/training/Dockerfile.cuda121.bak services/training/Dockerfile
   docker-compose build training
   docker-compose up -d training
   ```

2. **Investigate**: Check logs for compatibility issues
   ```bash
   docker logs heimdall-training
   ```

3. **Document**: Record issue in GitHub issue tracker

4. **Fix Forward**: Address root cause and retry migration

## Monitoring & Alerts

### Deprecation Detection

**Manual**: Check deprecation notices quarterly
- Review `docker logs heimdall-training` startup messages
- Check NVIDIA's deprecation policy: https://docs.nvidia.com/ngc/ngc-deprecated-image-support-policy/

**Automated** (Future Enhancement):
- Parse container startup logs for "DEPRECATION" warnings
- Alert if deprecation notice detected
- Integrate with monitoring stack (Prometheus/Grafana)

### Health Checks

Monitor these metrics continuously:
- Training service health endpoint: `/health`
- GPU utilization during training tasks
- CUDA out-of-memory errors
- Training task success rate

## Dependencies

Services dependent on GPU/CUDA:

1. **Training Service** (Primary)
   - Neural network training
   - Model evaluation
   - Synthetic data generation
   - ONNX export

2. **Inference Service** (Secondary)
   - Uses ONNX Runtime (CPU/GPU agnostic)
   - Loads models exported by training service
   - Not directly affected by CUDA changes

## Future Considerations

### NGC Container Migration

For production stability, consider migrating to NGC containers:

**Pros**:
- Official NVIDIA support
- Pre-optimized for performance
- Regular updates with security patches
- Production hardening available

**Cons**:
- Larger image size (~8GB vs ~4GB)
- Less granular version control
- Requires NGC account for some features

**Timeline**: Evaluate in Q2 2025 for production deployment

### Multi-Architecture Support

Currently x86_64 only. Consider adding ARM64 support if needed:
- NVIDIA Jetson/Orin platforms
- AWS Graviton instances with GPU
- Apple Silicon (via Metal backend, no CUDA)

## References

- [NVIDIA CUDA Container Images](https://hub.docker.com/r/nvidia/cuda)
- [NVIDIA NGC PyTorch Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [PyTorch-CUDA Compatibility Matrix](https://github.com/eminsafa/pytorch-cuda-compatibility)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [NVIDIA Deprecation Policy](https://docs.nvidia.com/ngc/ngc-deprecated-image-support-policy/)

## Changelog

### 2024-11-04 - Initial Migration
- **From**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (deprecated)
- **To**: `nvidia/cuda:12.6.0-runtime-ubuntu22.04`
- **PyTorch**: 2.2.0 (cu121) → 2.5.0+ (cu126)
- **Reason**: Avoid service disruption from image deletion
- **Status**: ✅ Completed

---

**Maintained by**: DevOps Team  
**Review Frequency**: Quarterly  
**Next Review**: 2025-02-04
