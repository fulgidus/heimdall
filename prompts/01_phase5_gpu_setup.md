# Prompt 01: Phase 5 - GPU Configuration for Training Service

## Context
Heimdall Phase 5 Training Pipeline is 80% complete (backend implementation done). Before testing training or building API/frontend, GPU support must be enabled in Docker for the training service.

## Current State
- **Training service**: Running but unhealthy, no GPU access
- **Host GPU**: NVIDIA drivers loaded, but `nvidia-smi` shows driver/library version mismatch (NVML 570.195)
- **PyTorch in container**: v2.9.0+cu128, CUDA available: False
- **Docker compose**: No GPU runtime configured for training service

## Architectural Decisions

### GPU Configuration Strategy
1. **GPU Runtime**: Use Docker Compose `deploy.resources.reservations.devices` (Compose v2 format) instead of deprecated `runtime: nvidia`
2. **Capabilities**: Request `gpu` capability for NVIDIA GPU access
3. **Device Access**: Request all GPUs or specific device IDs
4. **Fallback**: Training can run on CPU but will be very slow (ConvNeXt-Large = 200M params)

### Why GPU Is Critical
- **Model**: ConvNeXt-Large (200M parameters, 88.6% ImageNet accuracy)
- **Training time**: GPU required for reasonable training speed
- **Memory**: RTX 3090 24GB (from architecture notes) - adequate for batch_size=32
- **Inference**: ONNX export targets CPU inference (<500ms requirement already met)

## Tasks

### 1. Fix Host GPU Driver Mismatch
**Problem**: NVML library version mismatch prevents `nvidia-smi` from working
**Solution**: 
- Reboot host to reload kernel modules with correct driver
- OR: Reinstall NVIDIA drivers to match NVML library version 570.195
- Verify: `nvidia-smi` should show GPU info without errors

### 2. Add GPU Support to docker-compose.yml
**File**: `docker-compose.yml`
**Service**: `training`
**Changes**:
```yaml
training:
  # ... existing config ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

**Alternative** (specific GPU):
```yaml
devices:
  - driver: nvidia
    device_ids: ['0']  # First GPU only
    capabilities: [gpu]
```

### 3. Verify GPU Access in Container
**Test commands**:
```bash
# Rebuild and restart training service
docker compose up -d --build training

# Verify CUDA access
docker exec heimdall-training python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output**:
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

### 4. Update Training Service Health Check (Optional)
**Current**: Celery ping check
**Enhancement**: Add GPU availability check to readiness probe
**File**: `services/training/src/main.py`

Add to `/ready` endpoint:
```python
# Check GPU availability if configured
if settings.accelerator == "gpu":
    import torch
    if not torch.cuda.is_available():
        result.add_dependency("gpu", False, "CUDA not available")
```

## Validation Criteria
- [ ] `nvidia-smi` runs without errors on host
- [ ] `docker exec heimdall-training nvidia-smi` shows GPU info
- [ ] PyTorch reports `torch.cuda.is_available() == True` in container
- [ ] Training service health check passes
- [ ] System remains fully operational (no breaking changes)

## Non-Breaking Requirement
All existing services must remain functional. GPU configuration is additive only - if GPU unavailable, training falls back to CPU (with warning).

## Next Steps
After GPU setup complete → Proceed to **Prompt 02: Test Existing Training Pipeline**
