# MLflowLogger Import Fix - PyTorch Lightning 2.0+ Migration

## Issue Description

**Error**: `ImportError: cannot import name 'MLflowLogger' from 'pytorch_lightning.loggers'`

**Root Cause**: PyTorch Lightning 2.0+ restructured the package architecture:
- The `pytorch-lightning` package became a meta-package
- Actual functionality moved to the `lightning` umbrella package
- Module path changed: `pytorch_lightning` → `lightning.pytorch`
- Class name changed: `MLflowLogger` → `MLFlowLogger` (capital F, lowercase l)

## Solution Applied

### 1. Dependency Update

**File**: `requirements.txt`

```diff
- pytorch-lightning>=2.0.0
+ lightning>=2.0.0
```

The `lightning` package includes all PyTorch Lightning functionality under `lightning.pytorch`.

### 2. Import Updates

All Python files using PyTorch Lightning were updated:

#### Before (Old Import Pattern)
```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLflowLogger
```

#### After (New Import Pattern)
```python
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
```

### 3. Files Modified

1. **train.py**
   - Updated all `pytorch_lightning` imports to `lightning.pytorch`
   - Changed `MLflowLogger` to `MLFlowLogger`
   - Removed try-except import fallback (no longer needed)

2. **src/models/lightning_module.py**
   - Updated import: `import lightning.pytorch as pl`
   - Updated logger import: `from lightning.pytorch.loggers import MLFlowLogger`

3. **src/train.py**
   - Updated import: `import lightning.pytorch as pl`
   - Updated callbacks import: `from lightning.pytorch.callbacks import ...`

## Verification

### Import Tests Passed ✅
- `import lightning.pytorch as pl` - Works
- `from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor` - Works
- `from lightning.pytorch.loggers import MLFlowLogger` - Works
- All modified Python files compile successfully

### Expected CI Behavior
When CI runs:
1. `pip install lightning>=2.0.0` will install the correct package
2. All import statements will resolve successfully
3. Test collection will complete without ImportError
4. Tests can proceed normally

## Important Notes

### Class Name Change
⚠️ **Note the capitalization change**:
- Old: `MLflowLogger` (lowercase 'f' and 'l')
- New: `MLFlowLogger` (capital 'F', lowercase 'l')

This is a deliberate API change in Lightning 2.0+.

### Package Compatibility
- `lightning>=2.0.0` is compatible with PyTorch 2.0+
- The `lightning` package includes `pytorch`, `fabric`, and other Lightning components
- Using `lightning` instead of `pytorch-lightning` is the recommended approach for new projects

### Backward Compatibility
For projects that need to support both old and new versions:
```python
try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import MLFlowLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import MLflowLogger as MLFlowLogger
```

However, since we're targeting `>=2.0.0`, we use the new import pattern exclusively.

## References

- [PyTorch Lightning 2.0 Release Notes](https://github.com/Lightning-AI/lightning/releases/tag/2.0.0)
- [Lightning Package Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Migration Guide](https://lightning.ai/docs/pytorch/stable/upgrade/migration_guide.html)

## Testing Commands

To verify the fix locally:

```bash
# Install the lightning package
pip install lightning>=2.0.0

# Test imports
python -c "import lightning.pytorch as pl; from lightning.pytorch.loggers import MLFlowLogger; print('Success!')"

# Run the training service tests
cd services/training
pytest tests/ -v
```

## Date Fixed
2025-10-22

## Related Files
- `services/training/requirements.txt`
- `services/training/train.py`
- `services/training/src/models/lightning_module.py`
- `services/training/src/train.py`
