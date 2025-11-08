# Prompt 04: Phase 5 - .heimdall Export Format Implementation

## Context
GPU configured (Prompt 01 ✓), pipeline tested (Prompt 02 ✓), API built (Prompt 03 ✓). Now implement portable export format for trained models.

## Current State
- ONNX export works (`services/training/src/onnx_export.py`)
- Models stored in MinIO + MLflow registry
- No portable "bundle" format for model distribution
- Frontend needs downloadable format with model + metadata

## Architectural Decisions

### Export Format: .heimdall Bundle
**Decision**: Create JSON-based bundle format with base64-encoded ONNX

**Rationale**:
- Single file distribution (easy sharing between Heimdall instances)
- Human-readable metadata (JSON)
- No external dependencies (base64 ONNX embedded)
- Versioned format for backward compatibility
- Selectable components (user chooses what to include)

### Bundle Structure
```json
{
  "format_version": "1.0.0",
  "bundle_metadata": {
    "bundle_id": "uuid-v4",
    "created_at": "ISO-8601 timestamp",
    "created_by": "user_id or 'system'",
    "description": "Optional bundle description",
    "heimdall_version": "project version"
  },
  "model": {
    "model_id": "uuid from database",
    "model_name": "localization-net-v1",
    "version": "1.0.0",
    "architecture": "convnext_large",
    "framework": "pytorch",
    "onnx_opset": 17,
    "onnx_model_base64": "base64-encoded ONNX binary",
    "input_shape": [1, 3, 128, 32],
    "output_shape": [[1, 2], [1, 2]],
    "parameters_count": 200000000
  },
  "training_config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function": "gaussian_nll",
    "validation_split": 0.2,
    "dataset_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "dataset_names": ["Synthetic Dataset 10k", "Real Data 5k", "Mixed Dataset 3k"]
  },
  "performance_metrics": {
    "final_train_loss": 0.0123,
    "final_val_loss": 0.0145,
    "final_train_accuracy": 0.95,
    "final_val_accuracy": 0.93,
    "best_epoch": 42,
    "training_duration_seconds": 3600,
    "inference_latency_ms": 45.3,
    "onnx_speedup_factor": 2.1
  },
  "normalization_stats": {
    "feature_means": [0.485, 0.456, 0.406],
    "feature_stds": [0.229, 0.224, 0.225]
  },
  "sample_predictions": [
    {
      "sample_id": "uuid",
      "input_metadata": {
        "frequency_mhz": 145.0,
        "tx_power_dbm": 37.0,
        "snr_db": 15.3
      },
      "ground_truth": {
        "latitude": 41.9028,
        "longitude": 12.4964
      },
      "prediction": {
        "latitude": 41.9031,
        "longitude": 12.4960,
        "uncertainty_x": 25.3,
        "uncertainty_y": 28.7
      },
      "error_meters": 42.1
    }
  ]
}
```

### Component Selection
**Decision**: Users can select which components to include in export

**Options**:
- `model` (required): ONNX model binary
- `training_config`: Training hyperparameters
- `metrics`: Performance metrics
- `normalization_stats`: Feature normalization parameters
- `sample_predictions`: Example predictions (1-10 samples)

**Why**: Flexibility for different use cases (full bundle vs. model-only)

## Tasks

### Task 1: Create Export Module
**File**: `services/training/src/export/heimdall_format.py` (NEW)

**Classes**:
```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64

class BundleMetadata(BaseModel):
    bundle_id: str
    created_at: str
    created_by: str
    description: Optional[str] = None
    heimdall_version: str

class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    version: str
    architecture: str
    framework: str = "pytorch"
    onnx_opset: int
    onnx_model_base64: str
    input_shape: List[int]
    output_shape: List[List[int]]
    parameters_count: int

class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    loss_function: str
    validation_split: float
    dataset_id: str
    dataset_name: str

class PerformanceMetrics(BaseModel):
    final_train_loss: float
    final_val_loss: float
    final_train_accuracy: float
    final_val_accuracy: float
    best_epoch: int
    training_duration_seconds: int
    inference_latency_ms: float
    onnx_speedup_factor: float

class NormalizationStats(BaseModel):
    feature_means: List[float]
    feature_stds: List[float]

class SamplePrediction(BaseModel):
    sample_id: str
    input_metadata: Dict[str, Any]
    ground_truth: Dict[str, float]
    prediction: Dict[str, float]
    error_meters: float

class HeimdallBundle(BaseModel):
    format_version: str = "1.0.0"
    bundle_metadata: BundleMetadata
    model: ModelInfo
    training_config: Optional[TrainingConfig] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    normalization_stats: Optional[NormalizationStats] = None
    sample_predictions: Optional[List[SamplePrediction]] = None


class HeimdallExporter:
    """Export trained models in .heimdall format."""
    
    def __init__(self, db_session, minio_client):
        self.db = db_session
        self.minio = minio_client
    
    def export_model(
        self,
        model_id: str,
        include_config: bool = True,
        include_metrics: bool = True,
        include_normalization: bool = True,
        include_samples: bool = True,
        num_samples: int = 5,
        description: Optional[str] = None
    ) -> HeimdallBundle:
        """
        Export model as .heimdall bundle.
        
        Args:
            model_id: UUID of trained model
            include_config: Include training config
            include_metrics: Include performance metrics
            include_normalization: Include normalization stats
            include_samples: Include sample predictions
            num_samples: Number of samples to include (1-10)
            description: Optional bundle description
        
        Returns:
            HeimdallBundle ready for JSON serialization
        """
        # Load model from database
        model = self._load_model_from_db(model_id)
        
        # Download ONNX from MinIO
        onnx_bytes = self._download_onnx(model['onnx_path'])
        onnx_base64 = base64.b64encode(onnx_bytes).decode('utf-8')
        
        # Build bundle
        bundle = HeimdallBundle(
            bundle_metadata=self._create_metadata(description),
            model=self._create_model_info(model, onnx_base64),
            training_config=self._load_training_config(model_id) if include_config else None,
            performance_metrics=self._load_metrics(model_id) if include_metrics else None,
            normalization_stats=self._load_normalization_stats() if include_normalization else None,
            sample_predictions=self._load_sample_predictions(model_id, num_samples) if include_samples else None
        )
        
        return bundle
    
    def save_bundle(self, bundle: HeimdallBundle, output_path: str):
        """Save bundle to .heimdall file (JSON)."""
        with open(output_path, 'w') as f:
            f.write(bundle.model_dump_json(indent=2))
    
    def load_bundle(self, bundle_path: str) -> HeimdallBundle:
        """Load bundle from .heimdall file."""
        with open(bundle_path, 'r') as f:
            return HeimdallBundle.model_validate_json(f.read())
    
    def extract_onnx(self, bundle: HeimdallBundle, output_path: str):
        """Extract ONNX model from bundle to file."""
        onnx_bytes = base64.b64decode(bundle.model.onnx_model_base64)
        with open(output_path, 'wb') as f:
            f.write(onnx_bytes)
```

### Task 2: Add Export API Endpoint
**File**: `services/backend/src/routers/training.py`

**Add endpoint**:
```python
@router.get("/models/{model_id}/export", response_class=Response)
async def export_model_heimdall(
    model_id: str,
    include_config: bool = True,
    include_metrics: bool = True,
    include_normalization: bool = True,
    include_samples: bool = True,
    num_samples: int = 5,
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Export trained model in .heimdall format.
    
    Returns downloadable JSON file with model + metadata.
    """
    from ..export.heimdall_format import HeimdallExporter
    
    exporter = HeimdallExporter(db, minio_client)
    bundle = exporter.export_model(
        model_id=model_id,
        include_config=include_config,
        include_metrics=include_metrics,
        include_normalization=include_normalization,
        include_samples=include_samples,
        num_samples=num_samples,
        description=description
    )
    
    # Return as downloadable file
    json_content = bundle.model_dump_json(indent=2)
    filename = f"{bundle.model.model_name}-{bundle.model.version}.heimdall"
    
    return Response(
        content=json_content,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
```

### Task 3: Add Import API Endpoint
**File**: `services/backend/src/routers/training.py`

**Add endpoint**:
```python
@router.post("/models/import")
async def import_model_heimdall(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Import trained model from .heimdall file.
    
    Extracts ONNX, stores in MinIO, registers in MLflow.
    """
    from ..export.heimdall_format import HeimdallExporter
    
    # Read uploaded file
    content = await file.read()
    
    # Parse bundle
    exporter = HeimdallExporter(db, minio_client)
    bundle = HeimdallBundle.model_validate_json(content.decode('utf-8'))
    
    # Extract and store ONNX
    onnx_bytes = base64.b64decode(bundle.model.onnx_model_base64)
    onnx_path = f"imported/{bundle.model.model_name}-{bundle.model.version}.onnx"
    minio_client.put_object(
        bucket_name="heimdall-models",
        object_name=onnx_path,
        data=io.BytesIO(onnx_bytes),
        length=len(onnx_bytes)
    )
    
    # Register in database
    model_id = str(uuid.uuid4())
    db.execute("""
        INSERT INTO heimdall.models (id, name, version, onnx_path, architecture, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
    """, (model_id, bundle.model.model_name, bundle.model.version, onnx_path, bundle.model.architecture))
    db.commit()
    
    return {
        "model_id": model_id,
        "message": "Model imported successfully",
        "onnx_path": onnx_path
    }
```

### Task 4: Add CLI Export Tool
**File**: `services/training/src/export_cli.py` (NEW)

**Script**:
```python
#!/usr/bin/env python3
"""CLI tool for exporting models in .heimdall format."""

import argparse
from export.heimdall_format import HeimdallExporter
from db.connection import get_db_session
from storage.minio_client import get_minio_client

def main():
    parser = argparse.ArgumentParser(description="Export Heimdall model")
    parser.add_argument("model_id", help="Model UUID to export")
    parser.add_argument("-o", "--output", required=True, help="Output .heimdall file path")
    parser.add_argument("--no-config", action="store_true", help="Exclude training config")
    parser.add_argument("--no-metrics", action="store_true", help="Exclude metrics")
    parser.add_argument("--no-samples", action="store_true", help="Exclude sample predictions")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples (1-10)")
    parser.add_argument("--description", help="Bundle description")
    
    args = parser.parse_args()
    
    db = get_db_session()
    minio = get_minio_client()
    exporter = HeimdallExporter(db, minio)
    
    bundle = exporter.export_model(
        model_id=args.model_id,
        include_config=not args.no_config,
        include_metrics=not args.no_metrics,
        include_samples=not args.no_samples,
        num_samples=args.samples,
        description=args.description
    )
    
    exporter.save_bundle(bundle, args.output)
    print(f"✓ Exported model to {args.output}")
    print(f"  Bundle size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Export full bundle
python3 services/training/src/export_cli.py <model_id> -o model.heimdall

# Export model only (minimal)
python3 services/training/src/export_cli.py <model_id> -o model.heimdall \
  --no-config --no-metrics --no-samples

# Export with custom description
python3 services/training/src/export_cli.py <model_id> -o model.heimdall \
  --description "Production model v1.0 - trained on 50k samples"
```

### Task 5: Test Export/Import Flow
**Test sequence**:

1. **Export via API**:
```bash
MODEL_ID="<uuid from database>"
curl -X GET "http://localhost:8001/api/v1/training/models/${MODEL_ID}/export?num_samples=3" \
  -o model-test.heimdall
```

2. **Inspect bundle**:
```bash
# Check format
jq '.format_version' model-test.heimdall

# Check model info
jq '.model | {name, version, architecture, parameters_count}' model-test.heimdall

# Check metrics
jq '.performance_metrics' model-test.heimdall
```

3. **Extract ONNX**:
```python
from export.heimdall_format import HeimdallExporter

exporter = HeimdallExporter(None, None)
bundle = exporter.load_bundle("model-test.heimdall")
exporter.extract_onnx(bundle, "extracted-model.onnx")
```

4. **Import via API**:
```bash
curl -X POST http://localhost:8001/api/v1/training/models/import \
  -F "file=@model-test.heimdall"
```

5. **Verify imported model**:
```bash
# Check MinIO
mc ls minio/heimdall-models/imported/

# Check database
psql -h localhost -U heimdall -d heimdall -c \
  "SELECT id, name, version, onnx_path FROM heimdall.models ORDER BY created_at DESC LIMIT 5;"
```

## Validation Criteria

### Must Pass
- [ ] Export API returns valid JSON file
- [ ] Bundle passes JSON schema validation
- [ ] ONNX model decodes successfully from base64
- [ ] Extracted ONNX model runs inference correctly
- [ ] Import API accepts .heimdall file
- [ ] Imported model appears in database + MinIO
- [ ] CLI tool exports successfully
- [ ] Bundle size reasonable (<50MB for typical model)
- [ ] Component selection works (include/exclude options)

### Format Validation
- [ ] `format_version` is "1.0.0"
- [ ] All required fields present
- [ ] Base64 encoding valid (no corruption)
- [ ] Sample predictions match ground truth format
- [ ] Normalization stats valid (means/stds arrays)

## Non-Breaking Requirement
- Export/import is additive functionality
- Existing ONNX export still works
- MLflow registry unaffected
- No changes to training pipeline

## Success Criteria
When all validation passes, .heimdall export format is complete. Proceed to **Prompt 05: Frontend Training UI**.

## Deliverables
Document:
- Sample .heimdall file (with 1-2 sample predictions)
- Export API test results
- Import API test results
- Bundle file sizes (full vs. minimal)
- ONNX extraction verification
- Any format design decisions
