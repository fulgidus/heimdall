# 🧠 PHASE 5: Training Pipeline - Quick Start Guide

**Phase Status**: 🟢 READY TO START  
**Duration**: 3 giorni  
**Assignee**: Agent-ML (fulgidus)  
**Depends On**: Phase 1 ✅, Phase 3 ✅, Phase 4 ✅  
**Critical Path**: YES (blocca Phase 6)  
**Generated**: 2025-10-22  

---

## 📋 Quick Reference

| Aspetto                  | Dettagli                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------- |
| **Obiettivo**            | Implementare pipeline PyTorch Lightning per localizzazione radio con stima incertezza |
| **Output**               | Modello ONNX + MLflow registry + documentazione                                       |
| **Repository**           | `services/training/`                                                                  |
| **Infrastructure**       | PostgreSQL, MinIO, MLflow, Redis                                                      |
| **Linguaggio**           | Python 3.11                                                                           |
| **Framework**            | PyTorch Lightning                                                                     |
| **Test Coverage Target** | >85%                                                                                  |

---

## 🎯 Checkpoint System (Phase 5)

```
CP5.1: Model forward pass ✓
        ↓
CP5.2: Dataset loader ✓
        ↓
CP5.3: Training loop ✓
        ↓
CP5.4: ONNX export ✓
        ↓
CP5.5: MLflow registration ✓
        ↓
PHASE 5 COMPLETE → PHASE 6 START
```

---

## 📂 Project Structure

```
services/training/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point (FastAPI + Celery)
│   ├── config.py                  # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── localization_net.py    # Neural network architecture (T5.1)
│   │   └── lightning_module.py    # PyTorch Lightning wrapper (T5.5)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # HeimdallDataset (T5.3)
│   │   └── features.py            # Feature extraction (T5.2)
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── training_task.py       # Celery task (T5.8)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── losses.py              # Gaussian NLL loss (T5.4)
│   │   └── mlflow_logger.py       # MLflow integration (T5.6)
│   └── routers/
│       ├── __init__.py
│       └── health.py              # Health check endpoint
├── tests/
│   ├── __init__.py
│   ├── test_features.py           # Feature extraction tests (T5.9)
│   ├── test_dataset.py            # Dataset tests (T5.9)
│   ├── test_model.py              # Model forward pass (T5.9)
│   ├── test_loss.py               # Loss function tests (T5.9)
│   ├── test_mlflow.py             # MLflow logging (T5.9)
│   ├── test_onnx.py               # ONNX export (T5.9)
│   └── fixtures.py                # Pytest fixtures
├── docs/
│   └── TRAINING.md                # Architecture & hyperparameters (T5.10)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📝 Task Breakdown (10 Tasks, 3 giorni)

### **T5.1: Design Neural Network Architecture** (Day 1, Morning)
**Status**: 🔴 NOT STARTED  
**Effort**: 2 hours  
**Deliverable**: `src/models/localization_net.py`

```python
# Target structure:
class LocalizationNet(nn.Module):
    def __init__(self, input_features=128, output_dim=2):
        # CNN backbone: ResNet-18 pretrained
        # Regression head: 2 outputs (lat, lon)
        # Uncertainty head: 2 outputs (sigma_x, sigma_y)
        pass
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        # output: (batch, 2), (batch, 2) → [lat, lon], [sigma_lat, sigma_lon]
        pass
```

**Decisions**:
- ✅ ResNet-18 CNN backbone (proven for signal processing)
- ✅ Dual output heads (position + uncertainty)
- ✅ Input: mel-spectrogram (128 freq bins, 32 time steps)
- ✅ Output: (latitude, longitude, sigma_x, sigma_y)

**Definition**: Uncertainty ≈ confidence in prediction (lower = more confident)

---

### **T5.2: Feature Extraction Utilities** (Day 1, Morning)
**Status**: 🔴 NOT STARTED  
**Effort**: 2 hours  
**Deliverable**: `src/data/features.py`

```python
def iq_to_mel_spectrogram(iq_data, sr=1e6, n_mels=128, n_fft=2048):
    """
    Args:
        iq_data: Complex numpy array, shape (n_samples,)
        sr: Sample rate (Hz)
        n_mels: Number of mel frequency bins
        n_fft: FFT size
    
    Returns:
        mel_spec: numpy array, shape (n_mels, n_frames)
    """
    pass

def compute_mfcc(iq_data, sr=1e6, n_mfcc=13):
    """MFCC features (alternative to mel-spectrogram)"""
    pass

def normalize_features(features, mean=None, std=None):
    """Z-score normalization"""
    pass
```

**Parametri**:
- Mel bins: 128 (balance tra risoluzione e memoria)
- FFT size: 2048 (50ms @ 1 MHz SR)
- Time window: 32 frames (~1.6s di dati IQ)

---

### **T5.3: HeimdallDataset PyTorch Dataset** (Day 1, Afternoon)
**Status**: 🔴 NOT STARTED  
**Effort**: 2 hours  
**Deliverable**: `src/data/dataset.py`

```python
class HeimdallDataset(torch.utils.data.Dataset):
    def __init__(self, db_connection, minio_client, approved_sessions=None):
        """
        Load approved recordings from database + MinIO
        
        Args:
            db_connection: PostgreSQL connection
            minio_client: MinIO S3 client
            approved_sessions: List of session IDs to include (for training subset)
        """
        pass
    
    def __len__(self):
        # Return number of recordings
        pass
    
    def __getitem__(self, idx):
        # Load IQ data from MinIO
        # Compute features
        # Return: (features, ground_truth_position, uncertainty_estimate)
        pass
```

**Data Format**:
```python
# Each item returns:
{
    'features': torch.Tensor(shape=[128, 32]),      # Mel-spectrogram
    'position': torch.Tensor(shape=[2]),             # [lat, lon] (degrees)
    'session_id': str,
    'receiver_id': int,
}
```

---

### **T5.4: Gaussian Negative Log-Likelihood Loss** (Day 1, Afternoon)
**Status**: 🔴 NOT STARTED  
**Effort**: 1.5 hours  
**Deliverable**: `src/utils/losses.py`

```python
class GaussianNLLLoss(nn.Module):
    """
    Loss function that penalizes overconfidence.
    
    For each prediction (mu, sigma), compute:
    NLL = 0.5 * ((y - mu) / sigma)^2 + log(sigma)
    
    This encourages:
    - Accurate predictions (small residual)
    - Honest uncertainty (not too small sigma)
    """
    
    def forward(self, pred_mu, pred_sigma, target):
        # pred_mu: (batch, 2)
        # pred_sigma: (batch, 2)
        # target: (batch, 2)
        # return: scalar loss
        pass
```

**Motivation**: MSE loss non penalizza incertezza; Gaussian NLL = migliore.

**Formula**:
$$L = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{(\mathbf{y}_i - \boldsymbol{\mu}_i)^2}{2\sigma_i^2} + \log(\sigma_i) \right]$$

---

### **T5.5: PyTorch Lightning Module** (Day 1, Evening)
**Status**: 🔴 NOT STARTED  
**Effort**: 2 hours  
**Deliverable**: `src/models/lightning_module.py`

```python
class LocalizationLitModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = LocalizationNet()
        self.loss_fn = GaussianNLLLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        # Forward pass
        # Compute loss
        # Log metrics
        # Return loss
        pass
    
    def validation_step(self, batch, batch_idx):
        # Similar to training_step but no backprop
        pass
    
    def configure_optimizers(self):
        # Adam with LR scheduler
        pass
```

**Hyperparameters**:
```python
{
    'lr': 1e-3,              # Learning rate
    'batch_size': 32,        # Batch size
    'epochs': 100,           # Max epochs
    'early_stopping': 10,    # Patience
    'weight_decay': 1e-4,    # L2 regularization
}
```

---

### **T5.6: MLflow Tracking Integration** (Day 2, Morning)
**Status**: 🔴 NOT STARTED  
**Effort**: 1.5 hours  
**Deliverable**: `src/utils/mlflow_logger.py`

```python
class MLflowTracker:
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def log_hparams(self, hparams):
        """Log hyperparameters"""
        for key, value in hparams.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step):
        """Log validation metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step)
    
    def log_artifact(self, file_path):
        """Log model artifact"""
        mlflow.log_artifact(file_path)
    
    def end_run(self):
        mlflow.end_run()
```

**Integration with Lightning**:
```python
mlflow_logger = MLFlowLogger(
    experiment_name='heimdall_localization',
    tracking_uri='postgresql://...',
)
trainer = pl.Trainer(logger=mlflow_logger)
```

---

### **T5.7: ONNX Export & Upload** (Day 2, Afternoon)
**Status**: 🔴 NOT STARTED  
**Effort**: 1.5 hours  
**Deliverable**: `src/utils/onnx_exporter.py`

```python
def export_to_onnx(model, input_shape, output_path):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: LocalizationNet instance
        input_shape: Tuple (batch_size, channels, height, width)
        output_path: Path to save .onnx file
    """
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['position', 'uncertainty'],
        dynamic_axes={'mel_spectrogram': {0: 'batch_size'}},
        opset_version=14,
    )

def upload_onnx_to_minio(onnx_path, minio_client, bucket='heimdall-models'):
    """Upload ONNX to MinIO s3://heimdall-models/localization-v0.1.0.onnx"""
    minio_client.fput_object(
        bucket,
        'localization-v0.1.0.onnx',
        onnx_path,
    )
```

---

### **T5.8: Training Entry Point & Celery Task** (Day 2, Evening)
**Status**: 🔴 NOT STARTED  
**Effort**: 2 hours  
**Deliverable**: `src/tasks/training_task.py` + `src/main.py`

```python
# src/main.py
from fastapi import FastAPI
from src.tasks.training_task import train_model_task

app = FastAPI(title="Training Service")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "training"}

@app.post("/train")
async def trigger_training(config: TrainingConfig):
    """
    Trigger model training.
    Returns task_id for status monitoring.
    """
    task = train_model_task.delay(config.dict())
    return {"task_id": task.id, "status": "submitted"}

@app.get("/train/{task_id}")
async def get_training_status(task_id: str):
    """Check training progress"""
    task = train_model_task.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.info,
    }

# src/tasks/training_task.py
@app.celery_app.task(bind=True)
def train_model_task(self, config):
    """
    Full training pipeline:
    1. Load approved recordings from DB
    2. Create DataLoaders
    3. Train model with Lightning
    4. Export to ONNX
    5. Upload to MinIO
    6. Register in MLflow
    7. Return model version
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # 1. Load data
        dataset = HeimdallDataset(db, minio)
        train_loader = DataLoader(dataset, batch_size=32)
        
        # 2. Setup training
        model = LocalizationLitModule(config)
        trainer = pl.Trainer(max_epochs=100)
        
        # 3. Train
        trainer.fit(model, train_loader)
        
        # 4. Export + upload
        export_to_onnx(model, ...)
        upload_onnx_to_minio(...)
        
        # 5. Register
        mlflow.register_model(...)
        
        return {'status': 'completed', 'version': 'v0.1.0'}
    
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
```

---

### **T5.9: Comprehensive Test Suite** (Day 2-3)
**Status**: 🔴 NOT STARTED  
**Effort**: 3 hours  
**Deliverable**: `tests/test_*.py` (6 files)

**Test Coverage Target**: >85%

```bash
tests/
├── test_features.py        # Feature extraction (500 lines)
├── test_dataset.py         # Dataset loading (400 lines)
├── test_model.py           # Model architecture (300 lines)
├── test_loss.py            # Loss functions (200 lines)
├── test_mlflow.py          # MLflow tracking (300 lines)
├── test_onnx.py            # ONNX export (250 lines)
└── fixtures.py             # Shared pytest fixtures
```

**Example Tests**:

```python
# test_features.py
def test_iq_to_mel_spectrogram_shape():
    iq_data = np.random.randn(16000)  # 1Hz @ 16kHz
    mel_spec = iq_to_mel_spectrogram(iq_data)
    assert mel_spec.shape == (128, 32)

# test_dataset.py
def test_dataset_load_returns_correct_shape():
    dataset = HeimdallDataset(...)
    sample = dataset[0]
    assert sample['features'].shape == (128, 32)
    assert sample['position'].shape == (2,)

# test_model.py
def test_localization_net_forward():
    model = LocalizationNet()
    x = torch.randn(4, 3, 128, 32)
    out = model(x)
    assert out.shape == (4, 4)  # 2 position + 2 uncertainty
```

---

### **T5.10: Documentation & Architecture Guide** (Day 3)
**Status**: 🔴 NOT STARTED  
**Effort**: 1 hour  
**Deliverable**: `docs/TRAINING.md`

**Contents**:
```markdown
# Training Pipeline Documentation

## Architecture Overview
- Neural network design (ResNet-18 CNN)
- Loss function justification (Gaussian NLL)
- Feature extraction process

## Hyperparameters
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 100
- Early stopping: 10 epochs

## Training Procedure
1. Load approved sessions from PostgreSQL
2. Create DataLoaders (train/val split 80/20)
3. Initialize Lightning trainer
4. Train model
5. Export ONNX
6. Register in MLflow

## Inference Checklist
- Model loaded from MinIO
- Preprocessed input validated
- Output shape confirmed
- Uncertainty calibration tested
```

---

## 🚀 Getting Started

### Step 1: Verify Infrastructure

```bash
# SSH into workspace
cd c:\Users\aless\Documents\Projects\heimdall

# Check Docker containers
docker-compose ps

# Verify database
psql -h localhost -U heimdall_user -d heimdall -c "\dt"

# Check MLflow
curl http://localhost:5000/api/2.0/mlflow/runs/list
```

### Step 2: Setup Service

```bash
cd services/training

# Install dependencies
pip install -r requirements.txt

# Run baseline tests (should all pass before Phase 5)
pytest tests/ -v --tb=short
```

### Step 3: Start Development

```bash
# Start with T5.1 (Neural Network Architecture)
# See TASK BREAKDOWN above

# After each task, run tests:
pytest tests/test_<module>.py -v

# Track progress in PHASE5_PROGRESS.md
```

---

## ✅ Phase 5 Success Criteria

| Checkpoint | Validation                         | Status |
| ---------- | ---------------------------------- | ------ |
| **CP5.1**  | Model forward pass (output shapes) | 🔴 TODO |
| **CP5.2**  | Dataset loader (data pipeline)     | 🔴 TODO |
| **CP5.3**  | Training loop (no errors)          | 🔴 TODO |
| **CP5.4**  | ONNX export to MinIO               | 🔴 TODO |
| **CP5.5**  | MLflow model registration          | 🔴 TODO |

---

## 📞 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision pytorch-lightning
```

### Issue: "Connection refused" to PostgreSQL
```bash
docker-compose up -d postgres
docker-compose ps  # Verify healthy
```

### Issue: ONNX export fails
```bash
# Ensure model is on CPU
model = model.cpu()
# Retry export with verbose mode
torch.onnx.export(..., verbose=True)
```

---

## 📞 Next Steps

1. ✅ Read this document completely
2. ✅ Verify infrastructure operational
3. 🔴 Start T5.1: Neural Network Architecture
4. 🔴 Continue T5.2-T5.10 in order
5. 🔴 Validate all checkpoints
6. 🔴 Proceed to Phase 6 when complete

---

**Generated**: 2025-10-22  
**Next Review**: After T5.5  
**Estimated Completion**: 2025-10-25 (3 days)
