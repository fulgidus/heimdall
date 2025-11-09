# HeimdallNetPro 🚀: Experimental Architecture with Performer Attention

**Status:** Experimental | **Badge:** RESEARCH  
**Author:** Alessio Corsi  
**Created:** 2025-11-09  
**Version:** 0.1 (Alpha)  
**Base Architecture:** HeimdallNet v1.0

---

## 🎯 Overview

**HeimdallNetPro** is an experimental variant of [HeimdallNet](HEIMDALLNET.md) that replaces pooling-based aggregation with **Performer linear attention**. This enables explicit modeling of receiver-to-receiver interactions while maintaining numerical stability at small batch sizes.

### Motivation

HeimdallNet v1.0 achieves excellent accuracy (±8-15m) but sacrifices self-attention due to **NaN loss issues** with standard `MultiheadAttention` at batch sizes 1-2. HeimdallNetPro aims to recover the benefits of attention while ensuring production-grade stability.

### Key Differences from HeimdallNet

| Aspect | HeimdallNet v1.0 | HeimdallNetPro |
|--------|------------------|----------------|
| **Aggregation** | Pooling-only (max/mean/quality-weighted) | **Performer attention + pooling** |
| **Receiver Interactions** | ❌ None (implicit via geometry) | ✅ Explicit (learned attention weights) |
| **Complexity** | O(N) pooling | O(N) linear attention |
| **Numerical Stability** | ✅ Proven (no NaN at batch=1) | ⏳ Testing (kernel approximation) |
| **Status** | Production-ready | Experimental |

### Target Improvements

| Metric | HeimdallNet v1.0 | HeimdallNetPro Goal |
|--------|------------------|---------------------|
| **Mean Accuracy** | ±8-15m | **±5-10m (30-50% improvement)** |
| **Gradient Stability** | ✅ No NaN | ✅ No NaN (Performer kernel) |
| **Inference Latency** | 40-60ms | <70ms (acceptable overhead) |
| **Training Stability** | ✅ Batch 1-512 | ⏳ Testing batch 1-512 |

---

## 🧠 Why Performer?

### Problem with Standard Attention

Standard `MultiheadAttention` uses softmax for attention weights:

```python
attention_weights = softmax(Q @ K.T / sqrt(d))  # Can overflow/underflow
output = attention_weights @ V
```

**Issues at Batch Size 1-2:**
- Softmax overflow/underflow with small batches
- Gradient instability during backpropagation
- NaN loss after 1-2 epochs

### Performer Solution

Performer approximates attention using a **positive random feature kernel**:

```python
# Standard attention: O(N²) complexity, numerically unstable
attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

# Performer: O(N) complexity, numerically stable
φ(x) = ReLU(Wx + b)  # Positive random features
attention_approx(Q, K, V) = φ(Q) @ (φ(K).T @ V)  # Linear in N!
```

**Key Advantages:**
1. ✅ **Numerical Stability:** No softmax overflow/underflow
2. ✅ **Linear Complexity:** O(N) vs O(N²) for N receivers
3. ✅ **Production-Ready:** Used by Google, Hugging Face
4. ✅ **Explicit Interactions:** Models geometric relationships for triangulation

### Performer Configuration

```python
from performer_pytorch import SelfAttention

attention = SelfAttention(
    dim=256,                      # Feature dimension
    heads=8,                      # Number of attention heads
    dropout=0.1,                  # Dropout probability
    causal=False,                 # Non-causal (set processing)
    nb_features=64,               # Random features for kernel
    generalized_attention=True,   # More expressive kernel
    kernel_fn=nn.ReLU()          # Positive kernel (stability)
)
```

**Parameters Explained:**
- `nb_features=64`: Number of random features for kernel approximation (higher = more accurate, slower)
- `generalized_attention=True`: Uses orthogonal random features (better approximation)
- `kernel_fn=nn.ReLU()`: Positive kernel ensures numerical stability

---

## 🏗️ Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  HeimdallNetPro Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Variable 1-10 Receivers (same as HeimdallNet)      │
│                                                             │
│  ┌───────────────────────────────────────────────┐          │
│  │  Component 1: PerReceiverEncoder (UNCHANGED)  │          │
│  │  Output: (batch, N, 256)                      │          │
│  └───────────────────────────────────────────────┘          │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────┐          │
│  │  Component 2: SetAttentionAggregatorPro (NEW!)│          │
│  │  ┌─────────────────────────────────────────┐  │          │
│  │  │ 1. Pre-Normalization (LayerNorm)       │  │          │
│  │  │    → Stabilize inputs for attention     │  │          │
│  │  │                                         │  │          │
│  │  │ 2. Performer SelfAttention (8 heads)   │  │          │
│  │  │    → Explicit receiver interactions     │  │          │
│  │  │    → Linear O(N) complexity             │  │          │
│  │  │    → Numerically stable kernel          │  │          │
│  │  │                                         │  │          │
│  │  │ 3. Residual Connection + Post-Norm     │  │          │
│  │  │    → Gradient flow + stability          │  │          │
│  │  │                                         │  │          │
│  │  │ 4. Multi-Strategy Pooling (fallback)   │  │          │
│  │  │    • Max Pooling (best receiver)        │  │          │
│  │  │    • Mean Pooling (consensus)           │  │          │
│  │  │    • Quality-Weighted (SNR-aware)       │  │          │
│  │  │    → Concat + Fusion                    │  │          │
│  │  └─────────────────────────────────────────┘  │          │
│  │  Output: (batch, 256)                         │          │
│  └───────────────────────────────────────────────┘          │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────┐          │
│  │  Components 3-5: UNCHANGED from HeimdallNet  │          │
│  │  • GeometryEncoder                           │          │
│  │  • Global Fusion                             │          │
│  │  • Dual-Head Output                          │          │
│  └───────────────────────────────────────────────┘          │
│                                                             │
│  Output: Position (batch, 2) + Uncertainty (batch, 2)       │
└─────────────────────────────────────────────────────────────┘
```

### SetAttentionAggregatorPro (Detailed)

**Architectural Flow:**

```python
# Input: (batch, N, 256) receiver features
x = receiver_features

# Step 1: Pre-normalization (stabilize inputs)
x_norm = LayerNorm(x)

# Step 2: Performer attention (model interactions)
attended = PerformerSelfAttention(x_norm, mask=mask)
# attended.shape: (batch, N, 256)

# Step 3: Residual + post-normalization (gradient flow)
x_residual = x + Dropout(attended)
x_residual = LayerNorm(x_residual)

# Step 4: Multi-strategy pooling (proven fallback)
max_pooled = max(x_residual, dim=1)      # (batch, 256)
mean_pooled = mean(x_residual, dim=1)    # (batch, 256)
quality_pooled = weighted_mean(x_residual, quality_scores)  # (batch, 256)

# Step 5: Fusion
aggregated = (max_pooled + mean_pooled + quality_pooled) / 3
# aggregated.shape: (batch, 256)
```

**Key Design Decisions:**

1. **Pre-Normalization:** Stabilizes inputs to Performer (prevents large activations)
2. **Residual Connection:** Ensures gradient flow even if attention fails
3. **Pooling Fallback:** Provides safety net if attention learns poorly
4. **Equal Weighting:** Avoids bias toward any single strategy

---

## 🔬 Technical Specifications

### Model Differences from HeimdallNet

| Component | HeimdallNet v1.0 | HeimdallNetPro | Delta |
|-----------|------------------|----------------|-------|
| **PerReceiverEncoder** | 1.2M params | 1.2M params | - |
| **SetAttentionAggregator** | 264K params (pooling) | **520K params** (+Performer) | **+256K** |
| **GeometryEncoder** | 149K params | 149K params | - |
| **Global Fusion** | 395K params | 395K params | - |
| **Output Heads** | 1K params | 1K params | - |
| **Total Parameters** | 2.2M | **2.5M** | **+13%** |

### Memory Requirements

| Task | HeimdallNet v1.0 | HeimdallNetPro (Estimated) |
|------|------------------|----------------------------|
| **Training (batch=32)** | 3.0 GB VRAM | 3.5 GB VRAM (+17%) |
| **Inference (batch=1)** | 0.8 GB VRAM | 0.9 GB VRAM (+12%) |

### Inference Speed (Estimated)

| Hardware | HeimdallNet v1.0 | HeimdallNetPro (Goal) |
|----------|------------------|------------------------|
| **RTX 3080 (batch=1)** | 40-60ms | <70ms (acceptable) |
| **RTX 3080 (batch=32)** | 80-120ms | <150ms |

---

## 🧪 Experimental Validation Plan

### Phase 1: Smoke Test (10 minutes)

**Goal:** Verify basic functionality

```bash
# Rebuild training container with performer-pytorch
docker-compose build training

# Test model instantiation
docker exec heimdall-training python3 -c "
from src.models.heimdall_net import create_heimdall_net_pro
import torch

model = create_heimdall_net_pro(max_receivers=7)
print(f'✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

# Test forward pass
iq = torch.randn(4, 3, 2, 1024)
features = torch.randn(4, 3, 6)
positions = torch.randn(4, 3, 3)
receiver_ids = torch.tensor([[0, 2, 4]] * 4)
mask = torch.ones(4, 3, dtype=torch.bool)

pred_pos, pred_unc = model(iq, features, positions, receiver_ids, mask)
print(f'✅ Forward pass: pos={pred_pos.shape}, unc={pred_unc.shape}')

# Check for NaN
assert not torch.isnan(pred_pos).any(), '❌ NaN in position'
assert not torch.isnan(pred_unc).any(), '❌ NaN in uncertainty'
print('✅ No NaN detected')
"
```

**Success Criteria:**
- ✅ Model instantiates without errors
- ✅ Forward pass returns correct shapes
- ✅ No NaN in outputs

---

### Phase 2: Stability Test (2 epochs, 30 minutes)

**Goal:** Verify gradient stability and NaN-free training

```bash
# Launch 2-epoch training job
docker exec heimdall-training python3 -c "
from src.tasks.training_task import train_model_task

result = train_model_task.apply_async(kwargs={
    'dataset_id': 'test_dataset_001',
    'model_architecture': 'heimdall_net_pro',
    'batch_size': 8,
    'learning_rate': 0.001,
    'total_epochs': 2,
    'checkpoint_dir': '/tmp/heimdall_pro_stability_test',
    'early_stopping_patience': 10
}).get(timeout=1800)

print(f'Task result: {result}')
"

# Monitor logs for NaN warnings
docker logs -f heimdall-training | grep -E '(NaN|loss=)'
```

**Success Criteria:**
- ✅ Training completes 2 epochs without crashes
- ✅ Loss decreases (no NaN)
- ✅ Gradient norms < 10.0 (healthy)
- ✅ No NaN warnings in logs
- ✅ Checkpoint saved successfully

---

### Phase 3: A/B Comparison Test (5 epochs, 2 hours)

**Goal:** Compare HeimdallNetPro vs HeimdallNet accuracy

**Test Setup:**
```python
# Train both models on same dataset
experiments = [
    {
        'name': 'HeimdallNet_v1.0_baseline',
        'model_architecture': 'heimdall_net',
        'batch_size': 32,
        'total_epochs': 5
    },
    {
        'name': 'HeimdallNetPro_v0.1_experimental',
        'model_architecture': 'heimdall_net_pro',
        'batch_size': 32,
        'total_epochs': 5
    }
]

# Launch parallel training jobs
# Compare final metrics
```

**Success Criteria:**
- ✅ HeimdallNetPro converges (loss decreases)
- ✅ No NaN loss during training
- ✅ Final accuracy **≥20% improvement** over HeimdallNet
  - HeimdallNet: ±8-15m → HeimdallNetPro: ±5-10m
- ✅ Inference latency <70ms (acceptable overhead)

**Comparison Metrics:**

| Metric | HeimdallNet v1.0 | HeimdallNetPro | Target |
|--------|------------------|----------------|--------|
| **Mean Error** | 11.5m | ? | ≤8.0m (30% improvement) |
| **P50 Error** | 10.2m | ? | ≤7.0m |
| **P95 Error** | 18.3m | ? | ≤12.0m |
| **Loss (final)** | 0.15-0.25 | ? | ≤0.20 |
| **Gradient Norms** | <5.0 | ? | <10.0 |
| **Inference Time** | 40-60ms | ? | <70ms |

---

## 🚦 Decision Criteria: Production Promotion

### Promote to Production If:

1. ✅ **Stability:** No NaN loss for 10+ epochs across batch sizes 1-512
2. ✅ **Accuracy:** ≥20% improvement over HeimdallNet (±8-15m → ±5-10m)
3. ✅ **Speed:** Inference latency <70ms (RTX 3080, batch=1)
4. ✅ **Robustness:** Works with 2-10 receivers, handles dropout gracefully
5. ✅ **Convergence:** Consistent training across 3+ independent runs

### Demote to Archive If:

1. ❌ **NaN Loss:** Frequent NaN issues during training
2. ❌ **No Improvement:** Accuracy similar or worse than HeimdallNet
3. ❌ **Too Slow:** Inference >100ms (unacceptable for real-time)
4. ❌ **Unstable:** Training diverges or oscillates frequently

### Keep Experimental If:

1. 🟡 **Mixed Results:** Better accuracy but slower/unstable
2. 🟡 **Conditional Improvement:** Works well only for specific scenarios (e.g., 7+ receivers)
3. 🟡 **Research Value:** Insights useful for future architectures

---

## 📊 Implementation Details

### Code Location

- **Model:** `services/training/src/models/heimdall_net.py` (lines 665-985)
- **Factory:** `create_heimdall_net_pro()` (line 950)
- **Registry:** `services/training/src/models/model_registry.py` (to be added)

### Usage Example

```python
from src.models.heimdall_net import create_heimdall_net_pro
import torch

# Create model
model = create_heimdall_net_pro(
    max_receivers=7,
    use_calibration=True,
    dropout=0.1
)

# Same API as HeimdallNet (drop-in replacement)
iq = torch.randn(4, 3, 2, 1024)
features = torch.randn(4, 3, 6)
positions = torch.randn(4, 3, 3)
receiver_ids = torch.tensor([[0, 2, 4]] * 4)
mask = torch.ones(4, 3, dtype=torch.bool)

pred_pos, pred_unc = model(iq, features, positions, receiver_ids, mask)
print(f"Position: {pred_pos.shape}, Uncertainty: {pred_unc.shape}")
```

### Training Configuration

```json
{
  "model_architecture": "heimdall_net_pro",
  "batch_size": 32,
  "learning_rate": 0.001,
  "total_epochs": 50,
  "optimizer": "adamw",
  "scheduler": "reduce_on_plateau",
  "early_stopping_patience": 10
}
```

---

## 🔍 Debugging & Monitoring

### Key Metrics to Watch

**During Training:**
1. **Loss:** Should decrease smoothly, no spikes
2. **Gradient Norms:** Should be 0.1-10.0 (healthy range)
3. **NaN Count:** Should be 0 (check logs)
4. **Attention Weights:** Visualize receiver importance
5. **Uncertainty Calibration:** σ should match actual error

**During Inference:**
1. **Latency:** P50/P95/P99 (should be <70ms)
2. **Throughput:** Requests/second
3. **Accuracy:** Mean error, P95 error
4. **Cache Hit Rate:** Redis caching effectiveness

### Logging Commands

```bash
# Monitor training progress
docker logs -f heimdall-training | grep -E '(epoch|loss|NaN)'

# Check for NaN in specific layers
docker exec heimdall-training python3 -c "
import torch
from src.models.heimdall_net import create_heimdall_net_pro

model = create_heimdall_net_pro()
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f'❌ NaN in {name}')
"

# Profile inference time
docker exec heimdall-training python3 -c "
import torch, time
from src.models.heimdall_net import create_heimdall_net_pro

model = create_heimdall_net_pro().cuda().eval()
inputs = [
    torch.randn(1, 3, 2, 1024).cuda(),
    torch.randn(1, 3, 6).cuda(),
    torch.randn(1, 3, 3).cuda(),
    torch.tensor([[0, 2, 4]]).cuda(),
    torch.ones(1, 3, dtype=torch.bool).cuda()
]

# Warm-up
for _ in range(10):
    with torch.no_grad():
        model(*inputs)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f'Mean: {sum(times)/len(times):.1f}ms')
print(f'P50: {sorted(times)[49]:.1f}ms')
print(f'P95: {sorted(times)[94]:.1f}ms')
"
```

---

## 📈 Experimental Results

### A/B Test Results (2025-11-09)

**Test Configuration:**
- **Dataset:** Synthetic 7-receiver dataset (1000 samples, 80/20 train/val split)
- **Training:** 5 epochs, batch size 32, AdamW optimizer
- **Hardware:** RTX 3080 (10GB VRAM)
- **Comparison:** HeimdallNet v1.0 vs HeimdallNetPro v0.1

**Results Summary:**

| Metric | HeimdallNet v1.0 | HeimdallNetPro v0.1 | Delta | Status |
|--------|------------------|---------------------|-------|--------|
| **Position MSE** | 1.0303 | 1.0088 | **-2.1%** | 🟡 Marginal improvement |
| **Uncertainty MSE** | 0.4232 | 0.4271 | **+0.9%** | 🔴 Slightly worse |
| **Inference Time** | 4.56ms | 5.56ms | **+22%** | 🔴 Slower |
| **Parameters** | 2.2M | 2.5M | +13% | - |
| **VRAM Usage** | 0.8 GB | 0.9 GB | +12% | - |

**Success Criteria Check:**

| Criterion | Target | Result | Pass? |
|-----------|--------|--------|-------|
| **Training Stability** | No NaN loss | ✅ No NaN detected | ✅ PASS |
| **Inference Latency** | <70ms | 5.56ms | ✅ PASS |
| **Accuracy Improvement** | ≥20% better | 2.1% better | ❌ FAIL |

**Decision: KEEP EXPERIMENTAL** ⏳

**Rationale:**
- ✅ **Stability:** No NaN issues during training or inference (major achievement)
- ✅ **Speed:** Inference latency well within acceptable range (<70ms target)
- ❌ **Accuracy:** Only 2.1% improvement doesn't justify 13% parameter increase and added complexity
- 🟡 **Potential:** Shows promise but needs further investigation

**Recommendations for Future Work:**
1. **Extended Training:** Test with 20-50 epochs to see if attention learns better long-term
2. **Real WebSDR Data:** Synthetic data may not expose attention's full potential
3. **Hyperparameter Tuning:** Optimize `nb_features`, `num_heads`, dropout rates
4. **Attention Analysis:** Visualize learned attention weights to understand receiver interactions
5. **Larger Datasets:** Test with 10K-100K samples to see if attention scales better

**Why Only 2.1% Improvement?**
- **Small Dataset:** 1000 samples may not be enough for attention to learn meaningful patterns
- **Short Training:** 5 epochs limits attention mechanism's learning capacity
- **Synthetic Data:** Uniform noise/geometry may not require complex receiver interactions
- **Pooling Fallback:** Strong pooling baseline may dominate the aggregated features

**Next Steps:**
- Archive A/B test script: `/tmp/test_heimdallnetpro_ab.py`
- Monitor community feedback on Performer architecture
- Revisit after Phase 8 (Kubernetes deployment) with production data

---

## 🤝 Contributing & Future Work

### Potential Improvements

1. **Adaptive Attention:** Learn when to use attention vs pooling
2. **Multi-Scale Performer:** Different feature granularities
3. **Graph Neural Networks:** Explicit receiver graph structure
4. **Hyperparameter Tuning:** Optimize `nb_features`, `num_heads`
5. **Knowledge Distillation:** Compress to HeimdallNet size

### Experiments to Try

- [x] A/B test vs HeimdallNet (completed 2025-11-09)
- [ ] Vary `nb_features` (32, 64, 128) - accuracy vs speed tradeoff
- [ ] Test different kernel functions (ReLU, ELU, softmax)
- [ ] Compare generalized vs standard Performer
- [ ] Ablation: Attention-only (no pooling fallback)
- [ ] Ensemble: HeimdallNet + HeimdallNetPro
- [ ] Extended training (20-50 epochs) on real WebSDR data
- [ ] Attention weight visualization

---

## 📚 References

### Papers

1. **Performer:** Choromanski et al. (2020) - "Rethinking Attention with Performers"  
   [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)

2. **Linear Attention:** Katharopoulos et al. (2020) - "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"  
   [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)

3. **Efficient Transformers:** Tay et al. (2020) - "Efficient Transformers: A Survey"  
   [arXiv:2009.06732](https://arxiv.org/abs/2009.06732)

### Implementation

- **performer-pytorch:** [GitHub](https://github.com/lucidrains/performer-pytorch)
- **Original Performer:** [Google Research](https://github.com/google-research/google-research/tree/master/performer)

### Related Documentation

- [HeimdallNet v1.0](HEIMDALLNET.md) - Base architecture
- [Model Registry](../services/training/src/models/model_registry.py)
- [Training API](TRAINING_API.md)

---

## 📋 Changelog

### Version 0.1 (2025-11-09)
- ✅ Initial experimental release
- ✅ Performer SelfAttention integration
- ✅ Pre/post normalization + residual connections
- ✅ Multi-strategy pooling fallback
- ✅ Comprehensive NaN validation
- ✅ Same API as HeimdallNet (drop-in replacement)
- ✅ Smoke test passed (model instantiation, forward pass, no NaN)
- ✅ Stability test passed (2 epochs, no NaN loss)
- ✅ A/B test completed (2.1% accuracy improvement, insufficient for production)
- ✅ Debug logs removed from production code
- 📊 **Status: Experimental** - Requires extended validation with real WebSDR data

---

## 📄 License

**Heimdall Project**  
License: CC Non-Commercial  
Author: fulgidus (alessio.corsi@gmail.com)

---

**Status: Experimental** ⏳ - A/B testing shows marginal improvement (+2.1% accuracy). Not ready for production but shows promise for future research.

**Next Steps:**
1. Extended training (20-50 epochs) with larger datasets
2. Real WebSDR data validation
3. Attention weight analysis and visualization
4. Hyperparameter optimization

**Questions?** Open an issue on GitHub or contact alessio.corsi@gmail.com

**Want to experiment?** Follow the [Experimental Validation Plan](#-experimental-validation-plan) above.
