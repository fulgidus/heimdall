# Performance Optimization Tips

## Overview

Strategies and techniques to optimize Heimdall system performance.

## Query Optimization

### Database Queries

```python
# ❌ Bad: N+1 query problem
tasks = Task.query.all()
for task in tasks:
    results = Result.query.filter_by(task_id=task.id).all()

# ✅ Good: Eager loading
tasks = Task.query.options(joinedload(Task.results)).all()
```

### Index Strategy

```sql
-- Create indexes on frequently queried columns
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_measurements_created ON signal_measurements(created_at DESC);
CREATE INDEX idx_results_confidence ON task_results(confidence DESC);

-- Composite indexes for common filters
CREATE INDEX idx_measurements_station_freq 
ON signal_measurements(station_name, frequency);

-- Analyze query plans
EXPLAIN ANALYZE SELECT * FROM signal_measurements 
WHERE station_name = 'Giaveno' AND frequency = 145.5;
```

## GPU Optimization

### ONNX Model Export

```python
import torch
import onnx

# Convert PyTorch to ONNX
model = torch.load('model.pt')
dummy_input = torch.randn(1, 128, 431)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=13,
    do_constant_folding=True
)

# Verify model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### Model Quantization

```python
import torch.quantization as quantization

# Dynamic quantization
quantized_model = quantization.quantize_dynamic(
    model,
    qconfig_spec=quantization.QConfig(
        activation=quantization.HistogramObserver,
        weight=quantization.PerChannelMinMaxObserver,
    ),
    dtype=torch.qint8
)

# Result: ~3x smaller, 1.5-2x faster
```

### Batch Processing

```python
def inference_batch(signals):
    """Process multiple signals in batch for efficiency."""
    
    # Stack signals
    batch = np.stack(signals)  # (N, 128, 431)
    
    # Convert to tensor
    tensor = torch.from_numpy(batch).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
    
    # Results
    locations = outputs.cpu().numpy()
    
    return locations
```

## Caching Strategies

### Redis Caching

```python
from functools import wraps
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def cache_result(ttl=3600):
    """Cache function result in Redis."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{args}:{kwargs}"
            
            # Try cache
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache
            result = func(*args, **kwargs)
            redis_client.setex(key, ttl, json.dumps(result))
            
            return result
        return wrapper
    return decorator

@cache_result(ttl=3600)
def get_task_result(task_id):
    """Get result (cached for 1 hour)."""
    return db.query_result(task_id)
```

## API Response Optimization

### Compression

```python
from fastapi.middleware.gzip import GZIPMiddleware

app.add_middleware(GZIPMiddleware, minimum_size=1000)
```

### Pagination

```python
# Instead of returning all results
@app.get("/results")
async def get_results(skip: int = 0, limit: int = 50):
    """Get paginated results."""
    return db.query_results(skip=skip, limit=limit)
```

### Field Selection

```python
@app.get("/results/{task_id}")
async def get_result(task_id: str, fields: str = ""):
    """Get result with optional field selection."""
    
    result = db.query_result(task_id)
    
    # Return only requested fields
    if fields:
        requested = fields.split(',')
        result = {k: v for k, v in result.items() if k in requested}
    
    return result
```

## Infrastructure Optimization

### Container Resource Limits

```yaml
# Kubernetes resource limits
apiVersion: v1
kind: Pod
metadata:
  name: ml-inference
spec:
  containers:
    - name: inference
      resources:
        requests:
          memory: "2Gi"
          cpu: "1000m"
        limits:
          memory: "4Gi"
          cpu: "2000m"
```

### Connection Pooling

```python
# Database connection pool
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections after 1 hour
)
```

### Worker Tuning

```python
# Celery worker configuration
app.conf.update(
    worker_prefetch_multiplier=1,      # Reduce prefetch for long tasks
    worker_max_tasks_per_child=1000,   # Restart after 1000 tasks
    task_acks_late=True,               # ACK after task completion
    task_reject_on_worker_lost=True,   # Reject on worker loss
)
```

## Signal Processing Optimization

### Vectorization

```python
# ❌ Slow: Loops in Python
result = []
for i in range(len(signals)):
    result.append(np.fft.fft(signals[i]))

# ✅ Fast: Vectorized operations
result = np.fft.fft(signals, axis=1)
```

### Data Type Optimization

```python
# Use appropriate dtypes to save memory
# float32 instead of float64 when precision allows
signals = signals.astype(np.float32)

# Complex64 for IQ data instead of complex128
iq_data = iq_data.astype(np.complex64)
```

## Profiling & Benchmarking

### Profiling with cProfile

```bash
# Profile entire script
python -m cProfile -s cumulative -o profile_stats script.py

# View results
python -c "import pstats; p = pstats.Stats('profile_stats'); p.print_stats(10)"
```

### Memory Profiling

```bash
pip install memory-profiler

python -m memory_profiler script.py
```

### Benchmarking

```python
import timeit

# Benchmark function
def benchmark(func, *args, **kwargs):
    """Run function and measure time."""
    start = timeit.default_timer()
    result = func(*args, **kwargs)
    end = timeit.default_timer()
    
    print(f"{func.__name__}: {end - start:.4f}s")
    return result

# Compare approaches
t1 = timeit.timeit(lambda: list_comprehension(), number=10000)
t2 = timeit.timeit(lambda: loop_approach(), number=10000)
print(f"List comprehension {t1/t2:.1f}x faster")
```

## Monitoring Performance

### Key Metrics

```python
import prometheus_client

# Create metrics
request_latency = prometheus_client.Histogram(
    'request_latency_seconds',
    'Request latency'
)

task_duration = prometheus_client.Histogram(
    'task_duration_seconds',
    'Task processing duration'
)

# Record metrics
with request_latency.time():
    # Handle request
    pass
```

### Performance Alerts

```yaml
# Prometheus alert rules
groups:
  - name: performance
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, request_latency) > 0.1
        for: 5m
        
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.8
        for: 5m
```

## Common Bottlenecks & Solutions

| Bottleneck | Symptom         | Solution                             |
| ---------- | --------------- | ------------------------------------ |
| Database   | Slow queries    | Add indexes, optimize queries        |
| GPU        | Low utilization | Batch processing, check CUDA         |
| Memory     | OOM errors      | Reduce batch size, clear cache       |
| Network    | Timeouts        | Increase timeout, reduce payload     |
| CPU        | High usage      | Parallelize, use compiled extensions |

## Best Practices

1. **Measure before optimizing**: Profile first
2. **Optimize hotspots**: 20% of code uses 80% of resources
3. **Use appropriate algorithms**: O(n) vs O(n²) matters
4. **Cache aggressively**: But invalidate correctly
5. **Monitor continuously**: Catch regressions early

---

**Related**: [Performance Benchmarks](./performance_benchmarks.md) | [Deployment Guide](./deployment_instructions.md)

**Last Updated**: October 2025
