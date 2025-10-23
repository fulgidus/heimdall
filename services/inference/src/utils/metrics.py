"""Prometheus metrics for Phase 6 Inference Service."""
from prometheus_client import Counter, Histogram, Gauge
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ============================================================================
# INFERENCE METRICS
# ============================================================================

# Histogram: inference latency in milliseconds
inference_latency = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds (end-to-end)",
    buckets=[10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000],
)

# Histogram: preprocessing latency
preprocessing_latency = Histogram(
    "preprocessing_latency_ms",
    "IQ preprocessing latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100],
)

# Histogram: ONNX runtime latency (pure inference)
onnx_latency = Histogram(
    "onnx_latency_ms",
    "Pure ONNX runtime latency in milliseconds",
    buckets=[5, 10, 25, 50, 75, 100, 150, 200],
)

# ============================================================================
# CACHE METRICS
# ============================================================================

cache_hits = Counter(
    "cache_hits_total",
    "Total cache hits",
)

cache_misses = Counter(
    "cache_misses_total",
    "Total cache misses",
)

# Gauge: current cache hit rate (0-1)
cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Current cache hit rate (0-1)",
)

# Gauge: Redis memory usage (bytes)
redis_memory_bytes = Gauge(
    "redis_memory_bytes",
    "Redis memory usage in bytes",
)

# ============================================================================
# REQUEST METRICS
# ============================================================================

requests_total = Counter(
    "inference_requests_total",
    "Total inference requests by endpoint",
    ["endpoint"],
)

errors_total = Counter(
    "inference_errors_total",
    "Total inference errors by type",
    ["error_type"],
)

# Gauge: active concurrent requests
active_requests = Gauge(
    "inference_active_requests",
    "Number of active/concurrent inference requests",
)

# ============================================================================
# MODEL METRICS
# ============================================================================

model_reloads = Counter(
    "model_reloads_total",
    "Total model reloads from MLflow",
)

model_loads = Gauge(
    "model_loaded",
    "Is model currently loaded (1=yes, 0=no)",
)

model_inference_count = Counter(
    "model_inference_count_total",
    "Total inferences performed with this model",
)

model_accuracy_meters = Gauge(
    "model_accuracy_meters",
    "Model accuracy from training metadata (sigma in meters)",
)

# ============================================================================
# METRIC HELPER FUNCTIONS
# ============================================================================

def record_inference_latency(duration_ms: float):
    """Record end-to-end inference latency."""
    inference_latency.observe(duration_ms)


def record_preprocessing_latency(duration_ms: float):
    """Record IQ preprocessing latency."""
    preprocessing_latency.observe(duration_ms)


def record_onnx_latency(duration_ms: float):
    """Record pure ONNX runtime latency."""
    onnx_latency.observe(duration_ms)


def record_cache_hit():
    """Record cache hit and update hit rate."""
    cache_hits.inc()
    _update_cache_hit_rate()


def record_cache_miss():
    """Record cache miss and update hit rate."""
    cache_misses.inc()
    _update_cache_hit_rate()


def _update_cache_hit_rate():
    """Update cache hit rate gauge."""
    try:
        # Get current hit/miss counts
        total_hits = cache_hits._value.get() if hasattr(cache_hits, '_value') else 0
        total_misses = cache_misses._value.get() if hasattr(cache_misses, '_value') else 0
        total = total_hits + total_misses
        
        if total > 0:
            rate = total_hits / total
            cache_hit_rate.set(rate)
            logger.debug(f"Cache hit rate: {rate:.2%} ({total_hits}/{total})")
    except Exception as e:
        logger.warning(f"Could not update cache hit rate: {e}")


def record_request_error(error_type: str):
    """Record inference error."""
    errors_total.labels(error_type=error_type).inc()


def set_redis_memory(bytes_value: float):
    """Set Redis memory usage gauge."""
    redis_memory_bytes.set(bytes_value)


def record_model_reload():
    """Record model reload event."""
    model_reloads.inc()


def set_model_loaded(loaded: bool):
    """Set model loaded status (1=loaded, 0=not loaded)."""
    model_loads.set(1 if loaded else 0)


def record_model_inference():
    """Record successful model inference."""
    model_inference_count.inc()


def set_model_accuracy(accuracy_m: float):
    """Set model accuracy from metadata."""
    model_accuracy_meters.set(accuracy_m)


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

@contextmanager
def InferenceMetricsContext(endpoint: str):
    """
    Context manager for recording inference metrics.
    
    Usage:
        with InferenceMetricsContext("predict"):
            # Run inference
            result = model.predict(data)
        # Metrics automatically recorded
    
    Args:
        endpoint: API endpoint name for labeling
    """
    start_time = time.time()
    active_requests.inc()
    requests_total.labels(endpoint=endpoint).inc()
    
    try:
        yield
    except Exception as e:
        error_type = type(e).__name__
        record_request_error(error_type)
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_inference_latency(duration_ms)
        active_requests.dec()
        logger.debug(f"Endpoint {endpoint} completed in {duration_ms:.2f}ms")


@contextmanager
def PreprocessingMetricsContext():
    """Context manager for preprocessing latency."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_preprocessing_latency(duration_ms)


@contextmanager
def ONNXMetricsContext():
    """Context manager for ONNX runtime latency."""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        record_onnx_latency(duration_ms)
