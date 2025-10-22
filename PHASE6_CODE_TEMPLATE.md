# 🔧 PHASE 6: Code Structure Template

**Purpose**: Quick reference for implementing Phase 6 tasks  
**Target**: Copy-paste starting points for each file

---

## 📁 Directory Structure

```
services/inference/
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── src/
│   ├── __init__.py
│   ├── main.py                          # Entry point with graceful shutdown
│   ├── config.py                        # Configuration from .env
│   ├── models/
│   │   ├── __init__.py
│   │   ├── onnx_loader.py              # T6.1: ONNX Model Loader
│   │   └── schemas.py                  # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predict.py                  # T6.2: /predict endpoint
│   │   ├── model.py                    # T6.8: /model/info endpoint
│   │   └── health.py                   # /health endpoint
│   └── utils/
│       ├── __init__.py
│       ├── uncertainty.py              # T6.3: Ellipse calculations
│       ├── model_versioning.py         # T6.5: Model versioning
│       ├── metrics.py                  # T6.6: Prometheus metrics
│       └── preprocessing.py             # Feature extraction
└── tests/
    ├── __init__.py
    ├── test_onnx_loader.py             # T6.1 tests
    ├── test_predict_endpoints.py       # T6.2 tests
    ├── test_uncertainty.py             # T6.3 tests
    ├── test_batch_predict.py           # T6.4 tests
    ├── test_model_versioning.py        # T6.5 tests
    ├── test_metrics.py                 # T6.6 tests
    ├── load_test_inference.py          # T6.7 load test
    ├── integration_test_mlflow.py      # Integration tests
    └── conftest.py                     # Pytest fixtures
```

---

## 🔨 T6.1: services/inference/src/models/onnx_loader.py

```python
import logging
from typing import Dict, Optional
import numpy as np
import onnxruntime as ort
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ONNXModelLoader:
    """Load and manage ONNX model from MLflow registry."""
    
    def __init__(
        self,
        mlflow_uri: str,
        model_name: str = "localization_model",
        stage: str = "Production",
    ):
        """
        Args:
            mlflow_uri: MLflow tracking URI (e.g., "http://mlflow:5000")
            model_name: Registered model name
            stage: Model stage ("Production", "Staging", "None")
        """
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self.stage = stage
        self.session = None
        self.model_metadata = None
        
        # Initialize MLflow client
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ONNX model from MLflow registry."""
        try:
            # Get latest model version in stage
            model_versions = self.client.search_model_versions(
                f"name='{self.model_name}'"
            )
            
            if not model_versions:
                raise ValueError(f"Model {self.model_name} not found in registry")
            
            # Find version in requested stage
            version = None
            for mv in model_versions:
                if mv.current_stage == self.stage:
                    version = mv
                    break
            
            if version is None:
                raise ValueError(
                    f"Model {self.model_name} not found in stage {self.stage}"
                )
            
            # Download model artifact
            model_uri = f"models:/{self.model_name}/{self.stage}"
            local_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Initialize ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            model_path = f"{local_path}/model.onnx"
            self.session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            
            # Store metadata
            self.model_metadata = {
                "model_name": self.model_name,
                "version": version.version,
                "stage": self.stage,
                "run_id": version.run_id,
                "created_at": str(version.creation_timestamp),
                "status": version.status,
            }
            
            logger.info(
                f"Loaded {self.model_name} v{version.version} "
                f"from {self.stage} stage"
            )
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Run ONNX inference.
        
        Args:
            features: Input features (numpy array, shape depends on model)
        
        Returns:
            Dict with keys:
                - position: (latitude, longitude)
                - uncertainty: (sigma_x, sigma_y, theta)
                - confidence: float (probability of correct prediction)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Ensure correct shape and dtype
            if features.ndim == 1:
                features = features[np.newaxis, ...]
            
            # Get model input/output names
            input_name = self.session.get_inputs()[0].name
            output_names = [out.name for out in self.session.get_outputs()]
            
            # Run inference
            outputs = self.session.run(
                output_names,
                {input_name: features.astype(np.float32)},
            )
            
            # Parse outputs (assuming model outputs: position, uncertainty)
            # Adapt based on actual Phase 5 model output format
            position = outputs[0][0]  # (lat, lon)
            uncertainty = outputs[1][0] if len(outputs) > 1 else None  # (sigma_x, sigma_y)
            confidence = outputs[2][0] if len(outputs) > 2 else 1.0
            
            return {
                "position": {
                    "latitude": float(position[0]),
                    "longitude": float(position[1]),
                },
                "uncertainty": {
                    "sigma_x": float(uncertainty[0]) if uncertainty is not None else 0.0,
                    "sigma_y": float(uncertainty[1]) if uncertainty is not None else 0.0,
                    "theta": float(uncertainty[2]) if len(uncertainty) > 2 else 0.0,
                },
                "confidence": float(confidence),
            }
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def get_metadata(self) -> Dict:
        """Return model metadata."""
        return self.model_metadata or {}
    
    def reload(self) -> None:
        """Reload model from registry (for graceful updates)."""
        logger.info("Reloading model from registry...")
        self._load_model()
```

---

## 🔨 T6.2: services/inference/src/models/schemas.py

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request for single prediction."""
    iq_data: List[List[float]] = Field(..., description="IQ data as [[I1, Q1], [I2, Q2], ...]")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    cache_enabled: bool = Field(True, description="Enable Redis caching")


class UncertaintyResponse(BaseModel):
    """Uncertainty ellipse parameters."""
    sigma_x: float = Field(..., description="Sigma X in meters")
    sigma_y: float = Field(..., description="Sigma Y in meters")
    theta: float = Field(0.0, description="Rotation angle in degrees")
    confidence_interval: float = Field(0.68, description="1-sigma = 68%")


class PositionResponse(BaseModel):
    """Predicted position."""
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""
    position: PositionResponse
    uncertainty: UncertaintyResponse
    confidence: float = Field(..., description="Model confidence 0-1")
    model_version: str = Field(..., description="Model version used")
    inference_time_ms: float = Field(..., description="Inference latency")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    iq_samples: List[List[List[float]]] = Field(..., description="List of IQ data samples")


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse]
    total_time_ms: float
    samples_per_second: float


class ModelInfoResponse(BaseModel):
    """Model information endpoint response."""
    name: str
    version: str
    stage: str
    created_at: datetime
    mlflow_run_id: str
    accuracy_meters: Optional[float] = None
    training_samples: Optional[int] = None
    last_reloaded: datetime
    inference_count: int
    avg_latency_ms: float
    cache_hit_rate: float
```

---

## 🔨 T6.3: services/inference/src/utils/uncertainty.py

```python
import numpy as np
from typing import Dict
import math


def compute_uncertainty_ellipse(
    sigma_x: float,
    sigma_y: float,
    covariance_xy: float = 0.0,
    confidence_level: float = 0.68,  # 1-sigma
) -> Dict:
    """
    Convert (sigma_x, sigma_y) to uncertainty ellipse parameters.
    
    Args:
        sigma_x: Standard deviation in X direction (meters)
        sigma_y: Standard deviation in Y direction (meters)
        covariance_xy: Covariance between X and Y
        confidence_level: Confidence interval (0.68 for 1-sigma)
    
    Returns:
        Dict with ellipse parameters for Mapbox visualization:
        {
            "semi_major_axis": float,    # In meters
            "semi_minor_axis": float,    # In meters
            "rotation_angle": float,     # In degrees
            "confidence_interval": float,
            "geojson_circle": {         # For Mapbox
                "type": "Feature",
                "geometry": {...},
                "properties": {...}
            }
        }
    """
    # Create covariance matrix
    cov_matrix = np.array([
        [sigma_x**2, covariance_xy],
        [covariance_xy, sigma_y**2],
    ])
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Semi-major and semi-minor axes (in meters)
    semi_major = math.sqrt(eigenvalues[0])
    semi_minor = math.sqrt(eigenvalues[1])
    
    # Rotation angle (in degrees)
    rotation_rad = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    rotation_deg = math.degrees(rotation_rad)
    
    return {
        "semi_major_axis": float(semi_major),
        "semi_minor_axis": float(semi_minor),
        "rotation_angle": float(rotation_deg),
        "confidence_interval": float(confidence_level),
        "area_m2": float(math.pi * semi_major * semi_minor),
    }


def ellipse_to_geojson(
    center_lat: float,
    center_lon: float,
    semi_major_m: float,
    semi_minor_m: float,
    rotation_deg: float,
    num_points: int = 64,
) -> Dict:
    """
    Convert ellipse to GeoJSON Feature for Mapbox.
    
    Args:
        center_lat, center_lon: Center point
        semi_major_m: Semi-major axis in meters
        semi_minor_m: Semi-minor axis in meters
        rotation_deg: Rotation angle in degrees
        num_points: Number of points to approximate ellipse
    
    Returns:
        GeoJSON Feature with ellipse geometry
    """
    # Earth radius in meters
    EARTH_RADIUS_M = 6371000
    
    # Convert meters to degrees
    lat_scale = semi_major_m / EARTH_RADIUS_M
    lon_scale = semi_minor_m / (EARTH_RADIUS_M * math.cos(math.radians(center_lat)))
    
    # Generate ellipse points
    angles = np.linspace(0, 2 * np.pi, num_points)
    rotation_rad = math.radians(rotation_deg)
    
    coordinates = []
    for angle in angles:
        x = semi_major_m * math.cos(angle)
        y = semi_minor_m * math.sin(angle)
        
        # Rotate
        x_rot = x * math.cos(rotation_rad) - y * math.sin(rotation_rad)
        y_rot = x * math.sin(rotation_rad) + y * math.cos(rotation_rad)
        
        # Convert to lat/lon
        lat = center_lat + y_rot / EARTH_RADIUS_M
        lon = center_lon + x_rot / (EARTH_RADIUS_M * math.cos(math.radians(center_lat)))
        
        coordinates.append([lon, lat])
    
    # Close polygon
    coordinates.append(coordinates[0])
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        },
        "properties": {
            "name": "Uncertainty Ellipse",
            "semi_major_m": float(semi_major_m),
            "semi_minor_m": float(semi_minor_m),
            "rotation_deg": float(rotation_deg),
        },
    }
```

---

## 🔨 T6.6: services/inference/src/utils/metrics.py

```python
from prometheus_client import Counter, Histogram, Gauge
import time


# Inference metrics
inference_latency = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=[50, 100, 200, 300, 400, 500, 750, 1000],
)

cache_hits = Counter(
    "cache_hits_total",
    "Total cache hits",
)

cache_misses = Counter(
    "cache_misses_total",
    "Total cache misses",
)

requests_total = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint"],
)

errors_total = Counter(
    "inference_errors_total",
    "Total inference errors",
    ["error_type"],
)

model_reloads = Counter(
    "model_reloads_total",
    "Total model reloads",
)

active_requests = Gauge(
    "inference_active_requests",
    "Active inference requests",
)

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Cache hit rate (0-1)",
)


def record_inference_time(duration_ms: float):
    """Record inference latency."""
    inference_latency.observe(duration_ms)


def record_cache_hit():
    """Record cache hit."""
    cache_hits.inc()
    _update_cache_hit_rate()


def record_cache_miss():
    """Record cache miss."""
    cache_misses.inc()
    _update_cache_hit_rate()


def _update_cache_hit_rate():
    """Update cache hit rate gauge."""
    total_hits = cache_hits._value.get()
    total_misses = cache_misses._value.get()
    total = total_hits + total_misses
    if total > 0:
        cache_hit_rate.set(total_hits / total)


class InferenceMetricsContext:
    """Context manager for recording inference metrics."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        active_requests.inc()
        requests_total.labels(endpoint=self.endpoint).inc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        record_inference_time(duration_ms)
        active_requests.dec()
        
        if exc_type is not None:
            errors_total.labels(error_type=exc_type.__name__).inc()
            return False
```

---

## 📚 Reference Implementation Checklist

### T6.1: ONNX Loader
- [ ] Create services/inference/src/models/onnx_loader.py
- [ ] Copy ONNXModelLoader class above
- [ ] Customize output parsing based on Phase 5 model format
- [ ] Add docstrings and logging
- [ ] Create test file

### T6.2: Predict Endpoint
- [ ] Copy schemas.py above to services/inference/src/models/
- [ ] Create services/inference/src/routers/predict.py
- [ ] Implement @app.post("/predict") endpoint
- [ ] Add Redis cache integration
- [ ] Add error handling
- [ ] Create unit tests

### T6.3: Uncertainty
- [ ] Copy uncertainty.py above to services/inference/src/utils/
- [ ] Test ellipse calculations with known values
- [ ] Verify GeoJSON output for Mapbox
- [ ] Create unit tests

### T6.6: Metrics
- [ ] Copy metrics.py above to services/inference/src/utils/
- [ ] Integrate into predict endpoint (context manager)
- [ ] Expose /metrics endpoint in main.py
- [ ] Verify Prometheus scraping
- [ ] Update Prometheus config

---

## 🎯 Next Steps After Setup

1. Create base files structure
2. Implement T6.1 (ONNX Loader)
3. Implement T6.2 (Predict Endpoint)
4. Run unit tests
5. Test with Docker
6. Proceed with remaining tasks

---

**Happy coding! 🚀**

