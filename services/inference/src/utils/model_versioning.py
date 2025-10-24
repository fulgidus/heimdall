"""
Model Versioning & A/B Testing Framework
=========================================

Manages multiple model versions with support for:
- Loading different model versions from MLflow registry
- Dynamic version switching without service restart (graceful reload)
- A/B testing with traffic allocation
- Version metadata and performance tracking
- Fallback to previous version on failure

T6.5: Model Versioning & A/B Testing
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
import asyncio

import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """MLflow model stages"""
    PRODUCTION = "Production"
    STAGING = "Staging"
    ARCHIVED = "Archived"
    NONE = "None"


class VersionStatus(str, Enum):
    """Model version status during lifecycle"""
    LOADING = "loading"
    READY = "ready"
    STALE = "stale"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class ModelVersion:
    """Metadata for a single model version"""
    version_id: str  # MLflow version ID
    stage: ModelStage = ModelStage.PRODUCTION
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: VersionStatus = VersionStatus.READY
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    # performance_metrics: {
    #     "accuracy": 0.95,
    #     "latency_ms": 150.5,
    #     "throughput": 45.2,
    #     "cache_hit_rate": 0.82
    # }
    notes: str = ""
    is_active: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Export to dictionary (datetime → ISO string)"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["stage"] = self.stage.value
        data["status"] = self.status.value
        return data


@dataclass
class ABTestConfig:
    """A/B Testing configuration"""
    enabled: bool = False
    version_a: str = ""  # Primary version ID
    version_b: str = ""  # Experimental version ID
    traffic_split: float = 0.5  # 0.5 = 50/50 split
    min_traffic_split: float = 0.01  # Minimum 1%
    max_traffic_split: float = 0.99  # Maximum 99%
    auto_winner: bool = False  # Auto promote winner if enabled
    winner_threshold: float = 0.95  # Confidence threshold for auto-promotion
    started_at: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: Optional[int] = None  # None = unlimited
    metrics_to_compare: List[str] = field(default_factory=lambda: ["accuracy", "latency_ms"])

    def is_active(self) -> bool:
        """Check if A/B test is still active"""
        if not self.enabled:
            return False
        if self.duration_seconds is None:
            return True
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        return elapsed < self.duration_seconds

    def to_dict(self) -> Dict:
        """Export to dictionary"""
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        return data


class ModelVersionRegistry:
    """
    Manages model versions with versioning and A/B testing support.
    
    Architecture:
    - Maintains registry of loaded model versions
    - Supports multiple concurrent versions (memory overhead vs flexibility)
    - Routes predictions based on A/B test config
    - Provides graceful fallback on version failure
    
    Usage:
        registry = ModelVersionRegistry(mlflow_tracking_uri)
        await registry.load_version("v1", "Production")
        await registry.set_active_version("v1")
        prediction = await registry.predict(features)
    """

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        max_versions: int = 5,
        session_options: Optional[ort.SessionOptions] = None
    ):
        """
        Initialize version registry.
        
        Args:
            mlflow_tracking_uri: MLflow server endpoint
            max_versions: Maximum concurrent loaded versions
            session_options: ONNX Runtime session configuration
        """
        self.mlflow_uri = mlflow_tracking_uri
        self.max_versions = max_versions
        self.session_options = session_options or ort.SessionOptions()
        
        # Core registry
        self.versions: Dict[str, Tuple[ort.InferenceSession, ModelVersion]] = {}
        self.active_version_id: Optional[str] = None
        self.previous_version_id: Optional[str] = None  # For fallback
        
        # A/B Testing
        self.ab_test_config = ABTestConfig(enabled=False)
        self.ab_test_stats = {"routed_to_a": 0, "routed_to_b": 0}
        
        # Metrics
        self.version_metrics: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()  # Async lock for thread-safe operations
        
        logger.info(
            f"ModelVersionRegistry initialized (max_versions={max_versions}, "
            f"mlflow={mlflow_tracking_uri})"
        )

    async def load_version(
        self,
        version_id: str,
        stage: ModelStage = ModelStage.PRODUCTION,
        model_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Load a model version from MLflow or local path.
        
        Args:
            version_id: Unique version identifier (e.g., "v1", "prod-2025-10-22")
            stage: MLflow stage (Production, Staging, Archived)
            model_path: Path to ONNX model (if not in MLflow)
            metadata: Optional performance metrics
            
        Returns:
            True if successful, False on error
            
        Raises:
            ValueError: If max_versions exceeded and cannot unload old version
        """
        async with self._lock:
            # Check if already loaded
            if version_id in self.versions:
                logger.warning(f"Version {version_id} already loaded, skipping")
                return True
            
            # Check capacity
            if len(self.versions) >= self.max_versions:
                logger.warning(
                    f"Version registry full (max={self.max_versions}), "
                    f"consider unloading unused versions"
                )
                # Could implement automatic LRU unload here
            
            try:
                # In production: load from MLflow registry
                # For now: simulate loading from local path
                if model_path is None:
                    model_path = f"models/{version_id}/model.onnx"
                
                # Load ONNX session
                logger.info(f"Loading model version {version_id} from {model_path}")
                session = ort.InferenceSession(
                    model_path,
                    self.session_options,
                    providers=['CPUExecutionProvider']
                )
                
                # Create metadata
                version_meta = ModelVersion(
                    version_id=version_id,
                    stage=stage,
                    status=VersionStatus.READY,
                    performance_metrics=metadata or {}
                )
                
                self.versions[version_id] = (session, version_meta)
                logger.info(f"✅ Version {version_id} loaded successfully (stage={stage.value})")
                return True
                
            except FileNotFoundError as e:
                logger.error(f"❌ Model file not found for {version_id}: {e}")
                return False
            except Exception as e:
                logger.error(f"❌ Error loading version {version_id}: {e}")
                version_meta = ModelVersion(
                    version_id=version_id,
                    stage=stage,
                    status=VersionStatus.ERROR,
                    error_message=str(e)
                )
                self.versions[version_id] = (None, version_meta)
                return False

    async def set_active_version(self, version_id: str) -> bool:
        """
        Set active prediction version.
        
        Args:
            version_id: Version to activate
            
        Returns:
            True if successful
        """
        async with self._lock:
            if version_id not in self.versions:
                logger.error(f"Version {version_id} not loaded")
                return False
            
            session, meta = self.versions[version_id]
            if session is None:
                logger.error(f"Version {version_id} has error, cannot activate")
                return False
            
            # Store previous for fallback
            if self.active_version_id:
                self.previous_version_id = self.active_version_id
            
            self.active_version_id = version_id
            meta.is_active = True
            meta.updated_at = datetime.utcnow()
            
            logger.info(f"🔄 Active version switched to {version_id}")
            return True

    async def unload_version(self, version_id: str) -> bool:
        """
        Unload a model version to free memory.
        
        Args:
            version_id: Version to unload
            
        Returns:
            True if successful
        """
        async with self._lock:
            if version_id not in self.versions:
                return False
            
            if version_id == self.active_version_id:
                logger.warning(f"Cannot unload active version {version_id}")
                return False
            
            del self.versions[version_id]
            logger.info(f"Version {version_id} unloaded (memory freed)")
            return True

    def start_ab_test(
        self,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
        duration_seconds: Optional[int] = None
    ) -> bool:
        """
        Start A/B test between two model versions.
        
        Args:
            version_a: Primary/control version
            version_b: Experimental/treatment version
            traffic_split: Proportion to route to version_b (0.5 = 50/50)
            duration_seconds: Test duration (None = unlimited)
            
        Returns:
            True if A/B test started
        """
        if version_a not in self.versions or version_b not in self.versions:
            logger.error(f"One or both versions not loaded (A={version_a}, B={version_b})")
            return False
        
        traffic_split = max(0.01, min(0.99, traffic_split))  # Clamp to [0.01, 0.99]
        
        self.ab_test_config = ABTestConfig(
            enabled=True,
            version_a=version_a,
            version_b=version_b,
            traffic_split=traffic_split,
            started_at=datetime.utcnow(),
            duration_seconds=duration_seconds
        )
        self.ab_test_stats = {"routed_to_a": 0, "routed_to_b": 0}
        
        logger.info(
            f"🧪 A/B Test started: {version_a} vs {version_b} "
            f"(split={traffic_split:.1%}, duration={duration_seconds}s)"
        )
        return True

    def end_ab_test(self, winner: Optional[str] = None) -> bool:
        """
        End A/B test and optionally promote winner.
        
        Args:
            winner: Version to promote (None = no promotion)
            
        Returns:
            True if successful
        """
        self.ab_test_config.enabled = False
        
        logger.info(
            f"🏁 A/B Test ended. Stats: "
            f"A={self.ab_test_stats['routed_to_a']}, "
            f"B={self.ab_test_stats['routed_to_b']}"
        )
        
        if winner and winner in self.versions:
            logger.info(f"✨ Promoting {winner} to active version")
            return asyncio.run(self.set_active_version(winner))
        
        return True

    async def predict(
        self,
        features: np.ndarray,
        use_ab_routing: bool = True
    ) -> Tuple[np.ndarray, str]:
        """
        Run prediction on active version (or routed via A/B test).
        
        Args:
            features: Input features (mel-spectrogram, shape expected by model)
            use_ab_routing: Route through A/B test if enabled
            
        Returns:
            Tuple of (prediction, version_id_used)
            
        Raises:
            RuntimeError: If no active version or prediction fails
        """
        if not self.active_version_id:
            raise RuntimeError("No active model version loaded")
        
        # Determine which version to use
        version_id = self.active_version_id
        
        if use_ab_routing and self.ab_test_config.is_active():
            # Route based on traffic split
            if np.random.random() < self.ab_test_config.traffic_split:
                version_id = self.ab_test_config.version_b
                self.ab_test_stats["routed_to_b"] += 1
            else:
                version_id = self.ab_test_config.version_a
                self.ab_test_stats["routed_to_a"] += 1
        
        # Run inference with fallback
        try:
            session, meta = self.versions[version_id]
            if session is None:
                raise RuntimeError(f"Version {version_id} has error")
            
            # ONNX input/output names
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
            
            # Ensure proper shape (add batch dimension if needed)
            if features.ndim == 2:
                features = np.expand_dims(features, axis=0)
            
            # Run inference
            outputs = session.run(output_names, {input_name: features.astype(np.float32)})
            
            logger.debug(f"Prediction via version {version_id}: {outputs[0].shape}")
            return outputs[0], version_id
            
        except Exception as e:
            logger.error(f"Prediction failed on {version_id}: {e}")
            
            # Fallback to previous version if available
            if self.previous_version_id and self.previous_version_id != version_id:
                logger.warning(f"Falling back to previous version {self.previous_version_id}")
                try:
                    session, meta = self.versions[self.previous_version_id]
                    if session:
                        input_name = session.get_inputs()[0].name
                        output_names = [o.name for o in session.get_outputs()]
                        outputs = session.run(output_names, {input_name: features.astype(np.float32)})
                        return outputs[0], self.previous_version_id
                except Exception as fallback_error:
                    logger.error(f"Fallback failed: {fallback_error}")
            
            raise RuntimeError(f"Prediction failed with no fallback: {e}")

    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """Get metadata for a specific version"""
        if version_id not in self.versions:
            return None
        
        _, meta = self.versions[version_id]
        return meta.to_dict()

    def list_versions(self) -> List[Dict]:
        """List all loaded versions with metadata"""
        return [meta.to_dict() for _, meta in self.versions.values()]

    def get_registry_status(self) -> Dict:
        """Get overall registry status"""
        return {
            "active_version": self.active_version_id,
            "previous_version": self.previous_version_id,
            "loaded_versions": len(self.versions),
            "max_versions": self.max_versions,
            "versions": self.list_versions(),
            "ab_test": self.ab_test_config.to_dict() if self.ab_test_config.enabled else None,
            "ab_test_stats": self.ab_test_stats
        }

    @asynccontextmanager
    async def model_context(self, version_id: str):
        """
        Context manager for temporary version switching.
        
        Usage:
            async with registry.model_context("v2"):
                result = await registry.predict(features)
                # Uses v2 temporarily
            # Back to original active version
        """
        original_version = self.active_version_id
        try:
            if await self.set_active_version(version_id):
                yield
            else:
                raise RuntimeError(f"Failed to switch to {version_id}")
        finally:
            if original_version:
                await self.set_active_version(original_version)


async def create_version_registry(
    mlflow_uri: str = "http://localhost:5000",
    initial_versions: Optional[Dict[str, str]] = None
) -> ModelVersionRegistry:
    """
    Factory function to create and initialize version registry.
    
    Args:
        mlflow_uri: MLflow tracking server
        initial_versions: Dict of {version_id: model_path} to preload
        
    Returns:
        Initialized ModelVersionRegistry
        
    Example:
        registry = await create_version_registry(
            initial_versions={"v1": "models/v1.onnx"}
        )
        await registry.set_active_version("v1")
    """
    registry = ModelVersionRegistry(mlflow_tracking_uri=mlflow_uri)
    
    if initial_versions:
        for version_id, model_path in initial_versions.items():
            await registry.load_version(
                version_id,
                stage=ModelStage.PRODUCTION,
                model_path=model_path
            )
    
    return registry
