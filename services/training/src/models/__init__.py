"""
Heimdall ML Model Architectures.

This module exports all available model architectures for RF source localization.

Available Models:
- Spectrogram-based: LocalizationNet, LocalizationNetViT
- IQ-Raw CNNs: IQResNet18, IQResNet50, IQResNet101, IQVGGNet
- IQ EfficientNet: IQEfficientNet
- IQ Transformer: IQTransformer
- IQ Temporal: IQWaveNet
- Hybrid: IQHybridNet
- Feature-based: TriangulationMLP

Registry:
- MODEL_REGISTRY: Complete registry with all 11 architectures
- model_registry utilities for querying and comparison
"""

# Spectrogram-based models
from .localization_net import LocalizationNet, LocalizationNetViT

# IQ-Raw CNN models
from .iq_cnn_models import (
    IQResNet18,
    IQResNet50,
    IQResNet101,
    IQVGGNet,
    IQ_MODEL_REGISTRY,
    get_iq_model,
)

# IQ EfficientNet
from .iq_efficientnet import IQEfficientNetB4 as IQEfficientNet

# IQ Transformer
from .iq_transformer import IQTransformer

# IQ Temporal models
from .iq_wavenet import IQWaveNet

# Hybrid models
from .hybrid_models import IQHybridNet

# Model registry
from .model_registry import (
    MODEL_REGISTRY,
    ModelArchitectureInfo,
    PerformanceMetrics,
    get_model_info,
    list_models,
    compare_models,
    get_recommended_model,
    get_models_by_badge,
    get_all_model_ids,
    get_model_count,
    model_info_to_dict,
)

__all__ = [
    # Spectrogram-based
    "LocalizationNet",
    "LocalizationNetViT",
    
    # IQ-Raw CNNs
    "IQResNet18",
    "IQResNet50",
    "IQResNet101",
    "IQVGGNet",
    
    # IQ Advanced
    "IQEfficientNet",
    "IQTransformer",
    "IQWaveNet",
    
    # Hybrid
    "IQHybridNet",
    
    # Registry
    "MODEL_REGISTRY",
    "IQ_MODEL_REGISTRY",
    "ModelArchitectureInfo",
    "PerformanceMetrics",
    
    # Utilities
    "get_model_info",
    "get_iq_model",
    "list_models",
    "compare_models",
    "get_recommended_model",
    "get_models_by_badge",
    "get_all_model_ids",
    "get_model_count",
    "model_info_to_dict",
]
