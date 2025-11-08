"""
Model Factory for Heimdall Training Pipeline

This module provides a factory function to instantiate any model from MODEL_REGISTRY
with proper parameter handling and validation.

Usage:
    from src.models.model_factory import create_model_from_registry
    
    model = create_model_from_registry(
        model_id="heimdall_net",
        max_receivers=7,
        dropout=0.1
    )
"""

import structlog
import torch.nn as nn
from typing import Optional, Any

from .model_registry import MODEL_REGISTRY, get_model_info
from .localization_net import LocalizationNet, LocalizationNetViT
from .iq_cnn_models import IQResNet18, IQResNet50, IQResNet101, IQVGGNet
from .iq_efficientnet import IQEfficientNetB4
from .iq_transformer import IQTransformer
from .iq_wavenet import IQWaveNet
from .hybrid_models import IQHybridNet
from .heimdall_net import HeimdallNet, create_heimdall_net
from .triangulator import TriangulationModel
# LocalizationEnsembleFlagship imported inline in factory to avoid circular imports

logger = structlog.get_logger(__name__)


def create_model_from_registry(
    model_id: str,
    max_receivers: int = 7,
    dropout: float = 0.1,
    **kwargs: Any
) -> nn.Module:
    """
    Factory function to create any model from MODEL_REGISTRY.
    
    Args:
        model_id: Model identifier from MODEL_REGISTRY (e.g., "heimdall_net", "iq_resnet18")
        max_receivers: Maximum number of receivers (default: 7)
        dropout: Dropout rate (default: 0.1)
        **kwargs: Additional model-specific parameters
    
    Returns:
        Initialized PyTorch model ready for training
    
    Raises:
        ValueError: If model_id not found in registry
        TypeError: If model creation fails due to invalid parameters
    
    Examples:
        >>> # Create HeimdallNet
        >>> model = create_model_from_registry("heimdall_net", max_receivers=7)
        
        >>> # Create IQ Transformer with custom params
        >>> model = create_model_from_registry(
        ...     "iq_transformer",
        ...     max_receivers=10,
        ...     num_heads=8,
        ...     num_layers=6
        ... )
        
        >>> # Create Triangulation MLP
        >>> model = create_model_from_registry("triangulation_model", max_receivers=7)
    """
    
    # Validate model_id
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_id}' not found in registry. "
            f"Available models: {available}"
        )
    
    model_info = get_model_info(model_id)
    logger.info(
        f"Creating model from registry",
        model_id=model_id,
        display_name=model_info.display_name,
        data_type=model_info.data_type,
        architecture_type=model_info.architecture_type
    )
    
    # ------------------------------------------------------------------------
    # SPECTROGRAM-BASED MODELS (🖼️)
    # ------------------------------------------------------------------------
    if model_id == "localization_net_convnext_large":
        model = LocalizationNet(
            backbone_size="large",  # Fixed: use backbone_size not backbone_name
            pretrained=kwargs.get("pretrained", True),
            freeze_backbone=kwargs.get("freeze_backbone", False),
            uncertainty_min=kwargs.get("uncertainty_min", 0.01),
            uncertainty_max=kwargs.get("uncertainty_max", 1.0)
        )
    
    elif model_id == "localization_net_vit":
        model = LocalizationNetViT(
            pretrained=kwargs.get("pretrained", True),
            freeze_backbone=kwargs.get("freeze_backbone", False),
            uncertainty_min=kwargs.get("uncertainty_min", 0.01),
            uncertainty_max=kwargs.get("uncertainty_max", 1.0),
            model_size=kwargs.get("model_size", "b_16")  # Fixed: use model_size
        )
    
    # ------------------------------------------------------------------------
    # IQ-RAW CNN MODELS (📡)
    # ------------------------------------------------------------------------
    elif model_id == "iq_resnet18":
        model = IQResNet18(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            embedding_dim=kwargs.get("embedding_dim", 128),
            dropout=dropout
        )
    
    elif model_id == "iq_resnet50":
        model = IQResNet50(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            embedding_dim=kwargs.get("embedding_dim", 128),  # Fixed: default is 128
            dropout=dropout
        )
    
    elif model_id == "iq_resnet101":
        model = IQResNet101(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            embedding_dim=kwargs.get("embedding_dim", 128),  # Fixed: default is 128
            dropout=dropout
        )
    
    elif model_id == "iq_vggnet":
        model = IQVGGNet(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            embedding_dim=kwargs.get("embedding_dim", 128),
            dropout=dropout
        )
    
    # ------------------------------------------------------------------------
    # IQ EFFICIENTNET (📡)
    # ------------------------------------------------------------------------
    elif model_id == "iq_efficientnet_b4":
        model = IQEfficientNetB4(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            embedding_dim=kwargs.get("embedding_dim", 192),
            dropout=dropout
        )
    
    # ------------------------------------------------------------------------
    # IQ TRANSFORMER (🧠)
    # ------------------------------------------------------------------------
    elif model_id == "iq_transformer":
        model = IQTransformer(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            patch_size=kwargs.get("patch_size", 64),
            embed_dim=kwargs.get("embed_dim", 768),  # Fixed: use embed_dim not d_model
            num_layers=kwargs.get("num_layers", 12),  # Fixed: default is 12
            num_heads=kwargs.get("num_heads", 12),  # Fixed: default is 12
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            dropout=dropout,
            uncertainty_min=kwargs.get("uncertainty_min", 0.01),
            uncertainty_max=kwargs.get("uncertainty_max", 1.0)
        )
    
    # ------------------------------------------------------------------------
    # IQ TEMPORAL (🌊)
    # ------------------------------------------------------------------------
    elif model_id == "iq_wavenet":
        model = IQWaveNet(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            hidden_channels=kwargs.get("hidden_channels", 128),  # Fixed: add hidden_channels
            num_layers=kwargs.get("num_layers", 10),
            kernel_size=kwargs.get("kernel_size", 3),
            attention_heads=kwargs.get("attention_heads", 8),  # Fixed: add attention_heads
            dropout=dropout,
            uncertainty_min=kwargs.get("uncertainty_min", 0.01),
            uncertainty_max=kwargs.get("uncertainty_max", 1.0)
        )
    
    # ------------------------------------------------------------------------
    # HYBRID MODELS (🔬)
    # ------------------------------------------------------------------------
    elif model_id == "iq_hybrid_net":
        model = IQHybridNet(
            max_receivers=max_receivers,
            iq_sequence_length=kwargs.get("iq_sequence_length", 1024),
            embed_dim=kwargs.get("embed_dim", 512),  # Fixed: use embed_dim
            num_transformer_heads=kwargs.get("num_transformer_heads", 8),  # Fixed: correct param name
            num_transformer_layers=kwargs.get("num_transformer_layers", 6),  # Fixed: default is 6
            dropout=dropout,
            uncertainty_min=kwargs.get("uncertainty_min", 0.01),
            uncertainty_max=kwargs.get("uncertainty_max", 1.0)
        )
    
    # ------------------------------------------------------------------------
    # MULTI-MODAL MODELS (👁️)
    # ------------------------------------------------------------------------
    elif model_id == "heimdall_net":
        model = create_heimdall_net(
            max_receivers=max_receivers,
            use_calibration=kwargs.get("use_calibration", True),
            dropout=dropout
        )
    
    # ------------------------------------------------------------------------
    # FEATURE-BASED MODELS (🧮)
    # ------------------------------------------------------------------------
    elif model_id == "triangulation_model":
        model = TriangulationModel(
            encoder_input_dim=kwargs.get("encoder_input_dim", 6),
            encoder_hidden_dim=kwargs.get("encoder_hidden_dim", 64),
            encoder_output_dim=kwargs.get("encoder_output_dim", 32),
            attention_heads=kwargs.get("attention_heads", 4),
            head_hidden_dim=kwargs.get("head_hidden_dim", 64),
            dropout=dropout
        )
    
    # ------------------------------------------------------------------------
    # ENSEMBLE MODELS (🏆)
    # ------------------------------------------------------------------------
    elif model_id == "ensemble_flagship":
        # Fixed: Use LocalizationEnsembleFlagship with correct params
        from .ensemble_flagship import LocalizationEnsembleFlagship
        model = LocalizationEnsembleFlagship(
            max_receivers=max_receivers,
            iq_seq_len=kwargs.get("iq_sequence_length", 1024),
            use_pretrained=kwargs.get("use_pretrained", False),
            freeze_base_models=kwargs.get("freeze_base_models", False)
        )
    
    else:
        # Should never reach here due to validation above
        raise ValueError(f"Model creation not implemented for: {model_id}")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(
        f"Model created successfully",
        model_id=model_id,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        parameters_millions=round(total_params / 1e6, 2)
    )
    
    return model


def get_model_input_requirements(model_id: str) -> dict[str, bool]:
    """
    Get the input data requirements for a given model.
    
    Args:
        model_id: Model identifier from MODEL_REGISTRY
    
    Returns:
        Dictionary indicating which data types are required:
        {
            "iq_raw": bool,
            "spectrogram": bool,
            "features": bool,
            "positions": bool,
            "receiver_ids": bool,
            "mask": bool
        }
    
    Examples:
        >>> reqs = get_model_input_requirements("heimdall_net")
        >>> print(reqs)
        {'iq_raw': True, 'features': True, 'positions': True, ...}
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    
    model_info = get_model_info(model_id)
    data_type = model_info.data_type
    
    # Map data types to requirements
    requirements = {
        "iq_raw": False,
        "spectrogram": False,
        "features": False,
        "positions": False,
        "receiver_ids": False,
        "mask": False,
    }
    
    if data_type == "spectrogram":
        requirements["spectrogram"] = True
    
    elif data_type == "iq_raw":
        requirements["iq_raw"] = True
        requirements["mask"] = True  # Most IQ models support variable receivers
    
    elif data_type == "features":
        requirements["features"] = True
        requirements["positions"] = True  # Features include receiver positions
        requirements["mask"] = True
    
    elif data_type == "hybrid":
        requirements["iq_raw"] = True
        requirements["features"] = True
        requirements["mask"] = True
    
    elif data_type == "multi_modal":
        # HeimdallNet requires all data types
        requirements["iq_raw"] = True
        requirements["features"] = True
        requirements["positions"] = True
        requirements["receiver_ids"] = True
        requirements["mask"] = True
    
    return requirements


def validate_model_config(model_id: str, config: dict) -> tuple[bool, Optional[str]]:
    """
    Validate that a training config is compatible with a given model.
    
    Args:
        model_id: Model identifier from MODEL_REGISTRY
        config: Training configuration dictionary
    
    Returns:
        (is_valid, error_message): Tuple of validation result and optional error message
    
    Examples:
        >>> config = {"batch_size": 32, "max_receivers": 7}
        >>> is_valid, error = validate_model_config("heimdall_net", config)
        >>> if not is_valid:
        ...     print(f"Invalid config: {error}")
    """
    if model_id not in MODEL_REGISTRY:
        return False, f"Model '{model_id}' not found in registry"
    
    model_info = get_model_info(model_id)
    
    # Check batch size
    recommended_batch = model_info.recommended_batch_size
    actual_batch = config.get("batch_size", 32)
    
    if actual_batch > recommended_batch * 4:
        logger.warning(
            f"Batch size {actual_batch} is much larger than recommended "
            f"{recommended_batch} for {model_id}. May cause OOM errors."
        )
    
    # Check max_receivers for IQ-based models
    if model_info.data_type in ["iq_raw", "hybrid", "multi_modal"]:
        max_receivers = config.get("max_receivers", 7)
        if max_receivers > 10:
            return False, "max_receivers cannot exceed 10 (system limitation)"
        if max_receivers < 1:
            return False, "max_receivers must be at least 1"
    
    # All checks passed
    return True, None


__all__ = [
    "create_model_from_registry",
    "get_model_input_requirements",
    "validate_model_config",
]
