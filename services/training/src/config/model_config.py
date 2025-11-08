"""
Model configuration and selection utilities.

Provides convenient factory functions and configs for different backbone architectures.
Allows easy experimentation with different models without changing core code.
"""

from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class BackboneArchitecture(Enum):
    """Available backbone architectures for LocalizationNet."""

    # ConvNeXt variants (RECOMMENDED - Modern 2022 architecture)
    CONVNEXT_TINY = "convnext_tiny"  # 29M params, lightweight
    CONVNEXT_SMALL = "convnext_small"  # 50M params, balanced
    CONVNEXT_MEDIUM = "convnext_base"  # 89M params, good accuracy
    CONVNEXT_LARGE = "convnext_large"  # 200M params, best accuracy ⭐

    # ResNet variants (Traditional, well-tested)
    RESNET_50 = "resnet50"  # 26M params, conservative upgrade
    RESNET_101 = "resnet101"  # 45M params, heavier

    # Vision Transformers (Experimental, better long-range dependencies)
    VIT_BASE = "vit_b_16"  # 86M params, transformer-based
    VIT_LARGE = "vit_l_16"  # 306M params, very large

    # EfficientNet (Balanced, good for edge deployment)
    EFFICIENTNET_B3 = "efficientnet_b3"  # 12M params, lightweight
    EFFICIENTNET_B4 = "efficientnet_b4"  # 19M params, balanced


@dataclass
class ModelConfig:
    """Configuration for LocalizationNet."""

    backbone: BackboneArchitecture = BackboneArchitecture.CONVNEXT_LARGE
    pretrained: bool = True
    freeze_backbone: bool = False
    uncertainty_min: float = 0.01
    uncertainty_max: float = 1.0

    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    num_training_steps: int = 10000

    # Data parameters
    n_mels: int = 128
    n_frames: int = 32

    def __post_init__(self):
        """Validate configuration."""
        if self.uncertainty_min < 0 or self.uncertainty_max < self.uncertainty_min:
            raise ValueError("Invalid uncertainty bounds")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay")


# Predefined configurations for common use cases
CONFIGS = {
    # Production configs
    "production_high_accuracy": ModelConfig(
        backbone=BackboneArchitecture.CONVNEXT_LARGE,
        pretrained=True,
        freeze_backbone=False,
        learning_rate=1e-3,
        num_training_steps=50000,
    ),
    "production_balanced": ModelConfig(
        backbone=BackboneArchitecture.CONVNEXT_MEDIUM,
        pretrained=True,
        freeze_backbone=False,
        learning_rate=1e-3,
        num_training_steps=30000,
    ),
    "production_lightweight": ModelConfig(
        backbone=BackboneArchitecture.EFFICIENTNET_B4,
        pretrained=True,
        freeze_backbone=False,
        learning_rate=1e-3,
        num_training_steps=20000,
    ),
    # Development/testing configs
    "dev_fast": ModelConfig(
        backbone=BackboneArchitecture.CONVNEXT_SMALL,
        pretrained=True,
        freeze_backbone=True,  # Frozen backbone for faster training
        learning_rate=5e-4,
        num_training_steps=5000,
    ),
    "dev_test": ModelConfig(
        backbone=BackboneArchitecture.EFFICIENTNET_B3,
        pretrained=False,  # No pretraining for quick tests
        freeze_backbone=False,
        learning_rate=1e-3,
        num_training_steps=1000,
    ),
    # Experimental configs
    "experimental_vit": ModelConfig(
        backbone=BackboneArchitecture.VIT_BASE,
        pretrained=True,
        freeze_backbone=False,
        learning_rate=1e-4,  # ViT typically needs lower LR
        num_training_steps=50000,
    ),
    "experimental_resnet": ModelConfig(
        backbone=BackboneArchitecture.RESNET_101,
        pretrained=True,
        freeze_backbone=False,
        learning_rate=1e-3,
        num_training_steps=30000,
    ),
}


def get_model_config(config_name: str) -> ModelConfig:
    """
    Get a predefined model configuration by name.

    Args:
        config_name (str): Name of the configuration

    Returns:
        ModelConfig: Configuration object

    Raises:
        KeyError: If config_name not found
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise KeyError(f"Unknown config '{config_name}'. Available: {available}")

    config = CONFIGS[config_name]
    logger.info(
        "model_config_loaded",
        config_name=config_name,
        backbone=config.backbone.value,
        learning_rate=config.learning_rate,
    )

    return config


def get_backbone_info(backbone: BackboneArchitecture) -> dict:
    """
    Get information about a backbone architecture.

    Returns:
        Dict with: params_millions, imagenet_top1, inference_ms, vram_gb
    """
    info_map = {
        # ConvNeXt (Modern 2022 architecture)
        BackboneArchitecture.CONVNEXT_TINY: {
            "params_millions": 29,
            "imagenet_top1": "81.9%",
            "inference_ms": 15,
            "vram_gb": 4,
            "description": "Lightweight, fast",
        },
        BackboneArchitecture.CONVNEXT_SMALL: {
            "params_millions": 50,
            "imagenet_top1": "83.6%",
            "inference_ms": 20,
            "vram_gb": 6,
            "description": "Balanced speed/accuracy",
        },
        BackboneArchitecture.CONVNEXT_MEDIUM: {
            "params_millions": 89,
            "imagenet_top1": "86.2%",
            "inference_ms": 30,
            "vram_gb": 8,
            "description": "Very good accuracy",
        },
        BackboneArchitecture.CONVNEXT_LARGE: {
            "params_millions": 200,
            "imagenet_top1": "88.6%",
            "inference_ms": 45,
            "vram_gb": 12,
            "description": "Best accuracy ⭐ RECOMMENDED",
        },
        # ResNet (Traditional, well-tested)
        BackboneArchitecture.RESNET_50: {
            "params_millions": 26,
            "imagenet_top1": "76.1%",
            "inference_ms": 25,
            "vram_gb": 8,
            "description": "Well-tested, conservative",
        },
        BackboneArchitecture.RESNET_101: {
            "params_millions": 45,
            "imagenet_top1": "77.4%",
            "inference_ms": 35,
            "vram_gb": 10,
            "description": "Larger ResNet",
        },
        # Vision Transformers (Experimental)
        BackboneArchitecture.VIT_BASE: {
            "params_millions": 86,
            "imagenet_top1": "84.1%",
            "inference_ms": 55,
            "vram_gb": 10,
            "description": "Transformer-based, good long-range",
        },
        BackboneArchitecture.VIT_LARGE: {
            "params_millions": 306,
            "imagenet_top1": "85.9%",
            "inference_ms": 90,
            "vram_gb": 16,
            "description": "Very large, max accuracy",
        },
        # EfficientNet (Balanced)
        BackboneArchitecture.EFFICIENTNET_B3: {
            "params_millions": 12,
            "imagenet_top1": "81.6%",
            "inference_ms": 15,
            "vram_gb": 4,
            "description": "Lightweight, good balance",
        },
        BackboneArchitecture.EFFICIENTNET_B4: {
            "params_millions": 19,
            "imagenet_top1": "83.4%",
            "inference_ms": 20,
            "vram_gb": 6,
            "description": "Balanced efficiency",
        },
    }

    if backbone not in info_map:
        raise ValueError(f"Unknown backbone: {backbone}")

    return info_map[backbone]


def print_backbone_comparison():
    """Print a comparison table of all available backbones."""

    print("\n" + "=" * 100)
    print("BACKBONE ARCHITECTURE COMPARISON")
    print("=" * 100)
    print(
        f"\n{'Architecture':<20} {'Params (M)':<12} {'ImageNet':<12} {'Inference':<12} {'VRAM':<8} {'Description':<30}"
    )
    print("-" * 100)

    for backbone in BackboneArchitecture:
        info = get_backbone_info(backbone)
        print(
            f"{backbone.value:<20} {info['params_millions']:<12} {info['imagenet_top1']:<12} "
            f"{info['inference_ms']}ms{'':<8} {info['vram_gb']}GB{'':<4} {info['description']:<30}"
        )

    print("=" * 100)
    print("\n⭐ RECOMMENDED for your hardware: ConvNeXt-Large")
    print("   - 200M params, 88.6% ImageNet, 45ms inference")
    print("   - Best accuracy with acceptable speed")
    print("   - Fits comfortably in RTX 3090 (12GB VRAM needed)\n")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print_backbone_comparison()

    # Test loading a config
    config = get_model_config("production_high_accuracy")
    logger.info("Config loaded successfully", config=config)
