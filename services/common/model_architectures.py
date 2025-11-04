"""
Model Architecture Registry for Heimdall Training System

Central registry of all available model architectures with metadata:
- Display name (shown in UI)
- Data type requirement (feature_based, iq_raw, or both)
- Description
- Default hyperparameters

This registry is shared between backend API and training service.
"""

from typing import Literal, TypedDict


class ArchitectureMetadata(TypedDict):
    """Metadata for a model architecture."""
    display_name: str
    data_type: Literal["feature_based", "iq_raw", "both"]
    description: str
    default_params: dict


# Complete registry of all available architectures
MODEL_ARCHITECTURES: dict[str, ArchitectureMetadata] = {
    # Feature-based models (use extracted features: mel-spec, MFCC, etc.)
    "triangulation": {
        "display_name": "Triangulation Network (Features)",
        "data_type": "feature_based",
        "description": "Attention-based model using extracted RF features (SNR, PSD, frequency offset)",
        "default_params": {
            "embed_dim": 32,
            "num_heads": 4,
            "dropout": 0.2,
        }
    },
    
    "localization_net": {
        "display_name": "ConvNeXt Localization (Features)",
        "data_type": "feature_based",
        "description": "ConvNeXt-Large backbone with mel-spectrogram features (200M params, state-of-the-art)",
        "default_params": {
            "pretrained": True,
            "freeze_backbone": False,
            "backbone_size": "large",
        }
    },
    
    # IQ-raw models (process raw IQ samples directly)
    "iq_resnet18": {
        "display_name": "IQ ResNet-18 (IQ only)",
        "data_type": "iq_raw",
        "description": "ResNet-18 adapted for raw IQ samples with attention aggregation over receivers",
        "default_params": {
            "max_receivers": 10,
            "iq_sequence_length": 1024,
            "embedding_dim": 128,
            "dropout": 0.3,
        }
    },
    
    "iq_vggnet": {
        "display_name": "IQ VGG-Style (IQ only)",
        "data_type": "iq_raw",
        "description": "VGG-style CNN for IQ samples, simpler and faster training than ResNet",
        "default_params": {
            "max_receivers": 10,
            "iq_sequence_length": 1024,
            "embedding_dim": 128,
            "dropout": 0.3,
        }
    },
}


def get_architecture_metadata(architecture_name: str) -> ArchitectureMetadata:
    """
    Get metadata for a model architecture.
    
    Args:
        architecture_name: Architecture identifier
    
    Returns:
        Architecture metadata
    
    Raises:
        KeyError: If architecture not found
    """
    if architecture_name not in MODEL_ARCHITECTURES:
        available = list(MODEL_ARCHITECTURES.keys())
        raise KeyError(
            f"Unknown architecture: {architecture_name}. "
            f"Available: {available}"
        )
    
    return MODEL_ARCHITECTURES[architecture_name]


def list_architectures(data_type: str | None = None) -> list[dict]:
    """
    List all available architectures, optionally filtered by data type.
    
    Args:
        data_type: Filter by data type ('feature_based', 'iq_raw', or None for all)
    
    Returns:
        List of architectures with their metadata
    """
    architectures = []
    
    for name, metadata in MODEL_ARCHITECTURES.items():
        # Filter by data type if specified
        if data_type and metadata["data_type"] != data_type and metadata["data_type"] != "both":
            continue
        
        architectures.append({
            "name": name,
            **metadata
        })
    
    return architectures


def is_compatible(architecture_name: str, dataset_type: str) -> bool:
    """
    Check if an architecture is compatible with a dataset type.
    
    Args:
        architecture_name: Architecture identifier
        dataset_type: Dataset type ('feature_based' or 'iq_raw')
    
    Returns:
        True if compatible, False otherwise
    """
    try:
        metadata = get_architecture_metadata(architecture_name)
        arch_data_type = metadata["data_type"]
        
        # 'both' is compatible with everything
        if arch_data_type == "both":
            return True
        
        # Exact match
        return arch_data_type == dataset_type
    
    except KeyError:
        return False
