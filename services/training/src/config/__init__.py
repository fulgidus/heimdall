"""Configuration package for training service."""

from .model_config import ModelConfig, BackboneArchitecture
from ..config import settings

__all__ = ["ModelConfig", "BackboneArchitecture", "settings"]
