"""
IQ-Raw CNN Models for RF Source Localization

These models process raw IQ samples directly (without feature extraction).
Suitable for training on iq_raw datasets with random receiver geometry.

Architecture families:
- IQResNet: Residual networks adapted for complex IQ data
- IQVGGNet: VGG-style architecture for IQ processing
- IQEfficientNet: Efficient convolutional networks for IQ data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)


class IQResNet18(nn.Module):
    """
    ResNet-18 adapted for raw IQ samples from multiple receivers.
    
    Input: (batch_size, num_receivers, 2, seq_len)
        - num_receivers: variable (5-10 for iq_raw datasets)
        - 2 channels: I and Q components
        - seq_len: IQ sample sequence length (default: 1024)
    
    Output: (batch_size, 4)
        - [latitude, longitude, sigma_x, sigma_y]
    
    Architecture:
    - Separate CNN branch per receiver (shared weights)
    - ResNet-18 backbone for each branch
    - Attention aggregation over receiver embeddings
    - Dual-head output (position + uncertainty)
    """
    
    def __init__(
        self,
        max_receivers: int = 10,
        iq_sequence_length: int = 1024,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.max_receivers = max_receivers
        self.iq_sequence_length = iq_sequence_length
        self.embedding_dim = embedding_dim
        
        # Per-receiver IQ encoder (ResNet-18 style)
        # Input: (batch * num_rx, 2, seq_len)
        self.iq_encoder = nn.Sequential(
            # Conv block 1
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_residual_block(64, 64, stride=1),
            self._make_residual_block(64, 128, stride=2),
            self._make_residual_block(128, 256, stride=2),
            self._make_residual_block(256, embedding_dim, stride=2),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Attention aggregation over receivers
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Position head
        self.position_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),  # *3 for [mean, max, attention]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # [lat, lon]
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # [sigma_x, sigma_y]
        )
        
        logger.info(
            "IQResNet18 initialized (IQ only)",
            max_receivers=max_receivers,
            iq_sequence_length=iq_sequence_length,
            embedding_dim=embedding_dim,
            data_type="iq_raw"
        )
    
    def _make_residual_block(self, in_channels, out_channels, stride):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
        )
    
    def forward(self, iq_samples: torch.Tensor, receiver_mask: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            iq_samples: (batch, max_receivers, 2, seq_len) - IQ samples
            receiver_mask: (batch, max_receivers) - True = mask out (no signal)
        
        Returns:
            positions: (batch, 2) - [lat, lon]
            uncertainties: (batch, 2) - [sigma_x, sigma_y]
        """
        batch_size, num_receivers, _, seq_len = iq_samples.shape
        
        # Encode each receiver's IQ samples
        # Flatten batch and receiver dimensions
        iq_flat = iq_samples.view(-1, 2, seq_len)  # (batch * num_rx, 2, seq_len)
        embeddings_flat = self.iq_encoder(iq_flat)  # (batch * num_rx, embed_dim, 1)
        embeddings_flat = embeddings_flat.squeeze(-1)  # (batch * num_rx, embed_dim)
        
        # Reshape back to (batch, num_rx, embed_dim)
        embeddings = embeddings_flat.view(batch_size, num_receivers, -1)
        
        # Attention aggregation with masking
        attn_out, _ = self.attention(
            embeddings, embeddings, embeddings,
            key_padding_mask=receiver_mask  # True = ignore
        )
        
        # Aggregate: mean + max pooling
        mask_expanded = receiver_mask.unsqueeze(-1).expand_as(attn_out)
        attn_masked = attn_out.masked_fill(mask_expanded, 0.0)
        
        mean_features = attn_masked.sum(dim=1) / (~receiver_mask).sum(dim=1, keepdim=True).float()
        
        attn_masked_max = attn_out.masked_fill(mask_expanded, -1e9)
        max_features, _ = attn_masked_max.max(dim=1)
        
        # Attention statistics
        attn_std = torch.std(attn_masked, dim=1)
        
        # Concatenate features
        aggregated = torch.cat([mean_features, max_features, attn_std], dim=-1)
        
        # Predictions
        positions = self.position_head(aggregated)
        uncertainties = F.softplus(self.uncertainty_head(aggregated))
        uncertainties = torch.clamp(uncertainties, min=0.01, max=1.0)
        
        return positions, uncertainties


class IQVGGNet(nn.Module):
    """
    VGG-style architecture for IQ samples.
    
    Simpler than ResNet, faster training, good baseline.
    Input/output same as IQResNet18.
    """
    
    def __init__(
        self,
        max_receivers: int = 10,
        iq_sequence_length: int = 1024,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.max_receivers = max_receivers
        self.iq_sequence_length = iq_sequence_length
        
        # VGG-style encoder for IQ samples
        self.iq_encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Simplified aggregation (mean + max pooling, no attention)
        # Position head
        self.position_head = nn.Sequential(
            nn.Linear(256 * 2, 256),  # *2 for mean + max
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        logger.info(
            "IQVGGNet initialized (IQ only)",
            max_receivers=max_receivers,
            iq_sequence_length=iq_sequence_length,
            data_type="iq_raw"
        )
    
    def forward(self, iq_samples: torch.Tensor, receiver_mask: torch.Tensor) -> tuple:
        """Forward pass (same signature as IQResNet18)."""
        batch_size, num_receivers, _, seq_len = iq_samples.shape
        
        # Encode IQ samples
        iq_flat = iq_samples.view(-1, 2, seq_len)
        embeddings_flat = self.iq_encoder(iq_flat)
        embeddings_flat = embeddings_flat.squeeze(-1)
        embeddings = embeddings_flat.view(batch_size, num_receivers, -1)
        
        # Simple aggregation: mean + max pooling
        mask_expanded = receiver_mask.unsqueeze(-1).expand_as(embeddings)
        embeddings_masked = embeddings.masked_fill(mask_expanded, 0.0)
        
        mean_features = embeddings_masked.sum(dim=1) / (~receiver_mask).sum(dim=1, keepdim=True).float()
        
        embeddings_masked_max = embeddings.masked_fill(mask_expanded, -1e9)
        max_features, _ = embeddings_masked_max.max(dim=1)
        
        aggregated = torch.cat([mean_features, max_features], dim=-1)
        
        # Predictions
        positions = self.position_head(aggregated)
        uncertainties = F.softplus(self.uncertainty_head(aggregated))
        uncertainties = torch.clamp(uncertainties, min=0.01, max=1.0)
        
        return positions, uncertainties


# Model registry for easy instantiation
IQ_MODEL_REGISTRY = {
    "iq_resnet18": {
        "class": IQResNet18,
        "display_name": "IQ ResNet-18 (IQ only)",
        "data_type": "iq_raw",
        "description": "ResNet-18 adapted for raw IQ samples with attention aggregation"
    },
    "iq_vggnet": {
        "class": IQVGGNet,
        "display_name": "IQ VGG-Style (IQ only)",
        "data_type": "iq_raw",
        "description": "VGG-style CNN for IQ samples, faster training than ResNet"
    },
}


def get_iq_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create IQ models.
    
    Args:
        model_name: Model name from IQ_MODEL_REGISTRY
        **kwargs: Model-specific parameters
    
    Returns:
        Instantiated model
    """
    if model_name not in IQ_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown IQ model: {model_name}. "
            f"Available: {list(IQ_MODEL_REGISTRY.keys())}"
        )
    
    model_info = IQ_MODEL_REGISTRY[model_name]
    model_class = model_info["class"]
    
    return model_class(**kwargs)
