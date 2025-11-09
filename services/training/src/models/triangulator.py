"""
Triangulation model with attention mechanism for RF source localization.

Architecture:
- ReceiverEncoder: Encodes per-receiver measurements (SNR, PSD, freq_offset, position, signal_present)
- AttentionAggregator: Multi-head attention over receiver embeddings with masking
- TriangulationHead: Predicts [lat, lon, log_variance]

Input: Variable-length list of receiver measurements
Output: [latitude, longitude, log_variance]

Loss: Gaussian Negative Log-Likelihood (penalizes overconfidence)
"""

import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)

# Constants
MAX_RECEIVERS = 7  # Maximum number of WebSDR receivers in the network
DEGREES_TO_KM = 111.0  # Approximate conversion (1 degree latitude ≈ 111 km)


class ReceiverEncoder(nn.Module):
    """
    Encode per-receiver measurements into fixed-dimensional embeddings.
    
    Input per receiver: [snr, psd, freq_offset, rx_lat, rx_lon, signal_present] (6D)
    Output per receiver: 32D embedding
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 32, dropout: float = 0.2):
        """
        Initialize receiver encoder.
        
        Args:
            input_dim: Input feature dimension (default: 6)
            hidden_dim: Hidden layer dimension (default: 64)
            output_dim: Output embedding dimension (default: 32)
            dropout: Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(
            "ReceiverEncoder initialized",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode receiver measurements.
        
        Args:
            x: Receiver features (batch_size, num_receivers, input_dim)
        
        Returns:
            Receiver embeddings (batch_size, num_receivers, output_dim)
        """
        # x shape: (batch_size, num_receivers, input_dim)
        # Flatten batch and receiver dimensions for linear layer
        batch_size, num_receivers, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # (batch_size * num_receivers, input_dim)
        
        # Encode
        embeddings_flat = self.encoder(x_flat)  # (batch_size * num_receivers, output_dim)
        
        # Reshape back
        embeddings = embeddings_flat.view(batch_size, num_receivers, -1)  # (batch_size, num_receivers, output_dim)
        
        return embeddings


class AttentionAggregator(nn.Module):
    """
    Multi-head attention aggregator over receiver embeddings.
    
    Learns to weight receivers based on geometry and signal quality.
    Masking ensures failed receivers (signal_present=0) don't contribute.
    """
    
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize attention aggregator.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) format
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(
            "AttentionAggregator initialized",
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate receiver embeddings with attention.
        
        Args:
            x: Receiver embeddings (batch_size, num_receivers, embed_dim)
            mask: Boolean mask (batch_size, num_receivers) where True = masked (no signal)
        
        Returns:
            Aggregated features (batch_size, embed_dim * 4)
        """
        # x shape: (batch_size, num_receivers, embed_dim)
        # mask shape: (batch_size, num_receivers) - True means mask out (no signal)
        
        # Ensure mask is boolean for ONNX compatibility
        # (during tracing, masks may come in as float)
        mask = mask.bool()
        
        # Multi-head attention
        # key_padding_mask: True values are masked (ignored in attention)
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,  # (batch_size, num_receivers)
            need_weights=True
        )
        
        # Residual connection + layer norm
        attn_output = self.layer_norm(attn_output + x)
        
        # Aggregate: mean pooling over receivers (ignoring masked receivers)
        # Expand mask to match embedding dimension
        # Explicitly maintain boolean dtype for ONNX compatibility
        mask_expanded = mask.unsqueeze(-1).expand_as(attn_output).bool()  # (batch_size, num_receivers, embed_dim)
        
        # Replace masked positions with zeros
        attn_output_masked = attn_output.masked_fill(mask_expanded, 0.0)
        
        # Count non-masked receivers per batch
        num_valid = (~mask).sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        num_valid = torch.clamp(num_valid, min=1.0)  # Avoid division by zero
        
        # Mean pooling
        mean_features = attn_output_masked.sum(dim=1) / num_valid.expand(-1, attn_output_masked.size(2))  # (batch_size, embed_dim)
        
        # Max pooling (reuse mask_expanded which is already boolean)
        attn_output_masked_max = attn_output_masked.masked_fill(mask_expanded, -1e9)
        max_features, _ = torch.max(attn_output_masked_max, dim=1)  # (batch_size, embed_dim)
        
        # Attention weights statistics
        # Explicitly maintain boolean dtype for ONNX compatibility
        attn_weights_mask = mask.unsqueeze(1).expand_as(attn_weights).bool()
        attn_weights_masked = attn_weights.masked_fill(attn_weights_mask, 0.0)
        attn_std = torch.std(attn_weights_masked, dim=-1).mean(dim=1)  # (batch_size,)
        
        # Concatenate: [mean, max, std_repeated, num_valid_normalized]
        std_expanded = attn_std.unsqueeze(-1).expand(-1, mean_features.size(1))  # (batch_size, embed_dim)
        num_valid_normalized = (num_valid / MAX_RECEIVERS).expand(-1, mean_features.size(1))  # (batch_size, embed_dim)
        
        aggregated = torch.cat([mean_features, max_features, std_expanded, num_valid_normalized], dim=1)  # (batch_size, embed_dim * 4)
        
        return aggregated


class TriangulationHead(nn.Module):
    """
    Prediction head for triangulation.
    
    Input: Aggregated receiver features (embed_dim * 4)
    Output: [latitude, longitude, log_variance]
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.2):
        """
        Initialize triangulation head.
        
        Args:
            input_dim: Input feature dimension (embed_dim * 4)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # [lat, lon, log_variance]
        )
        
        logger.info(
            "TriangulationHead initialized",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict position and uncertainty.
        
        Args:
            x: Aggregated features (batch_size, input_dim)
        
        Returns:
            Tuple of (position, log_variance)
            - position: (batch_size, 2) - [latitude, longitude]
            - log_variance: (batch_size, 1) - log of positional variance
        """
        output = self.head(x)  # (batch_size, 3)
        
        position = output[:, :2]  # (batch_size, 2) - [lat, lon]
        log_variance = output[:, 2:3]  # (batch_size, 1) - log(variance)
        
        return position, log_variance


class TriangulationModel(nn.Module):
    """
    Complete triangulation model with attention.
    
    Architecture:
    1. ReceiverEncoder: 6D → 32D per receiver
    2. AttentionAggregator: Attention over receivers → 128D
    3. TriangulationHead: 128D → [lat, lon, log_var]
    """
    
    def __init__(
        self,
        encoder_input_dim: int = 6,
        encoder_hidden_dim: int = 64,
        encoder_output_dim: int = 32,
        attention_heads: int = 4,
        head_hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize triangulation model.
        
        Args:
            encoder_input_dim: Input feature dimension per receiver
            encoder_hidden_dim: Encoder hidden dimension
            encoder_output_dim: Encoder output embedding dimension
            attention_heads: Number of attention heads
            head_hidden_dim: Triangulation head hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = ReceiverEncoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )
        
        self.aggregator = AttentionAggregator(
            embed_dim=encoder_output_dim,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        self.head = TriangulationHead(
            input_dim=encoder_output_dim * 4,  # mean + max + std + num_valid
            hidden_dim=head_hidden_dim,
            dropout=dropout
        )
        
        logger.info(
            "TriangulationModel initialized",
            total_params=f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M",
            encoder_params=f"{sum(p.numel() for p in self.encoder.parameters())/1e3:.1f}K",
            aggregator_params=f"{sum(p.numel() for p in self.aggregator.parameters())/1e3:.1f}K",
            head_params=f"{sum(p.numel() for p in self.head.parameters())/1e3:.1f}K"
        )
    
    def forward(self, receiver_features: torch.Tensor, signal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            receiver_features: (batch_size, num_receivers, 6)
                Features: [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
            signal_mask: (batch_size, num_receivers)
                Boolean mask where True = no signal (masked out)
        
        Returns:
            Tuple of (position, log_variance)
            - position: (batch_size, 2) - [latitude, longitude]
            - log_variance: (batch_size, 1) - log of positional variance
        """
        # Encode receivers
        embeddings = self.encoder(receiver_features)  # (batch_size, num_receivers, encoder_output_dim)
        
        # Aggregate with attention
        aggregated = self.aggregator(embeddings, signal_mask)  # (batch_size, encoder_output_dim * 4)
        
        # Predict position and uncertainty
        position, log_variance = self.head(aggregated)
        
        return position, log_variance


def gaussian_nll_loss(
    predicted_position: torch.Tensor,
    log_variance: torch.Tensor,
    true_position: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood loss for localization with uncertainty.
    
    Loss = 0.5 * (mse / variance + log_variance)
    
    This loss:
    - Penalizes large errors (MSE term)
    - Penalizes overconfidence when errors are large (variance term)
    - Encourages calibrated uncertainty estimates
    
    Args:
        predicted_position: (batch_size, 2) - [lat, lon]
        log_variance: (batch_size, 1) - log of positional variance
        true_position: (batch_size, 2) - ground truth [lat, lon]
    
    Returns:
        Scalar loss
    """
    # Calculate squared error
    mse = torch.sum((predicted_position - true_position) ** 2, dim=1, keepdim=True)  # (batch_size, 1)
    
    # Clamp log_variance to prevent exp() explosion (numerical stability)
    # Range: [-10, 10] → variance range: [4.5e-5, 22026]
    # This prevents NaN from exp(large_number) while allowing reasonable uncertainty range
    log_variance = torch.clamp(log_variance, min=-10.0, max=10.0)
    
    # Convert log_variance to variance
    variance = torch.exp(log_variance)  # (batch_size, 1)
    
    # Add epsilon to variance for numerical stability (prevent division by zero)
    epsilon = 1e-6
    variance = variance + epsilon
    
    # Gaussian NLL
    nll = 0.5 * (mse / variance + log_variance)
    
    # Mean over batch
    return nll.mean()


def haversine_distance_torch(lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Haversine distance in PyTorch (for metrics).
    
    Args:
        lat1, lon1: (batch_size,) - predicted coordinates (degrees)
        lat2, lon2: (batch_size,) - true coordinates (degrees)
    
    Returns:
        Distance in meters (batch_size,) - SI unit
    """
    R = 6371000.0  # Earth radius in meters (SI unit)
    
    # Convert to radians
    lat1_rad = torch.deg2rad(lat1)
    lat2_rad = torch.deg2rad(lat2)
    delta_lat = torch.deg2rad(lat2 - lat1)
    delta_lon = torch.deg2rad(lon2 - lon1)
    
    # Haversine formula
    a = (
        torch.sin(delta_lat / 2) ** 2
        + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(delta_lon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    distance = R * c
    return distance
