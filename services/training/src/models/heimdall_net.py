"""
HeimdallNet: Adaptive Multi-Receiver Localization Network

Named after Heimdall, Norse god guardian of Bifrost, known for his 
extraordinary sight and hearing - fitting for a model that "sees" RF 
signals across distributed receivers.

Architecture:
    - Variable-length input (1-10 receivers)
    - Permutation-invariant set processing
    - Learnable per-receiver embeddings (antenna characteristics)
    - Multi-modal fusion (IQ raw + extracted features)
    - Geometry-aware spatial encoding
    - Dual-head output (position + uncertainty)

Key Features:
    - Handles receiver dropout gracefully
    - Shared IQ processing (physics-based)
    - Per-receiver identity embeddings (hardware-specific)
    - Quality-weighted aggregation (SNR-aware)
    - Outstanding accuracy: ±8-15m (target)

Author: Heimdall Team
Created: 2025-11-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Component 1: EfficientNet-B2 1D Encoder for IQ Signals
# ============================================================================

class MBConv1D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for 1D IQ signals."""
    
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        squeeze_channels = max(1, in_channels // 4)
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, squeeze_channels),
            nn.SiLU(inplace=True),
            nn.Linear(squeeze_channels, hidden_dim),
            nn.Sigmoid()
        ])
        
        self.se_block = nn.Sequential(*layers[4:])
        self.conv_block = nn.Sequential(*layers[:4])
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Convolution block
        out = self.conv_block(x)
        
        # SE block
        B, C, L = out.shape
        se_weight = self.se_block(out).view(B, C, 1)
        out = out * se_weight
        
        # Projection
        out = self.project(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        return out


class EfficientNetB2_1D(nn.Module):
    """EfficientNet-B2 adapted for 1D IQ signals."""
    
    def __init__(self, in_channels=2, out_dim=256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks (EfficientNet-B2 configuration scaled for 1D)
        self.blocks = nn.ModuleList([
            # Stage 1: 32 -> 16 channels
            MBConv1D(32, 16, expand_ratio=1, stride=1),
            
            # Stage 2: 16 -> 24 channels
            MBConv1D(16, 24, expand_ratio=6, stride=2),
            MBConv1D(24, 24, expand_ratio=6, stride=1),
            
            # Stage 3: 24 -> 40 channels
            MBConv1D(24, 40, expand_ratio=6, stride=2),
            MBConv1D(40, 40, expand_ratio=6, stride=1),
            
            # Stage 4: 40 -> 80 channels
            MBConv1D(40, 80, expand_ratio=6, stride=2),
            MBConv1D(80, 80, expand_ratio=6, stride=1),
            MBConv1D(80, 80, expand_ratio=6, stride=1),
            
            # Stage 5: 80 -> 112 channels
            MBConv1D(80, 112, expand_ratio=6, stride=1),
            MBConv1D(112, 112, expand_ratio=6, stride=1),
        ])
        
        # Head
        self.head = nn.Sequential(
            nn.Conv1d(112, 192, kernel_size=1, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(192, out_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 2, L) - IQ samples (L = any length, e.g., 1024, 200000)
        Returns:
            features: (batch, out_dim)
        """
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.head(x)


# ============================================================================
# Component 2: Per-Receiver Encoder with Learnable Identity Embeddings
# ============================================================================

class PerReceiverEncoder(nn.Module):
    """
    Encodes each receiver's data with receiver-specific identity embeddings.
    
    Combines:
        - Shared IQ encoder (universal RF physics)
        - Shared feature encoder (universal measurements)
        - Learnable receiver embeddings (antenna characteristics)
        - Optional per-receiver calibration layers
    """
    
    def __init__(self, 
                 max_receivers=10,
                 iq_dim=256, 
                 feature_dim=128,
                 receiver_embed_dim=64,
                 use_calibration=True):
        super().__init__()
        
        self.max_receivers = max_receivers
        self.use_calibration = use_calibration
        
        # Learnable receiver embeddings (captures antenna characteristics)
        self.receiver_embeddings = nn.Embedding(
            num_embeddings=max_receivers,
            embedding_dim=receiver_embed_dim
        )
        
        # Shared IQ processing (physics is universal)
        self.iq_encoder = EfficientNetB2_1D(
            in_channels=2,
            out_dim=iq_dim
        )
        
        # Shared feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(6, 64),  # SNR, PSD, freq_offset, lat, lon, alt
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, feature_dim)
        )
        
        # Receiver-aware fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(iq_dim + feature_dim + receiver_embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # Optional: receiver-specific calibration layers
        if use_calibration:
            self.calibration_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.GELU()
                ) for _ in range(max_receivers)
            ])
    
    def forward(self, iq_raw, features, receiver_id):
        """
        Args:
            iq_raw: (batch, 2, L) - IQ data (L = any length, e.g., 1024, 200000)
            features: (batch, 6) - [SNR, PSD, freq_offset, lat, lon, alt]
            receiver_id: (batch,) - receiver ID (0 to max_receivers-1)
        Returns:
            embedding: (batch, 256) - receiver-aware embedding
        """
        # Get learnable receiver-specific embedding
        receiver_embed = self.receiver_embeddings(receiver_id)  # (B, 64)
        
        # Process IQ with shared encoder
        iq_embed = self.iq_encoder(iq_raw)  # (B, 256)
        
        # Process features with shared encoder
        feat_embed = self.feature_encoder(features)  # (B, 128)
        
        # Fuse: IQ + features + receiver identity
        combined = torch.cat([iq_embed, feat_embed, receiver_embed], dim=-1)
        fused = self.adaptive_fusion(combined)  # (B, 256)
        
        # Optional: per-receiver calibration
        if self.use_calibration:
            batch_size = iq_raw.size(0)
            calibrated = []
            for b in range(batch_size):
                rid = receiver_id[b].item()
                calibrated.append(self.calibration_layers[rid](fused[b:b+1]))
            fused = torch.cat(calibrated, dim=0)
        
        return fused


# ============================================================================
# Component 3: Set Attention Aggregator (Permutation-Invariant)
# ============================================================================

class SetAttentionAggregator(nn.Module):
    """
    Aggregates variable number of receivers with permutation invariance.
    
    Uses multiple pooling strategies:
        - Max pooling (captures strongest signal)
        - Mean pooling (average characteristics)
        - Quality-weighted pooling (SNR-aware attention)
    
    NOTE: Originally used MultiheadAttention but it was unstable with 
    variable masking patterns (NaN issues). Simplified to pure pooling.
    """
    
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Per-receiver transformation (replaces self-attention)
        self.receiver_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Quality-weighted pooling network
        self.quality_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Combine multiple pooling strategies
        self.pool_combiner = nn.Sequential(
            nn.Linear(dim * 3, dim),  # max + mean + weighted
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, receiver_embeddings, mask=None):
        """
        Args:
            receiver_embeddings: (batch, N_receivers, 256)
            mask: (batch, N_receivers) - True = active, False = padding
        Returns:
            aggregated: (batch, 256) - global representation
        """
        logger.error(f"[AGGREGATOR] Input embeddings: min={receiver_embeddings.min():.6e}, max={receiver_embeddings.max():.6e}, has_nan={torch.isnan(receiver_embeddings).any()}")
        
        # Transform each receiver embedding independently
        transformed = self.receiver_transform(receiver_embeddings)  # (B, N, 256)
        logger.error(f"[AGGREGATOR] After transform: min={transformed.min():.6e}, max={transformed.max():.6e}, has_nan={torch.isnan(transformed).any()}")
        
        # Apply mask to embeddings if provided
        # MASK SEMANTICS: True = active receiver, False = padding
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            transformed = transformed * mask_expanded
            logger.error(f"[AGGREGATOR] After masking: min={transformed.min():.6e}, max={transformed.max():.6e}, has_nan={torch.isnan(transformed).any()}")
        
        # Max pooling (captures strongest signal)
        max_pool = torch.max(transformed, dim=1)[0]  # (B, 256)
        logger.error(f"[AGGREGATOR] Max pool: min={max_pool.min():.6e}, max={max_pool.max():.6e}, has_nan={torch.isnan(max_pool).any()}")
        
        # Mean pooling (average characteristics)
        if mask is not None:
            sum_pool = torch.sum(transformed, dim=1)  # (B, 256)
            active_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
            mean_pool = sum_pool / active_count
        else:
            mean_pool = torch.mean(transformed, dim=1)
        logger.error(f"[AGGREGATOR] Mean pool: min={mean_pool.min():.6e}, max={mean_pool.max():.6e}, has_nan={torch.isnan(mean_pool).any()}")
        
        # Quality-weighted pooling
        quality_scores = self.quality_net(transformed)  # (B, N, 1)
        logger.error(f"[AGGREGATOR] Quality scores (raw): min={quality_scores.min():.6e}, max={quality_scores.max():.6e}, has_nan={torch.isnan(quality_scores).any()}")
        
        if mask is not None:
            # Mask out padding before softmax (True = active, False = padding)
            quality_scores = quality_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        quality_weights = F.softmax(quality_scores, dim=1)  # (B, N, 1)
        logger.error(f"[AGGREGATOR] Quality weights: min={quality_weights.min():.6e}, max={quality_weights.max():.6e}, has_nan={torch.isnan(quality_weights).any()}")
        
        weighted_pool = torch.sum(transformed * quality_weights, dim=1)  # (B, 256)
        logger.error(f"[AGGREGATOR] Weighted pool: min={weighted_pool.min():.6e}, max={weighted_pool.max():.6e}, has_nan={torch.isnan(weighted_pool).any()}")
        
        # Combine all three pooling strategies
        combined = torch.cat([max_pool, mean_pool, weighted_pool], dim=-1)  # (B, 768)
        logger.error(f"[AGGREGATOR] Combined: min={combined.min():.6e}, max={combined.max():.6e}, has_nan={torch.isnan(combined).any()}")
        
        result = self.pool_combiner(combined)  # (B, 256)
        logger.error(f"[AGGREGATOR] Final result: min={result.min():.6e}, max={result.max():.6e}, has_nan={torch.isnan(result).any()}")
        
        return result


# ============================================================================
# Component 4: Geometry Encoder (Spatial Awareness)
# ============================================================================

class GeometryEncoder(nn.Module):
    """
    Encodes spatial geometry of receiver network.
    
    Captures:
        - Pairwise distances between receivers
        - Relative bearings (angles)
        - Altitude differences
        - Network topology features
    """
    
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()
        
        # Pairwise distance encoder
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 64)
        )
        
        # Angle encoder (bearing between receivers)
        self.angle_encoder = nn.Sequential(
            nn.Linear(2, 32),  # cos(θ), sin(θ)
            nn.GELU(),
            nn.Linear(32, 64)
        )
        
        # Altitude difference encoder
        self.altitude_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # Aggregate geometry information
        self.geometry_aggregator = nn.Sequential(
            nn.Linear(160, dim),  # 64 + 64 + 32
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, positions, mask=None):
        """
        Args:
            positions: (batch, N, 3) - [lat, lon, alt] for each receiver
            mask: (batch, N) - active receivers
        Returns:
            geometry_embedding: (batch, dim)
        """
        B, N, _ = positions.shape
        
        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)  # (B, N, 1, 3)
        pos_j = positions.unsqueeze(1)  # (B, 1, N, 3)
        
        # Euclidean distance (lat, lon in degrees → approximate km)
        # Simplified: treat as Euclidean (proper geodesic for production)
        distances = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)  # (B, N, N, 1)
        
        # Compute bearings (angle from receiver i to receiver j)
        delta = pos_j - pos_i  # (B, N, N, 3)
        angles = torch.atan2(delta[..., 1], delta[..., 0])  # (B, N, N)
        angle_features = torch.stack([
            torch.cos(angles), 
            torch.sin(angles)
        ], dim=-1)  # (B, N, N, 2)
        
        # Altitude differences
        alt_diff = delta[..., 2:3]  # (B, N, N, 1)
        
        # Encode features
        dist_embed = self.distance_encoder(distances)  # (B, N, N, 64)
        angle_embed = self.angle_encoder(angle_features)  # (B, N, N, 64)
        alt_embed = self.altitude_encoder(alt_diff)  # (B, N, N, 32)
        
        # Concatenate
        geometry_feat = torch.cat([dist_embed, angle_embed, alt_embed], dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
            geometry_feat = geometry_feat * pair_mask.unsqueeze(-1).float()
            
            # Mean pooling over valid pairs
            sum_geom = geometry_feat.sum(dim=(1, 2))
            count = pair_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
            geometry_embedding = sum_geom / count.squeeze(-1)
        else:
            # Mean pooling (permutation-invariant)
            geometry_embedding = geometry_feat.mean(dim=(1, 2))
        
        return self.geometry_aggregator(geometry_embedding)


# ============================================================================
# Component 5: Main HeimdallNet Model
# ============================================================================

class HeimdallNet(nn.Module):
    """
    HeimdallNet: Multi-modal adaptive localization network.
    
    Architecture:
        1. Per-receiver encoding (IQ + features + identity)
        2. Set aggregation (permutation-invariant)
        3. Geometry encoding (spatial awareness)
        4. Global fusion
        5. Dual-head output (position + uncertainty)
    
    Key Features:
        - Variable receivers (1-10)
        - Receiver dropout resilient
        - Permutation invariant
        - Multi-modal fusion
        - Uncertainty quantification
    
    Target Performance:
        - Accuracy: ±8-15m (68% confidence)
        - Inference: 40-60ms
        - Parameters: ~25M
        - VRAM: 6-8GB training, 1.8GB inference
    """
    
    def __init__(self,
                 max_receivers=10,
                 iq_dim=256,
                 feature_dim=128,
                 receiver_embed_dim=64,
                 hidden_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 use_calibration=True):
        super().__init__()
        
        self.max_receivers = max_receivers
        
        # Component 1: Per-receiver encoder
        self.receiver_encoder = PerReceiverEncoder(
            max_receivers=max_receivers,
            iq_dim=iq_dim,
            feature_dim=feature_dim,
            receiver_embed_dim=receiver_embed_dim,
            use_calibration=use_calibration
        )
        
        # Component 2: Set aggregation
        self.set_aggregator = SetAttentionAggregator(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Component 3: Geometry encoder
        self.geometry_encoder = GeometryEncoder(
            dim=hidden_dim,
            dropout=dropout
        )
        
        # Component 4: Global fusion
        self.global_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # aggregated + geometry
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Component 5: Dual-head output
        self.position_head = nn.Linear(hidden_dim, 2)  # (x, y)
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # (σ_x, σ_y)
            nn.Softplus()  # Ensure positive uncertainties
        )
    
    def forward(self, 
                iq_data: torch.Tensor,
                features: torch.Tensor,
                positions: torch.Tensor,
                receiver_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            iq_data: (batch, N_receivers, 2, L) - IQ samples (L = any length)
            features: (batch, N_receivers, 6) - [SNR, PSD, freq, lat, lon, alt]
            positions: (batch, N_receivers, 3) - [lat, lon, alt]
            receiver_ids: (batch, N_receivers) - receiver IDs (0 to max_receivers-1)
            mask: (batch, N_receivers) - True = active, False = padding
        
        Returns:
            pred_position: (batch, 2) - predicted (x, y)
            pred_uncertainty: (batch, 2) - predicted (σ_x, σ_y)
        """
        B, N, _, _ = iq_data.shape
        
        # DEBUG: Check inputs
        logger.error(f"[FORWARD] Input iq_data: min={iq_data.min():.6e}, max={iq_data.max():.6e}, has_nan={torch.isnan(iq_data).any()}")
        logger.error(f"[FORWARD] Input features: min={features.min():.6e}, max={features.max():.6e}, has_nan={torch.isnan(features).any()}")
        logger.error(f"[FORWARD] Input positions: min={positions.min():.6e}, max={positions.max():.6e}, has_nan={torch.isnan(positions).any()}")
        
        # Encode each receiver (shared weights + identity embeddings)
        receiver_embeddings = []
        for i in range(N):
            embed = self.receiver_encoder(
                iq_data[:, i],      # (B, 2, L)
                features[:, i],     # (B, 6)
                receiver_ids[:, i]  # (B,)
            )
            # DEBUG: Check each embedding
            if torch.isnan(embed).any():
                logger.error(f"[FORWARD] NaN detected in receiver {i} embedding!")
                logger.error(f"[FORWARD] Receiver {i} iq_data: min={iq_data[:, i].min():.6e}, max={iq_data[:, i].max():.6e}")
                logger.error(f"[FORWARD] Receiver {i} features: {features[:, i]}")
            receiver_embeddings.append(embed)
        
        receiver_embeddings = torch.stack(receiver_embeddings, dim=1)  # (B, N, 256)
        logger.error(f"[FORWARD] Receiver embeddings: min={receiver_embeddings.min():.6e}, max={receiver_embeddings.max():.6e}, has_nan={torch.isnan(receiver_embeddings).any()}")
        
        # Aggregate receivers (permutation-invariant)
        aggregated = self.set_aggregator(receiver_embeddings, mask)  # (B, 256)
        logger.error(f"[FORWARD] Aggregated: min={aggregated.min():.6e}, max={aggregated.max():.6e}, has_nan={torch.isnan(aggregated).any()}")
        
        # Encode geometry
        geometry = self.geometry_encoder(positions, mask)  # (B, 256)
        logger.error(f"[FORWARD] Geometry: min={geometry.min():.6e}, max={geometry.max():.6e}, has_nan={torch.isnan(geometry).any()}")
        
        # Global fusion
        global_repr = self.global_fusion(
            torch.cat([aggregated, geometry], dim=-1)
        )  # (B, 256)
        logger.error(f"[FORWARD] Global repr: min={global_repr.min():.6e}, max={global_repr.max():.6e}, has_nan={torch.isnan(global_repr).any()}")
        
        # Dual-head output
        pred_position = self.position_head(global_repr)
        pred_uncertainty = self.uncertainty_head(global_repr)
        
        logger.error(f"[FORWARD] Pred position: min={pred_position.min():.6e}, max={pred_position.max():.6e}, has_nan={torch.isnan(pred_position).any()}")
        logger.error(f"[FORWARD] Pred uncertainty: min={pred_uncertainty.min():.6e}, max={pred_uncertainty.max():.6e}, has_nan={torch.isnan(pred_uncertainty).any()}")
        
        return pred_position, pred_uncertainty


# ============================================================================
# Factory Function
# ============================================================================

def create_heimdall_net(
    max_receivers: int = 10,
    use_calibration: bool = True,
    dropout: float = 0.1
) -> HeimdallNet:
    """
    Factory function to create HeimdallNet model.
    
    Args:
        max_receivers: Maximum number of receivers (default: 10)
        use_calibration: Use per-receiver calibration layers (default: True)
        dropout: Dropout probability (default: 0.1)
    
    Returns:
        HeimdallNet model instance
    
    Example:
        >>> model = create_heimdall_net(max_receivers=7)
        >>> # Input shapes (L = IQ sample length, flexible)
        >>> iq = torch.randn(4, 3, 2, 200000)    # batch=4, receivers=3, 200k samples
        >>> feats = torch.randn(4, 3, 6)
        >>> pos = torch.randn(4, 3, 3)
        >>> ids = torch.tensor([[0, 2, 4]] * 4)   # SDR IDs: 0, 2, 4
        >>> mask = torch.ones(4, 3, dtype=torch.bool)
        >>> 
        >>> # Forward pass
        >>> pred_pos, pred_unc = model(iq, feats, pos, ids, mask)
        >>> print(pred_pos.shape)  # (4, 2)
        >>> print(pred_unc.shape)  # (4, 2)
    """
    return HeimdallNet(
        max_receivers=max_receivers,
        iq_dim=256,
        feature_dim=128,
        receiver_embed_dim=64,
        hidden_dim=256,
        num_heads=8,
        dropout=dropout,
        use_calibration=use_calibration
    )


# ============================================================================
# Utility: Count Parameters
# ============================================================================

def count_parameters(model: nn.Module) -> dict:
    """Count trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Breakdown by component
    breakdown = {}
    for name, module in model.named_children():
        breakdown[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'total_millions': total / 1e6,
        'breakdown': breakdown
    }


if __name__ == '__main__':
    # Test model instantiation and forward pass
    print("🌉 HeimdallNet Test")
    print("=" * 60)
    
    # Create model
    model = create_heimdall_net(max_receivers=10)
    
    # Count parameters
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total_millions']:.2f}M")
    print("\nBreakdown:")
    for name, count in param_info['breakdown'].items():
        print(f"  {name}: {count/1e6:.2f}M")
    
    # Test forward pass with variable receivers
    print("\n" + "=" * 60)
    print("Testing with 3 active receivers (typical scenario):")
    
    batch_size = 4
    num_receivers = 3
    
    # Create dummy inputs
    iq_data = torch.randn(batch_size, num_receivers, 2, 1024)
    features = torch.randn(batch_size, num_receivers, 6)
    positions = torch.randn(batch_size, num_receivers, 3)
    receiver_ids = torch.tensor([[0, 2, 4]] * batch_size)  # SDRs 0, 2, 4 active
    mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred_pos, pred_unc = model(iq_data, features, positions, receiver_ids, mask)
    
    print(f"✅ Input shapes:")
    print(f"   IQ data: {iq_data.shape}")
    print(f"   Features: {features.shape}")
    print(f"   Positions: {positions.shape}")
    print(f"   Receiver IDs: {receiver_ids.shape}")
    print(f"\n✅ Output shapes:")
    print(f"   Predicted position: {pred_pos.shape}")
    print(f"   Predicted uncertainty: {pred_unc.shape}")
    print(f"\n✅ Sample predictions:")
    print(f"   Position: {pred_pos[0].numpy()}")
    print(f"   Uncertainty: {pred_unc[0].numpy()}")
    
    print("\n" + "=" * 60)
    print("👁️ HeimdallNet ready for training!")
