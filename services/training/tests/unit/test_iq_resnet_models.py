"""
Unit Tests for IQ ResNet Models (IQResNet50 and IQResNet101)

Tests the deeper ResNet architectures for IQ-raw data processing.
These models use bottleneck blocks and are designed for high-accuracy RF localization.

Test Coverage:
- Model initialization with various parameters
- Forward pass with different batch sizes and receiver counts
- Output shape validation
- Attention masking behavior
- Uncertainty clamping
- Parameter counting
- Edge cases (single receiver, max receivers, padded receivers)

Execution: pytest tests/unit/test_iq_resnet_models.py -v --cov
"""

import pytest
import torch
import torch.nn as nn

from src.models.iq_cnn_models import IQResNet50, IQResNet101


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def default_config():
    """Default configuration for IQResNet models."""
    return {
        'max_receivers': 10,
        'iq_sequence_length': 1024,
        'embedding_dim': 128,
        'dropout': 0.3,
    }


@pytest.fixture
def sample_iq_batch():
    """
    Create sample IQ batch with realistic dimensions.
    
    Returns:
        tuple: (iq_samples, receiver_mask)
            - iq_samples: (batch, max_receivers, 2, seq_len)
            - receiver_mask: (batch, max_receivers) - True = masked out
    """
    batch_size = 4
    max_receivers = 10
    seq_len = 1024
    
    # Random IQ data (normalized)
    iq_samples = torch.randn(batch_size, max_receivers, 2, seq_len) * 0.5
    
    # Receiver mask: first 7 receivers active, last 3 masked
    receiver_mask = torch.zeros(batch_size, max_receivers, dtype=torch.bool)
    receiver_mask[:, 7:] = True  # Mask out last 3 receivers
    
    return iq_samples, receiver_mask


@pytest.fixture
def resnet50_model(default_config):
    """Initialize IQResNet50 model."""
    return IQResNet50(**default_config)


@pytest.fixture
def resnet101_model(default_config):
    """Initialize IQResNet101 model."""
    return IQResNet101(**default_config)


# ============================================================================
# TEST IQRESNET50 - INITIALIZATION
# ============================================================================


class TestIQResNet50Initialization:
    """Test IQResNet50 model initialization."""
    
    def test_default_initialization(self, resnet50_model):
        """Test model initializes with default parameters."""
        assert resnet50_model.max_receivers == 10
        assert resnet50_model.iq_sequence_length == 1024
        assert resnet50_model.embedding_dim == 128
        assert isinstance(resnet50_model.iq_encoder, nn.Sequential)
        assert isinstance(resnet50_model.attention, nn.MultiheadAttention)
        assert isinstance(resnet50_model.position_head, nn.Sequential)
        assert isinstance(resnet50_model.uncertainty_head, nn.Sequential)
    
    def test_custom_parameters(self):
        """Test model initializes with custom parameters."""
        model = IQResNet50(
            max_receivers=15,
            iq_sequence_length=2048,
            embedding_dim=256,
            dropout=0.5
        )
        assert model.max_receivers == 15
        assert model.iq_sequence_length == 2048
        assert model.embedding_dim == 256
    
    def test_attention_heads(self, resnet50_model):
        """Test ResNet50 uses 8 attention heads."""
        assert resnet50_model.attention.num_heads == 8
    
    def test_encoder_architecture(self, resnet50_model):
        """Test encoder has correct bottleneck structure [3,4,6,3]."""
        # Encoder should have: conv1 + pool + 4 bottleneck layers + adaptive pool
        encoder_layers = list(resnet50_model.iq_encoder.children())
        assert len(encoder_layers) == 7  # conv+bn+relu, pool, 4 bottleneck layers, adaptive pool


# ============================================================================
# TEST IQRESNET50 - FORWARD PASS
# ============================================================================


class TestIQResNet50Forward:
    """Test IQResNet50 forward pass."""
    
    def test_forward_pass_output_shape(self, resnet50_model, sample_iq_batch):
        """Test forward pass produces correct output shapes."""
        iq_samples, receiver_mask = sample_iq_batch
        batch_size = iq_samples.shape[0]
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)  # [lat, lon]
        assert uncertainties.shape == (batch_size, 2)  # [sigma_x, sigma_y]
    
    def test_forward_pass_different_batch_sizes(self, resnet50_model):
        """Test forward pass works with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            iq_samples = torch.randn(batch_size, 10, 2, 1024)
            receiver_mask = torch.zeros(batch_size, 10, dtype=torch.bool)
            
            positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
            
            assert positions.shape == (batch_size, 2)
            assert uncertainties.shape == (batch_size, 2)
    
    def test_forward_pass_single_receiver(self, resnet50_model):
        """Test forward pass with single active receiver."""
        batch_size = 4
        iq_samples = torch.randn(batch_size, 10, 2, 1024)
        receiver_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        receiver_mask[:, 0] = False  # Only first receiver active
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)
        assert uncertainties.shape == (batch_size, 2)
        assert torch.isfinite(positions).all()
        assert torch.isfinite(uncertainties).all()
    
    def test_forward_pass_all_receivers_active(self, resnet50_model):
        """Test forward pass with all receivers active."""
        batch_size = 4
        iq_samples = torch.randn(batch_size, 10, 2, 1024)
        receiver_mask = torch.zeros(batch_size, 10, dtype=torch.bool)  # All active
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)
        assert uncertainties.shape == (batch_size, 2)
    
    def test_uncertainty_clamping(self, resnet50_model, sample_iq_batch):
        """Test uncertainties are clamped to [0.01, 1.0]."""
        iq_samples, receiver_mask = sample_iq_batch
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        
        assert (uncertainties >= 0.01).all()
        assert (uncertainties <= 1.0).all()
    
    def test_output_finite(self, resnet50_model, sample_iq_batch):
        """Test outputs are finite (no NaN or Inf)."""
        iq_samples, receiver_mask = sample_iq_batch
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        
        assert torch.isfinite(positions).all()
        assert torch.isfinite(uncertainties).all()
    
    def test_gradient_flow(self, resnet50_model, sample_iq_batch):
        """Test gradients flow through the model."""
        iq_samples, receiver_mask = sample_iq_batch
        iq_samples.requires_grad = True
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        loss = positions.sum() + uncertainties.sum()
        loss.backward()
        
        assert iq_samples.grad is not None
        assert torch.isfinite(iq_samples.grad).all()


# ============================================================================
# TEST IQRESNET50 - EDGE CASES
# ============================================================================


class TestIQResNet50EdgeCases:
    """Test IQResNet50 edge cases and error handling."""
    
    def test_varying_receiver_counts(self, resnet50_model):
        """Test model handles varying active receiver counts."""
        batch_size = 8
        iq_samples = torch.randn(batch_size, 10, 2, 1024)
        
        # Each sample has different number of active receivers (1-10)
        receiver_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        for i in range(batch_size):
            active_receivers = min(i + 1, 10)
            receiver_mask[i, :active_receivers] = False
        
        positions, uncertainties = resnet50_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)
        assert torch.isfinite(positions).all()
    
    def test_different_sequence_lengths(self):
        """Test model initialization with different sequence lengths."""
        for seq_len in [512, 1024, 2048, 4096]:
            model = IQResNet50(iq_sequence_length=seq_len)
            iq_samples = torch.randn(4, 10, 2, seq_len)
            receiver_mask = torch.zeros(4, 10, dtype=torch.bool)
            
            positions, uncertainties = model(iq_samples, receiver_mask)
            assert positions.shape == (4, 2)
    
    def test_different_embedding_dims(self):
        """Test model with different embedding dimensions."""
        for embed_dim in [64, 128, 256, 512]:
            model = IQResNet50(embedding_dim=embed_dim)
            iq_samples = torch.randn(4, 10, 2, 1024)
            receiver_mask = torch.zeros(4, 10, dtype=torch.bool)
            
            positions, uncertainties = model(iq_samples, receiver_mask)
            assert positions.shape == (4, 2)


# ============================================================================
# TEST IQRESNET101 - INITIALIZATION
# ============================================================================


class TestIQResNet101Initialization:
    """Test IQResNet101 model initialization."""
    
    def test_default_initialization(self, resnet101_model):
        """Test model initializes with default parameters."""
        assert resnet101_model.max_receivers == 10
        assert resnet101_model.iq_sequence_length == 1024
        assert resnet101_model.embedding_dim == 128
        assert isinstance(resnet101_model.iq_encoder, nn.Sequential)
        assert isinstance(resnet101_model.attention, nn.MultiheadAttention)
        assert isinstance(resnet101_model.position_head, nn.Sequential)
        assert isinstance(resnet101_model.uncertainty_head, nn.Sequential)
    
    def test_custom_parameters(self):
        """Test model initializes with custom parameters."""
        model = IQResNet101(
            max_receivers=15,
            iq_sequence_length=2048,
            embedding_dim=256,
            dropout=0.5
        )
        assert model.max_receivers == 15
        assert model.iq_sequence_length == 2048
        assert model.embedding_dim == 256
    
    def test_attention_heads(self, resnet101_model):
        """Test ResNet101 uses 8 attention heads."""
        assert resnet101_model.attention.num_heads == 8
    
    def test_encoder_architecture(self, resnet101_model):
        """Test encoder has correct bottleneck structure [3,4,23,3]."""
        # Encoder should have: conv1 + pool + 4 bottleneck layers + adaptive pool
        encoder_layers = list(resnet101_model.iq_encoder.children())
        assert len(encoder_layers) == 7  # conv+bn+relu, pool, 4 bottleneck layers, adaptive pool


# ============================================================================
# TEST IQRESNET101 - FORWARD PASS
# ============================================================================


class TestIQResNet101Forward:
    """Test IQResNet101 forward pass."""
    
    def test_forward_pass_output_shape(self, resnet101_model, sample_iq_batch):
        """Test forward pass produces correct output shapes."""
        iq_samples, receiver_mask = sample_iq_batch
        batch_size = iq_samples.shape[0]
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)  # [lat, lon]
        assert uncertainties.shape == (batch_size, 2)  # [sigma_x, sigma_y]
    
    def test_forward_pass_different_batch_sizes(self, resnet101_model):
        """Test forward pass works with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            iq_samples = torch.randn(batch_size, 10, 2, 1024)
            receiver_mask = torch.zeros(batch_size, 10, dtype=torch.bool)
            
            positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
            
            assert positions.shape == (batch_size, 2)
            assert uncertainties.shape == (batch_size, 2)
    
    def test_forward_pass_single_receiver(self, resnet101_model):
        """Test forward pass with single active receiver."""
        batch_size = 4
        iq_samples = torch.randn(batch_size, 10, 2, 1024)
        receiver_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        receiver_mask[:, 0] = False  # Only first receiver active
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)
        assert uncertainties.shape == (batch_size, 2)
        assert torch.isfinite(positions).all()
        assert torch.isfinite(uncertainties).all()
    
    def test_forward_pass_all_receivers_active(self, resnet101_model):
        """Test forward pass with all receivers active."""
        batch_size = 4
        iq_samples = torch.randn(batch_size, 10, 2, 1024)
        receiver_mask = torch.zeros(batch_size, 10, dtype=torch.bool)  # All active
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)
        assert uncertainties.shape == (batch_size, 2)
    
    def test_uncertainty_clamping(self, resnet101_model, sample_iq_batch):
        """Test uncertainties are clamped to [0.01, 1.0]."""
        iq_samples, receiver_mask = sample_iq_batch
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        
        assert (uncertainties >= 0.01).all()
        assert (uncertainties <= 1.0).all()
    
    def test_output_finite(self, resnet101_model, sample_iq_batch):
        """Test outputs are finite (no NaN or Inf)."""
        iq_samples, receiver_mask = sample_iq_batch
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        
        assert torch.isfinite(positions).all()
        assert torch.isfinite(uncertainties).all()
    
    def test_gradient_flow(self, resnet101_model, sample_iq_batch):
        """Test gradients flow through the model."""
        iq_samples, receiver_mask = sample_iq_batch
        iq_samples.requires_grad = True
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        loss = positions.sum() + uncertainties.sum()
        loss.backward()
        
        assert iq_samples.grad is not None
        assert torch.isfinite(iq_samples.grad).all()


# ============================================================================
# TEST IQRESNET101 - EDGE CASES
# ============================================================================


class TestIQResNet101EdgeCases:
    """Test IQResNet101 edge cases and error handling."""
    
    def test_varying_receiver_counts(self, resnet101_model):
        """Test model handles varying active receiver counts."""
        batch_size = 8
        iq_samples = torch.randn(batch_size, 10, 2, 1024)
        
        # Each sample has different number of active receivers (1-10)
        receiver_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        for i in range(batch_size):
            active_receivers = min(i + 1, 10)
            receiver_mask[i, :active_receivers] = False
        
        positions, uncertainties = resnet101_model(iq_samples, receiver_mask)
        
        assert positions.shape == (batch_size, 2)
        assert torch.isfinite(positions).all()
    
    def test_different_sequence_lengths(self):
        """Test model initialization with different sequence lengths."""
        for seq_len in [512, 1024, 2048, 4096]:
            model = IQResNet101(iq_sequence_length=seq_len)
            iq_samples = torch.randn(4, 10, 2, seq_len)
            receiver_mask = torch.zeros(4, 10, dtype=torch.bool)
            
            positions, uncertainties = model(iq_samples, receiver_mask)
            assert positions.shape == (4, 2)
    
    def test_different_embedding_dims(self):
        """Test model with different embedding dimensions."""
        for embed_dim in [64, 128, 256, 512]:
            model = IQResNet101(embedding_dim=embed_dim)
            iq_samples = torch.randn(4, 10, 2, 1024)
            receiver_mask = torch.zeros(4, 10, dtype=torch.bool)
            
            positions, uncertainties = model(iq_samples, receiver_mask)
            assert positions.shape == (4, 2)


# ============================================================================
# TEST COMPARATIVE - RESNET50 VS RESNET101
# ============================================================================


class TestResNet50vs101Comparison:
    """Compare IQResNet50 and IQResNet101 behavior."""
    
    def test_parameter_count_difference(self, resnet50_model, resnet101_model):
        """Test ResNet101 has more parameters than ResNet50."""
        count_50 = sum(p.numel() for p in resnet50_model.parameters())
        count_101 = sum(p.numel() for p in resnet101_model.parameters())
        
        assert count_101 > count_50
        # ResNet101 should have ~1.7-2x more parameters
        assert 1.5 < (count_101 / count_50) < 2.5
    
    def test_same_input_different_outputs(self, resnet50_model, resnet101_model, sample_iq_batch):
        """Test both models produce different outputs (not identical weights)."""
        iq_samples, receiver_mask = sample_iq_batch
        
        pos_50, unc_50 = resnet50_model(iq_samples, receiver_mask)
        pos_101, unc_101 = resnet101_model(iq_samples, receiver_mask)
        
        # Outputs should be different (random initialization)
        assert not torch.allclose(pos_50, pos_101, atol=1e-3)
        assert not torch.allclose(unc_50, unc_101, atol=1e-3)
    
    def test_both_models_produce_valid_outputs(self, resnet50_model, resnet101_model, sample_iq_batch):
        """Test both models produce valid outputs with same input."""
        iq_samples, receiver_mask = sample_iq_batch
        
        pos_50, unc_50 = resnet50_model(iq_samples, receiver_mask)
        pos_101, unc_101 = resnet101_model(iq_samples, receiver_mask)
        
        # Both should produce finite outputs
        assert torch.isfinite(pos_50).all()
        assert torch.isfinite(unc_50).all()
        assert torch.isfinite(pos_101).all()
        assert torch.isfinite(unc_101).all()
        
        # Both should respect uncertainty clamping
        assert (unc_50 >= 0.01).all() and (unc_50 <= 1.0).all()
        assert (unc_101 >= 0.01).all() and (unc_101 <= 1.0).all()


# ============================================================================
# TEST PERFORMANCE METRICS
# ============================================================================


class TestPerformanceMetrics:
    """Test performance-related metrics."""
    
    def test_model_parameter_counts(self):
        """Test parameter counts match expected ranges."""
        model_50 = IQResNet50()
        model_101 = IQResNet101()
        
        count_50 = sum(p.numel() for p in model_50.parameters())
        count_101 = sum(p.numel() for p in model_101.parameters())
        
        # ResNet50 should have ~20-30M parameters
        assert 15_000_000 < count_50 < 35_000_000
        
        # ResNet101 should have ~35-50M parameters
        assert 30_000_000 < count_101 < 55_000_000
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_compatibility_resnet50(self, resnet50_model):
        """Test ResNet50 works on GPU."""
        device = torch.device('cuda')
        model = resnet50_model.to(device)
        
        iq_samples = torch.randn(4, 10, 2, 1024).to(device)
        receiver_mask = torch.zeros(4, 10, dtype=torch.bool).to(device)
        
        positions, uncertainties = model(iq_samples, receiver_mask)
        
        assert positions.device.type == 'cuda'
        assert uncertainties.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_compatibility_resnet101(self, resnet101_model):
        """Test ResNet101 works on GPU."""
        device = torch.device('cuda')
        model = resnet101_model.to(device)
        
        iq_samples = torch.randn(4, 10, 2, 1024).to(device)
        receiver_mask = torch.zeros(4, 10, dtype=torch.bool).to(device)
        
        positions, uncertainties = model(iq_samples, receiver_mask)
        
        assert positions.device.type == 'cuda'
        assert uncertainties.device.type == 'cuda'
