"""
Unit tests for HeimdallNet multi-modal localization model.

Tests cover:
- Model instantiation
- Forward pass with various receiver counts
- Parameter counting
- Input validation
- Output shape verification
"""

import pytest
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "heimdall_net",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'heimdall_net.py')
)
heimdall_net = importlib.util.module_from_spec(spec)
sys.modules["heimdall_net"] = heimdall_net
spec.loader.exec_module(heimdall_net)


class TestHeimdallNetInstantiation:
    """Test model creation and configuration."""
    
    def test_create_default(self):
        """Test creating HeimdallNet with default parameters."""
        model = heimdall_net.create_heimdall_net()
        assert model is not None
        assert isinstance(model, heimdall_net.HeimdallNet)
        
    def test_create_custom_receivers(self):
        """Test creating HeimdallNet with custom max_receivers."""
        for max_receivers in [5, 7, 10]:
            model = heimdall_net.create_heimdall_net(max_receivers=max_receivers)
            assert model.max_receivers == max_receivers
            
    def test_create_without_calibration(self):
        """Test creating HeimdallNet without per-receiver calibration."""
        model = heimdall_net.create_heimdall_net(use_calibration=False)
        assert not model.receiver_encoder.use_calibration
        
    def test_parameter_count(self):
        """Test parameter counting matches expected ~2.2M."""
        model = heimdall_net.create_heimdall_net()
        params = heimdall_net.count_parameters(model)
        
        # Should be around 2.2M parameters
        assert 2.0 <= params['total_millions'] <= 2.5
        
        # Check components exist
        expected_components = [
            'receiver_encoder', 'set_aggregator', 
            'geometry_encoder', 'global_fusion',
            'position_head', 'uncertainty_head'
        ]
        for comp in expected_components:
            assert comp in params['breakdown']


class TestHeimdallNetForwardPass:
    """Test forward pass with various inputs."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        model = heimdall_net.create_heimdall_net(max_receivers=10)
        model.eval()  # Set to eval mode
        return model
    
    def test_forward_3_receivers(self, model):
        """Test forward pass with 3 active receivers (typical scenario)."""
        batch_size = 4
        num_receivers = 3
        
        # Create dummy inputs
        iq_data = torch.randn(batch_size, num_receivers, 2, 1024)
        features = torch.randn(batch_size, num_receivers, 6)
        positions = torch.randn(batch_size, num_receivers, 3)
        receiver_ids = torch.tensor([[0, 2, 4]] * batch_size)
        mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
        
        # Forward pass
        with torch.no_grad():
            pred_pos, pred_unc = model(iq_data, features, positions, receiver_ids, mask)
        
        # Check output shapes
        assert pred_pos.shape == (batch_size, 2)
        assert pred_unc.shape == (batch_size, 2)
        
        # Check uncertainty is positive
        assert (pred_unc > 0).all()
        
    def test_forward_variable_receivers(self, model):
        """Test forward pass with different receiver counts."""
        batch_size = 2
        
        for num_receivers in [1, 3, 5, 7, 10]:
            iq_data = torch.randn(batch_size, num_receivers, 2, 1024)
            features = torch.randn(batch_size, num_receivers, 6)
            positions = torch.randn(batch_size, num_receivers, 3)
            receiver_ids = torch.randint(0, 10, (batch_size, num_receivers))
            mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
            
            with torch.no_grad():
                pred_pos, pred_unc = model(iq_data, features, positions, receiver_ids, mask)
            
            assert pred_pos.shape == (batch_size, 2)
            assert pred_unc.shape == (batch_size, 2)
    
    def test_forward_with_masking(self, model):
        """Test forward pass with partial receiver masking."""
        batch_size = 4
        num_receivers = 5
        
        iq_data = torch.randn(batch_size, num_receivers, 2, 1024)
        features = torch.randn(batch_size, num_receivers, 6)
        positions = torch.randn(batch_size, num_receivers, 3)
        receiver_ids = torch.tensor([[0, 1, 2, 3, 4]] * batch_size)
        
        # Mask out last 2 receivers (only 3 active)
        mask = torch.tensor([[True, True, True, False, False]] * batch_size)
        
        with torch.no_grad():
            pred_pos, pred_unc = model(iq_data, features, positions, receiver_ids, mask)
        
        assert pred_pos.shape == (batch_size, 2)
        assert pred_unc.shape == (batch_size, 2)
        
    def test_forward_single_sample(self, model):
        """Test forward pass with batch_size=1."""
        iq_data = torch.randn(1, 3, 2, 1024)
        features = torch.randn(1, 3, 6)
        positions = torch.randn(1, 3, 3)
        receiver_ids = torch.tensor([[0, 2, 4]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        
        with torch.no_grad():
            pred_pos, pred_unc = model(iq_data, features, positions, receiver_ids, mask)
        
        assert pred_pos.shape == (1, 2)
        assert pred_unc.shape == (1, 2)


class TestHeimdallNetComponents:
    """Test individual model components."""
    
    def test_efficientnet_encoder(self):
        """Test EfficientNet-B2 1D encoder."""
        encoder = heimdall_net.EfficientNetB2_1D(in_channels=2, out_dim=256)
        
        # Test forward pass
        x = torch.randn(4, 2, 1024)  # (batch, [I,Q], seq_len)
        with torch.no_grad():
            out = encoder(x)
        
        assert out.shape == (4, 256)
        
    def test_per_receiver_encoder(self):
        """Test per-receiver encoder with identity embeddings."""
        encoder = heimdall_net.PerReceiverEncoder(
            max_receivers=10,
            iq_dim=256,
            feature_dim=128,
            receiver_embed_dim=64,
            use_calibration=True
        )
        
        batch_size = 4
        iq_raw = torch.randn(batch_size, 2, 1024)
        features = torch.randn(batch_size, 6)
        receiver_id = torch.randint(0, 10, (batch_size,))
        
        with torch.no_grad():
            out = encoder(iq_raw, features, receiver_id)
        
        assert out.shape == (batch_size, 256)
        
    def test_set_attention_aggregator(self):
        """Test set attention aggregation."""
        aggregator = heimdall_net.SetAttentionAggregator(dim=256, num_heads=8)
        
        batch_size = 4
        num_receivers = 3
        embeddings = torch.randn(batch_size, num_receivers, 256)
        mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
        
        with torch.no_grad():
            out = aggregator(embeddings, mask)
        
        assert out.shape == (batch_size, 256)
        
    def test_geometry_encoder(self):
        """Test geometry encoder."""
        encoder = heimdall_net.GeometryEncoder(dim=256)
        
        batch_size = 4
        num_receivers = 3
        positions = torch.randn(batch_size, num_receivers, 3)
        mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
        
        with torch.no_grad():
            out = encoder(positions, mask)
        
        assert out.shape == (batch_size, 256)


class TestHeimdallNetPermutationInvariance:
    """Test that model is permutation-invariant."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        model = heimdall_net.create_heimdall_net(max_receivers=10)
        model.eval()
        return model
    
    def test_permutation_invariance(self, model):
        """Test that receiver order doesn't affect output (with fixed IDs)."""
        batch_size = 2
        num_receivers = 3
        
        # Create inputs
        iq_data = torch.randn(batch_size, num_receivers, 2, 1024)
        features = torch.randn(batch_size, num_receivers, 6)
        positions = torch.randn(batch_size, num_receivers, 3)
        receiver_ids = torch.tensor([[0, 2, 4]] * batch_size)
        mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
        
        # Forward pass with original order
        with torch.no_grad():
            pred_pos_1, pred_unc_1 = model(iq_data, features, positions, receiver_ids, mask)
        
        # Permute receivers (swap 0 and 2)
        perm_indices = torch.tensor([2, 1, 0])
        iq_data_perm = iq_data[:, perm_indices]
        features_perm = features[:, perm_indices]
        positions_perm = positions[:, perm_indices]
        receiver_ids_perm = receiver_ids[:, perm_indices]
        mask_perm = mask[:, perm_indices]
        
        # Forward pass with permuted order
        with torch.no_grad():
            pred_pos_2, pred_unc_2 = model(iq_data_perm, features_perm, positions_perm, receiver_ids_perm, mask_perm)
        
        # Results should be the same (within numerical tolerance)
        assert torch.allclose(pred_pos_1, pred_pos_2, rtol=1e-4, atol=1e-4)
        assert torch.allclose(pred_unc_1, pred_unc_2, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
