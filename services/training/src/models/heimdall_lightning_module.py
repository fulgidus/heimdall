"""
HeimdallNetLitModule: Custom PyTorch Lightning module for HeimdallNet.

HeimdallNet requires multiple inputs (IQ data, features, positions, receiver IDs, mask)
extracted from the IQ dataloader batch, unlike standard models that take a single tensor.

This module handles:
- Multi-modal input extraction from batch
- Forward pass with all required arguments
- Gaussian NLL loss computation
- Optimization and scheduling
- Validation and logging
"""

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, Dict, Any
import numpy as np
import structlog

from src.models.heimdall_net import HeimdallNet
from src.utils.losses import GaussianNLLLoss

logger = structlog.get_logger(__name__)


class HeimdallNetLitModule(pl.LightningModule):
    """
    PyTorch Lightning module for HeimdallNet multi-modal localization.
    
    HeimdallNet Architecture:
        - Per-receiver encoding (IQ + features + identity embeddings)
        - Set aggregation (permutation-invariant)
        - Geometry encoding (spatial awareness)
        - Global fusion
        - Dual-head output (position + uncertainty)
    
    Input Requirements:
        - iq_data: (batch, N_receivers, 2, seq_len) - IQ samples
        - features: (batch, N_receivers, 6) - [SNR, PSD, freq_offset, lat, lon, alt]
        - positions: (batch, N_receivers, 3) - [lat, lon, alt]
        - receiver_ids: (batch, N_receivers) - receiver IDs (0 to max_receivers-1)
        - mask: (batch, N_receivers) - active receivers mask
    
    Target Performance:
        - Accuracy: ±8-15m (68% confidence)
        - Inference: 40-60ms
        - Parameters: ~25M
    """
    
    def __init__(
        self,
        max_receivers: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_steps: int = 0,
        num_training_steps: int = 10000,
        use_calibration: bool = True,
        dropout: float = 0.1,
        uncertainty_bounds: Tuple[float, float] = (0.01, 1.0),
    ):
        """
        Initialize HeimdallNetLitModule.
        
        Args:
            max_receivers: Maximum number of receivers (default: 10)
            learning_rate: Initial learning rate (default: 1e-3)
            weight_decay: L2 regularization weight (default: 1e-5)
            warmup_steps: Number of warmup steps (default: 0)
            num_training_steps: Total training steps for cosine scheduling
            use_calibration: Use per-receiver calibration layers (default: True)
            dropout: Dropout probability (default: 0.1)
            uncertainty_bounds: (min_sigma, max_sigma) for uncertainty clamping
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.max_receivers = max_receivers
        
        # Model: HeimdallNet
        self.model = HeimdallNet(
            max_receivers=max_receivers,
            iq_dim=256,
            feature_dim=128,
            receiver_embed_dim=64,
            hidden_dim=256,
            num_heads=8,
            dropout=dropout,
            use_calibration=use_calibration
        )
        
        # Loss: Gaussian NLL
        self.loss_fn = GaussianNLLLoss(reduction="mean")
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        
        # Hyperparameter saving
        self.save_hyperparameters()
        
        logger.info(
            "heimdall_lightning_module_initialized",
            model="HeimdallNet",
            max_receivers=max_receivers,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            use_calibration=use_calibration,
        )
    
    def _extract_batch_inputs(self, batch: Tuple) -> Dict[str, torch.Tensor]:
        """
        Extract all required inputs for HeimdallNet from IQ dataloader batch.
        
        IQ Dataloader returns:
            - iq_samples: (batch, num_receivers, 2, seq_len)
            - receiver_positions: (batch, num_receivers, 3) [lat, lon, alt]
            - signal_mask: (batch, num_receivers) - True = active
            - target_position: (batch, 2) [lat, lon]
            - metadata: Dict
        
        HeimdallNet requires:
            - iq_data: (batch, num_receivers, 2, seq_len)
            - features: (batch, num_receivers, 6) - [SNR, PSD, freq_offset, lat, lon, alt]
            - positions: (batch, num_receivers, 3)
            - receiver_ids: (batch, num_receivers)
            - mask: (batch, num_receivers)
        
        Args:
            batch: Tuple from IQ dataloader
        
        Returns:
            Dict with all required tensors for HeimdallNet forward pass
        """
        iq_samples, receiver_positions, signal_mask, target_position, metadata = batch
        
        # IQ data is already in correct format
        iq_data = iq_samples  # (B, N, 2, seq_len)
        
        # Positions are already in correct format
        positions = receiver_positions  # (B, N, 3)
        
        # Mask is already in correct format
        mask = signal_mask  # (B, N) - True = active
        
        # Extract features from metadata (or compute dummy features)
        # For now, create dummy features with receiver positions
        batch_size, num_receivers, _ = positions.shape
        
        # Features: [SNR, PSD, freq_offset, lat, lon, alt]
        # Use positions + dummy signal metrics
        features = torch.zeros(batch_size, num_receivers, 6, device=positions.device)
        features[:, :, 0] = 20.0  # Dummy SNR (20 dB)
        features[:, :, 1] = -80.0  # Dummy PSD (-80 dBm)
        features[:, :, 2] = 0.0  # Dummy freq offset (0 Hz)
        features[:, :, 3:6] = positions  # lat, lon, alt
        
        # Receiver IDs: Assign sequential IDs (0, 1, 2, ..., N-1)
        receiver_ids = torch.arange(num_receivers, device=positions.device).unsqueeze(0).expand(batch_size, -1)
        
        # Clamp receiver IDs to valid range [0, max_receivers-1]
        receiver_ids = torch.clamp(receiver_ids, 0, self.max_receivers - 1)
        
        return {
            "iq_data": iq_data,
            "features": features,
            "positions": positions,
            "receiver_ids": receiver_ids,
            "mask": mask,
            "target_position": target_position,
        }
    
    def forward(
        self,
        iq_data: torch.Tensor,
        features: torch.Tensor,
        positions: torch.Tensor,
        receiver_ids: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through HeimdallNet.
        
        Args:
            iq_data: (batch, N, 2, seq_len)
            features: (batch, N, 6)
            positions: (batch, N, 3)
            receiver_ids: (batch, N)
            mask: (batch, N) - optional
        
        Returns:
            Tuple of (predicted_positions, predicted_uncertainties)
        """
        return self.model(iq_data, features, positions, receiver_ids, mask)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step called for each batch.
        
        Args:
            batch: Tuple from IQ dataloader
            batch_idx: Batch index
        
        Returns:
            Loss tensor for backward pass
        """
        # Extract all required inputs
        inputs = self._extract_batch_inputs(batch)
        
        # Forward pass
        positions, uncertainties = self.forward(
            inputs["iq_data"],
            inputs["features"],
            inputs["positions"],
            inputs["receiver_ids"],
            inputs["mask"]
        )
        
        # Compute loss
        target_position = inputs["target_position"]
        loss, stats = self.loss_fn.forward_with_stats(positions, uncertainties, target_position)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mae", stats["mae"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/sigma_mean", stats["sigma_mean"], on_step=False, on_epoch=True)
        self.log("train/mse_term", stats["mse_term"], on_step=False, on_epoch=True)
        self.log("train/log_term", stats["log_term"], on_step=False, on_epoch=True)
        
        # Store for epoch-level statistics
        self.train_losses.append(loss.item())
        
        if batch_idx % 100 == 0:
            logger.debug(
                "training_step",
                batch_idx=batch_idx,
                loss=loss.item(),
                mae=stats["mae"],
                sigma_mean=stats["sigma_mean"],
            )
        
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        """Validation step called for each validation batch."""
        # Extract all required inputs
        inputs = self._extract_batch_inputs(batch)
        
        # Forward pass
        with torch.no_grad():
            positions, uncertainties = self.forward(
                inputs["iq_data"],
                inputs["features"],
                inputs["positions"],
                inputs["receiver_ids"],
                inputs["mask"]
            )
        
        # Compute loss
        target_position = inputs["target_position"]
        loss, stats = self.loss_fn.forward_with_stats(positions, uncertainties, target_position)
        
        # Compute additional metrics
        mae = torch.abs(positions - target_position).mean()
        position_error = torch.norm(positions - target_position, dim=1)  # Euclidean distance
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/mae", mae, on_epoch=True)
        self.log("val/position_error_mean", position_error.mean(), on_epoch=True)
        self.log("val/position_error_std", position_error.std(), on_epoch=True)
        self.log("val/sigma_mean", stats["sigma_mean"], on_epoch=True)
        
        # Store for averaging
        self.val_losses.append(loss.item())
        self.val_maes.append(mae.item())
        
        return {
            "loss": loss,
            "mae": mae,
            "position_error": position_error,
        }
    
    def test_step(self, batch, batch_idx: int):
        """Test step (evaluation on test set)."""
        # Extract all required inputs
        inputs = self._extract_batch_inputs(batch)
        
        with torch.no_grad():
            positions, uncertainties = self.forward(
                inputs["iq_data"],
                inputs["features"],
                inputs["positions"],
                inputs["receiver_ids"],
                inputs["mask"]
            )
        
        target_position = inputs["target_position"]
        loss, stats = self.loss_fn.forward_with_stats(positions, uncertainties, target_position)
        
        mae = torch.abs(positions - target_position).mean()
        position_error = torch.norm(positions - target_position, dim=1)
        
        self.log("test/loss", loss)
        self.log("test/mae", mae)
        self.log("test/position_error_mean", position_error.mean())
        self.log("test/position_error_std", position_error.std())
        
        return {
            "loss": loss,
            "mae": mae,
            "position_error": position_error,
            "positions": positions,
            "labels": target_position,
            "uncertainties": uncertainties,
        }
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses:
        - Optimizer: AdamW
        - Scheduler: Cosine annealing with warmup
        """
        # Optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(self.num_training_steps - self.warmup_steps, 1),
            eta_min=1e-6,
        )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        
        logger.info(
            "optimizer_configured",
            optimizer="AdamW",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            scheduler="CosineAnnealing",
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if len(self.train_losses) > 0:
            avg_train_loss = np.mean(self.train_losses[-100:])
            logger.debug("epoch_end", avg_train_loss=avg_train_loss)
        
        # Clear metrics
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
    
    def get_model_for_export(self) -> HeimdallNet:
        """
        Get the underlying model for export.
        
        Used for ONNX export and inference deployment.
        """
        return self.model


if __name__ == "__main__":
    """Test HeimdallNetLitModule instantiation and forward pass."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger.info("🌉 Testing HeimdallNetLitModule")
    
    # Create module
    module = HeimdallNetLitModule(
        max_receivers=7,
        learning_rate=1e-3,
        num_training_steps=1000,
    )
    
    # Create dummy batch (IQ dataloader format)
    batch_size = 4
    num_receivers = 3
    seq_len = 1024
    
    iq_samples = torch.randn(batch_size, num_receivers, 2, seq_len)
    receiver_positions = torch.randn(batch_size, num_receivers, 3)
    signal_mask = torch.ones(batch_size, num_receivers, dtype=torch.bool)
    target_position = torch.randn(batch_size, 2)
    metadata = {"session_id": ["s1"] * batch_size}
    
    batch = (iq_samples, receiver_positions, signal_mask, target_position, metadata)
    
    # Test training step
    loss = module.training_step(batch, 0)
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"
    logger.info(f"✅ Training step passed! Loss: {loss.item():.4f}")
    
    # Test validation step
    val_outputs = module.validation_step(batch, 0)
    assert "loss" in val_outputs
    assert "mae" in val_outputs
    logger.info("✅ Validation step passed!")
    
    # Test forward pass
    inputs = module._extract_batch_inputs(batch)
    positions, uncertainties = module(
        inputs["iq_data"],
        inputs["features"],
        inputs["positions"],
        inputs["receiver_ids"],
        inputs["mask"]
    )
    assert positions.shape == (batch_size, 2)
    assert uncertainties.shape == (batch_size, 2)
    assert (uncertainties > 0).all()
    logger.info("✅ Forward pass passed!")
    
    # Test optimizer configuration
    optimizer_config = module.configure_optimizers()
    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config
    logger.info("✅ Optimizer configuration passed!")
    
    logger.info("✅ HeimdallNetLitModule verification complete!")
