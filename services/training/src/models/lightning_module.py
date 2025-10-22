"""
LocalizationLitModule: PyTorch Lightning training module.

Encapsulates the entire training loop:
- forward pass (LocalizationNet)
- loss computation (GaussianNLLLoss)
- optimization (Adam)
- learning rate scheduling
- validation and logging
- model checkpointing via MLflow

This module orchestrates training with Lightning, handling:
- Distributed training (multi-GPU ready)
- Gradient accumulation
- Mixed precision training (optional)
- Experiment tracking via MLflow
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLflowLogger
import structlog
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

from src.models.localization_net import LocalizationNet
from src.utils.losses import GaussianNLLLoss

logger = structlog.get_logger(__name__)


class LocalizationLitModule(pl.LightningModule):
    """
    PyTorch Lightning module for RF source localization training.
    
    Architecture:
    - Backbone: LocalizationNet (ResNet-18)
    - Loss: Gaussian Negative Log-Likelihood
    - Optimizer: Adam with warmup
    - LR Scheduler: CosineAnnealing with restarts
    
    Metrics tracked:
    - Training: MSE, NLL, MAE
    - Validation: MSE, NLL, MAE, calibration metrics
    - Test: Final evaluation
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        warmup_steps: int = 0,
        num_training_steps: int = 10000,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        uncertainty_bounds: Tuple[float, float] = (0.01, 1.0),
        backbone_size: str = 'large',
    ):
        """
        Initialize LocalizationLitModule.
        
        Args:
            learning_rate (float): Initial learning rate
            weight_decay (float): L2 regularization weight
            warmup_steps (int): Number of warmup steps
            num_training_steps (int): Total training steps for cosine scheduling
            pretrained_backbone (bool): Use ImageNet pretrained ConvNeXt-Large
            freeze_backbone (bool): Freeze backbone during training
            uncertainty_bounds (Tuple[float, float]): (min_sigma, max_sigma)
            backbone_size (str): ConvNeXt size ('tiny', 'small', 'medium', 'large')
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.backbone_size = backbone_size
        
        # Model - Now using ConvNeXt-Large (200M params, 88.6% ImageNet accuracy)
        # vs previous ResNet-18 (11M params, 69.8% accuracy)
        self.model = LocalizationNet(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            uncertainty_min=uncertainty_bounds[0],
            uncertainty_max=uncertainty_bounds[1],
            backbone_size=backbone_size,
        )
        
        # Loss
        self.loss_fn = GaussianNLLLoss(reduction='mean')
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        
        # Hyperparameter saving
        self.save_hyperparameters()
        
        logger.info(
            "lightning_module_initialized",
            backbone="ConvNeXt-Large",
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            pretrained_backbone=pretrained_backbone,
            backbone_size=backbone_size,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step called for each batch.
        
        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index
        
        Returns:
            Loss tensor for backward pass
        """
        features, labels, metadata = batch
        
        # Forward pass
        positions, uncertainties = self(features)
        
        # Compute loss
        loss, stats = self.loss_fn.forward_with_stats(
            positions, uncertainties, labels
        )
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/mae', stats['mae'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/sigma_mean', stats['sigma_mean'], on_step=False, on_epoch=True)
        self.log('train/mse_term', stats['mse_term'], on_step=False, on_epoch=True)
        self.log('train/log_term', stats['log_term'], on_step=False, on_epoch=True)
        
        # Store for epoch-level statistics
        self.train_losses.append(loss.item())
        
        if batch_idx % 100 == 0:
            logger.debug(
                "training_step",
                batch_idx=batch_idx,
                loss=loss.item(),
                mae=stats['mae'],
                sigma_mean=stats['sigma_mean'],
            )
        
        return loss
    
    def validation_step(self, batch, batch_idx: int):
        """Validation step called for each validation batch."""
        
        features, labels, metadata = batch
        
        # Forward pass
        with torch.no_grad():
            positions, uncertainties = self(features)
        
        # Compute loss
        loss, stats = self.loss_fn.forward_with_stats(
            positions, uncertainties, labels
        )
        
        # Compute additional metrics
        mae = torch.abs(positions - labels).mean()
        position_error = torch.norm(positions - labels, dim=1)  # Euclidean distance
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.log('val/mae', mae, on_epoch=True)
        self.log('val/position_error_mean', position_error.mean(), on_epoch=True)
        self.log('val/position_error_std', position_error.std(), on_epoch=True)
        self.log('val/sigma_mean', stats['sigma_mean'], on_epoch=True)
        
        # Store for averaging
        self.val_losses.append(loss.item())
        self.val_maes.append(mae.item())
        
        return {
            'loss': loss,
            'mae': mae,
            'position_error': position_error,
        }
    
    def test_step(self, batch, batch_idx: int):
        """Test step (evaluation on test set)."""
        
        features, labels, metadata = batch
        
        with torch.no_grad():
            positions, uncertainties = self(features)
        
        loss, stats = self.loss_fn.forward_with_stats(
            positions, uncertainties, labels
        )
        
        mae = torch.abs(positions - labels).mean()
        position_error = torch.norm(positions - labels, dim=1)
        
        self.log('test/loss', loss)
        self.log('test/mae', mae)
        self.log('test/position_error_mean', position_error.mean())
        self.log('test/position_error_std', position_error.std())
        
        return {
            'loss': loss,
            'mae': mae,
            'position_error': position_error,
            'positions': positions,
            'labels': labels,
            'uncertainties': uncertainties,
        }
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Uses:
        - Optimizer: Adam with weight decay (AdamW equivalent)
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
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        
        logger.info(
            "optimizer_configured",
            optimizer="AdamW",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            scheduler="CosineAnnealing",
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config,
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
    
    def get_model_for_export(self) -> LocalizationNet:
        """
        Get the underlying model for export.
        
        Used for ONNX export and inference deployment.
        """
        return self.model


# Verification function
def verify_lightning_module():
    """Verification function for Lightning module."""
    
    logger.info("Starting Lightning module verification...")
    
    # Create module
    module = LocalizationLitModule(
        learning_rate=1e-3,
        num_training_steps=1000,
        pretrained_backbone=False,
    )
    
    # Create dummy batch
    features = torch.randn(8, 3, 128, 32)
    labels = torch.randn(8, 2)
    metadata = {'session_id': ['s1'] * 8}
    batch = (features, labels, metadata)
    
    # Test training step
    loss = module.training_step(batch, 0)
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() > 0, "Loss should be positive"
    
    logger.info("✅ Training step passed!")
    
    # Test validation step
    val_outputs = module.validation_step(batch, 0)
    assert 'loss' in val_outputs
    assert 'mae' in val_outputs
    
    logger.info("✅ Validation step passed!")
    
    # Test forward pass
    positions, uncertainties = module(features)
    assert positions.shape == (8, 2)
    assert uncertainties.shape == (8, 2)
    assert (uncertainties > 0).all()
    
    logger.info("✅ Forward pass passed!")
    
    # Test optimizer configuration
    optimizer_config = module.configure_optimizers()
    assert 'optimizer' in optimizer_config
    assert 'lr_scheduler' in optimizer_config
    
    logger.info("✅ Optimizer configuration passed!")
    
    logger.info("✅ Lightning module verification complete!")
    
    return module


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    module = verify_lightning_module()
