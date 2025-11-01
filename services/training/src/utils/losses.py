"""
Gaussian Negative Log-Likelihood Loss for uncertainty-aware regression.

This custom loss function is designed for the localization task where we want to:
1. Predict accurate position estimates (latitude, longitude)
2. Estimate uncertainty for each prediction (sigma_x, sigma_y)
3. Penalize overconfidence (small sigma with large error)

The loss combines position error with uncertainty calibration.

Loss formula:
L = (y - mu)^2 / (2 * sigma^2) + log(sigma)

Where:
- y: ground truth position
- mu: predicted position
- sigma: predicted uncertainty (standard deviation)

Interpretation:
- First term: MSE weighted by inverse of uncertainty
- Second term: Regularization that prevents collapse of sigma to zero

This encourages the model to:
- Make accurate predictions (minimize position error)
- Produce well-calibrated uncertainty (not too overconfident)
"""

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for uncertainty-aware regression.

    Suitable for tasks where the model should output both predictions and
    uncertainty estimates.

    Args:
        reduction (str): 'mean' or 'sum' for batch reduction
        eps (float): Small value to avoid numerical issues
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")

        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.

        Args:
            predictions (torch.Tensor): Predicted positions, shape (batch_size, 2)
            uncertainties (torch.Tensor): Predicted uncertainties (sigmas), shape (batch_size, 2)
                                         Must be positive
            targets (torch.Tensor): Ground truth positions, shape (batch_size, 2)

        Returns:
            torch.Tensor: Loss value (scalar if reduction='mean'/'sum', else shape (batch_size, 2))

        Example:
            >>> loss_fn = GaussianNLLLoss()
            >>> pred = torch.randn(8, 2)
            >>> sigma = torch.abs(torch.randn(8, 2)) + 0.1  # Ensure positive
            >>> target = torch.randn(8, 2)
            >>> loss = loss_fn(pred, sigma, target)
        """

        # Ensure uncertainties are positive
        uncertainties = torch.clamp(uncertainties, min=self.eps)

        # Compute residuals
        residuals = targets - predictions  # (batch_size, 2)

        # Gaussian NLL = (residual^2) / (2 * sigma^2) + log(sigma)
        # First term: weighted MSE
        mse_term = (residuals**2) / (2 * uncertainties**2)

        # Second term: log of uncertainty (regularization)
        log_term = torch.log(uncertainties)

        # Total loss per element
        loss_per_element = mse_term + log_term

        # Aggregate based on reduction strategy
        if self.reduction == "mean":
            return loss_per_element.mean()
        elif self.reduction == "sum":
            return loss_per_element.sum()
        else:  # 'none'
            return loss_per_element

    def forward_with_stats(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss and return detailed statistics.

        Useful for monitoring and debugging during training.

        Returns:
            Tuple[torch.Tensor, Dict]:
                - Loss value
                - Statistics dict with components
        """

        # Compute loss
        loss = self.forward(predictions, uncertainties, targets)

        # Compute components separately
        uncertainties_safe = torch.clamp(uncertainties, min=self.eps)
        residuals = targets - predictions
        mse_term = (residuals**2) / (2 * uncertainties_safe**2)
        log_term = torch.log(uncertainties_safe)

        # Statistics
        stats = {
            "loss": loss.item() if loss.dim() == 0 else loss.mean().item(),
            "mse_term": mse_term.mean().item(),
            "log_term": log_term.mean().item(),
            "mae": torch.abs(residuals).mean().item(),
            "sigma_mean": uncertainties.mean().item(),
            "sigma_min": uncertainties.min().item(),
            "sigma_max": uncertainties.max().item(),
            "residual_mean": residuals.mean().item(),
            "residual_std": residuals.std().item(),
        }

        return loss, stats


class HuberNLLLoss(nn.Module):
    """
    Huber loss variant for uncertainty-aware regression.

    More robust to outliers than pure Gaussian NLL.

    Args:
        delta (float): Huber loss delta parameter (transition point)
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.eps = 1e-6

    def forward(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Huber NLL loss.

        More robust than Gaussian NLL for data with outliers.
        """

        uncertainties = torch.clamp(uncertainties, min=self.eps)
        residuals = targets - predictions

        # Normalize by uncertainty
        normalized_residuals = residuals / uncertainties

        # Huber loss on normalized residuals
        huber_loss = F.huber_loss(
            normalized_residuals,
            torch.zeros_like(normalized_residuals),
            delta=self.delta,
            reduction="none",
        )

        # Add log term
        log_term = torch.log(uncertainties)
        loss_per_element = huber_loss + log_term

        if self.reduction == "mean":
            return loss_per_element.mean()
        elif self.reduction == "sum":
            return loss_per_element.sum()
        else:
            return loss_per_element


class QuantileLoss(nn.Module):
    """
    Quantile loss for predicting confidence intervals.

    Can be used alongside Gaussian NLL for more flexible uncertainty modeling.

    Args:
        quantiles (list): List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
    """

    def __init__(self, quantiles: list = None):
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        super().__init__()
        self.quantiles = quantiles
        self.eps = 1e-6

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions (torch.Tensor): Predicted quantiles, shape (batch, len(quantiles), 2)
            targets (torch.Tensor): Ground truth, shape (batch, 2)

        Returns:
            torch.Tensor: Quantile loss
        """

        total_loss = 0
        for i, q in enumerate(self.quantiles):
            residuals = targets - predictions[:, i, :]
            # Quantile loss: max(q * residual, (q - 1) * residual)
            loss = torch.max(q * residuals, (q - 1) * residuals).mean()
            total_loss += loss

        return total_loss / len(self.quantiles)


def verify_gaussian_nll_loss():
    """Verification function for Gaussian NLL loss."""

    logger.info("Starting Gaussian NLL loss verification...")

    # Create loss function
    loss_fn = GaussianNLLLoss(reduction="mean")

    # Create dummy data
    batch_size = 8
    predictions = torch.randn(batch_size, 2)
    uncertainties = torch.abs(torch.randn(batch_size, 2)) + 0.1  # Ensure positive
    targets = torch.randn(batch_size, 2)

    # Compute loss
    loss = loss_fn(predictions, uncertainties, targets)

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"

    logger.info("✅ Basic loss computation passed!")

    # Test with stats
    loss_with_stats, stats = loss_fn.forward_with_stats(predictions, uncertainties, targets)

    logger.info("loss_statistics", stats=stats)

    # Test edge cases
    # Case 1: Perfect prediction with low uncertainty
    perfect_pred = torch.tensor([[0.0, 0.0]])
    perfect_sigma = torch.tensor([[0.1, 0.1]])
    perfect_target = torch.tensor([[0.0, 0.0]])

    loss_perfect = loss_fn(perfect_pred, perfect_sigma, perfect_target)
    logger.info("loss_perfect_prediction", value=loss_perfect.item())

    # Case 2: Bad prediction with high uncertainty (should be better than low uncertainty)
    bad_pred = torch.tensor([[1.0, 1.0]])
    high_sigma = torch.tensor([[1.0, 1.0]])
    bad_target = torch.tensor([[0.0, 0.0]])

    loss_high_sigma = loss_fn(bad_pred, high_sigma, bad_target)
    logger.info("loss_high_uncertainty", value=loss_high_sigma.item())

    # Case 3: Same error with low uncertainty (should be worse)
    low_sigma = torch.tensor([[0.1, 0.1]])
    loss_low_sigma = loss_fn(bad_pred, low_sigma, bad_target)
    logger.info("loss_low_uncertainty", value=loss_low_sigma.item())

    assert (
        loss_low_sigma > loss_high_sigma
    ), "Low uncertainty should have higher loss for same error"

    logger.info("✅ Gaussian NLL loss verification complete!")

    return loss_fn


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    loss_fn = verify_gaussian_nll_loss()
