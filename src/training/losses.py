"""Loss functions for stack count prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class HuberLoss(nn.Module):
    """Huber Loss for robust regression."""
    
    def __init__(self, delta: float = 1.0):
        """Initialize Huber Loss.
        
        Args:
            delta: Threshold for switching from L2 to L1 loss
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Huber Loss.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            
        Returns:
            Huber loss scalar
        """
        residual = torch.abs(predictions - targets)
        
        # Quadratic for small errors, linear for large errors
        quadratic = torch.where(
            residual <= self.delta,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        
        return quadratic.mean()


class ConfidenceCalibrationLoss(nn.Module):
    """Binary Cross Entropy loss for confidence calibration."""
    
    def __init__(self, label_smoothing: float = 0.0):
        """Initialize confidence calibration loss.
        
        Args:
            label_smoothing: Label smoothing factor
        """
        super(ConfidenceCalibrationLoss, self).__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        confidence_pred: torch.Tensor,
        count_pred: torch.Tensor,
        count_true: torch.Tensor
    ) -> torch.Tensor:
        """Calculate confidence calibration loss.
        
        Target confidence is based on prediction accuracy:
        - Low absolute error → high confidence
        - High absolute error → low confidence
        
        Args:
            confidence_pred: Predicted confidence scores (B,)
            count_pred: Predicted counts (B,)
            count_true: True counts (B,)
            
        Returns:
            BCE loss scalar
        """
        # Calculate absolute error
        abs_error = torch.abs(count_pred - count_true)
        
        # Calculate target confidence based on error
        # Using sigmoid: confidence = 1 / (1 + error/mean_count)
        mean_count = count_true.abs().mean() + 1e-6
        target_confidence = 1.0 / (1.0 + abs_error / mean_count)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            target_confidence = (
                target_confidence * (1 - self.label_smoothing) +
                0.5 * self.label_smoothing
            )
        
        # Binary Cross Entropy
        loss = F.binary_cross_entropy(confidence_pred, target_confidence)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss for counting and confidence prediction."""
    
    def __init__(
        self,
        primary_weight: float = 0.8,
        confidence_weight: float = 0.2,
        huber_delta: float = 1.0,
        label_smoothing: float = 0.0
    ):
        """Initialize combined loss.
        
        Args:
            primary_weight: Weight for primary (counting) loss
            confidence_weight: Weight for confidence calibration loss
            huber_delta: Delta parameter for Huber loss
            label_smoothing: Label smoothing for confidence loss
        """
        super(CombinedLoss, self).__init__()
        
        self.primary_weight = primary_weight
        self.confidence_weight = confidence_weight
        
        self.huber_loss = HuberLoss(delta=huber_delta)
        self.confidence_loss = ConfidenceCalibrationLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        count_pred: torch.Tensor,
        confidence_pred: torch.Tensor,
        count_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate combined loss.
        
        Args:
            count_pred: Predicted counts (B,)
            confidence_pred: Predicted confidence scores (B,)
            count_true: True counts (B,)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Calculate individual losses
        primary_loss = self.huber_loss(count_pred, count_true)
        conf_loss = self.confidence_loss(confidence_pred, count_pred, count_true)
        
        # Combined loss
        total_loss = (
            self.primary_weight * primary_loss +
            self.confidence_weight * conf_loss
        )
        
        # Return components for logging
        loss_components = {
            'total_loss': total_loss,
            'primary_loss': primary_loss,
            'confidence_loss': conf_loss
        }
        
        return total_loss, loss_components


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for counting."""
    
    def __init__(self, count_weighting: bool = False):
        """Initialize weighted MSE loss.
        
        Args:
            count_weighting: Whether to weight by true count
        """
        super(WeightedMSELoss, self).__init__()
        self.count_weighting = count_weighting
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted MSE loss.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            
        Returns:
            Weighted MSE loss scalar
        """
        squared_error = (predictions - targets) ** 2
        
        if self.count_weighting:
            # Weight by true count (higher counts get higher weight)
            weights = targets.abs() / (targets.abs().mean() + 1e-6)
            weighted_error = squared_error * weights
            return weighted_error.mean()
        
        return squared_error.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (optional for confidence)."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Focal Loss.
        
        Args:
            predictions: Predicted probabilities
            targets: Target probabilities
            
        Returns:
            Focal loss scalar
        """
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()


class UncertaintyLoss(nn.Module):
    """Loss that encourages accurate uncertainty estimates."""
    
    def __init__(self, lambda_reg: float = 0.1):
        """Initialize uncertainty loss.
        
        Args:
            lambda_reg: Regularization weight
        """
        super(UncertaintyLoss, self).__init__()
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        count_mean: torch.Tensor,
        count_std: torch.Tensor,
        count_true: torch.Tensor
    ) -> torch.Tensor:
        """Calculate uncertainty regularization loss.
        
        Encourages:
        - Low uncertainty for accurate predictions
        - High uncertainty for inaccurate predictions
        
        Args:
            count_mean: Mean count predictions from MC Dropout (B,)
            count_std: Std from MC Dropout (B,)
            count_true: True counts (B,)
            
        Returns:
            Uncertainty loss scalar
        """
        # Calculate prediction error
        error = torch.abs(count_mean - count_true)
        
        # Normalize error
        normalized_error = error / (count_true.abs().mean() + 1e-6)
        
        # We want uncertainty to correlate with error
        # High error should have high uncertainty
        uncertainty_loss = F.mse_loss(count_std, normalized_error)
        
        return self.lambda_reg * uncertainty_loss


def create_loss_from_config(config: Dict) -> CombinedLoss:
    """Create loss function from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CombinedLoss instance
    """
    loss_config = config.get('training', {}).get('loss_weights', {})
    
    return CombinedLoss(
        primary_weight=loss_config.get('primary_loss', 0.8),
        confidence_weight=loss_config.get('confidence_loss', 0.2),
        huber_delta=1.0,
        label_smoothing=0.0
    )


# Test loss functions
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # Create dummy data
    batch_size = 8
    count_pred = torch.randn(batch_size) * 50 + 50
    count_true = torch.randn(batch_size) * 50 + 50
    confidence_pred = torch.sigmoid(torch.randn(batch_size))
    
    # Test Huber Loss
    print("\n1. Testing Huber Loss...")
    huber_loss = HuberLoss(delta=1.0)
    loss = huber_loss(count_pred, count_true)
    print(f"Huber Loss: {loss.item():.4f}")
    
    # Test Confidence Calibration Loss
    print("\n2. Testing Confidence Calibration Loss...")
    conf_loss = ConfidenceCalibrationLoss()
    loss = conf_loss(confidence_pred, count_pred, count_true)
    print(f"Confidence Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\n3. Testing Combined Loss...")
    combined_loss = CombinedLoss(
        primary_weight=0.8,
        confidence_weight=0.2
    )
    total_loss, loss_components = combined_loss(
        count_pred, confidence_pred, count_true
    )
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Primary Loss: {loss_components['primary_loss'].item():.4f}")
    print(f"Confidence Loss: {loss_components['confidence_loss'].item():.4f}")
    
    # Test Uncertainty Loss
    print("\n4. Testing Uncertainty Loss...")
    uncertainty_loss = UncertaintyLoss(lambda_reg=0.1)
    count_mean = torch.randn(batch_size) * 50 + 50
    count_std = torch.abs(torch.randn(batch_size)) * 5
    loss = uncertainty_loss(count_mean, count_std, count_true)
    print(f"Uncertainty Loss: {loss.item():.4f}")
    
    print("\nAll loss function tests passed!")
