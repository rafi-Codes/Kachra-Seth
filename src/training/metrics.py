"""Training and evaluation metrics for stack count prediction."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class CountingMetrics:
    """Metrics for counting task."""
    
    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Error.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            
        Returns:
            MAE scalar
        """
        return torch.abs(predictions - targets).mean()
    
    @staticmethod
    def mape(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Mean Absolute Percentage Error.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE scalar (as percentage)
        """
        abs_error = torch.abs(predictions - targets)
        percentage_error = abs_error / (targets.abs() + epsilon)
        return percentage_error.mean() * 100
    
    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Root Mean Square Error.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            
        Returns:
            RMSE scalar
        """
        return torch.sqrt(((predictions - targets) ** 2).mean())
    
    @staticmethod
    def within_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        percentage: float = 5.0
    ) -> torch.Tensor:
        """Percentage of predictions within X% of true value.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            percentage: Percentage threshold
            
        Returns:
            Accuracy scalar (0-1)
        """
        abs_error = torch.abs(predictions - targets)
        threshold = percentage / 100.0 * targets.abs()
        within_threshold = (abs_error <= threshold).float()
        return within_threshold.mean()
    
    @staticmethod
    def absolute_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        tolerance: int = 5
    ) -> torch.Tensor:
        """Percentage of predictions within absolute tolerance.
        
        Args:
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            tolerance: Absolute tolerance
            
        Returns:
            Accuracy scalar (0-1)
        """
        abs_error = torch.abs(predictions - targets)
        within_tolerance = (abs_error <= tolerance).float()
        return within_tolerance.mean()


class ConfidenceMetrics:
    """Metrics for confidence calibration."""
    
    @staticmethod
    def expected_calibration_error(
        confidence_scores: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Calculate Expected Calibration Error (ECE).
        
        Args:
            confidence_scores: Predicted confidence scores (B,)
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            n_bins: Number of bins for calibration
            
        Returns:
            Tuple of (ECE scalar, bin_statistics_list)
        """
        # Calculate accuracy (binary: accurate if within 5%)
        accuracy = CountingMetrics.within_accuracy(predictions, targets, percentage=5.0)
        
        # Sort by confidence
        indices = torch.argsort(confidence_scores)
        confidence_sorted = confidence_scores[indices]
        accuracy_sorted = accuracy[indices]
        
        # Calculate ECE
        ece = 0.0
        bin_stats = []
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            # Get bin indices
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            in_bin = (confidence_sorted >= lower) & (confidence_sorted < upper)
            
            if in_bin.sum() > 0:
                bin_confidence = confidence_sorted[in_bin].mean()
                bin_accuracy = accuracy_sorted[in_bin].mean()
                bin_weight = in_bin.float().mean()
                
                bin_stats.append({
                    'bin': i,
                    'lower': lower.item(),
                    'upper': upper.item(),
                    'confidence': bin_confidence.item(),
                    'accuracy': bin_accuracy.item(),
                    'weight': bin_weight.item(),
                    'count': in_bin.sum().item()
                })
                
                ece += bin_weight * torch.abs(bin_confidence - bin_accuracy)
        
        return ece, bin_stats
    
    @staticmethod
    def brier_score(
        confidence_scores: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Brier score for confidence calibration.
        
        Args:
            confidence_scores: Predicted confidence scores (B,)
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            
        Returns:
            Brier score scalar
        """
        # Calculate accuracy
        accuracy = CountingMetrics.within_accuracy(predictions, targets, percentage=5.0)
        
        # Brier score: mean squared error between confidence and accuracy
        brier = ((confidence_scores - accuracy) ** 2).mean()
        
        return brier
    
    @staticmethod
    def negative_log_likelihood(
        confidence_scores: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Calculate negative log likelihood for confidence.
        
        Args:
            confidence_scores: Predicted confidence scores (B,)
            predictions: Predicted counts (B,)
            targets: True counts (B,)
            epsilon: Small value to avoid log(0)
            
        Returns:
            NLL scalar
        """
        # Calculate target confidence based on error
        abs_error = torch.abs(predictions - targets)
        mean_count = targets.abs().mean() + epsilon
        target_confidence = 1.0 / (1.0 + abs_error / mean_count)
        
        # Negative log likelihood
        nll = -(
            target_confidence * torch.log(confidence_scores + epsilon) +
            (1 - target_confidence) * torch.log(1 - confidence_scores + epsilon)
        ).mean()
        
        return nll


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.count_predictions = []
        self.count_targets = []
        self.confidence_predictions = []
        self.count_means = []
        self.count_stds = []
        self.losses = defaultdict(list)
    
    def update(
        self,
        count_pred: torch.Tensor,
        count_true: torch.Tensor,
        confidence_pred: Optional[torch.Tensor] = None,
        count_mean: Optional[torch.Tensor] = None,
        count_std: Optional[torch.Tensor] = None,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Update tracked metrics with new batch.
        
        Args:
            count_pred: Predicted counts
            count_true: True counts
            confidence_pred: Predicted confidence (optional)
            count_mean: Mean count from MC Dropout (optional)
            count_std: Std from MC Dropout (optional)
            loss_dict: Dictionary of losses (optional)
        """
        # Move to CPU and convert to numpy
        self.count_predictions.append(count_pred.detach().cpu())
        self.count_targets.append(count_true.detach().cpu())
        
        if confidence_pred is not None:
            self.confidence_predictions.append(confidence_pred.detach().cpu())
        
        if count_mean is not None:
            self.count_means.append(count_mean.detach().cpu())
        
        if count_std is not None:
            self.count_stds.append(count_std.detach().cpu())
        
        if loss_dict is not None:
            for key, value in loss_dict.items():
                self.losses[key].append(value.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics from tracked data.
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Concatenate all batches
        if len(self.count_predictions) == 0:
            return metrics
        
        count_pred = torch.cat(self.count_predictions)
        count_true = torch.cat(self.count_targets)
        
        # Counting metrics
        metrics['mae'] = CountingMetrics.mae(count_pred, count_true).item()
        metrics['mape'] = CountingMetrics.mape(count_pred, count_true).item()
        metrics['rmse'] = CountingMetrics.rmse(count_pred, count_true).item()
        metrics['within_5pct'] = CountingMetrics.within_accuracy(count_pred, count_true, 5.0).item()
        metrics['within_10pct'] = CountingMetrics.within_accuracy(count_pred, count_true, 10.0).item()
        metrics['within_abs_5'] = CountingMetrics.absolute_accuracy(count_pred, count_true, 5).item()
        
        # Confidence metrics
        if len(self.confidence_predictions) > 0:
            confidence_pred = torch.cat(self.confidence_predictions)
            
            ece, _ = ConfidenceMetrics.expected_calibration_error(
                confidence_pred, count_pred, count_true
            )
            metrics['confidence_ece'] = ece.item()
            
            brier = ConfidenceMetrics.brier_score(
                confidence_pred, count_pred, count_true
            )
            metrics['confidence_brier'] = brier.item()
        
        # Uncertainty metrics
        if len(self.count_means) > 0 and len(self.count_stds) > 0:
            count_mean = torch.cat(self.count_means)
            count_std = torch.cat(self.count_stds)
            metrics['mean_uncertainty'] = count_std.mean().item()
        
        # Loss metrics
        for key, values in self.losses.items():
            if len(values) > 0:
                metrics[f'loss_{key}'] = torch.cat(values).mean().item()
        
        return metrics
    
    def get_strata_metrics(
        self,
        count_ranges: List[Tuple[int, int]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics stratified by count range.
        
        Args:
            count_ranges: List of (min, max) count ranges
            
        Returns:
            Dictionary mapping range names to metric dictionaries
        """
        strata_metrics = {}
        
        if len(self.count_predictions) == 0:
            return strata_metrics
        
        count_pred = torch.cat(self.count_predictions)
        count_true = torch.cat(self.count_targets)
        
        for i, (min_count, max_count) in enumerate(count_ranges):
            range_name = f"{min_count}-{max_count}"
            
            # Filter by range
            mask = (count_true >= min_count) & (count_true <= max_count)
            
            if mask.sum() > 0:
                range_pred = count_pred[mask]
                range_true = count_true[mask]
                
                strata_metrics[range_name] = {
                    'mae': CountingMetrics.mae(range_pred, range_true).item(),
                    'mape': CountingMetrics.mape(range_pred, range_true).item(),
                    'within_5pct': CountingMetrics.within_accuracy(range_pred, range_true, 5.0).item(),
                    'count': mask.sum().item()
                }
        
        return strata_metrics


class PerformanceThresholds:
    """Performance thresholds for model evaluation."""
    
    # Target metrics from requirements
    TARGETS = {
        'mae_small': 3.0,      # MAE ≤ 3 for stacks 5-50
        'mae_medium': 5.0,     # MAE ≤ 5 for stacks 51-150
        'mape_large': 8.0,     # MAPE ≤ 8% for stacks 151-500
        'confidence_ece': 0.05, # ECE ≤ 0.05
        'within_5pct': 0.85,   # Within 5% accuracy ≥ 85%
        'inference_latency_ms': 200  # Latency ≤ 200ms
    }
    
    @staticmethod
    def check_metrics(metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if metrics meet target thresholds.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Dictionary of threshold check results
        """
        results = {}
        
        # Check overall metrics
        if 'mae' in metrics:
            results['mae_met'] = metrics['mae'] <= PerformanceThresholds.TARGETS['mae_small']
        
        if 'confidence_ece' in metrics:
            results['ece_met'] = metrics['confidence_ece'] <= PerformanceThresholds.TARGETS['confidence_ece']
        
        if 'within_5pct' in metrics:
            results['accuracy_met'] = metrics['within_5pct'] >= PerformanceThresholds.TARGETS['within_5pct']
        
        return results
    
    @staticmethod
    def check_strata_metrics(strata_metrics: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """Check if stratified metrics meet thresholds.
        
        Args:
            strata_metrics: Dictionary of stratified metrics
            
        Returns:
            Dictionary of threshold check results
        """
        results = {}
        
        # Check small stacks (5-50)
        if '5-50' in strata_metrics:
            results['mae_small_met'] = strata_metrics['5-50']['mae'] <= PerformanceThresholds.TARGETS['mae_small']
        
        # Check medium stacks (51-150)
        if '51-150' in strata_metrics:
            results['mae_medium_met'] = strata_metrics['51-150']['mae'] <= PerformanceThresholds.TARGETS['mae_medium']
        
        # Check large stacks (151-500)
        if '151-500' in strata_metrics:
            results['mape_large_met'] = strata_metrics['151-500']['mape'] <= PerformanceThresholds.TARGETS['mape_large']
        
        return results
    
    @staticmethod
    def print_results(check_results: Dict[str, bool]):
        """Print threshold check results.
        
        Args:
            check_results: Dictionary of check results
        """
        print("\nPerformance Threshold Checks:")
        print("=" * 50)
        
        for metric, passed in check_results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{metric:25s}: {status}")
        
        print("=" * 50)


# Test metrics
if __name__ == "__main__":
    print("Testing Metrics...")
    
    # Create dummy data
    batch_size = 32
    count_pred = torch.randn(batch_size) * 50 + 50
    count_true = torch.randn(batch_size) * 50 + 50
    confidence_pred = torch.sigmoid(torch.randn(batch_size))
    
    # Test CountingMetrics
    print("\n1. Testing CountingMetrics...")
    mae = CountingMetrics.mae(count_pred, count_true)
    mape = CountingMetrics.mape(count_pred, count_true)
    rmse = CountingMetrics.rmse(count_pred, count_true)
    accuracy = CountingMetrics.within_accuracy(count_pred, count_true, 5.0)
    
    print(f"MAE: {mae.item():.4f}")
    print(f"MAPE: {mape.item():.4f}%")
    print(f"RMSE: {rmse.item():.4f}")
    print(f"Within 5% accuracy: {accuracy.item():.4f}")
    
    # Test ConfidenceMetrics
    print("\n2. Testing ConfidenceMetrics...")
    ece, bin_stats = ConfidenceMetrics.expected_calibration_error(
        confidence_pred, count_pred, count_true
    )
    print(f"ECE: {ece.item():.4f}")
    print(f"Number of bins: {len(bin_stats)}")
    
    brier = ConfidenceMetrics.brier_score(confidence_pred, count_pred, count_true)
    print(f"Brier score: {brier.item():.4f}")
    
    # Test MetricsTracker
    print("\n3. Testing MetricsTracker...")
    tracker = MetricsTracker()
    tracker.update(count_pred, count_true, confidence_pred)
    tracker.update(count_pred, count_true, confidence_pred)
    
    metrics = tracker.compute()
    print("Computed metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test PerformanceThresholds
    print("\n4. Testing PerformanceThresholds...")
    check_results = PerformanceThresholds.check_metrics(metrics)
    PerformanceThresholds.print_results(check_results)
    
    print("\nAll metric tests passed!")
