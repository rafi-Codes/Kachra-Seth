"""Example script demonstrating training pipeline usage."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import create_default_model
from src.training import (
    CombinedLoss,
    MetricsTracker,
    CountingMetrics,
    ConfidenceMetrics,
    WarmupCosineScheduler
)


def example_loss_functions():
    """Demonstrate loss functions."""
    print("=" * 60)
    print("Loss Functions Example")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 8
    count_pred = torch.randn(batch_size) * 50 + 50
    count_true = torch.randn(batch_size) * 50 + 50
    confidence_pred = torch.sigmoid(torch.randn(batch_size))
    
    # Combined loss
    loss_fn = CombinedLoss(primary_weight=0.8, confidence_weight=0.2)
    total_loss, loss_components = loss_fn(count_pred, confidence_pred, count_true)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Primary (Counting) Loss: {loss_components['primary_loss'].item():.4f}")
    print(f"Confidence Loss: {loss_components['confidence_loss'].item():.4f}")


def example_metrics():
    """Demonstrate metrics calculation."""
    print("\n" + "=" * 60)
    print("Metrics Example")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 32
    count_pred = torch.randn(batch_size) * 50 + 50
    count_true = torch.randn(batch_size) * 50 + 50
    confidence_pred = torch.sigmoid(torch.randn(batch_size))
    
    # Counting metrics
    mae = CountingMetrics.mae(count_pred, count_true)
    mape = CountingMetrics.mape(count_pred, count_true)
    accuracy = CountingMetrics.within_accuracy(count_pred, count_true, 5.0)
    
    print(f"MAE: {mae.item():.4f}")
    print(f"MAPE: {mape.item():.4f}%")
    print(f"Within 5% accuracy: {accuracy.item():.4f}")
    
    # Confidence metrics
    ece, _ = ConfidenceMetrics.expected_calibration_error(
        confidence_pred, count_pred, count_true
    )
    print(f"ECE: {ece.item():.4f}")


def example_metrics_tracker():
    """Demonstrate metrics tracking over multiple batches."""
    print("\n" + "=" * 60)
    print("Metrics Tracker Example")
    print("=" * 60)
    
    tracker = MetricsTracker()
    
    # Simulate multiple batches
    for _ in range(5):
        batch_size = 16
        count_pred = torch.randn(batch_size) * 50 + 50
        count_true = torch.randn(batch_size) * 50 + 50
        confidence_pred = torch.sigmoid(torch.randn(batch_size))
        
        loss = torch.randn(1)
        tracker.update(count_pred, count_true, confidence_pred, loss_dict={'loss': loss})
    
    # Compute overall metrics
    metrics = tracker.compute()
    
    print("Overall metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")


def example_lr_scheduler():
    """Demonstrate learning rate scheduler with warmup."""
    print("\n" + "=" * 60)
    print("Learning Rate Scheduler Example")
    print("=" * 60)
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=100
    )
    
    print("Learning rate schedule (first 20 epochs):")
    for epoch in range(1, 21):
        scheduler.step()
        lr = scheduler.get_lr()
        print(f"  Epoch {epoch:2d}: {lr:.2e}")


def example_simple_training_step():
    """Demonstrate a simple training step."""
    print("\n" + "=" * 60)
    print("Simple Training Step Example")
    print("=" * 60)
    
    # Create dummy model
    model = create_default_model(pretrained=False)
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 384, 384)
    counts = torch.randint(10, 100, (batch_size,)).float()
    
    # Forward pass
    outputs = model(images)
    count_pred = outputs['count']
    confidence_pred = outputs['confidence']
    
    # Calculate loss
    loss_fn = CombinedLoss()
    loss, loss_components = loss_fn(count_pred, confidence_pred, counts)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Count predictions: {count_pred}")
    print(f"True counts: {counts}")
    print(f"Confidence predictions: {confidence_pred}")


def main():
    """Run all examples."""
    print("Stack Count Prediction - Training Pipeline Examples")
    print("=" * 60)
    
    # Run examples
    example_loss_functions()
    example_metrics()
    example_metrics_tracker()
    example_lr_scheduler()
    example_simple_training_step()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
