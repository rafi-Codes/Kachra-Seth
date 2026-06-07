# Training Pipeline Documentation

## Overview

The training pipeline provides a complete solution for training the stack count prediction model with curriculum learning, comprehensive metrics tracking, and experiment management.

## Components

### 1. Loss Functions (`src/training/losses.py`)

**Huber Loss** - Robust regression loss for counting
- Less sensitive to outliers than MSE
- Quadratic for small errors, linear for large errors
- Delta parameter controls transition point

**Confidence Calibration Loss** - BCE for confidence scores
- Target confidence based on prediction accuracy
- Encourages well-calibrated confidence estimates
- Optional label smoothing for regularization

**Combined Loss** - Weighted sum of both losses
```python
total_loss = 0.8 * primary_loss + 0.2 * confidence_loss
```

### 2. Metrics (`src/training/metrics.py`)

**Counting Metrics:**
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- Within-X% accuracy
- Absolute tolerance accuracy

**Confidence Metrics:**
- ECE (Expected Calibration Error)
- Brier Score
- Negative Log Likelihood

**MetricsTracker:**
- Track metrics over batches
- Compute epoch-level statistics
- Stratified metrics by count range
- Performance threshold checking

### 3. Trainer (`src/training/trainer.py`)

**StackCountTrainer** - Main training class

Features:
- Curriculum learning (3 phases)
- Automatic phase transitions
- Validation after each epoch
- Early stopping with patience
- Model checkpointing
- Gradient clipping
- Progress bars with tqdm

### 4. Schedulers (`src/training/schedulers.py`)

**WarmupCosineScheduler** - Warmup + Cosine Annealing
- Linear warmup for first N epochs
- Cosine annealing for remaining epochs
- Prevents unstable early training

**Configuration-based scheduler creation:**
- cosine_annealing (default)
- warmup
- step
- exponential

### 5. Logger (`src/training/logger.py`)

**TrainingLogger** - Comprehensive logging
- Console logging with timestamps
- File logging
- TensorBoard integration
- Configuration logging
- Metric history tracking
- Experiment naming

**ExperimentTracker** - Compare experiments
- Track all experiments
- Find best performing configuration
- Generate comparison reports

## Usage

### Basic Training

```python
from src.training.train import main
import sys

# Run with default configuration
sys.argv = ['train.py', '--config', 'configs/config.yaml']
main()
```

### Command Line Training

```bash
# Basic training
python src/training/train.py --config configs/config.yaml

# Custom data paths
python src/training/train.py \
    --config configs/config.yaml \
    --data_dir data/raw \
    --annotations_dir data/annotations

# Resume from checkpoint
python src/training/train.py \
    --config configs/config.yaml \
    --resume models/checkpoints/best.pth

# Debug mode (small dataset, few epochs)
python src/training/train.py \
    --config configs/config.yaml \
    --debug

# CPU training
python src/training/train.py \
    --config configs/config.yaml \
    --device cpu
```

### Programmatic Training

```python
import torch
from src.models import create_model_from_config
from src.data import create_data_loaders_from_config
from src.training import StackCountTrainer, TrainingLogger
from src.utils.config import load_config

# Load configuration
config = load_config('configs/config.yaml')

# Create model
model = create_model_from_config(config)

# Create data loaders
data_loader = create_data_loaders_from_config(
    config=config,
    data_dir='data/raw',
    annotations_dir='data/annotations',
    current_phase=1
)

# Create trainer
trainer = StackCountTrainer(
    model=model,
    train_loader=data_loader.get_train_loader(),
    val_loader=data_loader.get_val_loader(),
    config=config,
    device='cuda'
)

# Train
history = trainer.train()
```

### Custom Training Loop

```python
from src.training import CombinedLoss, MetricsTracker
import torch

# Setup
model = create_default_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = CombinedLoss()
metrics = MetricsTracker()

# Training loop
for epoch in range(100):
    model.train()
    
    for batch in train_loader:
        images = batch['image'].cuda()
        counts = batch['count'].cuda()
        
        # Forward pass
        outputs = model(images)
        count_pred = outputs['count']
        confidence_pred = outputs['confidence']
        
        # Calculate loss
        loss, loss_components = loss_fn(count_pred, confidence_pred, counts)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        metrics.update(count_pred, counts, confidence_pred, loss_dict=loss_components)
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    print(f"Epoch {epoch}: MAE={epoch_metrics['mae']:.4f}")
```

## Curriculum Learning

### Phase 1 (Epochs 1-30)
- **Data**: Stacks with 5-50 items
- **Backbone**: Frozen except last 3 blocks
- **Purpose**: Learn basic counting patterns

### Phase 2 (Epochs 31-70)
- **Data**: Add stacks with 51-150 items
- **Backbone**: Unfreeze additional layers
- **Purpose**: Handle medium-sized stacks

### Phase 3 (Epochs 71-100)
- **Data**: All stacks (5-500 items)
- **Backbone**: Fully trainable
- **Purpose**: Generalize to all stack sizes

### Manual Phase Control

```python
# Update curriculum phase during training
if epoch == 31:
    data_loader.update_curriculum_phase(2)
elif epoch == 71:
    data_loader.update_curriculum_phase(3)
```

## Configuration

Training configuration in `configs/config.yaml`:

```yaml
training:
  # Loss weights
  loss_weights:
    primary_loss: 0.8
    confidence_loss: 0.2
  
  # Optimizer
  optimizer:
    name: "adamw"
    lr: 0.0001
    weight_decay: 0.00001
  
  # Learning rate scheduler
  scheduler:
    name: "cosine_annealing"
    warmup_epochs: 5
    T_max: 100
  
  # Training parameters
  batch_size: 32
  epochs: 100
  gradient_clip_norm: 1.0
  
  # Early stopping
  early_stopping:
    patience: 15
    monitor: "val_mae"
  
  # Curriculum learning
  curriculum:
    phase1:
      epochs: [1, 30]
      count_range: [5, 50]
    phase2:
      epochs: [31, 70]
      count_range: [5, 150]
    phase3:
      epochs: [71, 100]
      count_range: [5, 500]
```

## Metrics

### Target Performance

From requirements:
- MAE ≤ 3 for stacks 5-50
- MAE ≤ 5 for stacks 51-150
- MAPE ≤ 8% for stacks 151-500
- ECE ≤ 0.05
- Within 5% accuracy ≥ 85%

### Monitoring Metrics

Track during training:
- Train/Val Loss
- Train/Val MAE
- Val MAPE
- Val Within 5% accuracy
- Val ECE (confidence calibration)
- Stratified MAE by count range

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# View in browser
# http://localhost:6006
```

## Checkpointing

### Automatic Checkpointing
- **Best checkpoint**: Saved when validation MAE improves
- **Regular checkpoints**: Saved every 5 epochs
- **Location**: `models/checkpoints/`

### Load Checkpoint

```python
# In training script
checkpoint = torch.load('models/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Resume Training

```bash
python src/training/train.py \
    --config configs/config.yaml \
    --resume models/checkpoints/best.pth
```

## Experiment Management

### Tracking Experiments

```python
from src.training import ExperimentTracker

tracker = ExperimentTracker('logs/experiment_results.json')

tracker.add_experiment(
    experiment_name='experiment_1',
    config=config,
    metrics={'val_mae': 3.2, 'val_mape': 4.5},
    training_time=3600,
    notes="Baseline model"
)

# Get best experiment
best = tracker.get_best_experiment('val_mae', mode='min')

# Compare experiments
report = tracker.compare_experiments()
print(report)
```

## Troubleshooting

### Poor Convergence
- Check learning rate (try 1e-5 to 1e-4)
- Verify data quality and annotations
- Ensure backbone freezing is appropriate
- Try different batch sizes

### Overfitting
- Increase dropout rates
- Add more data augmentation
- Implement stronger regularization
- Reduce model capacity

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Use smaller backbone variant
- Enable mixed precision training

### Slow Training
- Use smaller backbone
- Reduce image size
- Optimize data loading
- Use mixed precision (AMP)

## Best Practices

1. **Always monitor validation metrics** - Don't just look at training loss
2. **Use curriculum learning** - Progress through count ranges gradually
3. **Check calibration** - Monitor ECE metric
4. **Save checkpoints regularly** - Don't lose progress
5. **Log experiments** - Track what works and what doesn't
6. **Validate on stratified metrics** - Ensure performance across all count ranges
7. **Use early stopping** - Prevent overfitting
8. **Monitor learning rate** - Ensure warmup is working correctly

## Next Steps

After training:
1. Evaluate on test set with comprehensive metrics
2. Analyze failure cases and edge cases
3. Perform hyperparameter tuning
4. Optimize for deployment
5. Build inference API
