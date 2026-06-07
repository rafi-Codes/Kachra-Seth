# Model Architecture Documentation

## Overview

The model architecture for stack count prediction uses a dual-head approach with a powerful backbone network, Monte Carlo Dropout for uncertainty estimation, and optional extensions for improved accuracy.

## Architecture Components

### 1. Backbone Networks (`src/models/backbones.py`)

#### EfficientNet-B4
- **Purpose**: Feature extraction from input images
- **Input**: (B, 3, 384, 384) - RGB images
- **Output**: (B, 1792) - Feature vector
- **Pretraining**: ImageNet
- **Freezing**: Early layers frozen initially, last 3 blocks trainable

#### ConvNeXt-Small (Alternative)
- **Purpose**: Modern CNN architecture alternative
- **Input**: (B, 3, 384, 384)
- **Output**: (B, 768) - Feature vector
- **Pretraining**: ImageNet-22K
- **Freezing**: Early stages frozen initially, last 2 stages trainable

#### ResNet-50 (Alternative)
- **Purpose**: Classic backbone option
- **Input**: (B, 3, 384, 384)
- **Output**: (B, 2048) - Feature vector
- **Pretraining**: ImageNet
- **Freezing**: Configurable layer freezing

### 2. Dual-Head Architecture (`src/models/heads.py`)

#### Counting Head (Regression)
```
Input (B, feature_dim) → Dense(512, ReLU) → Dropout(0.3) → 
Dense(256, ReLU) → Dense(1, linear) → Count (B,)
```
- **Purpose**: Predict exact item count
- **Output**: Single scalar per image
- **Loss**: Huber Loss for robust regression

#### Confidence Head (Calibration)
```
Input (B, feature_dim) → Dense(256, ReLU) → Dropout(0.2) → 
Dense(128, ReLU) → Dense(1, sigmoid) → Confidence (B,)
```
- **Purpose**: Predict confidence score [0, 1]
- **Output**: Single scalar per image
- **Loss**: Binary Cross Entropy for calibration

### 3. Main Model (`src/models/stack_count_model.py`)

The `StackCountModel` integrates all components:

```
Input Image (B, 3, 384, 384)
    ↓
Backbone (EfficientNet-B4/ConvNeXt-Small)
    ↓
Feature Vector (B, 1792/768)
    ↓
    ├──→ Counting Head → Count Prediction (B,)
    └──→ Confidence Head → Confidence Score (B,)
```

#### Key Features

**Monte Carlo Dropout**
- Enabled by default for uncertainty estimation
- 15 forward passes at inference
- Mean prediction = final count
- Standard deviation = uncertainty
- Confidence = sigmoid(1 / (1 + std))

**Curriculum Learning Support**
- Phase 1: Train on counts 5-50
- Phase 2: Add counts 51-150
- Phase 3: Add counts 151-500

**Optional Extensions**
- Density Map Branch (CSRNet-style) for large stacks
- Category Classification Head for mixed stacks
- Attention Mechanisms (SE, CBAM) for edge focus

### 4. Attention Mechanisms (`src/models/attention.py`)

#### SE (Squeeze-and-Excitation)
- Channel-wise attention
- Computationally efficient
- Focuses on important feature channels

#### CBAM (Convolutional Block Attention Module)
- Channel attention + Spatial attention
- More comprehensive attention mechanism
- Better for edge detection

#### Self-Attention
- Global context modeling
- Captures long-range dependencies
- Higher computational cost

#### Edge Attention
- Multi-scale edge detection
- Focuses on stack boundaries
- Specialized for counting task

### 5. Model Factory (`src/models/factory.py`)

Predefined configurations:
- **default**: EfficientNet-B4, balanced performance
- **convnext**: ConvNeXt-Small alternative
- **lightweight**: EfficientNet-B0 for fast inference
- **high_accuracy**: All features enabled
- **mobile**: Optimized for mobile deployment

## Usage Examples

### Basic Model Creation

```python
from src.models import create_default_model

# Create default model
model = create_default_model()

# Get parameter count
params = model.get_num_parameters()
print(f"Total parameters: {params['total']:,}")
```

### Custom Model Configuration

```python
from src.models import ModelFactory

model = ModelFactory.create_model(
    backbone_type='convnext',
    backbone_variant='small',
    counting_hidden=[512, 256, 128],
    confidence_hidden=[256, 128],
    activation='gelu',
    counting_dropout=0.4,
    use_density_map=True,
    mc_dropout=True
)
```

### Forward Pass

```python
import torch

# Create input tensor
x = torch.randn(4, 3, 384, 384)

# Forward pass
outputs = model(x)

count = outputs['count']  # Shape: (4,)
confidence = outputs['confidence']  # Shape: (4,)
```

### Monte Carlo Dropout Inference

```python
# Predict with uncertainty
predictions = model.predict_with_uncertainty(x, n_samples=15)

count_mean = predictions['count_mean']
count_std = predictions['count_std']
confidence = predictions['uncertainty_based_confidence']

print(f"Predicted count: {count_mean}")
print(f"Uncertainty: ±{count_std}")
print(f"Confidence: {confidence}")
```

### Get Prediction Summary

```python
summary = model.get_prediction_summary(x, n_samples=15)

print(f"Predicted: {summary['predicted_count']}")
print(f"Confidence: {summary['confidence_score']}")
print(f"Range: [{summary['count_range']['low']}, {summary['count_range']['high']}]")
print(f"Uncertainty level: {summary['uncertainty_level']}")
```

### Backbone Freezing

```python
# Freeze backbone (default)
model.freeze_backbone()

# Unfreeze last 3 layers
model.unfreeze_backbone(n_layers=3)

# Unfreeze all
model.unfreeze_backbone(n_layers=10)
```

### Model from Configuration

```python
from src.models import StackCountModel
from src.utils.config import load_config

# Load config
config = load_config('configs/config.yaml')

# Create model
model = StackCountModel.create_model_from_config(config)
```

## Configuration

Model configuration in `configs/config.yaml`:

```yaml
model:
  backbone:
    name: "efficientnet"  # or "convnext"
    variant: "b4"  # or "small"
    pretrained: true
    freeze_early_layers: true
  
  counting_head:
    hidden_layers: [512, 256]
    activation: "relu"
    dropout: 0.3
  
  confidence_head:
    hidden_layers: [256, 128]
    activation: "relu"
    dropout: 0.2
    output_activation: "sigmoid"
  
  use_density_map: false
  use_attention: false
  attention_type: "cbam"
```

## Performance Characteristics

### Parameter Counts

| Model Variant | Total Parameters | Trainable Parameters |
|--------------|------------------|---------------------|
| EfficientNet-B4 Default | ~19M | ~5M (initial) |
| ConvNeXt-Small | ~28M | ~8M (initial) |
| Lightweight (B0) | ~5M | ~3M |
| High Accuracy | ~20M+ | ~8M+ |

### Inference Speed

| Model Variant | Inference Time (ms) | Memory (GB) |
|--------------|---------------------|-------------|
| Lightweight | ~50 | ~1.5 |
| Default | ~150 | ~2.5 |
| High Accuracy | ~200+ | ~3.5+ |

## Training Strategy

### Phase 1: Initial Training (Epochs 1-30)
- Backbone: Frozen except last 3 blocks
- Data: Stacks with 5-50 items
- Learning Rate: 1e-4
- Purpose: Learn basic counting patterns

### Phase 2: Intermediate Training (Epochs 31-70)
- Backbone: Unfreeze additional layers
- Data: Add stacks with 51-150 items
- Learning Rate: Reduce with cosine annealing
- Purpose: Handle medium-sized stacks

### Phase 3: Advanced Training (Epochs 71-100)
- Backbone: Fully trainable
- Data: All stacks (5-500 items)
- Learning Rate: Continue annealing
- Purpose: Generalize to all stack sizes

## Loss Function

```python
total_loss = 0.8 * primary_loss + 0.2 * confidence_loss

# Primary loss: Huber Loss for robust regression
primary_loss = HuberLoss(predicted_count, true_count)

# Confidence loss: Binary Cross Entropy for calibration
confidence_loss = BCELoss(predicted_confidence, target_confidence)
```

## Best Practices

1. **Start with frozen backbone** - Train heads first
2. **Gradual unfreezing** - Unfreeze backbone layers progressively
3. **Use MC Dropout** - Essential for reliable uncertainty estimates
4. **Monitor calibration** - Check confidence ECE metric
5. **Validate edge cases** - Test on blurry images, mixed stacks
6. **Use curriculum learning** - Progress through count ranges
7. **Regular checkpointing** - Save models at each phase transition

## Troubleshooting

### Overfitting
- Increase dropout rates
- Add more data augmentation
- Use stronger regularization
- Reduce model complexity

### Underfitting
- Unfreeze more backbone layers
- Increase model capacity
- Train for more epochs
- Reduce regularization

### Poor Calibration
- Increase confidence loss weight
- Use temperature scaling
- Check MC Dropout samples
- Validate on diverse data

### Slow Inference
- Disable MC Dropout for deployment
- Use smaller backbone variant
- Optimize with ONNX/TFLite
- Reduce MC Dropout samples

## Next Steps

After implementing the model architecture:
1. Set up training pipeline with curriculum learning
2. Implement comprehensive evaluation metrics
3. Create inference API with uncertainty quantification
4. Optimize for deployment (ONNX, TFLite, CoreML)
