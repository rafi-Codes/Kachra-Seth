# Stack Count Prediction - User Guide

Complete guide for using the Stack Count Prediction AI model.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Preparation](#data-preparation)
5. [Training the Model](#training-the-model)
6. [Running Inference](#running-inference)
7. [API Deployment](#api-deployment)
8. [Demo Interface](#demo-interface)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

## Overview

The Stack Count Prediction system uses deep learning to analyze images of stacked objects (banknotes, books, papers, tiles, cards, plates, etc.) and returns:
- A precise integer count of items in the stack
- A confidence score from 0.0 to 1.0
- A count range estimate when confidence is below 0.85

### Key Features

- **Dual-Head Architecture**: Counting head + confidence calibration
- **Monte Carlo Dropout**: Uncertainty quantification (15 forward passes)
- **Curriculum Learning**: Progressive training through count ranges
- **Multi-Category Support**: Works with various stacked objects
- **Edge Detection**: Blur and quality warnings
- **REST API**: Production-ready FastAPI endpoint
- **Interactive Demo**: Gradio web interface

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 20GB+ disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/rafi-Codes/Kachra-Seth.git
cd Kachra-Seth
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python scripts/setup_environment.py
```

## Quick Start

### 1. Prepare Your Data

Organize your images in the following structure:

```
data/
├── raw/
│   ├── banknotes/
│   │   ├── stack_001.jpg
│   │   ├── stack_002.jpg
│   │   └── ...
│   ├── books/
│   ├── papers/
│   └── ...
└── annotations/
    └── your_annotations.json
```

### 2. Create Annotations

Create annotation files in JSON format:

```json
[
  {
    "image_id": "stack_001.jpg",
    "category": "banknotes",
    "true_count": 87,
    "annotator_1_count": 87,
    "annotator_2_count": 88,
    "agreement_score": 0.99,
    "stack_angle": "45_degree",
    "lighting": "natural",
    "occlusion_percent": 5
  }
]
```

### 3. Validate Annotations

```python
from src.data import AnnotationValidator

# Validate your annotations
is_valid, errors = AnnotationValidator.validate_annotation_file('data/annotations/your_annotations.json')

if not is_valid:
    print("Errors:", errors)
else:
    print("Annotations are valid!")
```

### 4. Split Data

```python
from src.data import AnnotationParser, AnnotationSplitter

# Load annotations
annotations = AnnotationParser.load_annotations('data/annotations/your_annotations.json')

# Split into train/val/test
train, val, test = AnnotationSplitter.split_by_image(
    annotations,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Save splits
AnnotationSplitter.save_splits(train, val, test, 'data/annotations')
```

### 5. Train the Model

```bash
python src/training/train.py --config configs/config.yaml
```

### 6. Run Inference

```python
from src.inference import load_predictor

# Load trained model
predictor = load_predictor('models/checkpoints/best.pth')

# Make prediction
result = predictor.predict('path/to/image.jpg')

print(f"Count: {result['predicted_count']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Range: {result['count_range']['low']} - {result['count_range']['high']}")
```

## Data Preparation

### Image Requirements

- **Format**: JPEG, PNG, or WebP
- **Resolution**: Minimum 256x256, recommended 512x512
- **Lighting**: Various conditions (natural, bright, dim, backlit)
- **Angles**: Top-down, 45-degree, and side views
- **Stack Sizes**: 5 to 500 items

### Annotation Guidelines

Each image requires:
- `image_id`: Filename of the image
- `category`: Type of object (banknotes, books, papers, tiles, cards, plates)
- `true_count`: Ground truth count
- `annotator_1_count`: First annotator's count
- `annotator_2_count`: Second annotator's count
- `agreement_score`: Agreement between annotators (0-1)
- `stack_angle`: Viewing angle
- `lighting`: Lighting condition
- `occlusion_percent`: Percentage of occlusion (0-100)

### Data Augmentation

The pipeline automatically applies:
- Random rotation (±15°)
- Zoom (0.8x - 1.2x)
- Brightness jitter
- Gaussian noise
- Horizontal flips

### Synthetic Data Generation

For large stacks (200-500 items), generate synthetic data:

```python
from src.data import SyntheticDataGenerator

generator = SyntheticDataGenerator(blender_path='blender')
annotations = generator.generate_batch(
    output_dir='data/synthetic',
    count_range=(200, 500)
)
```

## Training the Model

### Basic Training

```bash
python src/training/train.py --config configs/config.yaml
```

### Training with Custom Options

```bash
python src/training/train.py \
    --config configs/config.yaml \
    --data_dir data/raw \
    --annotations_dir data/annotations \
    --experiment_name my_experiment
```

### Resume Training

```bash
python src/training/train.py \
    --config configs/config.yaml \
    --resume models/checkpoints/best.pth
```

### Debug Mode

```bash
python src/training/train.py \
    --config configs/config.yaml \
    --debug
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

### Training Phases

The model uses curriculum learning:

- **Phase 1 (Epochs 1-30)**: Stacks with 5-50 items
- **Phase 2 (Epochs 31-70)**: Stacks with 5-150 items
- **Phase 3 (Epochs 71-100)**: All stacks (5-500 items)

### Expected Training Time

- **GPU (RTX 3090)**: ~2-3 hours for 100 epochs
- **GPU (RTX 3060)**: ~4-5 hours for 100 epochs
- **CPU**: ~20-30 hours for 100 epochs

## Running Inference

### Python API

#### Single Image Prediction

```python
from src.inference import load_predictor

# Load predictor
predictor = load_predictor('models/checkpoints/best.pth', device='cuda')

# Predict
result = predictor.predict('image.jpg')

# Access results
print(f"Predicted: {result['predicted_count']} items")
print(f"Confidence: {result['confidence_score']:.3f}")
print(f"Range: {result['count_range']['low']}-{result['count_range']['high']}")
```

#### Batch Prediction

```python
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = predictor.predict_batch(images)

for i, result in enumerate(results):
    print(f"Image {i}: {result['predicted_count']} items")
```

#### Different Input Types

```python
# From file path
result = predictor.predict('path/to/image.jpg')

# From numpy array
import numpy as np
image = np.array(Image.open('image.jpg'))
result = predictor.predict(image)

# From PIL Image
from PIL import Image
image = Image.open('image.jpg')
result = predictor.predict(image)
```

### Confidence Thresholds

```python
# Adjust confidence thresholds
predictor.set_confidence_thresholds({
    'high': 0.90,      # Default
    'moderate': 0.70,  # Default
    'low': 0.50        # Default
})
```

## API Deployment

### Start the REST API

```bash
python src/inference/api.py
```

Or with uvicorn:

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

#### Batch Predict

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

#### Model Info

```bash
curl http://localhost:8000/model_info
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

COPY models/checkpoints/best.pth models/checkpoints/

EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t stack-count-prediction .
docker run -p 8000:8000 stack-count-prediction
```

## Demo Interface

### Start Gradio Demo

```bash
python src/inference/gradio_demo.py
```

### With Custom Model

```bash
python src/inference/gradio_demo.py \
    --model_path models/checkpoints/best.pth \
    --device cuda \
    --share
```

The demo will be available at `http://localhost:7860`

## Configuration

### Main Configuration File

Edit `configs/config.yaml` to customize:

```yaml
# Data configuration
data:
  categories: ["banknotes", "books", "papers", "tiles", "cards", "plates"]
  image_size: 384
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

# Model configuration
model:
  backbone:
    name: "efficientnet"
    variant: "b4"
    pretrained: true
    freeze_early_layers: true

# Training configuration
training:
  batch_size: 32
  epochs: 100
  optimizer:
    lr: 0.0001
    weight_decay: 0.00001
  early_stopping:
    patience: 15
```

### Performance Targets

The model aims to achieve:
- MAE ≤ 3 for stacks 5-50 items
- MAE ≤ 5 for stacks 51-150 items
- MAPE ≤ 8% for stacks 151-500 items
- Confidence ECE ≤ 0.05
- Within-5% accuracy ≥ 85% on test set
- Inference latency ≤ 200ms on GPU

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size in config
- Use smaller model variant
- Enable gradient accumulation

**Model Not Loading**
- Check checkpoint path exists
- Verify device compatibility
- Ensure PyTorch version matches training

**Poor Predictions**
- Verify data quality
- Check annotation accuracy
- Ensure similar distribution to training data
- Consider retraining with more data

**Slow Inference**
- Reduce MC Dropout passes
- Use smaller model variant
- Ensure using GPU
- Optimize image preprocessing

### Getting Help

- Check documentation in `docs/` directory
- Review example scripts in `examples/`
- Examine error logs in `logs/`
- Open an issue on GitHub

## Best Practices

1. **Always validate annotations** before training
2. **Use curriculum learning** for better convergence
3. **Monitor validation metrics** during training
4. **Check confidence scores** in production
5. **Handle edge cases** (blur, no stack)
6. **Log predictions** for active learning
7. **Regularly retrain** with new data
8. **Monitor production performance**

## Advanced Usage

See detailed documentation:
- [Data Pipeline](docs/data_pipeline.md)
- [Model Architecture](docs/model_architecture.md)
- [Training Pipeline](docs/training_pipeline.md)
- [Inference Pipeline](docs/inference_pipeline.md)

## License

MIT License - See LICENSE file for details
