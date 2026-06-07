# Quick Start Guide

Get started with Stack Count Prediction in 5 minutes.

## Prerequisites

- Python 3.8+
- CUDA GPU (optional but recommended)
- Git

## Installation

```bash
# Clone repository
git clone https://github.com/rafi-Codes/Kachra-Seth.git
cd Kachra-Seth

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test (No Training Required)

### 1. Test Model Architecture

```python
from src.models import create_default_model

# Create model
model = create_default_model(pretrained=False)

# Test forward pass
import torch
x = torch.randn(1, 3, 384, 384)
outputs = model(x)

print(f"Count prediction: {outputs['count'].item()}")
print(f"Confidence: {outputs['confidence'].item()}")
```

### 2. Test Data Pipeline

```python
from src.data import AnnotationValidator, AnnotationParser

# Create sample annotation
sample_ann = {
    'image_id': 'test.jpg',
    'category': 'banknotes',
    'true_count': 50,
    'annotator_1_count': 50,
    'annotator_2_count': 50,
    'agreement_score': 1.0,
    'stack_angle': 'top_down',
    'lighting': 'natural',
    'occlusion_percent': 0
}

# Validate
is_valid, errors = AnnotationValidator.validate_annotation(sample_ann)
print(f"Valid: {is_valid}")
```

### 3. Test Inference (with dummy model)

```python
from src.models import create_default_model
from src.inference import StackCountPredictor
import numpy as np

# Create model
model = create_default_model(pretrained=False)
predictor = StackCountPredictor(model, device='cpu', mc_dropout_passes=5)

# Test with dummy image
dummy_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
result = predictor.predict(dummy_image)

print(f"Predicted: {result['predicted_count']}")
print(f"Confidence: {result['confidence_score']}")
```

## Training Your First Model

### 1. Prepare Your Data

Create this structure:
```
data/
├── raw/
│   └── your_images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── annotations/
    └── annotations.json
```

### 2. Create Annotations

```json
[
  {
    "image_id": "image1.jpg",
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

### 3. Split Data

```python
from src.data import AnnotationParser, AnnotationSplitter

annotations = AnnotationParser.load_annotations('data/annotations/annotations.json')
train, val, test = AnnotationSplitter.split_by_image(annotations)
AnnotationSplitter.save_splits(train, val, test, 'data/annotations')
```

### 4. Train

```bash
python src/training/train.py --config configs/config.yaml --debug
```

The `--debug` flag uses small dataset and few epochs for quick testing.

## Running Inference

### Python API

```python
from src.inference import load_predictor

# Load trained model
predictor = load_predictor('models/checkpoints/best.pth')

# Predict
result = predictor.predict('your_image.jpg')
print(f"Count: {result['predicted_count']}")
print(f"Confidence: {result['confidence_score']}")
```

### REST API

```bash
# Start API
python src/inference/api.py

# In another terminal, make prediction
curl -X POST "http://localhost:8000/predict" -F "file=@your_image.jpg"
```

### Gradio Demo

```bash
python src/inference/gradio_demo.py
```

Visit `http://localhost:7860` in your browser.

## Common Commands

```bash
# Train with default config
python src/training/train.py

# Train with custom data
python src/training/train.py --data_dir data/raw --annotations_dir data/annotations

# Resume from checkpoint
python src/training/train.py --resume models/checkpoints/best.pth

# Start API server
python src/inference/api.py

# Start Gradio demo
python src/inference/gradio_demo.py

# Monitor training
tensorboard --logdir logs/tensorboard
```

## Next Steps

- Read [USER_GUIDE.md](USER_GUIDE.md) for detailed documentation
- Check [examples/](examples/) directory for code examples
- Review [docs/](docs/) for in-depth guides

## Troubleshooting

**Import errors?** Make sure you're in the project root directory.

**CUDA not available?** Use `--device cpu` flag.

**Model not found?** Train a model first or download a pre-trained one.

## Support

Open an issue at: https://github.com/rafi-Codes/Kachra-Seth/issues
