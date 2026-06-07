# Stack Count Prediction System

Deep learning model for analyzing images of stacked objects (banknotes, books, papers, tiles, cards, etc.) and returning precise item counts with confidence scores.

## Features

- **Precise Counting**: Predicts exact integer counts of stacked items (5-500 items)
- **Confidence Scoring**: Provides confidence scores (0.0-1.0) for each prediction
- **Range Estimation**: Returns count ranges when confidence is below 0.85
- **Multi-Category**: Generalizes across object types without retraining
- **Uncertainty Quantification**: Uses Monte Carlo Dropout for reliable uncertainty estimates

## Model Architecture

- **Backbone**: EfficientNet-B4 or ConvNeXt-Small (ImageNet pretrained)
- **Dual Heads**: Parallel counting head and confidence head
- **Uncertainty**: Monte Carlo Dropout with 15 forward passes at inference
- **Optional**: Density Map branch (CSRNet-style) for large stacks (>100 items)

## Project Structure

```
stack-count-prediction/
├── data/
│   ├── raw/              # Original images
│   ├── processed/        # Augmented/preprocessed
│   ├── annotations/      # JSON annotation files
│   └── synthetic/        # Blender-rendered images
├── models/
│   ├── architectures/    # Model definitions
│   ├── checkpoints/      # Saved weights
│   └── onnx/             # Exported ONNX models
├── src/
│   ├── data/            # Data loading/augmentation
│   ├── models/          # Model implementations
│   ├── training/        # Training scripts
│   └── inference/       # Inference API
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks for experiments
├── tests/               # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Data Requirements

- **Categories**: banknotes, books, papers, tiles, cards, plates
- **Images**: 500+ per category with verified ground-truth counts
- **Variations**: lighting conditions, angles, stack sizes, occlusion, blur
- **Annotation format**: JSON with image metadata and true counts

## Training

```bash
python src/training/train.py --config configs/config.yaml
```

### Curriculum Learning
- **Phase 1** (epochs 1-30): Stacks with 5-50 items
- **Phase 2** (epochs 31-70): Add stacks with 51-150 items
- **Phase 3** (epochs 71-100): All counts including 150-500+

## Inference

### Python API
```python
from src.inference.predictor import StackCountPredictor

predictor = StackCountPredictor(model_path="models/checkpoints/best_model.pth")
result = predictor.predict("path/to/image.jpg")
print(result)
```

### REST API
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

### Demo Interface
```bash
# Gradio
python src/inference/gradio_demo.py

# Streamlit
streamlit run src/inference/streamlit_demo.py
```

## Output Format

```json
{
  "predicted_count": 87,
  "confidence_score": 0.91,
  "count_range": { "low": 83, "high": 91 },
  "category_detected": "banknotes",
  "uncertainty_level": "low",
  "processing_time_ms": 140
}
```

## Performance Targets

- MAE ≤ 3 for stack sizes 5-50
- MAE ≤ 5 for stack sizes 51-150
- MAPE ≤ 8% for stack sizes 151-500
- Confidence ECE ≤ 0.05
- Within-5% accuracy ≥ 85% on test set
- Inference latency ≤ 200ms on mid-range GPU

## Edge Cases

- No stack detected → error: "no_stack_found"
- Count exceeds 500 → flag: "high_density_stack"
- Blurry image → flag: "low_image_quality"
- Mixed stack → return count per detected category

## Deployment

- **REST API**: FastAPI + ONNX Runtime
- **Mobile**: TFLite and CoreML exports
- **Monitoring**: Active learning with user feedback logging

## License

MIT License
