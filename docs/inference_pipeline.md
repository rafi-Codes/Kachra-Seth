# Inference Pipeline Documentation

## Overview

The inference pipeline provides a complete solution for deploying the stack count prediction model, including a predictor class, REST API, and interactive demo interface.

## Components

### 1. Predictor (`src/inference/predictor.py`)

**StackCountPredictor** - Main inference class

Features:
- Load trained models from checkpoints
- Monte Carlo Dropout with uncertainty quantification
- Edge case detection (blur, quality issues)
- Confidence-based output formatting
- Batch inference support
- Customizable confidence thresholds

### 2. REST API (`src/inference/api.py`)

**FastAPI Application** - RESTful API endpoints

Endpoints:
- `POST /predict` - Single image prediction
- `POST /batch_predict` - Batch prediction
- `GET /model_info` - Model information
- `GET /health` - Health check
- `POST /load_model` - Load specific model
- `POST /update_thresholds` - Update confidence thresholds

### 3. Demo Interface (`src/inference/gradio_demo.py`)

**Gradio Interface** - Interactive web demo

Features:
- Drag-and-drop image upload
- Real-time prediction
- Detailed result display
- Confidence level visualization
- Quality warnings

## Usage

### Python API

```python
from src.inference import load_predictor

# Load predictor
predictor = load_predictor('models/checkpoints/best.pth', device='cuda')

# Single image prediction
result = predictor.predict('path/to/image.jpg')

print(f"Predicted count: {result['predicted_count']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Range: {result['count_range']['low']} - {result['count_range']['high']}")
```

### Batch Prediction

```python
# Multiple images
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = predictor.predict_batch(images)

for i, result in enumerate(results):
    print(f"Image {i}: {result['predicted_count']} items")
```

### REST API

**Start the server:**
```bash
python src/inference/api.py
```

**Or with uvicorn directly:**
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

**Make predictions:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Gradio Demo

**Start the demo:**
```bash
python src/inference/gradio_demo.py
```

**With options:**
```bash
python src/inference/gradio_demo.py \
    --model_path models/checkpoints/best.pth \
    --device cuda \
    --share  # Share publicly
```

## Output Format

### Prediction Response

```json
{
  "predicted_count": 87,
  "confidence_score": 0.91,
  "count_range": {
    "low": 83,
    "high": 91
  },
  "uncertainty_std": 2.5,
  "uncertainty_level": "low",
  "message": "Count: 87 items",
  "processing_time_ms": 140.5,
  "quality_warning": null,
  "flag": null
}
```

### Confidence Tiers

| Confidence Range | Level | Output |
|------------------|-------|--------|
| 0.90 - 1.00 | High | Show count only |
| 0.70 - 0.89 | Moderate | Show count + range |
| 0.50 - 0.69 | Low | Show range + suggest retake |
| < 0.50 | Very Low | Unable to predict reliably |

### Edge Cases

**Blur Detection:**
```json
{
  "quality_warning": "image may be blurry"
}
```

**High Density Stack:**
```json
{
  "flag": "high_density_stack"
}
```

**No Stack Detected:**
```json
{
  "error": "no_stack_found"
}
```

## Configuration

### Confidence Thresholds

```python
predictor.set_confidence_thresholds({
    'high': 0.90,
    'moderate': 0.70,
    'low': 0.50
})
```

### MC Dropout Settings

```python
# Number of MC Dropout samples
predictor = StackCountPredictor(
    model=model,
    mc_dropout_passes=15  # More samples = more accurate but slower
)
```

### Blur Detection

```python
predictor = StackCountPredictor(
    model=model,
    blur_threshold=100.0  # Lower = more strict
)
```

## Performance

### Inference Speed

| Model | Batch Size | Time (ms) | Device |
|-------|------------|-----------|--------|
| EfficientNet-B4 | 1 | ~150 | GPU |
| EfficientNet-B4 | 1 | ~500 | CPU |
| ConvNeXt-Small | 1 | ~120 | GPU |
| Lightweight (B0) | 1 | ~50 | GPU |

### MC Dropout Overhead

| Passes | Overhead | Accuracy |
|--------|----------|----------|
| 1 | 1.0x | Lower |
| 5 | 1.3x | Medium |
| 15 | 2.0x | High |
| 30 | 3.5x | Very High |

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

**AWS/GCP/Azure:**
1. Containerize the application
2. Push to container registry
3. Deploy to cloud service
4. Configure autoscaling

### ONNX Export (Future)

```python
# Export to ONNX for deployment
import torch

# Create dummy input
dummy_input = torch.randn(1, 3, 384, 384)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11
)
```

## Troubleshooting

### Model Not Loading
- Check checkpoint path exists
- Verify device compatibility
- Check PyTorch version matches training
- Ensure model architecture matches checkpoint

### Slow Inference
- Reduce MC Dropout passes
- Use smaller model variant
- Ensure using GPU if available
- Optimize image preprocessing

### Poor Predictions
- Verify model was trained properly
- Check image quality
- Ensure similar distribution to training data
- Consider retraining with more data

### Memory Issues
- Reduce batch size
- Use smaller model
- Enable gradient checkpointing
- Use CPU inference

## Best Practices

1. **Always check confidence scores** - Low confidence may indicate issues
2. **Monitor processing time** - Ensure acceptable latency
3. **Handle edge cases** - Blur detection, no stack, etc.
4. **Use MC Dropout** - Essential for reliable uncertainty
5. **Validate on real data** - Test with actual use cases
6. **Set appropriate thresholds** - Tune for your use case
7. **Log predictions** - Enable active learning
8. **Monitor performance** - Track metrics in production

## Security Considerations

- **Validate input images** - Check file types and sizes
- **Rate limiting** - Prevent abuse
- **Authentication** - Add API keys if needed
- **HTTPS** - Use secure connections
- **Input sanitization** - Prevent injection attacks

## Monitoring

### Key Metrics to Track
- Prediction latency
- Error rates
- Confidence distribution
- Edge case frequency
- User feedback

### Logging
```python
# Log prediction for monitoring
prediction_log = {
    'timestamp': datetime.now().isoformat(),
    'prediction': result['predicted_count'],
    'confidence': result['confidence_score'],
    'processing_time': result['processing_time_ms'],
    'user_feedback': None  # Collect from users
}
```

## Next Steps

After inference:
1. Collect user feedback for active learning
2. Monitor performance in production
3. Set up automated retraining pipeline
4. Optimize for specific deployment targets
5. Add monitoring and alerting
