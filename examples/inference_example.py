"""Example script demonstrating inference pipeline usage."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_default_model
from src.inference import StackCountPredictor


def example_basic_prediction():
    """Demonstrate basic prediction."""
    print("=" * 60)
    print("Basic Prediction Example")
    print("=" * 60)
    
    # Create model (untrained, just for demonstration)
    model = create_default_model(pretrained=False)
    
    # Create predictor
    predictor = StackCountPredictor(
        model=model,
        device='cpu',
        mc_dropout_passes=5  # Use fewer for demo
    )
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    # Run prediction
    result = predictor.predict(dummy_image)
    
    print("Prediction result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def example_image_types():
    """Demonstrate prediction with different image types."""
    print("\n" + "=" * 60)
    print("Different Image Types Example")
    print("=" * 60)
    
    model = create_default_model(pretrained=False)
    predictor = StackCountPredictor(model=model, device='cpu', mc_dropout_passes=5)
    
    # Test with numpy array
    print("\n1. NumPy array:")
    numpy_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    result = predictor.predict(numpy_image)
    print(f"  Count: {result['predicted_count']}")
    
    # Test with PIL Image
    print("\n2. PIL Image:")
    pil_image = Image.fromarray(numpy_image)
    result = predictor.predict(pil_image)
    print(f"  Count: {result['predicted_count']}")


def example_batch_prediction():
    """Demonstrate batch prediction."""
    print("\n" + "=" * 60)
    print("Batch Prediction Example")
    print("=" * 60)
    
    model = create_default_model(pretrained=False)
    predictor = StackCountPredictor(model=model, device='cpu', mc_dropout_passes=5)
    
    # Create multiple dummy images
    images = [
        np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    
    # Run batch prediction
    results = predictor.predict_batch(images)
    
    print("Batch results:")
    for i, result in enumerate(results):
        print(f"  Image {i}: Count={result['predicted_count']}, "
              f"Confidence={result['confidence_score']:.3f}")


def example_confidence_thresholds():
    """Demonstrate confidence threshold adjustment."""
    print("\n" + "=" * 60)
    print("Confidence Thresholds Example")
    print("=" * 60)
    
    model = create_default_model(pretrained=False)
    predictor = StackCountPredictor(model=model, device='cpu', mc_dropout_passes=5)
    
    # Check default thresholds
    print("Default thresholds:")
    for key, value in predictor.confidence_thresholds.items():
        print(f"  {key}: {value}")
    
    # Update thresholds
    print("\nUpdating thresholds...")
    predictor.set_confidence_thresholds({
        'high': 0.85,
        'moderate': 0.65,
        'low': 0.45
    })
    
    print("Updated thresholds:")
    for key, value in predictor.confidence_thresholds.items():
        print(f"  {key}: {value}")


def example_mc_dropout_control():
    """Demonstrate MC Dropout control."""
    print("\n" + "=" * 60)
    print("MC Dropout Control Example")
    print("=" * 60)
    
    model = create_default_model(pretrained=False)
    predictor = StackCountPredictor(model=model, device='cpu', mc_dropout_passes=5)
    
    # Check MC Dropout status
    print(f"MC Dropout enabled: {model.mc_dropout}")
    print(f"MC Dropout passes: {predictor.mc_dropout_passes}")
    
    # Disable MC Dropout
    print("\nDisabling MC Dropout...")
    predictor.disable_mc_dropout()
    print(f"MC Dropout enabled: {model.mc_dropout}")
    
    # Re-enable MC Dropout
    print("\nRe-enabling MC Dropout...")
    predictor.enable_mc_dropout()
    print(f"MC Dropout enabled: {model.mc_dropout}")


def main():
    """Run all examples."""
    print("Stack Count Prediction - Inference Pipeline Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_prediction()
    example_image_types()
    example_batch_prediction()
    example_confidence_thresholds()
    example_mc_dropout_control()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
