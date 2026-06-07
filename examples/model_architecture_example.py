"""Example script demonstrating model architecture usage."""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    StackCountModel,
    create_model,
    create_default_model,
    create_convnext_model,
    create_lightweight_model,
    create_high_accuracy_model,
    ModelFactory
)
from src.utils.config import load_config


def example_basic_model_creation():
    """Demonstrate basic model creation."""
    print("=" * 60)
    print("Basic Model Creation Example")
    print("=" * 60)
    
    # Create default model
    model = create_default_model()
    
    # Print model information
    params = model.get_num_parameters()
    print(f"Model created successfully!")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Backbone parameters: {params['backbone']:,}")
    print(f"Dual head parameters: {params['dual_head']:,}")


def example_model_variants():
    """Demonstrate different model variants."""
    print("\n" + "=" * 60)
    print("Model Variants Example")
    print("=" * 60)
    
    variants = [
        ('default', create_default_model),
        ('convnext', create_convnext_model),
        ('lightweight', create_lightweight_model),
        ('high_accuracy', create_high_accuracy_model)
    ]
    
    for variant_name, create_fn in variants:
        print(f"\n{variant_name.upper()}:")
        model = create_fn()
        params = model.get_num_parameters()
        print(f"  Parameters: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")


def example_forward_pass():
    """Demonstrate forward pass through the model."""
    print("\n" + "=" * 60)
    print("Forward Pass Example")
    print("=" * 60)
    
    # Create model
    model = create_default_model()
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 384, 384)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    outputs = model(x)
    
    print(f"Count predictions shape: {outputs['count'].shape}")
    print(f"Confidence predictions shape: {outputs['confidence'].shape}")
    print(f"Count values: {outputs['count']}")
    print(f"Confidence values: {outputs['confidence']}")


def example_monte_carlo_dropout():
    """Demonstrate Monte Carlo Dropout for uncertainty estimation."""
    print("\n" + "=" * 60)
    print("Monte Carlo Dropout Example")
    print("=" * 60)
    
    # Create model with MC Dropout enabled
    model = create_default_model(mc_dropout=True)
    
    # Create dummy input
    x = torch.randn(2, 3, 384, 384)
    
    print(f"Input shape: {x.shape}")
    
    # Predict with uncertainty
    predictions = model.predict_with_uncertainty(x, n_samples=15)
    
    print(f"Count mean: {predictions['count_mean']}")
    print(f"Count std (uncertainty): {predictions['count_std']}")
    print(f"Uncertainty-based confidence: {predictions['uncertainty_based_confidence']}")
    
    # Get prediction summary
    summary = model.get_prediction_summary(x, n_samples=15)
    print(f"\nPrediction Summary:")
    print(f"  Predicted count: {summary['predicted_count']}")
    print(f"  Confidence score: {summary['confidence_score']}")
    print(f"  Count range: [{summary['count_range']['low']}, {summary['count_range']['high']}]")
    print(f"  Uncertainty level: {summary['uncertainty_level']}")


def example_model_from_config():
    """Demonstrate creating model from configuration file."""
    print("\n" + "=" * 60)
    print("Model from Config Example")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_config('configs/config.yaml')
        print("Configuration loaded successfully")
        
        # Create model from config
        model = StackCountModel.create_model_from_config(config)
        
        params = model.get_num_parameters()
        print(f"Model created from config:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        
    except FileNotFoundError:
        print("Configuration file not found. Using default configuration instead.")
        model = create_default_model()
        params = model.get_num_parameters()
        print(f"Model created with defaults:")
        print(f"  Total parameters: {params['total']:,}")


def example_backbone_freezing():
    """Demonstrate backbone freezing and unfreezing."""
    print("\n" + "=" * 60)
    print("Backbone Freezing Example")
    print("=" * 60)
    
    # Create model with frozen backbone
    model = create_default_model(freeze_early_layers=True)
    
    params = model.get_num_parameters()
    print(f"With frozen backbone:")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
    # Unfreeze last 3 layers
    model.unfreeze_backbone(n_layers=3)
    
    params = model.get_num_parameters()
    print(f"After unfreezing last 3 layers:")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
    # Unfreeze all
    model.unfreeze_backbone(n_layers=10)  # More than total layers will unfreeze all
    
    params = model.get_num_parameters()
    print(f"After unfreezing all layers:")
    print(f"  Trainable parameters: {params['trainable']:,}")


def example_custom_model():
    """Demonstrate creating a custom model with specific parameters."""
    print("\n" + "=" * 60)
    print("Custom Model Example")
    print("=" * 60)
    
    # Create custom model
    custom_model = ModelFactory.create_model(
        backbone_type='convnext',
        backbone_variant='small',
        counting_hidden=[256, 128, 64],
        confidence_hidden=[128, 64],
        activation='gelu',
        counting_dropout=0.4,
        confidence_dropout=0.3,
        use_density_map=True,
        use_category_head=True,
        num_categories=6,
        mc_dropout=True
    )
    
    params = custom_model.get_num_parameters()
    print(f"Custom model created:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
    if 'density_map' in params:
        print(f"  Density map parameters: {params['density_map']:,}")
    if 'category_head' in params:
        print(f"  Category head parameters: {params['category_head']:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    outputs = custom_model(x)
    
    print(f"\nForward pass outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")


def main():
    """Run all examples."""
    print("Stack Count Prediction - Model Architecture Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_model_creation()
    example_model_variants()
    example_forward_pass()
    example_monte_carlo_dropout()
    example_model_from_config()
    example_backbone_freezing()
    example_custom_model()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
