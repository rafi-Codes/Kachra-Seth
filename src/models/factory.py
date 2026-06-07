"""Model factory for creating models with different configurations."""

import torch
from typing import Dict, Any, Optional
import warnings

from .stack_count_model import StackCountModel
from .backbones import get_backbone
from .attention import create_attention_module


class ModelFactory:
    """Factory class for creating stack count prediction models."""
    
    @staticmethod
    def create_model(
        backbone_type: str = 'efficientnet',
        backbone_variant: str = 'b4',
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        counting_hidden: list = [512, 256],
        confidence_hidden: list = [256, 128],
        activation: str = 'relu',
        counting_dropout: float = 0.3,
        confidence_dropout: float = 0.2,
        use_density_map: bool = False,
        use_category_head: bool = False,
        num_categories: int = 6,
        use_attention: bool = False,
        attention_type: str = 'cbam',
        attention_position: str = 'backbone',
        mc_dropout: bool = True,
        **kwargs
    ) -> StackCountModel:
        """Create a stack count prediction model.
        
        Args:
            backbone_type: Type of backbone ('efficientnet', 'convnext', 'resnet')
            backbone_variant: Backbone variant (e.g., 'b4', 'small', '50')
            pretrained: Whether to use pretrained weights
            freeze_early_layers: Whether to freeze early backbone layers
            counting_hidden: Hidden layers for counting head
            confidence_hidden: Hidden layers for confidence head
            activation: Activation function
            counting_dropout: Dropout rate for counting head
            confidence_dropout: Dropout rate for confidence head
            use_density_map: Whether to use density map branch
            use_category_head: Whether to use category classification head
            num_categories: Number of categories for classification
            use_attention: Whether to add attention blocks
            attention_type: Type of attention ('se', 'cbam', 'self_attention', 'edge')
            attention_position: Where to add attention ('backbone', 'head', 'both')
            mc_dropout: Whether to enable Monte Carlo Dropout
            **kwargs: Additional arguments
            
        Returns:
            StackCountModel instance
        """
        model = StackCountModel(
            backbone_type=backbone_type,
            backbone_variant=backbone_variant,
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            counting_hidden=counting_hidden,
            confidence_hidden=confidence_hidden,
            activation=activation,
            counting_dropout=counting_dropout,
            confidence_dropout=confidence_dropout,
            use_density_map=use_density_map,
            use_category_head=use_category_head,
            num_categories=num_categories,
            mc_dropout=mc_dropout,
            **kwargs
        )
        
        # Add attention blocks if requested
        if use_attention:
            model = ModelFactory._add_attention(
                model,
                attention_type,
                attention_position
            )
        
        return model
    
    @staticmethod
    def _add_attention(
        model: StackCountModel,
        attention_type: str,
        position: str
    ) -> StackCountModel:
        """Add attention blocks to the model.
        
        Args:
            model: Base model
            attention_type: Type of attention
            position: Where to add attention
            
        Returns:
            Model with attention added
        """
        # This is a placeholder for adding attention blocks
        # In a full implementation, you would modify the model architecture
        # to insert attention modules at the specified positions
        
        warnings.warn(
            f"Attention blocks ({attention_type}) not yet implemented in model architecture. "
            "This is a placeholder for future implementation."
        )
        
        return model
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> StackCountModel:
        """Create model from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            StackCountModel instance
        """
        return ModelFactory.create_model(**config)
    
    @staticmethod
    def create_efficientnet_b4(**kwargs) -> StackCountModel:
        """Create model with EfficientNet-B4 backbone (default configuration).
        
        Args:
            **kwargs: Additional model arguments
            
        Returns:
            StackCountModel with EfficientNet-B4
        """
        return ModelFactory.create_model(
            backbone_type='efficientnet',
            backbone_variant='b4',
            **kwargs
        )
    
    @staticmethod
    def create_convnext_small(**kwargs) -> StackCountModel:
        """Create model with ConvNeXt-Small backbone.
        
        Args:
            **kwargs: Additional model arguments
            
        Returns:
            StackCountModel with ConvNeXt-Small
        """
        return ModelFactory.create_model(
            backbone_type='convnext',
            backbone_variant='small',
            **kwargs
        )
    
    @staticmethod
    def create_resnet50(**kwargs) -> StackCountModel:
        """Create model with ResNet-50 backbone.
        
        Args:
            **kwargs: Additional model arguments
            
        Returns:
            StackCountModel with ResNet-50
        """
        return ModelFactory.create_model(
            backbone_type='resnet',
            backbone_variant='50',
            **kwargs
        )
    
    @staticmethod
    def create_lightweight_model(**kwargs) -> StackCountModel:
        """Create a lightweight model for faster inference.
        
        Args:
            **kwargs: Additional model arguments
            
        Returns:
            Lightweight StackCountModel
        """
        return ModelFactory.create_model(
            backbone_type='efficientnet',
            backbone_variant='b0',  # Smaller variant
            counting_hidden=[256, 128],  # Smaller hidden layers
            confidence_hidden=[128, 64],
            counting_dropout=0.2,
            confidence_dropout=0.1,
            use_density_map=False,
            mc_dropout=False,  # Disable MC Dropout for speed
            **kwargs
        )
    
    @staticmethod
    def create_high_accuracy_model(**kwargs) -> StackCountModel:
        """Create a high-accuracy model with all features enabled.
        
        Args:
            **kwargs: Additional model arguments
            
        Returns:
            High-accuracy StackCountModel
        """
        return ModelFactory.create_model(
            backbone_type='efficientnet',
            backbone_variant='b4',
            counting_hidden=[512, 256, 128],  # Deeper network
            confidence_hidden=[256, 128, 64],
            counting_dropout=0.4,
            confidence_dropout=0.3,
            use_density_map=True,
            use_category_head=True,
            use_attention=True,
            attention_type='cbam',
            mc_dropout=True,
            **kwargs
        )


def get_model_config_variants() -> Dict[str, Dict[str, Any]]:
    """Get predefined model configuration variants.
    
    Returns:
        Dictionary of model configuration variants
    """
    return {
        'default': {
            'backbone_type': 'efficientnet',
            'backbone_variant': 'b4',
            'pretrained': True,
            'freeze_early_layers': True,
            'counting_hidden': [512, 256],
            'confidence_hidden': [256, 128],
            'activation': 'relu',
            'counting_dropout': 0.3,
            'confidence_dropout': 0.2,
            'use_density_map': False,
            'use_category_head': False,
            'mc_dropout': True
        },
        'convnext': {
            'backbone_type': 'convnext',
            'backbone_variant': 'small',
            'pretrained': True,
            'freeze_early_layers': True,
            'counting_hidden': [512, 256],
            'confidence_hidden': [256, 128],
            'activation': 'relu',
            'counting_dropout': 0.3,
            'confidence_dropout': 0.2,
            'use_density_map': False,
            'use_category_head': False,
            'mc_dropout': True
        },
        'lightweight': {
            'backbone_type': 'efficientnet',
            'backbone_variant': 'b0',
            'pretrained': True,
            'freeze_early_layers': False,
            'counting_hidden': [256, 128],
            'confidence_hidden': [128, 64],
            'activation': 'relu',
            'counting_dropout': 0.2,
            'confidence_dropout': 0.1,
            'use_density_map': False,
            'use_category_head': False,
            'mc_dropout': False
        },
        'high_accuracy': {
            'backbone_type': 'efficientnet',
            'backbone_variant': 'b4',
            'pretrained': True,
            'freeze_early_layers': True,
            'counting_hidden': [512, 256, 128],
            'confidence_hidden': [256, 128, 64],
            'activation': 'relu',
            'counting_dropout': 0.4,
            'confidence_dropout': 0.3,
            'use_density_map': True,
            'use_category_head': True,
            'use_attention': True,
            'attention_type': 'cbam',
            'mc_dropout': True
        },
        'mobile': {
            'backbone_type': 'efficientnet',
            'backbone_variant': 'b0',
            'pretrained': True,
            'freeze_early_layers': False,
            'counting_hidden': [128, 64],
            'confidence_hidden': [64, 32],
            'activation': 'relu',
            'counting_dropout': 0.2,
            'confidence_dropout': 0.1,
            'use_density_map': False,
            'use_category_head': False,
            'mc_dropout': False
        }
    }


def create_model(variant: str = 'default', **kwargs) -> StackCountModel:
    """Create a model from a predefined variant.
    
    Args:
        variant: Name of the model variant ('default', 'convnext', 'lightweight', etc.)
        **kwargs: Additional arguments to override defaults
        
    Returns:
        StackCountModel instance
    """
    variants = get_model_config_variants()
    
    if variant not in variants:
        raise ValueError(
            f"Unknown variant: {variant}. Available: {list(variants.keys())}"
        )
    
    config = variants[variant].copy()
    config.update(kwargs)
    
    return ModelFactory.create_model(**config)


# Convenience functions
def create_default_model(**kwargs) -> StackCountModel:
    """Create default model configuration."""
    return create_model('default', **kwargs)


def create_convnext_model(**kwargs) -> StackCountModel:
    """Create ConvNeXt model."""
    return create_model('convnext', **kwargs)


def create_lightweight_model(**kwargs) -> StackCountModel:
    """Create lightweight model for fast inference."""
    return create_model('lightweight', **kwargs)


def create_high_accuracy_model(**kwargs) -> StackCountModel:
    """Create high-accuracy model with all features."""
    return create_model('high_accuracy', **kwargs)


def create_mobile_model(**kwargs) -> StackCountModel:
    """Create mobile-optimized model."""
    return create_model('mobile', **kwargs)


# Test the factory
if __name__ == "__main__":
    print("Testing Model Factory...")
    
    # Test default model creation
    print("\n1. Creating default model...")
    model = create_default_model()
    params = model.get_num_parameters()
    print(f"Total parameters: {params['total']:,}")
    
    # Test variant creation
    print("\n2. Creating lightweight model...")
    lightweight = create_lightweight_model()
    params = lightweight.get_num_parameters()
    print(f"Total parameters: {params['total']:,}")
    
    # Test high accuracy model
    print("\n3. Creating high accuracy model...")
    high_acc = create_high_accuracy_model()
    params = high_acc.get_num_parameters()
    print(f"Total parameters: {params['total']:,}")
    
    # Test from config
    print("\n4. Creating from custom config...")
    custom_config = {
        'backbone_type': 'convnext',
        'backbone_variant': 'small',
        'counting_hidden': [256, 128],
        'confidence_hidden': [128, 64]
    }
    custom_model = ModelFactory.create_from_config(custom_config)
    params = custom_model.get_num_parameters()
    print(f"Total parameters: {params['total']:,}")
    
    # Test model forward pass
    print("\n5. Testing forward pass...")
    x = torch.randn(2, 3, 384, 384)
    outputs = model(x)
    print(f"Count shape: {outputs['count'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    
    print("\nAll factory tests passed!")
