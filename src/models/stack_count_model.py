"""Main stack count prediction model with dual heads and Monte Carlo Dropout."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
import numpy as np

from .backbones import get_backbone
from .heads import DualHead, DensityMapHead, CategoryHead


class StackCountModel(nn.Module):
    """Main model for stack count prediction with dual heads."""
    
    def __init__(
        self,
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
        mc_dropout: bool = True,
        mc_dropout_rate: float = 0.3,
        **backbone_kwargs
    ):
        """Initialize stack count model.
        
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
            mc_dropout: Whether to enable Monte Carlo Dropout
            mc_dropout_rate: Dropout rate for MC Dropout
            **backbone_kwargs: Additional backbone arguments
        """
        super(StackCountModel, self).__init__()
        
        self.backbone_type = backbone_type
        self.backbone_variant = backbone_variant
        self.use_density_map = use_density_map
        self.use_category_head = use_category_head
        self.mc_dropout = mc_dropout
        
        # Create backbone
        self.backbone = get_backbone(
            backbone_type=backbone_type,
            variant=backbone_variant,
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            **backbone_kwargs
        )
        
        feature_dim = self.backbone.feature_dim
        
        # Create dual head
        self.dual_head = DualHead(
            input_dim=feature_dim,
            counting_hidden=counting_hidden,
            confidence_hidden=confidence_hidden,
            activation=activation,
            counting_dropout=counting_dropout if not mc_dropout else mc_dropout_rate,
            confidence_dropout=confidence_dropout if not mc_dropout else mc_dropout_rate
        )
        
        # Optional density map head
        if use_density_map:
            # Get the final feature map channels from backbone
            if backbone_type == 'efficientnet':
                final_channels = 1792  # EfficientNet-B4 final channels
            elif backbone_type == 'convnext':
                final_channels = 768  # ConvNeXt-Small final channels
            else:
                final_channels = 2048  # ResNet-50 final channels
            
            self.density_map_head = DensityMapHead(
                input_channels=final_channels,
                output_channels=1
            )
        else:
            self.density_map_head = None
        
        # Optional category head
        if use_category_head:
            self.category_head = CategoryHead(
                input_dim=feature_dim,
                num_categories=num_categories,
                hidden_layers=[256, 128],
                activation=activation,
                dropout=confidence_dropout,
                use_batch_norm=True
            )
        else:
            self.category_head = None
        
        # Enable MC Dropout mode
        if mc_dropout:
            self.enable_mc_dropout()
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo Dropout by setting layers to train mode."""
        self.mc_dropout = True
        # Set dropout layers to train mode
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_mc_dropout(self):
        """Disable Monte Carlo Dropout by setting layers to eval mode."""
        self.mc_dropout = False
        # Set dropout layers to eval mode
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            return_intermediate: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions and optional intermediate outputs
        """
        outputs = {}
        
        # Get backbone features
        if return_intermediate or self.use_density_map:
            features = self.backbone.get_intermediate_features(x)
            pooled_features = features['final_pooled']
            
            if return_intermediate:
                outputs['intermediate_features'] = features
            
            # Get density map if enabled
            if self.use_density_map:
                # Use the final spatial features
                final_block_key = f'block_{len(features)-2}' if self.backbone_type == 'efficientnet' else f'stage_{len(features)-2}'
                if final_block_key in features:
                    spatial_features = features[final_block_key]
                    outputs['density_map'] = self.density_map_head(spatial_features)
        else:
            pooled_features = self.backbone(x)
        
        # Get predictions from dual head
        count, confidence = self.dual_head(pooled_features)
        outputs['count'] = count
        outputs['confidence'] = confidence
        
        # Get category prediction if enabled
        if self.use_category_head:
            outputs['category_logits'] = self.category_head(pooled_features)
        
        return outputs
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 15,
        return_raw: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty using Monte Carlo Dropout.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            n_samples: Number of MC Dropout samples
            return_raw: Whether to return raw samples
            
        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        if not self.mc_dropout:
            # If MC Dropout is not enabled, do single forward pass
            with torch.no_grad():
                outputs = self.forward(x)
                outputs['count_mean'] = outputs['count']
                outputs['count_std'] = torch.zeros_like(outputs['count'])
                outputs['confidence_mean'] = outputs['confidence']
                return outputs
        
        self.enable_mc_dropout()
        
        # Collect multiple predictions
        count_samples = []
        confidence_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(x)
                count_samples.append(outputs['count'])
                confidence_samples.append(outputs['confidence'])
        
        # Stack samples
        count_samples = torch.stack(count_samples, dim=0)  # (n_samples, B)
        confidence_samples = torch.stack(confidence_samples, dim=0)  # (n_samples, B)
        
        # Calculate statistics
        count_mean = count_samples.mean(dim=0)
        count_std = count_samples.std(dim=0)
        confidence_mean = confidence_samples.mean(dim=0)
        
        # Calculate confidence from uncertainty
        # Higher uncertainty (std) → lower confidence
        # Using sigmoid mapping: confidence = sigmoid(1 / (1 + std))
        uncertainty_based_confidence = torch.sigmoid(1.0 / (1.0 + count_std))
        
        result = {
            'count_mean': count_mean,
            'count_std': count_std,
            'confidence_mean': confidence_mean,
            'uncertainty_based_confidence': uncertainty_based_confidence
        }
        
        if return_raw:
            result['count_samples'] = count_samples
            result['confidence_samples'] = confidence_samples
        
        return result
    
    def get_count_range(
        self,
        x: torch.Tensor,
        n_samples: int = 15,
        std_multiplier: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get count prediction with range estimate.
        
        Args:
            x: Input tensor
            n_samples: Number of MC Dropout samples
            std_multiplier: Multiplier for std to calculate range
            
        Returns:
            Tuple of (mean_count, low_bound, high_bound)
        """
        predictions = self.predict_with_uncertainty(x, n_samples)
        
        mean_count = predictions['count_mean']
        count_std = predictions['count_std']
        
        low_bound = mean_count - std_multiplier * count_std
        high_bound = mean_count + std_multiplier * count_std
        
        return mean_count, low_bound, high_bound
    
    def unfreeze_backbone(self, n_layers: int = 3):
        """Unfreeze the last n layers of the backbone.
        
        Args:
            n_layers: Number of layers to unfreeze
        """
        if hasattr(self.backbone, 'unfreeze_last_n_blocks'):
            self.backbone.unfreeze_last_n_blocks(n_layers)
        elif hasattr(self.backbone, 'unfreeze_last_n_stages'):
            self.backbone.unfreeze_last_n_stages(n_layers)
        else:
            self.backbone.unfreeze_all()
    
    def freeze_backbone(self):
        """Freeze all backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get the number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        dual_head_params = sum(p.numel() for p in self.dual_head.parameters())
        
        params_info = {
            'total': total_params,
            'trainable': trainable_params,
            'backbone': backbone_params,
            'dual_head': dual_head_params
        }
        
        if self.density_map_head is not None:
            params_info['density_map'] = sum(p.numel() for p in self.density_map_head.parameters())
        
        if self.category_head is not None:
            params_info['category_head'] = sum(p.numel() for p in self.category_head.parameters())
        
        return params_info
    
    def get_prediction_summary(
        self,
        x: torch.Tensor,
        n_samples: int = 15
    ) -> Dict[str, Any]:
        """Get a comprehensive prediction summary.
        
        Args:
            x: Input tensor
            n_samples: Number of MC Dropout samples
            
        Returns:
            Dictionary with comprehensive prediction information
        """
        predictions = self.predict_with_uncertainty(x, n_samples)
        
        count_mean = predictions['count_mean']
        count_std = predictions['count_std']
        confidence = predictions['uncertainty_based_confidence']
        
        # Determine uncertainty level
        uncertainty_levels = []
        for i in range(len(confidence)):
            if confidence[i] >= 0.90:
                uncertainty_levels.append('low')
            elif confidence[i] >= 0.70:
                uncertainty_levels.append('medium')
            else:
                uncertainty_levels.append('high')
        
        # Calculate count range
        low_bound = count_mean - count_std
        high_bound = count_mean + count_std
        
        return {
            'predicted_count': count_mean,
            'confidence_score': confidence,
            'count_range': {
                'low': low_bound,
                'high': high_bound
            },
            'uncertainty_std': count_std,
            'uncertainty_level': uncertainty_levels
        }


def create_model_from_config(config: Dict[str, Any]) -> StackCountModel:
    """Create model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StackCountModel instance
    """
    model_config = config.get('model', {})
    backbone_config = model_config.get('backbone', {})
    counting_config = model_config.get('counting_head', {})
    confidence_config = model_config.get('confidence_head', {})
    
    return StackCountModel(
        backbone_type=backbone_config.get('name', 'efficientnet'),
        backbone_variant=backbone_config.get('variant', 'b4'),
        pretrained=backbone_config.get('pretrained', True),
        freeze_early_layers=backbone_config.get('freeze_early_layers', True),
        counting_hidden=counting_config.get('hidden_layers', [512, 256]),
        confidence_hidden=confidence_config.get('hidden_layers', [256, 128]),
        activation=counting_config.get('activation', 'relu'),
        counting_dropout=counting_config.get('dropout', 0.3),
        confidence_dropout=confidence_config.get('dropout', 0.2),
        use_density_map=model_config.get('use_density_map', False),
        use_category_head=model_config.get('use_category_head', False),
        num_categories=6,
        mc_dropout=True,
        mc_dropout_rate=0.3
    )


# Test the model
if __name__ == "__main__":
    print("Testing StackCountModel...")
    
    # Create model
    model = StackCountModel(
        backbone_type='efficientnet',
        backbone_variant='b4',
        pretrained=False,
        mc_dropout=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Count shape: {outputs['count'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    
    # Test MC Dropout
    print("\nTesting Monte Carlo Dropout...")
    predictions = model.predict_with_uncertainty(x, n_samples=15)
    print(f"Count mean shape: {predictions['count_mean'].shape}")
    print(f"Count std shape: {predictions['count_std'].shape}")
    print(f"Uncertainty-based confidence: {predictions['uncertainty_based_confidence']}")
    
    # Test parameter count
    params = model.get_num_parameters()
    print(f"\nParameter count:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Backbone: {params['backbone']:,}")
    print(f"  Dual Head: {params['dual_head']:,}")
    
    print("\nModel tests passed!")
