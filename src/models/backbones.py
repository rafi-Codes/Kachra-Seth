"""Backbone models for stack count prediction."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
import warnings


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for feature extraction."""
    
    def __init__(
        self,
        variant: str = 'b4',
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        freeze_blocks: int = 0
    ):
        """Initialize EfficientNet backbone.
        
        Args:
            variant: EfficientNet variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_early_layers: Whether to freeze early layers
            freeze_blocks: Number of blocks to freeze from the start
        """
        super(EfficientNetBackbone, self).__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        self.freeze_early_layers = freeze_early_layers
        self.freeze_blocks = freeze_blocks
        
        # Load EfficientNet
        efficientnet_map = {
            'b0': models.efficientnet_b0,
            'b1': models.efficientnet_b1,
            'b2': models.efficientnet_b2,
            'b3': models.efficientnet_b3,
            'b4': models.efficientnet_b4,
            'b5': models.efficientnet_b5,
            'b6': models.efficientnet_b6,
            'b7': models.efficientnet_b7
        }
        
        if variant not in efficientnet_map:
            raise ValueError(f"Invalid EfficientNet variant: {variant}")
        
        model_fn = efficientnet_map[variant]
        self.efficientnet = model_fn(weights=models.EfficientNet_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get feature dimensions
        self.feature_dim = self._get_feature_dim()
        
        # Freeze layers if requested
        if freeze_early_layers or freeze_blocks > 0:
            self._apply_freezing()
    
    def _get_feature_dim(self) -> int:
        """Get the output feature dimension of the backbone."""
        # EfficientNet classifier input features
        return self.efficientnet.classifier[1].in_features
    
    def _apply_freezing(self):
        """Freeze backbone layers according to configuration."""
        if self.freeze_blocks > 0:
            # Freeze specific number of blocks
            blocks = list(self.efficientnet.features.children())
            for i, block in enumerate(blocks):
                if i < self.freeze_blocks:
                    for param in block.parameters():
                        param.requires_grad = False
        
        if self.freeze_early_layers:
            # Freeze all layers except the last 3 blocks
            blocks = list(self.efficientnet.features.children())
            for i, block in enumerate(blocks[:-3]):
                for param in block.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all backbone layers."""
        for param in self.efficientnet.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n blocks.
        
        Args:
            n: Number of blocks to unfreeze from the end
        """
        blocks = list(self.efficientnet.features.children())
        total_blocks = len(blocks)
        
        # Freeze all first
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Unfreeze last n blocks
        for i in range(max(0, total_blocks - n), total_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Extract features (remove classifier head)
        features = self.efficientnet.features(x)
        
        # Global average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        return features
    
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate features from different layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with intermediate features
        """
        features = {}
        current = x
        
        for i, block in enumerate(self.efficientnet.features):
            current = block(current)
            features[f'block_{i}'] = current
        
        # Global average pooling on final features
        pooled = nn.functional.adaptive_avg_pool2d(current, (1, 1))
        features['final_pooled'] = torch.flatten(pooled, 1)
        
        return features


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone for feature extraction."""
    
    def __init__(
        self,
        variant: str = 'small',
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        freeze_stages: int = 0
    ):
        """Initialize ConvNeXt backbone.
        
        Args:
            variant: ConvNeXt variant ('tiny', 'small', 'base', 'large')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_early_layers: Whether to freeze early layers
            freeze_stages: Number of stages to freeze from the start
        """
        super(ConvNeXtBackbone, self).__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        self.freeze_early_layers = freeze_early_layers
        self.freeze_stages = freeze_stages
        
        # Load ConvNeXt
        convnext_map = {
            'tiny': models.convnext_tiny,
            'small': models.convnext_small,
            'base': models.convnext_base,
            'large': models.convnext_large
        }
        
        if variant not in convnext_map:
            raise ValueError(f"Invalid ConvNeXt variant: {variant}")
        
        model_fn = convnext_map[variant]
        self.convnext = model_fn(weights=models.ConvNeXt_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get feature dimensions
        self.feature_dim = self._get_feature_dim()
        
        # Freeze layers if requested
        if freeze_early_layers or freeze_stages > 0:
            self._apply_freezing()
    
    def _get_feature_dim(self) -> int:
        """Get the output feature dimension of the backbone."""
        return self.convnext.classifier[2].in_features
    
    def _apply_freezing(self):
        """Freeze backbone stages according to configuration."""
        if self.freeze_stages > 0:
            # Freeze specific number of stages
            stages = list(self.convnext.features.children())
            for i, stage in enumerate(stages):
                if i < self.freeze_stages:
                    for param in stage.parameters():
                        param.requires_grad = False
        
        if self.freeze_early_layers:
            # Freeze all stages except the last 2
            stages = list(self.convnext.features.children())
            for i, stage in enumerate(stages[:-2]):
                for param in stage.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all backbone layers."""
        for param in self.convnext.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_stages(self, n: int):
        """Unfreeze the last n stages.
        
        Args:
            n: Number of stages to unfreeze from the end
        """
        stages = list(self.convnext.features.children())
        total_stages = len(stages)
        
        # Freeze all first
        for param in self.convnext.parameters():
            param.requires_grad = False
        
        # Unfreeze last n stages
        for i in range(max(0, total_stages - n), total_stages):
            for param in stages[i].parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Extract features (remove classifier head)
        features = self.convnext.features(x)
        
        # Global average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        return features
    
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate features from different stages.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with intermediate features
        """
        features = {}
        current = x
        
        for i, stage in enumerate(self.convnext.features):
            current = stage(current)
            features[f'stage_{i}'] = current
        
        # Global average pooling on final features
        pooled = nn.functional.adaptive_avg_pool2d(current, (1, 1))
        features['final_pooled'] = torch.flatten(pooled, 1)
        
        return features


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction (alternative option)."""
    
    def __init__(
        self,
        variant: str = '50',
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        freeze_layers: int = 0
    ):
        """Initialize ResNet backbone.
        
        Args:
            variant: ResNet variant ('18', '34', '50', '101', '152')
            pretrained: Whether to use ImageNet pretrained weights
            freeze_early_layers: Whether to freeze early layers
            freeze_layers: Number of layers to freeze from the start
        """
        super(ResNetBackbone, self).__init__()
        
        self.variant = variant
        self.pretrained = pretrained
        self.freeze_early_layers = freeze_early_layers
        self.freeze_layers = freeze_layers
        
        # Load ResNet
        resnet_map = {
            '18': models.resnet18,
            '34': models.resnet34,
            '50': models.resnet50,
            '101': models.resnet101,
            '152': models.resnet152
        }
        
        if variant not in resnet_map:
            raise ValueError(f"Invalid ResNet variant: {variant}")
        
        model_fn = resnet_map[variant]
        self.resnet = model_fn(weights=models.ResNet_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get feature dimensions
        self.feature_dim = self.resnet.fc.in_features
        
        # Remove the final FC layer
        self.resnet.fc = nn.Identity()
        
        # Freeze layers if requested
        if freeze_early_layers or freeze_layers > 0:
            self._apply_freezing()
    
    def _apply_freezing(self):
        """Freeze backbone layers according to configuration."""
        if self.freeze_layers > 0:
            # Freeze specific number of layers
            layers = list(self.resnet.children())
            for i, layer in enumerate(layers):
                if i < self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        if self.freeze_early_layers:
            # Freeze all layers except the last 2 layers
            layers = list(self.resnet.children())
            for i, layer in enumerate(layers[:-2]):
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all backbone layers."""
        for param in self.resnet.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        return self.resnet(x)


def get_backbone(
    backbone_type: str,
    variant: str = None,
    pretrained: bool = True,
    freeze_early_layers: bool = True,
    **kwargs
) -> nn.Module:
    """Factory function to create backbone models.
    
    Args:
        backbone_type: Type of backbone ('efficientnet', 'convnext', 'resnet')
        variant: Model variant (e.g., 'b4', 'small', '50')
        pretrained: Whether to use pretrained weights
        freeze_early_layers: Whether to freeze early layers
        **kwargs: Additional arguments for specific backbones
        
    Returns:
        Backbone model
    """
    backbone_type = backbone_type.lower()
    
    if backbone_type == 'efficientnet':
        variant = variant or 'b4'
        return EfficientNetBackbone(
            variant=variant,
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            **kwargs
        )
    elif backbone_type == 'convnext':
        variant = variant or 'small'
        return ConvNeXtBackbone(
            variant=variant,
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            **kwargs
        )
    elif backbone_type == 'resnet':
        variant = variant or '50'
        return ResNetBackbone(
            variant=variant,
            pretrained=pretrained,
            freeze_early_layers=freeze_early_layers,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


# Test the backbones
if __name__ == "__main__":
    # Test EfficientNet
    print("Testing EfficientNet-B4...")
    efficientnet = EfficientNetBackbone(variant='b4', pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    features = efficientnet(x)
    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {efficientnet.feature_dim}")
    
    # Test ConvNeXt
    print("\nTesting ConvNeXt-Small...")
    convnext = ConvNeXtBackbone(variant='small', pretrained=False)
    features = convnext(x)
    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {convnext.feature_dim}")
    
    print("\nBackbone tests passed!")
