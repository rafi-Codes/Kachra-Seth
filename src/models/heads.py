"""Model heads for stack count prediction."""

import torch
import torch.nn as nn
from typing import List, Optional


class CountingHead(nn.Module):
    """Counting head for regression of item counts."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [512, 256],
        activation: str = 'relu',
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """Initialize counting head.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'leaky_relu', 'gelu', 'swish')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(CountingHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation.lower()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final output layer (no activation, linear output)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.activation_name == 'relu':
            return nn.ReLU(inplace=True)
        elif self.activation_name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif self.activation_name == 'gelu':
            return nn.GELU()
        elif self.activation_name == 'swish':
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through counting head.
        
        Args:
            x: Input tensor of shape (B, input_dim)
            
        Returns:
            Predicted count tensor of shape (B, 1)
        """
        return self.head(x).squeeze(-1)  # Shape: (B,)


class ConfidenceHead(nn.Module):
    """Confidence head for prediction uncertainty estimation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        output_activation: str = 'sigmoid'
    ):
        """Initialize confidence head.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            output_activation: Output activation ('sigmoid', 'tanh')
        """
        super(ConfidenceHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation.lower()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.output_activation = output_activation.lower()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # Output activation
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'none':
            pass
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
        
        self.head = nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.activation_name == 'relu':
            return nn.ReLU(inplace=True)
        elif self.activation_name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif self.activation_name == 'gelu':
            return nn.GELU()
        elif self.activation_name == 'swish':
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through confidence head.
        
        Args:
            x: Input tensor of shape (B, input_dim)
            
        Returns:
            Confidence score tensor of shape (B,) in range [0, 1]
        """
        return self.head(x).squeeze(-1)  # Shape: (B,)


class DualHead(nn.Module):
    """Dual head combining counting and confidence heads."""
    
    def __init__(
        self,
        input_dim: int,
        counting_hidden: List[int] = [512, 256],
        confidence_hidden: List[int] = [256, 128],
        activation: str = 'relu',
        counting_dropout: float = 0.3,
        confidence_dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """Initialize dual head.
        
        Args:
            input_dim: Input feature dimension
            counting_hidden: Hidden layers for counting head
            confidence_hidden: Hidden layers for confidence head
            activation: Activation function
            counting_dropout: Dropout rate for counting head
            confidence_dropout: Dropout rate for confidence head
            use_batch_norm: Whether to use batch normalization
        """
        super(DualHead, self).__init__()
        
        self.counting_head = CountingHead(
            input_dim=input_dim,
            hidden_layers=counting_hidden,
            activation=activation,
            dropout=counting_dropout,
            use_batch_norm=use_batch_norm
        )
        
        self.confidence_head = ConfidenceHead(
            input_dim=input_dim,
            hidden_layers=confidence_hidden,
            activation=activation,
            dropout=confidence_dropout,
            use_batch_norm=use_batch_norm
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through dual head.
        
        Args:
            x: Input tensor of shape (B, input_dim)
            
        Returns:
            Tuple of (count_predictions, confidence_predictions)
        """
        count = self.counting_head(x)
        confidence = self.confidence_head(x)
        return count, confidence


class DensityMapHead(nn.Module):
    """Density map head for CSRNet-style counting (optional)."""
    
    def __init__(
        self,
        input_channels: int = 1792,  # EfficientNet-B4 final channels
        output_channels: int = 1
    ):
        """Initialize density map head.
        
        Args:
            input_channels: Number of input feature channels
            output_channels: Number of output channels (1 for density map)
        """
        super(DensityMapHead, self).__init__()
        
        # CSRNet-style decoder with dilated convolutions
        self.decoder = nn.Sequential(
            # First convolution
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Dilated convolutions
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through density map head.
        
        Args:
            x: Input feature tensor of shape (B, C, H, W)
            
        Returns:
            Density map tensor of shape (B, 1, H, W)
        """
        return self.decoder(x)


class CategoryHead(nn.Module):
    """Category classification head (optional for mixed stacks)."""
    
    def __init__(
        self,
        input_dim: int,
        num_categories: int = 6,
        hidden_layers: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        """Initialize category head.
        
        Args:
            input_dim: Input feature dimension
            num_categories: Number of categories
            hidden_layers: Hidden layer sizes
            activation: Activation function
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(CategoryHead, self).__init__()
        
        self.num_categories = num_categories
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_categories))
        
        self.head = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activation = activation.lower()
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through category head.
        
        Args:
            x: Input tensor of shape (B, input_dim)
            
        Returns:
            Category logits tensor of shape (B, num_categories)
        """
        return self.head(x)


def create_counting_head(config: dict) -> CountingHead:
    """Factory function to create counting head from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CountingHead instance
    """
    head_config = config.get('counting_head', {})
    
    return CountingHead(
        input_dim=config.get('feature_dim', 1792),
        hidden_layers=head_config.get('hidden_layers', [512, 256]),
        activation=head_config.get('activation', 'relu'),
        dropout=head_config.get('dropout', 0.3),
        use_batch_norm=head_config.get('use_batch_norm', True)
    )


def create_confidence_head(config: dict) -> ConfidenceHead:
    """Factory function to create confidence head from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ConfidenceHead instance
    """
    head_config = config.get('confidence_head', {})
    
    return ConfidenceHead(
        input_dim=config.get('feature_dim', 1792),
        hidden_layers=head_config.get('hidden_layers', [256, 128]),
        activation=head_config.get('activation', 'relu'),
        dropout=head_config.get('dropout', 0.2),
        use_batch_norm=head_config.get('use_batch_norm', True),
        output_activation=head_config.get('output_activation', 'sigmoid')
    )


def create_dual_head(config: dict) -> DualHead:
    """Factory function to create dual head from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DualHead instance
    """
    counting_config = config.get('counting_head', {})
    confidence_config = config.get('confidence_head', {})
    
    return DualHead(
        input_dim=config.get('feature_dim', 1792),
        counting_hidden=counting_config.get('hidden_layers', [512, 256]),
        confidence_hidden=confidence_config.get('hidden_layers', [256, 128]),
        activation=counting_config.get('activation', 'relu'),
        counting_dropout=counting_config.get('dropout', 0.3),
        confidence_dropout=confidence_config.get('dropout', 0.2),
        use_batch_norm=counting_config.get('use_batch_norm', True)
    )


# Test the heads
if __name__ == "__main__":
    # Test CountingHead
    print("Testing CountingHead...")
    counting_head = CountingHead(input_dim=1792, hidden_layers=[512, 256])
    x = torch.randn(4, 1792)
    count = counting_head(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {count.shape}")
    
    # Test ConfidenceHead
    print("\nTesting ConfidenceHead...")
    confidence_head = ConfidenceHead(input_dim=1792, hidden_layers=[256, 128])
    confidence = confidence_head(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {confidence.shape}")
    print(f"Output range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    # Test DualHead
    print("\nTesting DualHead...")
    dual_head = DualHead(input_dim=1792)
    count, conf = dual_head(x)
    print(f"Count shape: {count.shape}")
    print(f"Confidence shape: {conf.shape}")
    
    # Test DensityMapHead
    print("\nTesting DensityMapHead...")
    density_head = DensityMapHead(input_channels=1792)
    x_spatial = torch.randn(4, 1792, 12, 12)
    density = density_head(x_spatial)
    print(f"Input shape: {x_spatial.shape}")
    print(f"Output shape: {density.shape}")
    
    print("\nAll head tests passed!")
