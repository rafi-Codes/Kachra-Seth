"""Attention mechanisms (CBAM and SE) for improved feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer."""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16
    ):
        """Initialize SE layer.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super(SELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE layer.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W) with channel attention applied
        """
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel Attention Module (part of CBAM)."""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16
    ):
        """Initialize channel attention module.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Channel attention map of shape (B, C, 1, 1)
        """
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (part of CBAM)."""
    
    def __init__(
        self,
        kernel_size: int = 7
    ):
        """Initialize spatial attention module.
        
        Args:
            kernel_size: Kernel size for convolution
        """
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Spatial attention map of shape (B, 1, H, W)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        """Initialize CBAM module.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CBAM.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class SelfAttention2D(nn.Module):
    """Self-Attention mechanism for 2D feature maps."""
    
    def __init__(
        self,
        channels: int,
        head_dim: int = 64,
        num_heads: int = 8
    ):
        """Initialize self-attention module.
        
        Args:
            channels: Number of input channels
            head_dim: Dimension per attention head
            num_heads: Number of attention heads
        """
        super(SelfAttention2D, self).__init__()
        
        self.channels = channels
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.total_heads_dim = head_dim * num_heads
        
        assert self.total_heads_dim <= channels, \
            f"Total head dimension ({self.total_heads_dim}) must be <= channels ({channels})"
        
        # Query, Key, Value projections
        self.q_conv = nn.Conv2d(channels, self.total_heads_dim, 1)
        self.k_conv = nn.Conv2d(channels, self.total_heads_dim, 1)
        self.v_conv = nn.Conv2d(channels, self.total_heads_dim, 1)
        
        # Output projection
        self.out_conv = nn.Conv2d(self.total_heads_dim, channels, 1)
        
        self.scale = head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Project to Q, K, V
        q = self.q_conv(x).view(b, self.num_heads, self.head_dim, -1)
        k = self.k_conv(x).view(b, self.num_heads, self.head_dim, -1)
        v = self.v_conv(x).view(b, self.num_heads, self.head_dim, -1)
        
        # Transpose for attention computation
        q = q.permute(0, 1, 3, 2)  # (B, num_heads, hw, head_dim)
        k = k.permute(0, 1, 3, 2)  # (B, num_heads, hw, head_dim)
        v = v.permute(0, 1, 3, 2)  # (B, num_heads, hw, head_dim)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, hw, head_dim)
        
        # Reshape and project back
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(b, self.total_heads_dim, h, w)
        out = self.out_conv(out)
        
        # Residual connection
        return x + out


class EdgeAttention(nn.Module):
    """Edge-focused attention for stack boundary detection."""
    
    def __init__(
        self,
        channels: int,
        kernel_sizes: list = [3, 5, 7]
    ):
        """Initialize edge attention module.
        
        Args:
            channels: Number of input channels
            kernel_sizes: List of kernel sizes for multi-scale edge detection
        """
        super(EdgeAttention, self).__init__()
        
        self.edge_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=k, padding=k//2, groups=channels)
            for k in kernel_sizes
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(kernel_sizes), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through edge attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Edge attention map of shape (B, C, H, W)
        """
        edge_features = []
        
        for conv in self.edge_convs:
            edge_features.append(conv(x))
        
        # Concatenate multi-scale edge features
        edge_cat = torch.cat(edge_features, dim=1)
        
        # Fuse and generate attention map
        attention = self.fusion(edge_cat)
        
        return x * attention


class AttentionGate(nn.Module):
    """Attention Gate for feature fusion."""
    
    def __init__(
        self,
        gate_channels: int,
        signal_channels: int,
        inter_channels: Optional[int] = None
    ):
        """Initialize attention gate.
        
        Args:
            gate_channels: Number of channels in gate signal
            signal_channels: Number of channels in input signal
            inter_channels: Intermediate channels (default: signal_channels // 4)
        """
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = signal_channels // 4
        
        # Transform gate signal
        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, 1)
        
        # Transform input signal
        self.signal_conv = nn.Conv2d(signal_channels, inter_channels, 1)
        
        # Compute attention weights
        self.attention_conv = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.transform_signal = nn.Conv2d(signal_channels, gate_channels, 1)
    
    def forward(
        self,
        gate: torch.Tensor,
        signal: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through attention gate.
        
        Args:
            gate: Gate signal of shape (B, gate_channels, H_g, W_g)
            signal: Input signal of shape (B, signal_channels, H_s, W_s)
            
        Returns:
            Gated output of shape (B, gate_channels, H_s, W_s)
        """
        # Resize gate to match signal size
        if gate.shape[2:] != signal.shape[2:]:
            gate = F.interpolate(gate, size=signal.shape[2:], mode='bilinear', align_corners=False)
        
        # Transform signals
        gate_transformed = self.gate_conv(gate)
        signal_transformed = self.signal_conv(signal)
        
        # Combine and compute attention
        combined = gate_transformed + signal_transformed
        attention = self.attention_conv(combined)
        
        # Transform signal to match gate channels
        signal_transformed = self.transform_signal(signal)
        
        # Apply attention
        return signal_transformed * attention


def create_attention_module(
    attention_type: str,
    channels: int,
    **kwargs
) -> nn.Module:
    """Factory function to create attention modules.
    
    Args:
        attention_type: Type of attention ('se', 'cbam', 'self_attention', 'edge')
        channels: Number of input channels
        **kwargs: Additional arguments for specific attention types
        
    Returns:
        Attention module
    """
    attention_type = attention_type.lower()
    
    if attention_type == 'se':
        return SELayer(channels, reduction=kwargs.get('reduction', 16))
    elif attention_type == 'cbam':
        return CBAM(
            channels,
            reduction=kwargs.get('reduction', 16),
            kernel_size=kwargs.get('kernel_size', 7)
        )
    elif attention_type == 'self_attention':
        return SelfAttention2D(
            channels,
            head_dim=kwargs.get('head_dim', 64),
            num_heads=kwargs.get('num_heads', 8)
        )
    elif attention_type == 'edge':
        return EdgeAttention(
            channels,
            kernel_sizes=kwargs.get('kernel_sizes', [3, 5, 7])
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# Test attention modules
if __name__ == "__main__":
    print("Testing Attention Modules...")
    
    x = torch.randn(2, 256, 32, 32)
    
    # Test SE Layer
    print("Testing SE Layer...")
    se_layer = SELayer(256)
    out = se_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test CBAM
    print("\nTesting CBAM...")
    cbam = CBAM(256)
    out = cbam(x)
    print(f"Output shape: {out.shape}")
    
    # Test Self-Attention
    print("\nTesting Self-Attention...")
    self_attn = SelfAttention2D(256, head_dim=32, num_heads=4)
    out = self_attn(x)
    print(f"Output shape: {out.shape}")
    
    # Test Edge Attention
    print("\nTesting Edge Attention...")
    edge_attn = EdgeAttention(256)
    out = edge_attn(x)
    print(f"Output shape: {out.shape}")
    
    # Test Attention Gate
    print("\nTesting Attention Gate...")
    gate = torch.randn(2, 128, 16, 16)
    signal = torch.randn(2, 256, 32, 32)
    attn_gate = AttentionGate(128, 256)
    out = attn_gate(gate, signal)
    print(f"Gate shape: {gate.shape}")
    print(f"Signal shape: {signal.shape}")
    print(f"Output shape: {out.shape}")
    
    print("\nAll attention module tests passed!")
