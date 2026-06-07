"""Model architectures and implementations."""

from .backbones import (
    EfficientNetBackbone,
    ConvNeXtBackbone,
    ResNetBackbone,
    get_backbone
)
from .heads import (
    CountingHead,
    ConfidenceHead,
    DualHead,
    DensityMapHead,
    CategoryHead,
    create_counting_head,
    create_confidence_head,
    create_dual_head
)
from .attention import (
    SELayer,
    ChannelAttention,
    SpatialAttention,
    CBAM,
    SelfAttention2D,
    EdgeAttention,
    AttentionGate,
    create_attention_module
)
from .stack_count_model import (
    StackCountModel,
    create_model_from_config
)

__all__ = [
    # Backbones
    'EfficientNetBackbone',
    'ConvNeXtBackbone',
    'ResNetBackbone',
    'get_backbone',
    
    # Heads
    'CountingHead',
    'ConfidenceHead',
    'DualHead',
    'DensityMapHead',
    'CategoryHead',
    'create_counting_head',
    'create_confidence_head',
    'create_dual_head',
    
    # Attention
    'SELayer',
    'ChannelAttention',
    'SpatialAttention',
    'CBAM',
    'SelfAttention2D',
    'EdgeAttention',
    'AttentionGate',
    'create_attention_module',
    
    # Main Model
    'StackCountModel',
    'create_model_from_config'
]