"""Data loading and augmentation modules."""

from .dataset import StackCountDataset, CurriculumDataset
from .annotations import (
    AnnotationValidator,
    AnnotationParser,
    AnnotationSplitter
)
from .augmentation import (
    AugmentationPipeline,
    AdvancedAugmentation,
    AugmentationConfig,
    get_transforms
)
from .synthetic import (
    BlenderScriptGenerator,
    SyntheticDataGenerator,
    SyntheticAnnotationManager,
    generate_synthetic_dataset
)
from .visualization import (
    DataVisualizer,
    DatasetExplorer,
    visualize_dataset
)
from .preprocessing import (
    BlurDetector,
    ImageResizer,
    ImageNormalizer,
    Preprocessor,
    load_and_preprocess_image,
    check_image_quality
)
from .dataloader import (
    StackCountDataLoader,
    InferenceDataLoader,
    create_data_loaders_from_config,
    get_curriculum_phase_for_epoch
)

__all__ = [
    # Dataset classes
    'StackCountDataset',
    'CurriculumDataset',
    
    # Annotation handling
    'AnnotationValidator',
    'AnnotationParser',
    'AnnotationSplitter',
    
    # Augmentation
    'AugmentationPipeline',
    'AdvancedAugmentation',
    'AugmentationConfig',
    'get_transforms',
    
    # Synthetic data
    'BlenderScriptGenerator',
    'SyntheticDataGenerator',
    'SyntheticAnnotationManager',
    'generate_synthetic_dataset',
    
    # Visualization
    'DataVisualizer',
    'DatasetExplorer',
    'visualize_dataset',
    
    # Preprocessing
    'BlurDetector',
    'ImageResizer',
    'ImageNormalizer',
    'Preprocessor',
    'load_and_preprocess_image',
    'check_image_quality',
    
    # Data loaders
    'StackCountDataLoader',
    'InferenceDataLoader',
    'create_data_loaders_from_config',
    'get_curriculum_phase_for_epoch'
]