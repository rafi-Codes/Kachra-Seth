# Data Pipeline Documentation

## Overview

The data pipeline provides a comprehensive solution for loading, processing, augmenting, and managing the stack count prediction dataset. It supports curriculum learning, synthetic data generation, and includes tools for data exploration and validation.

## Components

### 1. Dataset Classes (`src/data/dataset.py`)

**StackCountDataset**: Main PyTorch Dataset class
- Loads images and annotations
- Supports filtering by category and count range
- Returns preprocessed images with metadata
- Provides dataset statistics

**CurriculumDataset**: Extends StackCountDataset for curriculum learning
- Supports 3-phase curriculum learning
- Phase 1: counts 5-50
- Phase 2: counts 5-150  
- Phase 3: counts 5-500
- Dynamic phase updates during training

### 2. Annotation Handling (`src/data/annotations.py`)

**AnnotationValidator**: Validates annotation format
- Ensures all required fields are present
- Validates data types and value ranges
- Checks category, angle, and lighting values
- Provides detailed error messages

**AnnotationParser**: Parses and manipulates annotations
- Load/save annotations from/to JSON
- Filter by category, count range, agreement score
- Calculate annotation statistics
- Create new annotations with helper functions

**AnnotationSplitter**: Splits data with no leakage
- Split by image (no leakage across splits)
- Split by category (ensures representation)
- Save splits to separate files
- Supports custom split ratios

### 3. Augmentation (`src/data/augmentation.py`)

**AugmentationPipeline**: Standard augmentation pipelines
- Training: rotation, zoom, brightness, noise, flip
- Validation: minimal preprocessing
- Test: no augmentation
- Inference: padding to maintain aspect ratio

**AdvancedAugmentation**: Specialized augmentation
- Light-robust augmentation
- Occlusion-robust augmentation
- Angle-robust augmentation

**AugmentationConfig**: Configuration-based transforms
- Load augmentation parameters from config
- Generate appropriate transforms for each phase

### 4. Synthetic Data Generation (`src/data/synthetic.py`)

**BlenderScriptGenerator**: Generates Blender Python scripts
- Single image rendering script
- Batch rendering script
- Support for various categories, angles, lighting
- Automatic annotation generation

**SyntheticDataGenerator**: Manages Blender execution
- Find Blender executable
- Generate single images
- Generate batch datasets
- Error handling and timeouts

**SyntheticAnnotationManager**: Manages synthetic annotations
- Merge real and synthetic annotations
- Filter synthetic vs real data
- Maintain data provenance

### 5. Visualization (`src/data/visualization.py`)

**DataVisualizer**: Plot various dataset characteristics
- Count distribution (histogram + box plot)
- Category distribution (bar + pie chart)
- Agreement score distribution
- Attribute distributions (lighting, angles, occlusion)
- Sample images with metadata

**DatasetExplorer**: Explore and analyze data
- Generate comprehensive summary reports
- Print formatted statistics
- Identify data gaps
- Coverage analysis

### 6. Preprocessing (`src/data/preprocessing.py`)

**BlurDetector**: Detect blurry images
- Laplacian variance method
- Configurable threshold
- Batch processing support

**ImageResizer**: Handle image resizing
- Resize with padding (maintains aspect ratio)
- Resize with crop
- Letterbox method

**ImageNormalizer**: Normalize images for model input
- ImageNet normalization (default)
- Custom normalization support
- Tensor and array support

**Preprocessor**: Complete preprocessing pipeline
- Load images
- Blur detection
- Resizing
- Normalization
- Batch processing

### 7. Data Loader (`src/data/dataloader.py`)

**StackCountDataLoader**: Main data loader interface
- Integrates all pipeline components
- Curriculum learning support
- Configurable transforms and splits
- Automatic phase updates

**InferenceDataLoader**: Specialized for inference
- Single image loading
- Batch loading
- Array input support
- Blur detection

**Utility Functions**:
- `create_data_loaders_from_config()`: One-step setup
- `get_curriculum_phase_for_epoch()`: Determine phase

## Usage Examples

### Basic Dataset Usage

```python
from src.data import StackCountDataset, get_transforms

# Create dataset
dataset = StackCountDataset(
    data_dir='data/raw',
    annotations_file='data/annotations/train_annotations.json',
    image_size=384,
    transform=get_transforms('train'),
    categories=['banknotes', 'books'],
    count_range=(5, 50),
    phase='train'
)

# Get sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Count: {sample['count']}")

# Get statistics
stats = dataset.get_statistics()
print(f"Mean count: {stats['mean_count']}")
```

### Annotation Handling

```python
from src.data import AnnotationParser, AnnotationSplitter

# Load annotations
annotations = AnnotationParser.load_annotations('annotations.json')

# Validate
is_valid, errors = AnnotationValidator.validate_annotation(annotation)

# Create new annotation
new_ann = AnnotationParser.create_annotation(
    image_id='stack_001.jpg',
    category='banknotes',
    true_count=87,
    annotator_1_count=87,
    annotator_2_count=88
)

# Split data
train, val, test = AnnotationSplitter.split_by_image(
    annotations,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### Data Loader with Config

```python
from src.data import create_data_loaders_from_config
from src.utils.config import load_config

# Load config
config = load_config('configs/config.yaml')

# Create data loaders
data_loader = create_data_loaders_from_config(
    config=config,
    data_dir='data/raw',
    annotations_dir='data/annotations',
    current_phase=1
)

# Get loaders
train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()

# Use in training loop
for batch in train_loader:
    images = batch['image']
    counts = batch['count']
    # Training logic here
```

### Curriculum Learning

```python
# Update curriculum phase during training
for epoch in range(1, 101):
    # Determine phase from epoch
    phase = get_curriculum_phase_for_epoch(epoch, config)
    
    # Update data loader if phase changed
    if phase != current_phase:
        data_loader.update_curriculum_phase(phase)
        current_phase = phase
    
    # Training epoch
    train_epoch(...)
```

### Visualization

```python
from src.data import visualize_dataset, AnnotationParser

# Load annotations
annotations = AnnotationParser.load_annotations('annotations.json')

# Generate visualizations
visualize_dataset(
    annotations=annotations,
    image_dir='data/raw',
    output_dir='data/visualizations',
    generate_plots=True
)
```

### Inference

```python
from src.data import InferenceDataLoader

# Create inference loader
inference_loader = InferenceDataLoader(
    target_size=(384, 384),
    blur_threshold=100.0
)

# Load single image
result = inference_loader.load_single_image('test_image.jpg')
image_tensor = result['processed_image']
blur_detected = result['blur_detected']
```

## Annotation Format

Each annotation must contain:

```json
{
  "image_id": "stack_001.jpg",
  "category": "banknotes",
  "true_count": 87,
  "annotator_1_count": 87,
  "annotator_2_count": 88,
  "agreement_score": 0.99,
  "stack_angle": "45_degree",
  "lighting": "natural",
  "occlusion_percent": 5
}
```

### Valid Categories
- banknotes
- books
- papers
- tiles
- cards
- plates

### Valid Stack Angles
- top_down
- 45_degree
- side

### Valid Lighting Conditions
- natural
- bright
- dim
- backlit

## Configuration

The data pipeline uses the configuration from `configs/config.yaml`:

```yaml
data:
  categories: [...]
  image_size: 384
  augmentations:
    rotation: { range: 15 }
    zoom: { min: 0.8, max: 1.2 }
    brightness: { jitter: 0.2 }
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
```

## Best Practices

1. **Always validate annotations** before use
2. **Generate visualizations** to understand data distribution
3. **Use curriculum learning** for better training stability
4. **Check for data gaps** before training
5. **Monitor blur detection** during inference
6. **Save processed data splits** to ensure reproducibility

## Next Steps

After setting up the data pipeline:
1. Collect and annotate real images
2. Generate synthetic data for large counts
3. Split data and validate distributions
4. Visualize to ensure good coverage
5. Proceed to model architecture implementation
