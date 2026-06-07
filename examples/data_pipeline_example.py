"""Example script demonstrating the data pipeline usage."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    StackCountDataset,
    AnnotationParser,
    AnnotationValidator,
    AnnotationSplitter,
    get_transforms,
    visualize_dataset,
    create_data_loaders_from_config
)
from src.utils.config import load_config


def example_annotation_handling():
    """Demonstrate annotation validation and parsing."""
    print("=" * 60)
    print("Annotation Handling Example")
    print("=" * 60)
    
    # Example annotation
    example_annotation = {
        'image_id': 'stack_001.jpg',
        'category': 'banknotes',
        'true_count': 87,
        'annotator_1_count': 87,
        'annotator_2_count': 88,
        'agreement_score': 0.99,
        'stack_angle': '45_degree',
        'lighting': 'natural',
        'occlusion_percent': 5
    }
    
    # Validate annotation
    is_valid, errors = AnnotationValidator.validate_annotation(example_annotation)
    print(f"Annotation valid: {is_valid}")
    if not is_valid:
        print("Errors:", errors)
    
    # Create annotation using helper
    new_annotation = AnnotationParser.create_annotation(
        image_id='stack_002.jpg',
        category='books',
        true_count=45,
        annotator_1_count=45,
        annotator_2_count=45,
        stack_angle='top_down',
        lighting='bright',
        occlusion_percent=0
    )
    print("Created annotation:", new_annotation)


def example_data_splitting():
    """Demonstrate data splitting."""
    print("\n" + "=" * 60)
    print("Data Splitting Example")
    print("=" * 60)
    
    # Create example annotations
    annotations = []
    for i in range(100):
        ann = AnnotationParser.create_annotation(
            image_id=f'image_{i}.jpg',
            category='banknotes' if i % 2 == 0 else 'books',
            true_count=50 + i,
            annotator_1_count=50 + i,
            annotator_2_count=50 + i
        )
        annotations.append(ann)
    
    # Split annotations
    train, val, test = AnnotationSplitter.split_by_image(
        annotations,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"Total annotations: {len(annotations)}")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


def example_dataset_creation():
    """Demonstrate dataset creation."""
    print("\n" + "=" * 60)
    print("Dataset Creation Example")
    print("=" * 60)
    
    # Note: This requires actual data and annotations
    # Example code structure:
    
    """
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
    
    # Get a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Count: {sample['count']}")
    print(f"Category: {sample['category']}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
    """
    
    print("Dataset creation requires actual data files.")
    print("See the code comments for the implementation pattern.")


def example_augmentation():
    """Demonstrate augmentation pipelines."""
    print("\n" + "=" * 60)
    print("Augmentation Example")
    print("=" * 60)
    
    # Get transforms for different phases
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    test_transform = get_transforms('test')
    
    print("Training transform:", type(train_transform))
    print("Validation transform:", type(val_transform))
    print("Test transform:", type(test_transform))


def example_data_loader():
    """Demonstrate data loader creation from config."""
    print("\n" + "=" * 60)
    print("Data Loader Example")
    print("=" * 60)
    
    # Load configuration
    try:
        config = load_config('configs/config.yaml')
        print("Loaded configuration successfully")
        
        # Example data loader creation
        """
        data_loader = create_data_loaders_from_config(
            config=config,
            data_dir='data/raw',
            annotations_dir='data/annotations',
            current_phase=1
        )
        
        # Get loaders
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        
        # Iterate through batches
        for batch in train_loader:
            images = batch['image']
            counts = batch['count']
            print(f"Batch images shape: {images.shape}")
            print(f"Batch counts: {counts}")
            break
        """
        
        print("Data loader creation requires actual data files.")
        print("See the code comments for the implementation pattern.")
        
    except FileNotFoundError:
        print("Configuration file not found. Run setup first.")


def example_visualization():
    """Demonstrate dataset visualization."""
    print("\n" + "=" * 60)
    print("Visualization Example")
    print("=" * 60)
    
    # Example visualization code
    """
    # Load annotations
    annotations = AnnotationParser.load_annotations('data/annotations/train_annotations.json')
    
    # Generate visualizations
    visualize_dataset(
        annotations=annotations,
        image_dir='data/raw',
        output_dir='data/visualizations',
        generate_plots=True
    )
    """
    
    print("Visualization requires actual data files.")
    print("See the code comments for the implementation pattern.")


def main():
    """Run all examples."""
    print("Stack Count Prediction - Data Pipeline Examples")
    print("=" * 60)
    
    # Run examples
    example_annotation_handling()
    example_data_splitting()
    example_dataset_creation()
    example_augmentation()
    example_data_loader()
    example_visualization()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
