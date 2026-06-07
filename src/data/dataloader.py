"""Data loader with curriculum learning support."""

import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .dataset import StackCountDataset, CurriculumDataset
from .augmentation import get_transforms, AugmentationConfig
from .annotations import AnnotationParser, AnnotationSplitter
from .preprocessing import Preprocessor


class StackCountDataLoader:
    """Main data loader for stack count prediction."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        annotations_dir: Union[str, Path],
        config: Optional[Dict] = None,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing images
            annotations_dir: Directory containing annotation files
            config: Configuration dictionary
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = Path(data_dir)
        self.annotations_dir = Path(annotations_dir)
        self.config = config or {}
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Load configuration
        self.data_config = self.config.get('data', {})
        self.training_config = self.config.get('training', {})
        
        # Initialize augmentation config
        self.augmentation_config = AugmentationConfig(self.data_config)
        
        # Load splits
        self.train_annotations, self.val_annotations, self.test_annotations = \
            self._load_data_splits()
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Create data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def _load_data_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load train/val/test annotation splits."""
        train_path = self.annotations_dir / 'train_annotations.json'
        val_path = self.annotations_dir / 'val_annotations.json'
        test_path = self.annotations_dir / 'test_annotations.json'
        
        if not all(path.exists() for path in [train_path, val_path, test_path]):
            raise FileNotFoundError(
                "Annotation split files not found. Please run data split first."
            )
        
        train_annotations = AnnotationParser.load_annotations(train_path)
        val_annotations = AnnotationParser.load_annotations(val_path)
        test_annotations = AnnotationParser.load_annotations(test_path)
        
        return train_annotations, val_annotations, test_annotations
    
    def create_datasets(
        self,
        current_phase: int = 1,
        categories: Optional[List[str]] = None,
        use_synthetic: bool = True
    ):
        """Create datasets for all splits.
        
        Args:
            current_phase: Current curriculum learning phase (1-3)
            categories: List of categories to include (None = all)
            use_synthetic: Whether to include synthetic data
        """
        image_size = self.data_config.get('image_size', 384)
        categories = categories or self.data_config.get('categories', [])
        
        # Get transforms
        train_transform = self.augmentation_config.get_training_pipeline()
        val_transform = self.augmentation_config.get_validation_pipeline()
        test_transform = self.augmentation_config.get_test_pipeline()
        
        # Create datasets
        self.train_dataset = CurriculumDataset(
            data_dir=self.data_dir,
            annotations_file=self.annotations_dir / 'train_annotations.json',
            image_size=image_size,
            transform=train_transform,
            categories=categories,
            current_phase=current_phase,
            phase='train'
        )
        
        self.val_dataset = StackCountDataset(
            data_dir=self.data_dir,
            annotations_file=self.annotations_dir / 'val_annotations.json',
            image_size=image_size,
            transform=val_transform,
            categories=categories,
            phase='val'
        )
        
        self.test_dataset = StackCountDataset(
            data_dir=self.data_dir,
            annotations_file=self.annotations_dir / 'test_annotations.json',
            image_size=image_size,
            transform=test_transform,
            categories=categories,
            phase='test'
        )
        
        print(f"Created datasets:")
        print(f"  Train: {len(self.train_dataset)} samples (Phase {current_phase})")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
    
    def create_data_loaders(self, batch_size: Optional[int] = None):
        """Create PyTorch data loaders.
        
        Args:
            batch_size: Batch size (None = use config)
        """
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise ValueError("Datasets not created. Call create_datasets() first.")
        
        batch_size = batch_size or self.training_config.get('batch_size', 32)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        print(f"Created data loaders with batch_size={batch_size}")
    
    def update_curriculum_phase(self, new_phase: int):
        """Update curriculum learning phase.
        
        Args:
            new_phase: New phase number (1-3)
        """
        if self.train_dataset is None:
            raise ValueError("Train dataset not created. Call create_datasets() first.")
        
        if isinstance(self.train_dataset, CurriculumDataset):
            self.train_dataset.update_phase(new_phase)
            
            # Recreate train loader with updated dataset
            batch_size = self.training_config.get('batch_size', 32)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
            print(f"Updated to curriculum phase {new_phase}")
        else:
            print("Warning: Train dataset is not a CurriculumDataset. Phase update ignored.")
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        if self.train_loader is None:
            raise ValueError("Train loader not created. Call create_data_loaders() first.")
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        if self.val_loader is None:
            raise ValueError("Val loader not created. Call create_data_loaders() first.")
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        if self.test_loader is None:
            raise ValueError("Test loader not created. Call create_data_loaders() first.")
        return self.test_loader
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics for all datasets."""
        stats = {
            'train': self.train_dataset.get_statistics() if self.train_dataset else None,
            'val': self.val_dataset.get_statistics() if self.val_dataset else None,
            'test': self.test_dataset.get_statistics() if self.test_dataset else None
        }
        return stats
    
    def split_annotations_from_file(
        self,
        annotations_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ):
        """Split annotations file into train/val/test sets.
        
        Args:
            annotations_file: Path to combined annotations file
            output_dir: Directory to save split files (default: annotations_dir)
        """
        output_dir = Path(output_dir) if output_dir else self.annotations_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        annotations = AnnotationParser.load_annotations(annotations_file)
        
        # Get split ratios from config
        train_ratio = self.data_config.get('train_split', 0.70)
        val_ratio = self.data_config.get('val_split', 0.15)
        test_ratio = self.data_config.get('test_split', 0.15)
        
        # Split annotations
        train, val, test = AnnotationSplitter.split_by_image(
            annotations,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Save splits
        AnnotationSplitter.save_splits(train, val, test, output_dir)
        
        # Reload splits
        self.train_annotations, self.val_annotations, self.test_annotations = \
            self._load_data_splits()


class InferenceDataLoader:
    """Data loader for inference."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (384, 384),
        blur_threshold: float = 100.0,
        normalize: bool = True
    ):
        """Initialize inference data loader.
        
        Args:
            target_size: Target image size
            blur_threshold: Blur detection threshold
            normalize: Whether to normalize images
        """
        self.preprocessor = Preprocessor(
            target_size=target_size,
            blur_threshold=blur_threshold,
            normalize=normalize
        )
        self.target_size = target_size
    
    def load_single_image(
        self,
        image_path: Union[str, Path],
        detect_blur: bool = True
    ) -> Dict:
        """Load and preprocess a single image for inference.
        
        Args:
            image_path: Path to image file
            detect_blur: Whether to perform blur detection
            
        Returns:
            Dictionary with preprocessed image and metadata
        """
        return self.preprocessor.preprocess(image_path, detect_blur=detect_blur)
    
    def load_batch(
        self,
        image_paths: List[Union[str, Path]],
        detect_blur: bool = True
    ) -> Dict:
        """Load and preprocess multiple images for inference.
        
        Args:
            image_paths: List of image paths
            detect_blur: Whether to perform blur detection
            
        Returns:
            Dictionary with batched images and metadata
        """
        return self.preprocessor.batch_preprocess(image_paths, detect_blur=detect_blur)
    
    def load_from_array(
        self,
        image_array: np.ndarray,
        detect_blur: bool = True
    ) -> Dict:
        """Load and preprocess image from numpy array.
        
        Args:
            image_array: Image as numpy array (H, W, C)
            detect_blur: Whether to perform blur detection
            
        Returns:
            Dictionary with preprocessed image and metadata
        """
        return self.preprocessor.preprocess(image_array, detect_blur=detect_blur)


def create_data_loaders_from_config(
    config: Dict,
    data_dir: Union[str, Path],
    annotations_dir: Union[str, Path],
    current_phase: int = 1
) -> StackCountDataLoader:
    """Convenience function to create data loaders from config.
    
    Args:
        config: Configuration dictionary
        data_dir: Directory containing images
        annotations_dir: Directory containing annotation files
        current_phase: Current curriculum learning phase
        
    Returns:
        StackCountDataLoader instance
    """
    # Extract hardware config
    hardware_config = config.get('hardware', {})
    num_workers = hardware_config.get('num_workers', 4)
    pin_memory = hardware_config.get('pin_memory', True)
    
    # Create data loader
    data_loader = StackCountDataLoader(
        data_dir=data_dir,
        annotations_dir=annotations_dir,
        config=config,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create datasets
    data_loader.create_datasets(current_phase=current_phase)
    
    # Create data loaders
    data_loader.create_data_loaders()
    
    return data_loader


def get_curriculum_phase_for_epoch(epoch: int, config: Dict) -> int:
    """Determine curriculum phase for a given epoch.
    
    Args:
        epoch: Current epoch number
        config: Configuration dictionary
        
    Returns:
        Curriculum phase number (1-3)
    """
    curriculum_config = config.get('training', {}).get('curriculum', {})
    
    phase1_epochs = curriculum_config.get('phase1', {}).get('epochs', [1, 30])
    phase2_epochs = curriculum_config.get('phase2', {}).get('epochs', [31, 70])
    phase3_epochs = curriculum_config.get('phase3', {}).get('epochs', [71, 100])
    
    if phase1_epochs[0] <= epoch <= phase1_epochs[1]:
        return 1
    elif phase2_epochs[0] <= epoch <= phase2_epochs[1]:
        return 2
    elif phase3_epochs[0] <= epoch <= phase3_epochs[1]:
        return 3
    else:
        return 3  # Default to phase 3 for epochs beyond defined range
