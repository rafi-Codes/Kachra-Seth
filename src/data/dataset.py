"""PyTorch Dataset class for stack count prediction."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class StackCountDataset(Dataset):
    """Dataset for stack count prediction task."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        image_size: int = 384,
        transform: Optional[transforms.Compose] = None,
        categories: Optional[List[str]] = None,
        count_range: Optional[Tuple[int, int]] = None,
        phase: str = "train"
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: Path to JSON annotations file
            image_size: Target image size for resizing
            transform: Optional transforms to apply
            categories: List of categories to include (None = all)
            count_range: Optional tuple (min_count, max_count) for filtering
            phase: Dataset phase ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.annotations_file = Path(annotations_file)
        self.image_size = image_size
        self.transform = transform
        self.categories = categories
        self.count_range = count_range
        self.phase = phase
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Filter annotations based on criteria
        self.annotations = self._filter_annotations()
        
        print(f"Loaded {len(self.annotations)} samples for {phase} phase")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON file."""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
        
        if not isinstance(annotations, list):
            raise ValueError("Annotations should be a list of dictionaries")
        
        return annotations
    
    def _filter_annotations(self) -> List[Dict]:
        """Filter annotations based on categories and count range."""
        filtered = []
        
        for ann in self.annotations:
            # Check category filter
            if self.categories is not None:
                if ann.get('category') not in self.categories:
                    continue
            
            # Check count range filter
            if self.count_range is not None:
                count = ann.get('true_count')
                if count is None:
                    continue
                min_count, max_count = self.count_range
                if count < min_count or count > max_count:
                    continue
            
            filtered.append(ann)
        
        return filtered
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - count: Tensor with the true count
                - category: String category label
                - metadata: Dictionary with additional info
        """
        ann = self.annotations[idx]
        
        # Load image
        image_path = self.data_dir / ann['image_id']
        if not image_path.exists():
            # Try alternative paths
            image_path = self.data_dir / ann.get('category', '') / ann['image_id']
        
        image = Image.open(image_path).convert('RGB')
        
        # Get count
        count = ann.get('true_count')
        if count is None:
            raise ValueError(f"Annotation missing 'true_count' for image {ann['image_id']}")
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: resize and convert to tensor
            default_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_transform(image)
        
        # Prepare metadata
        metadata = {
            'category': ann.get('category', 'unknown'),
            'stack_angle': ann.get('stack_angle', 'unknown'),
            'lighting': ann.get('lighting', 'unknown'),
            'occlusion_percent': ann.get('occlusion_percent', 0),
            'agreement_score': ann.get('agreement_score', 1.0),
            'image_id': ann.get('image_id', ''),
        }
        
        return {
            'image': image,
            'count': torch.tensor(count, dtype=torch.float32),
            'category': ann.get('category', 'unknown'),
            'metadata': metadata
        }
    
    def get_statistics(self) -> Dict[str, Union[float, int]]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with statistics like mean count, std, etc.
        """
        counts = [ann['true_count'] for ann in self.annotations]
        
        stats = {
            'num_samples': len(self.annotations),
            'mean_count': np.mean(counts),
            'std_count': np.std(counts),
            'min_count': np.min(counts),
            'max_count': np.max(counts),
            'median_count': np.median(counts),
        }
        
        # Category distribution
        categories = [ann.get('category', 'unknown') for ann in self.annotations]
        unique_categories, counts_cat = np.unique(categories, return_counts=True)
        stats['category_distribution'] = dict(zip(unique_categories, counts_cat))
        
        return stats
    
    def get_category_samples(self, category: str) -> List[Dict]:
        """Get all samples for a specific category."""
        return [ann for ann in self.annotations if ann.get('category') == category]


class CurriculumDataset(StackCountDataset):
    """Dataset with curriculum learning support based on count ranges."""
    
    def __init__(self, *args, current_phase: int = 1, **kwargs):
        """Initialize curriculum dataset.
        
        Args:
            current_phase: Current curriculum phase (1, 2, or 3)
            Phase 1: counts 5-50
            Phase 2: counts 5-150  
            Phase 3: counts 5-500
        """
        self.current_phase = current_phase
        
        # Define count ranges for each phase
        self.phase_ranges = {
            1: (5, 50),
            2: (5, 150),
            3: (5, 500)
        }
        
        # Set count range based on phase
        count_range = self.phase_ranges.get(current_phase, (5, 500))
        kwargs['count_range'] = count_range
        
        super().__init__(*args, **kwargs)
        
        print(f"Curriculum Phase {current_phase}: count range {count_range}")
    
    def update_phase(self, new_phase: int):
        """Update the curriculum phase and reload data.
        
        Args:
            new_phase: New phase number (1, 2, or 3)
        """
        if new_phase == self.current_phase:
            return
        
        self.current_phase = new_phase
        count_range = self.phase_ranges.get(new_phase, (5, 500))
        self.count_range = count_range
        self.annotations = self._filter_annotations()
        print(f"Updated to Phase {new_phase}: count range {count_range}, {len(self.annotations)} samples")
