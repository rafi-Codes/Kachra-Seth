"""Annotation format validation and parsing utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np


class AnnotationValidator:
    """Validate annotation format according to specification."""
    
    REQUIRED_FIELDS = [
        'image_id',
        'category',
        'true_count',
        'annotator_1_count',
        'annotator_2_count',
        'agreement_score',
        'stack_angle',
        'lighting',
        'occlusion_percent'
    ]
    
    VALID_CATEGORIES = [
        'banknotes',
        'books',
        'papers',
        'tiles',
        'cards',
        'plates'
    ]
    
    VALID_STACK_ANGLES = [
        'top_down',
        '45_degree',
        'side'
    ]
    
    VALID_LIGHTING = [
        'natural',
        'bright',
        'dim',
        'backlit'
    ]
    
    @classmethod
    def validate_annotation(cls, annotation: Dict) -> Tuple[bool, List[str]]:
        """Validate a single annotation entry.
        
        Args:
            annotation: Dictionary containing annotation data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in annotation:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate image_id
        if not isinstance(annotation['image_id'], str):
            errors.append("image_id must be a string")
        elif not annotation['image_id']:
            errors.append("image_id cannot be empty")
        
        # Validate category
        if annotation['category'] not in cls.VALID_CATEGORIES:
            errors.append(f"Invalid category: {annotation['category']}. Must be one of {cls.VALID_CATEGORIES}")
        
        # Validate true_count
        if not isinstance(annotation['true_count'], int):
            errors.append("true_count must be an integer")
        elif annotation['true_count'] < 5 or annotation['true_count'] > 500:
            errors.append("true_count must be between 5 and 500")
        
        # Validate annotator counts
        if not isinstance(annotation['annotator_1_count'], int):
            errors.append("annotator_1_count must be an integer")
        if not isinstance(annotation['annotator_2_count'], int):
            errors.append("annotator_2_count must be an integer")
        
        # Validate agreement_score
        if not isinstance(annotation['agreement_score'], (int, float)):
            errors.append("agreement_score must be a number")
        elif annotation['agreement_score'] < 0 or annotation['agreement_score'] > 1:
            errors.append("agreement_score must be between 0 and 1")
        
        # Validate stack_angle
        if annotation['stack_angle'] not in cls.VALID_STACK_ANGLES:
            errors.append(f"Invalid stack_angle: {annotation['stack_angle']}. Must be one of {cls.VALID_STACK_ANGLES}")
        
        # Validate lighting
        if annotation['lighting'] not in cls.VALID_LIGHTING:
            errors.append(f"Invalid lighting: {annotation['lighting']}. Must be one of {cls.VALID_LIGHTING}")
        
        # Validate occlusion_percent
        if not isinstance(annotation['occlusion_percent'], (int, float)):
            errors.append("occlusion_percent must be a number")
        elif annotation['occlusion_percent'] < 0 or annotation['occlusion_percent'] > 100:
            errors.append("occlusion_percent must be between 0 and 100")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_annotation_file(cls, file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate an entire annotation file.
        
        Args:
            file_path: Path to the JSON annotation file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, [f"File not found: {file_path}"]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {str(e)}"]
        
        if not isinstance(data, list):
            return False, ["Annotation file must contain a list of annotations"]
        
        if len(data) == 0:
            errors.append("Annotation file is empty")
        
        # Validate each annotation
        all_valid = True
        for i, annotation in enumerate(data):
            is_valid, ann_errors = cls.validate_annotation(annotation)
            if not is_valid:
                all_valid = False
                for error in ann_errors:
                    errors.append(f"Annotation {i}: {error}")
        
        return all_valid, errors


class AnnotationParser:
    """Parse and manipulate annotation data."""
    
    @staticmethod
    def load_annotations(file_path: Union[str, Path]) -> List[Dict]:
        """Load annotations from JSON file.
        
        Args:
            file_path: Path to the annotation file
            
        Returns:
            List of annotation dictionaries
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_annotations(annotations: List[Dict], file_path: Union[str, Path]):
        """Save annotations to JSON file.
        
        Args:
            annotations: List of annotation dictionaries
            file_path: Path to save the file
        """
        with open(file_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    @staticmethod
    def filter_by_category(
        annotations: List[Dict],
        categories: Union[str, List[str]]
    ) -> List[Dict]:
        """Filter annotations by category.
        
        Args:
            annotations: List of annotations
            categories: Category or list of categories to keep
            
        Returns:
            Filtered list of annotations
        """
        if isinstance(categories, str):
            categories = [categories]
        
        return [ann for ann in annotations if ann['category'] in categories]
    
    @staticmethod
    def filter_by_count_range(
        annotations: List[Dict],
        min_count: int,
        max_count: int
    ) -> List[Dict]:
        """Filter annotations by count range.
        
        Args:
            annotations: List of annotations
            min_count: Minimum count (inclusive)
            max_count: Maximum count (inclusive)
            
        Returns:
            Filtered list of annotations
        """
        return [
            ann for ann in annotations
            if min_count <= ann['true_count'] <= max_count
        ]
    
    @staticmethod
    def filter_by_agreement(
        annotations: List[Dict],
        min_agreement: float = 0.9
    ) -> List[Dict]:
        """Filter annotations by agreement score.
        
        Args:
            annotations: List of annotations
            min_agreement: Minimum agreement score (inclusive)
            
        Returns:
            Filtered list of annotations
        """
        return [
            ann for ann in annotations
            if ann['agreement_score'] >= min_agreement
        ]
    
    @staticmethod
    def get_statistics(annotations: List[Dict]) -> Dict:
        """Get statistics from annotations.
        
        Args:
            annotations: List of annotations
            
        Returns:
            Dictionary containing various statistics
        """
        counts = [ann['true_count'] for ann in annotations]
        categories = [ann['category'] for ann in annotations]
        agreements = [ann['agreement_score'] for ann in annotations]
        
        # Count statistics
        count_stats = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'median': np.median(counts),
            'total': len(counts)
        }
        
        # Category distribution
        unique_categories, category_counts = np.unique(categories, return_counts=True)
        category_dist = dict(zip(unique_categories, category_counts))
        
        # Agreement statistics
        agreement_stats = {
            'mean': np.mean(agreements),
            'std': np.std(agreements),
            'min': np.min(agreements),
            'max': np.max(agreements)
        }
        
        return {
            'count_statistics': count_stats,
            'category_distribution': category_dist,
            'agreement_statistics': agreement_stats,
            'num_annotations': len(annotations)
        }
    
    @staticmethod
    def create_annotation(
        image_id: str,
        category: str,
        true_count: int,
        annotator_1_count: int,
        annotator_2_count: int,
        stack_angle: str = "top_down",
        lighting: str = "natural",
        occlusion_percent: int = 0
    ) -> Dict:
        """Create a new annotation entry.
        
        Args:
            image_id: Image filename
            category: Object category
            true_count: Ground truth count
            annotator_1_count: First annotator's count
            annotator_2_count: Second annotator's count
            stack_angle: Stack angle
            lighting: Lighting condition
            occlusion_percent: Percentage of occlusion (0-100)
            
        Returns:
            Annotation dictionary
        """
        # Calculate agreement score
        if annotator_1_count == 0 and annotator_2_count == 0:
            agreement_score = 1.0
        else:
            agreement_score = 1.0 - abs(annotator_1_count - annotator_2_count) / max(annotator_1_count, annotator_2_count)
        
        return {
            'image_id': image_id,
            'category': category,
            'true_count': true_count,
            'annotator_1_count': annotator_1_count,
            'annotator_2_count': annotator_2_count,
            'agreement_score': round(agreement_score, 2),
            'stack_angle': stack_angle,
            'lighting': lighting,
            'occlusion_percent': occlusion_percent
        }


class AnnotationSplitter:
    """Split annotations into train/val/test sets with no leakage."""
    
    @staticmethod
    def split_by_image(
        annotations: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split annotations by image (no leakage).
        
        Args:
            annotations: List of annotations
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_annotations, val_annotations, test_annotations)
        """
        np.random.seed(random_seed)
        
        # Shuffle annotations
        shuffled = annotations.copy()
        np.random.shuffle(shuffled)
        
        # Calculate split indices
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]
        
        return train, val, test
    
    @staticmethod
    def split_by_category(
        annotations: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Dict[str, Tuple[List[Dict], List[Dict], List[Dict]]]:
        """Split annotations ensuring each category is represented in all sets.
        
        Args:
            annotations: List of annotations
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping categories to (train, val, test) tuples
        """
        categories = set(ann['category'] for ann in annotations)
        splits = {}
        
        for category in categories:
            category_annotations = [ann for ann in annotations if ann['category'] == category]
            train, val, test = AnnotationSplitter.split_by_image(
                category_annotations,
                train_ratio,
                val_ratio,
                test_ratio,
                random_seed
            )
            splits[category] = (train, val, test)
        
        return splits
    
    @staticmethod
    def save_splits(
        train: List[Dict],
        val: List[Dict],
        test: List[Dict],
        output_dir: Union[str, Path]
    ):
        """Save split annotations to separate files.
        
        Args:
            train: Training annotations
            val: Validation annotations
            test: Test annotations
            output_dir: Directory to save the files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        AnnotationParser.save_annotations(train, output_dir / 'train_annotations.json')
        AnnotationParser.save_annotations(val, output_dir / 'val_annotations.json')
        AnnotationParser.save_annotations(test, output_dir / 'test_annotations.json')
        
        print(f"Saved splits to {output_dir}")
        print(f"  Train: {len(train)} samples")
        print(f"  Val: {len(val)} samples")
        print(f"  Test: {len(test)} samples")
