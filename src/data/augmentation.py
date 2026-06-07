"""Data augmentation pipelines using Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict, Any


class AugmentationPipeline:
    """Data augmentation pipelines for different phases."""
    
    @staticmethod
    def get_training_augmentation(
        image_size: int = 384,
        rotation_range: int = 15,
        zoom_range: tuple = (0.8, 1.2),
        brightness_jitter: float = 0.2,
        noise_std: float = 0.01,
        p_rotate: float = 0.5,
        p_zoom: float = 0.5,
        p_brightness: float = 0.5,
        p_noise: float = 0.3,
        p_flip: float = 0.5
    ) -> A.Compose:
        """Get training augmentation pipeline.
        
        Args:
            image_size: Target image size
            rotation_range: Maximum rotation angle in degrees (±)
            zoom_range: Zoom range (min, max)
            brightness_jitter: Brightness jitter intensity
            noise_std: Standard deviation for Gaussian noise
            p_rotate: Probability of applying rotation
            p_zoom: Probability of applying zoom
            p_brightness: Probability of brightness adjustment
            p_noise: Probability of adding noise
            p_flip: Probability of horizontal flip
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=p_flip),
            A.Rotate(limit=rotation_range, p=p_rotate, border_mode=0),
            A.RandomScale(scale_limit=0.2, p=p_zoom),
            A.Resize(image_size, image_size),
            
            # Color transformations
            A.RandomBrightnessContrast(
                brightness_limit=brightness_jitter,
                contrast_limit=brightness_jitter,
                p=p_brightness
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=p_brightness
            ),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=p_noise),
            A.GaussianBlur(blur_limit=3, p=0.1),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_validation_augmentation(
        image_size: int = 384
    ) -> A.Compose:
        """Get validation augmentation pipeline (minimal).
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_test_augmentation(
        image_size: int = 384
    ) -> A.Compose:
        """Get test augmentation pipeline (no augmentation).
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_inference_augmentation(
        image_size: int = 384
    ) -> A.Compose:
        """Get inference augmentation pipeline.
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                value=(0, 0, 0)
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class AdvancedAugmentation:
    """Advanced augmentation techniques for specific scenarios."""
    
    @staticmethod
    def get_light_robust_augmentation(image_size: int = 384) -> A.Compose:
        """Augmentation pipeline robust to lighting conditions.
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_occlusion_robust_augmentation(image_size: int = 384) -> A.Compose:
        """Augmentation pipeline robust to occlusion.
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5
            ),
            A.RandomSizedCrop(
                min_max_height=(int(image_size * 0.8), image_size),
                height=image_size,
                width=image_size,
                p=0.5
            ),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_angle_robust_augmentation(image_size: int = 384) -> A.Compose:
        """Augmentation pipeline robust to viewing angles.
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations compose object
        """
        return A.Compose([
            A.Rotate(limit=45, p=0.8, border_mode=0),
            A.Perspective(scale=0.2, p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.1),
                rotate=0,
                shear=0,
                p=0.5
            ),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class AugmentationConfig:
    """Configuration for augmentation parameters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize augmentation config from dictionary.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def get_training_pipeline(self) -> A.Compose:
        """Get training pipeline from config."""
        aug_config = self.config.get('augmentations', {})
        
        return AugmentationPipeline.get_training_augmentation(
            image_size=self.config.get('image_size', 384),
            rotation_range=aug_config.get('rotation', {}).get('range', 15),
            zoom_range=(
                aug_config.get('zoom', {}).get('min', 0.8),
                aug_config.get('zoom', {}).get('max', 1.2)
            ),
            brightness_jitter=aug_config.get('brightness', {}).get('jitter', 0.2),
            noise_std=0.01 if aug_config.get('noise', {}).get('gaussian', False) else 0.0
        )
    
    def get_validation_pipeline(self) -> A.Compose:
        """Get validation pipeline from config."""
        return AugmentationPipeline.get_validation_augmentation(
            image_size=self.config.get('image_size', 384)
        )
    
    def get_test_pipeline(self) -> A.Compose:
        """Get test pipeline from config."""
        return AugmentationPipeline.get_test_augmentation(
            image_size=self.config.get('image_size', 384)
        )
    
    def get_inference_pipeline(self) -> A.Compose:
        """Get inference pipeline from config."""
        return AugmentationPipeline.get_inference_augmentation(
            image_size=self.config.get('image_size', 384)
        )


def get_transforms(phase: str, config: Optional[Dict] = None) -> A.Compose:
    """Get transforms for a specific phase.
    
    Args:
        phase: One of 'train', 'val', 'test', 'inference'
        config: Optional configuration dictionary
        
    Returns:
        Albumentations compose object
    """
    if config is not None:
        aug_config = AugmentationConfig(config)
        if phase == 'train':
            return aug_config.get_training_pipeline()
        elif phase == 'val':
            return aug_config.get_validation_pipeline()
        elif phase == 'test':
            return aug_config.get_test_pipeline()
        elif phase == 'inference':
            return aug_config.get_inference_pipeline()
    
    # Default fallback
    if phase == 'train':
        return AugmentationPipeline.get_training_augmentation()
    elif phase in ['val', 'test']:
        return AugmentationPipeline.get_validation_augmentation()
    elif phase == 'inference':
        return AugmentationPipeline.get_inference_augmentation()
    else:
        raise ValueError(f"Unknown phase: {phase}")
