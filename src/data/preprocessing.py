"""Data preprocessing utilities including blur detection and image resizing."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import torch


class BlurDetector:
    """Detect blurry images using Laplacian variance."""
    
    def __init__(self, threshold: float = 100.0):
        """Initialize blur detector.
        
        Args:
            threshold: Laplacian variance threshold (below = blurry)
        """
        self.threshold = threshold
    
    def detect_blur(self, image: Union[np.ndarray, str, Path]) -> Tuple[bool, float]:
        """Detect if an image is blurry.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            Tuple of (is_blurry, variance_score)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Determine if blurry
        is_blurry = laplacian_var < self.threshold
        
        return is_blurry, laplacian_var
    
    def batch_detect_blur(
        self,
        image_paths: list,
        return_details: bool = True
    ) -> Dict:
        """Detect blur in multiple images.
        
        Args:
            image_paths: List of image paths
            return_details: Whether to return detailed scores
            
        Returns:
            Dictionary with blur detection results
        """
        results = {
            'total': len(image_paths),
            'blurry': 0,
            'clear': 0,
            'details': {} if return_details else None
        }
        
        for img_path in image_paths:
            try:
                is_blurry, variance = self.detect_blur(img_path)
                
                if is_blurry:
                    results['blurry'] += 1
                else:
                    results['clear'] += 1
                
                if return_details:
                    results['details'][str(img_path)] = {
                        'is_blurry': is_blurry,
                        'variance': variance
                    }
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                if return_details:
                    results['details'][str(img_path)] = {
                        'error': str(e)
                    }
        
        return results


class ImageResizer:
    """Handle image resizing with padding."""
    
    @staticmethod
    def resize_with_padding(
        image: Union[np.ndarray, Image.Image],
        target_size: Tuple[int, int] = (384, 384),
        padding_color: Tuple[int, int, int] = (0, 0, 0),
        return_pil: bool = False
    ) -> Union[np.ndarray, Image.Image]:
        """Resize image maintaining aspect ratio with padding.
        
        Args:
            image: Input image as numpy array or PIL Image
            target_size: Target (width, height)
            padding_color: Padding color (R, G, B)
            return_pil: Whether to return PIL Image instead of numpy array
            
        Returns:
            Resized and padded image
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        target_width, target_height = target_size
        original_width, original_height = image.size
        
        # Calculate scaling factor
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with padding
        padded = Image.new('RGB', target_size, padding_color)
        
        # Calculate padding offsets (center the image)
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        
        # Paste resized image onto padded canvas
        padded.paste(resized, (pad_left, pad_top))
        
        if return_pil:
            return padded
        else:
            # Convert back to numpy array (BGR for OpenCV)
            return cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def resize_with_crop(
        image: Union[np.ndarray, Image.Image],
        target_size: Tuple[int, int] = (384, 384),
        return_pil: bool = False
    ) -> Union[np.ndarray, Image.Image]:
        """Resize image with center crop.
        
        Args:
            image: Input image as numpy array or PIL Image
            target_size: Target (width, height)
            return_pil: Whether to return PIL Image instead of numpy array
            
        Returns:
            Resized and cropped image
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Use PIL's thumbnail with center crop
        image.thumbnail(target_size, Image.LANCZOS)
        
        # Center crop
        left = (image.width - target_size[0]) // 2
        top = (image.height - target_size[1]) // 2
        right = left + target_size[0]
        bottom = top + target_size[1]
        
        cropped = image.crop((left, top, right, bottom))
        
        if return_pil:
            return cropped
        else:
            return cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def resize_letterbox(
        image: Union[np.ndarray, Image.Image],
        target_size: Tuple[int, int] = (384, 384),
        padding_color: Tuple[int, int, int] = (114, 114, 114),
        return_pil: bool = False
    ) -> Union[np.ndarray, Image.Image]:
        """Resize image using letterbox method (pad to maintain aspect ratio).
        
        Args:
            image: Input image as numpy array or PIL Image
            target_size: Target (width, height)
            padding_color: Padding color (R, G, B)
            return_pil: Whether to return PIL Image instead of numpy array
            
        Returns:
            Resized image with letterbox padding
        """
        return ImageResizer.resize_with_padding(
            image, target_size, padding_color, return_pil
        )


class ImageNormalizer:
    """Normalize images for model input."""
    
    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    @staticmethod
    def normalize(
        image: np.ndarray,
        mean: Optional[list] = None,
        std: Optional[list] = None
    ) -> np.ndarray:
        """Normalize image array.
        
        Args:
            image: Image array in range [0, 255] or [0, 1]
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
            
        Returns:
            Normalized image array
        """
        if mean is None:
            mean = ImageNormalizer.IMAGENET_MEAN
        if std is None:
            std = ImageNormalizer.IMAGENET_STD
        
        # Convert to float if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        
        normalized = (image - mean) / std
        
        return normalized
    
    @staticmethod
    def denormalize(
        image: np.ndarray,
        mean: Optional[list] = None,
        std: Optional[list] = None
    ) -> np.ndarray:
        """Denormalize image array.
        
        Args:
            image: Normalized image array
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
            
        Returns:
            Denormalized image array in range [0, 1]
        """
        if mean is None:
            mean = ImageNormalizer.IMAGENET_MEAN
        if std is None:
            std = ImageNormalizer.IMAGENET_STD
        
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        
        denormalized = image * std + mean
        denormalized = np.clip(denormalized, 0, 1)
        
        return denormalized
    
    @staticmethod
    def normalize_tensor(
        tensor: torch.Tensor,
        mean: Optional[list] = None,
        std: Optional[list] = None
    ) -> torch.Tensor:
        """Normalize PyTorch tensor.
        
        Args:
            tensor: Image tensor (C, H, W) in range [0, 1]
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
            
        Returns:
            Normalized tensor
        """
        if mean is None:
            mean = ImageNormalizer.IMAGENET_MEAN
        if std is None:
            std = ImageNormalizer.IMAGENET_STD
        
        mean = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
        
        return (tensor - mean) / std


class Preprocessor:
    """Complete preprocessing pipeline for inference."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (384, 384),
        blur_threshold: float = 100.0,
        normalize: bool = True
    ):
        """Initialize preprocessor.
        
        Args:
            target_size: Target image size
            blur_threshold: Blur detection threshold
            normalize: Whether to normalize images
        """
        self.target_size = target_size
        self.blur_detector = BlurDetector(threshold=blur_threshold)
        self.normalize = normalize
    
    def preprocess(
        self,
        image: Union[np.ndarray, str, Path],
        detect_blur: bool = True
    ) -> Dict:
        """Complete preprocessing pipeline.
        
        Args:
            image: Input image as array or path
            detect_blur: Whether to perform blur detection
            
        Returns:
            Dictionary containing processed image and metadata
        """
        result = {
            'blur_detected': False,
            'blur_score': None,
            'original_size': None,
            'processed_image': None
        }
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Store original size
        result['original_size'] = (img.shape[1], img.shape[0])
        
        # Blur detection
        if detect_blur:
            is_blurry, blur_score = self.blur_detector.detect_blur(img)
            result['blur_detected'] = is_blurry
            result['blur_score'] = blur_score
        
        # Resize with padding
        resized = ImageResizer.resize_with_padding(img, self.target_size)
        
        # Normalize if requested
        if self.normalize:
            processed = ImageNormalizer.normalize(resized)
        else:
            # Convert to float and scale to [0, 1]
            processed = resized.astype(np.float32) / 255.0
        
        # Convert to CHW format (C, H, W) for PyTorch
        processed = np.transpose(processed, (2, 0, 1))
        result['processed_image'] = torch.from_numpy(processed).float()
        
        return result
    
    def batch_preprocess(
        self,
        image_paths: list,
        detect_blur: bool = True
    ) -> Dict:
        """Preprocess multiple images.
        
        Args:
            image_paths: List of image paths
            detect_blur: Whether to perform blur detection
            
        Returns:
            Dictionary with batched results
        """
        results = {
            'images': [],
            'metadata': []
        }
        
        for img_path in image_paths:
            try:
                result = self.preprocess(img_path, detect_blur=detect_blur)
                results['images'].append(result['processed_image'])
                results['metadata'].append({
                    'path': str(img_path),
                    'blur_detected': result['blur_detected'],
                    'blur_score': result['blur_score'],
                    'original_size': result['original_size']
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results['metadata'].append({
                    'path': str(img_path),
                    'error': str(e)
                })
        
        # Stack images into batch
        if results['images']:
            results['images'] = torch.stack(results['images'])
        
        return results


def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (384, 384),
    blur_threshold: float = 100.0
) -> Dict:
    """Convenience function to load and preprocess a single image.
    
    Args:
        image_path: Path to image file
        target_size: Target image size
        blur_threshold: Blur detection threshold
        
    Returns:
        Dictionary with preprocessed image and metadata
    """
    preprocessor = Preprocessor(target_size=target_size, blur_threshold=blur_threshold)
    return preprocessor.preprocess(image_path, detect_blur=True)


def check_image_quality(
    image_path: Union[str, Path],
    blur_threshold: float = 100.0,
    min_resolution: Tuple[int, int] = (256, 256)
) -> Dict:
    """Check overall image quality.
    
    Args:
        image_path: Path to image file
        blur_threshold: Blur detection threshold
        min_resolution: Minimum acceptable resolution (width, height)
        
    Returns:
        Dictionary with quality assessment
    """
    result = {
        'path': str(image_path),
        'is_acceptable': True,
        'issues': []
    }
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        result['is_acceptable'] = False
        result['issues'].append('Could not load image')
        return result
    
    # Check resolution
    height, width = img.shape[:2]
    if width < min_resolution[0] or height < min_resolution[1]:
        result['is_acceptable'] = False
        result['issues'].append(f'Resolution too low: {width}x{height}')
    
    # Check blur
    blur_detector = BlurDetector(threshold=blur_threshold)
    is_blurry, blur_score = blur_detector.detect_blur(img)
    
    if is_blurry:
        result['is_acceptable'] = False
        result['issues'].append(f'Image blurry (score: {blur_score:.2f})')
    
    result['blur_score'] = blur_score
    result['resolution'] = (width, height)
    
    return result
