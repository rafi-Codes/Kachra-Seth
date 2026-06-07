"""Predictor class for stack count inference."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any
import time
import json

from ..models import StackCountModel
from ..data.preprocessing import Preprocessor, BlurDetector
from ..utils.constants import (
    CONFIDENCE_HIGH, CONFIDENCE_MODERATE, CONFIDENCE_LOW,
    BLUR_THRESHOLD_LAPLACIAN, MAX_COUNT,
    ERROR_NO_STACK, ERROR_HIGH_DENSITY, ERROR_LOW_QUALITY
)


class StackCountPredictor:
    """Predictor for stack count estimation."""
    
    def __init__(
        self,
        model: StackCountModel,
        device: str = 'cuda',
        mc_dropout_passes: int = 15,
        blur_threshold: float = 100.0,
        confidence_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize predictor.
        
        Args:
            model: Trained StackCountModel
            device: Device to run inference on
            mc_dropout_passes: Number of MC Dropout samples
            blur_threshold: Laplacian variance threshold for blur detection
            confidence_thresholds: Dictionary of confidence thresholds
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.mc_dropout_passes = mc_dropout_passes
        self.blur_threshold = blur_threshold
        
        # Confidence thresholds
        if confidence_thresholds is None:
            self.confidence_thresholds = {
                'high': CONFIDENCE_HIGH,      # 0.90
                'moderate': CONFIDENCE_MODERATE,  # 0.70
                'low': CONFIDENCE_LOW         # 0.50
            }
        else:
            self.confidence_thresholds = confidence_thresholds
        
        # Preprocessor
        self.preprocessor = Preprocessor(
            target_size=(384, 384),
            blur_threshold=self.blur_threshold,
            normalize=True
        )
        
        # Blur detector
        self.blur_detector = BlurDetector(threshold=self.blur_threshold)
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_image: bool = False
    ) -> Dict[str, Any]:
        """Predict stack count for a single image.
        
        Args:
            image: Input image (path, array, or PIL Image)
            return_image: Whether to return processed image
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess image
        preprocess_result = self.preprocessor.preprocess(
            image,
            detect_blur=True
        )
        
        # Check for edge cases
        edge_case = self._check_edge_cases(preprocess_result)
        if edge_case is not None:
            return {
                'error': edge_case,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        # Prepare input tensor
        image_tensor = preprocess_result['processed_image'].unsqueeze(0).to(self.device)
        
        # Run inference with MC Dropout
        with torch.no_grad():
            predictions = self.model.predict_with_uncertainty(
                image_tensor,
                n_samples=self.mc_dropout_passes
            )
        
        # Extract predictions
        count_mean = predictions['count_mean'].cpu().item()
        count_std = predictions['count_std'].cpu().item()
        confidence = predictions['uncertainty_based_confidence'].cpu().item()
        
        # Calculate count range
        count_range = self._calculate_count_range(count_mean, count_std)
        
        # Determine uncertainty level
        uncertainty_level = self._determine_uncertainty_level(confidence)
        
        # Determine message based on confidence
        message = self._generate_message(confidence, count_mean, count_range)
        
        # Check for high density
        high_density_flag = None
        if count_mean > MAX_COUNT:
            high_density_flag = ERROR_HIGH_DENSITY
        
        # Build result
        result = {
            'predicted_count': round(count_mean),
            'confidence_score': round(confidence, 3),
            'count_range': {
                'low': max(0, round(count_range[0])),
                'high': round(count_range[1])
            },
            'uncertainty_std': round(count_std, 2),
            'uncertainty_level': uncertainty_level,
            'message': message,
            'processing_time_ms': round((time.time() - start_time) * 1000, 1)
        }
        
        # Add optional fields
        if high_density_flag:
            result['flag'] = high_density_flag
        
        if preprocess_result['blur_detected']:
            result['quality_warning'] = 'image may be blurry'
        
        if return_image:
            result['processed_image'] = preprocess_result['processed_image']
        
        return result
    
    def predict_batch(
        self,
        images: list,
        return_images: bool = False
    ) -> list:
        """Predict stack counts for multiple images.
        
        Args:
            images: List of input images
            return_images: Whether to return processed images
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image in images:
            result = self.predict(image, return_image=return_images)
            results.append(result)
        
        return results
    
    def _check_edge_cases(self, preprocess_result: Dict) -> Optional[str]:
        """Check for edge cases that should trigger errors.
        
        Args:
            preprocess_result: Preprocessing result dictionary
            
        Returns:
            Error message if edge case detected, None otherwise
        """
        # Check for blurry image
        if preprocess_result['blur_detected']:
            # Don't reject outright, but flag it
            pass
        
        # Check for extremely low quality (could be added)
        # For now, we'll warn but allow inference
        
        return None
    
    def _calculate_count_range(
        self,
        count_mean: float,
        count_std: float,
        multiplier: float = 1.0
    ) -> Tuple[float, float]:
        """Calculate count range based on uncertainty.
        
        Args:
            count_mean: Mean prediction
            count_std: Standard deviation
            multiplier: Multiplier for std
            
        Returns:
            Tuple of (low, high) bounds
        """
        low = count_mean - multiplier * count_std
        high = count_mean + multiplier * count_std
        
        # Ensure non-negative
        low = max(0, low)
        
        return low, high
    
    def _determine_uncertainty_level(self, confidence: float) -> str:
        """Determine uncertainty level from confidence score.
        
        Args:
            confidence: Confidence score [0, 1]
            
        Returns:
            Uncertainty level string
        """
        if confidence >= self.confidence_thresholds['high']:
            return 'low'
        elif confidence >= self.confidence_thresholds['moderate']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'high'
        else:
            return 'very_high'
    
    def _generate_message(
        self,
        confidence: float,
        count: float,
        count_range: Tuple[float, float]
    ) -> str:
        """Generate user-friendly message based on confidence.
        
        Args:
            confidence: Confidence score
            count: Predicted count
            count_range: Count range tuple
            
        Returns:
            User-friendly message
        """
        count_rounded = round(count)
        range_low = max(0, round(count_range[0]))
        range_high = round(count_range[1])
        
        if confidence >= self.confidence_thresholds['high']:
            return f"Count: {count_rounded} items"
        elif confidence >= self.confidence_thresholds['moderate']:
            return f"Count: {count_rounded} items (range: {range_low}-{range_high})"
        elif confidence >= self.confidence_thresholds['low']:
            return f"Estimated range: {range_low}-{range_high} items. Please retake photo for better accuracy."
        else:
            return "Unable to predict reliably. Please provide a clearer image."
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo Dropout."""
        self.model.enable_mc_dropout()
    
    def disable_mc_dropout(self):
        """Disable Monte Carlo Dropout."""
        self.model.disable_mc_dropout()
    
    def set_confidence_thresholds(self, thresholds: Dict[str, float]):
        """Update confidence thresholds.
        
        Args:
            thresholds: Dictionary of thresholds
        """
        self.confidence_thresholds.update(thresholds)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        params = self.model.get_num_parameters()
        
        return {
            'model_type': self.model.backbone_type,
            'model_variant': self.model.backbone_variant,
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'mc_dropout_enabled': self.model.mc_dropout,
            'mc_dropout_passes': self.mc_dropout_passes,
            'device': self.device
        }


def load_predictor(
    model_path: Union[str, Path],
    device: str = 'cuda',
    mc_dropout_passes: int = 15
) -> StackCountPredictor:
    """Load predictor from saved model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to run on
        mc_dropout_passes: Number of MC Dropout samples
        
    Returns:
        StackCountPredictor instance
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get configuration
    config = checkpoint.get('config', {})
    
    # Create model from config
    model = StackCountModel.create_model_from_config(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create predictor
    predictor = StackCountPredictor(
        model=model,
        device=device,
        mc_dropout_passes=mc_dropout_passes
    )
    
    return predictor


# Test predictor
if __name__ == "__main__":
    print("Testing StackCountPredictor...")
    
    # Create dummy model
    from ..models import create_default_model
    model = create_default_model(pretrained=False)
    
    # Create predictor
    predictor = StackCountPredictor(
        model=model,
        device='cpu',
        mc_dropout_passes=5  # Use fewer for testing
    )
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    # Test prediction
    print("\nTesting prediction...")
    result = predictor.predict(dummy_image)
    
    print("Prediction result:")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # Test model info
    print("\nModel info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nPredictor test passed!")
