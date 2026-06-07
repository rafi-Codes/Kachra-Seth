"""Gradio demo interface for stack count prediction."""

import sys
from pathlib import Path
import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.predictor import load_predictor


def create_demo(
    model_path: str = "models/checkpoints/best.pth",
    device: str = 'cuda',
    share: bool = False
):
    """Create Gradio demo interface.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to run on
        share: Whether to share publicly
    """
    
    # Load predictor
    try:
        predictor = load_predictor(model_path, device=device)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Creating demo without model (for UI testing only)")
        predictor = None
    
    def predict_image(image: np.ndarray) -> Tuple[str, str]:
        """Predict stack count for uploaded image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (result_text, detailed_info)
        """
        if predictor is None:
            return "Model not loaded", "Please train a model first"
        
        if image is None:
            return "No image provided", "Please upload an image"
        
        try:
            # Run prediction
            result = predictor.predict(image)
            
            # Format result text
            result_text = result['message']
            
            # Format detailed info
            detailed_info = f"""
            **Prediction Details:**
            - Predicted Count: {result['predicted_count']}
            - Confidence Score: {result['confidence_score']:.3f}
            - Count Range: {result['count_range']['low']} - {result['count_range']['high']}
            - Uncertainty: {result['uncertainty_std']:.2f}
            - Uncertainty Level: {result['uncertainty_level']}
            - Processing Time: {result['processing_time_ms']}ms
            """
            
            if 'quality_warning' in result:
                detailed_info += f"\n- Warning: {result['quality_warning']}"
            
            if 'flag' in result:
                detailed_info += f"\n- Flag: {result['flag']}"
            
            return result_text, detailed_info
            
        except Exception as e:
            return f"Error: {str(e)}", str(e)
    
    # Create interface
    with gr.Blocks(
        title="Stack Count Prediction",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # Stack Count Prediction 📊
            
            Upload an image of stacked items (banknotes, books, papers, tiles, cards, etc.) 
            to get an accurate count with confidence estimation.
            
            ## Features
            - **Precise Counting**: Estimates exact item count
            - **Confidence Score**: Indicates prediction reliability
            - **Range Estimation**: Provides count range when uncertain
            - **Quality Detection**: Warns about blurry images
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=400
                )
                predict_btn = gr.Button(
                    "Count Items",
                    variant="primary",
                    size="lg"
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary"
                )
            
            with gr.Column():
                result_output = gr.Textbox(
                    label="Result",
                    lines=2,
                    scale=2
                )
                details_output = gr.Markdown(
                    label="Details"
                )
        
        gr.Markdown(
            """
            ## How to Use
            1. Upload an image showing a stack of items
            2. Click "Count Items" to run prediction
            3. View the count estimate and confidence
            4. If confidence is low, retake the photo for better accuracy
            
            ## Confidence Levels
            - **High (≥0.90)**: Very confident in the count
            - **Moderate (0.70-0.89)**: Reasonably confident, provides range
            - **Low (0.50-0.69)**: Uncertain, provides range and suggests retake
            - **Very Low (<0.50)**: Unable to predict reliably
            """
        )
        
        # Event handlers
        predict_btn.click(
            fn=predict_image,
            inputs=input_image,
            outputs=[result_output, details_output]
        )
        
        clear_btn.click(
            fn=lambda: (None, "", ""),
            outputs=[input_image, result_output, details_output]
        )
        
        # Model info
        if predictor is not None:
            model_info = predictor.get_model_info()
            gr.Markdown(
                f"""
                ---
                
                ## Model Information
                - **Type**: {model_info['model_type']}-{model_info['model_variant']}
                - **Parameters**: {model_info['total_parameters']:,}
                - **Device**: {model_info['device']}
                - **MC Dropout**: {model_info['mc_dropout_passes']} samples
                """
            )
    
    # Launch demo
    demo.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Gradio demo for stack count prediction')
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/checkpoints/best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Share demo publicly'
    )
    
    args = parser.parse_args()
    
    create_demo(
        model_path=args.model_path,
        device=args.device,
        share=args.share
    )
