"""FastAPI REST API for stack count prediction."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import io
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import time
from datetime import datetime

from .predictor import StackCountPredictor, load_predictor


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    """Response model for prediction."""
    predicted_count: int
    confidence_score: float
    count_range: dict
    uncertainty_std: float
    uncertainty_level: str
    message: str
    processing_time_ms: float
    quality_warning: Optional[str] = None
    flag: Optional[str] = None


class ModelInfo(BaseModel):
    """Response model for model information."""
    model_type: str
    model_variant: str
    total_parameters: int
    trainable_parameters: int
    mc_dropout_enabled: bool
    mc_dropout_passes: int
    device: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    image_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# Global predictor instance
predictor: Optional[StackCountPredictor] = None
model_path: Optional[str] = None


def create_app() -> FastAPI:
    """Create FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Stack Count Prediction API",
        description="API for predicting item counts in stacked objects",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        global predictor, model_path
        
        # Try to load best model
        default_model_path = "models/checkpoints/best.pth"
        
        if Path(default_model_path).exists():
            try:
                predictor = load_predictor(default_model_path, device='cuda')
                model_path = default_model_path
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                predictor = None
        else:
            print(f"Model not found at {default_model_path}")
            predictor = None
    
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint."""
        return {
            "message": "Stack Count Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "predict": "/predict (POST)",
                "batch_predict": "/batch_predict (POST)",
                "model_info": "/model_info (GET)",
                "health": "/health (GET)"
            }
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if predictor is not None else "unhealthy",
            model_loaded=predictor is not None,
            device=predictor.device if predictor else "none",
            timestamp=datetime.now().isoformat()
        )
    
    @app.get("/model_info", response_model=ModelInfo)
    async def get_model_info():
        """Get model information."""
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        info = predictor.get_model_info()
        return ModelInfo(**info)
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        """Predict stack count for a single image.
        
        Args:
            file: Uploaded image file
            
        Returns:
            Prediction response
        """
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = np.array(image)
            
            # Run prediction
            result = predictor.predict(image_np)
            
            # Handle errors
            if 'error' in result:
                raise HTTPException(status_code=400, detail=result['error'])
            
            # Return response
            return PredictionResponse(**result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_predict")
    async def batch_predict(files: List[UploadFile] = File(...)):
        """Predict stack counts for multiple images.
        
        Args:
            files: List of uploaded image files
            
        Returns:
            List of prediction responses
        """
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            results = []
            
            for file in files:
                # Read image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert('RGB')
                image_np = np.array(image)
                
                # Run prediction
                result = predictor.predict(image_np)
                
                # Add filename to result
                result['filename'] = file.filename
                results.append(result)
            
            return {"results": results}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/load_model")
    async def load_model_endpoint(model_file_path: str):
        """Load a specific model.
        
        Args:
            model_file_path: Path to model checkpoint
            
        Returns:
            Success message
        """
        global predictor, model_path
        
        try:
            predictor = load_predictor(model_file_path, device='cuda')
            model_path = model_file_path
            
            return {
                "message": f"Model loaded from {model_file_path}",
                "model_info": predictor.get_model_info()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    @app.post("/update_thresholds")
    async def update_thresholds(
        high: float = 0.90,
        moderate: float = 0.70,
        low: float = 0.50
    ):
        """Update confidence thresholds.
        
        Args:
            high: High confidence threshold
            moderate: Moderate confidence threshold
            low: Low confidence threshold
            
        Returns:
            Success message
        """
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictor.set_confidence_thresholds({
            'high': high,
            'moderate': moderate,
            'low': low
        })
        
        return {
            "message": "Thresholds updated",
            "new_thresholds": predictor.confidence_thresholds
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
