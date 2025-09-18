from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import torch
from datetime import datetime
import os
from pathlib import Path

# Import model wrapper
from model_wrapper import ModelWrapper

app = FastAPI(
    title="Guard-X AI Surveillance API",
    description="Human Detection API using YOLO/Custom Models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model wrapper
model_wrapper = ModelWrapper()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Starting Guard-X AI Backend...")
    await model_wrapper.load_models()
    print("✅ Models loaded successfully!")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "🚀 Guard-X AI Surveillance API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/api/detect",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

@app.post("/api/detect")
async def detect_humans(file: UploadFile = File(...)):
    """
    Human detection endpoint
    Upload an image and get human detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run detection
        detection_result = await model_wrapper.detect_humans(image)
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "boxes": detection_result["boxes"],
            "count": detection_result["count"],
            "confidence_scores": detection_result["confidences"],
            "timestamp": datetime.now().isoformat(),
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "model_used": detection_result["model_type"],
            "processing_time": detection_result["processing_time"]
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Detection failed: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    Returns system status and model information
    """
    health_status = await model_wrapper.get_health_status()
    
    return {
        "status": "operational" if health_status["models_loaded"] else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models": health_status["models"],
        "system": {
            "python_version": health_status["python_version"],
            "torch_version": health_status["torch_version"],
            "device": health_status["device"]
        }
    }

@app.get("/api/models")
async def list_models():
    """
    List available models
    """
    models_info = await model_wrapper.get_models_info()
    return {
        "available_models": models_info,
        "active_model": model_wrapper.active_model_name
    }

if __name__ == "__main__":
    print("🚀 Starting Guard-X FastAPI Server...")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
