import torch
import cv2
import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path
import asyncio

class ModelWrapper:
    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def load_models(self):
        """Load all available models"""
        models_dir = Path("models")
        
        # Try to load custom model first
        custom_model_path = models_dir / "best.pt"
        if custom_model_path.exists():
            await self._load_custom_model(custom_model_path)
        
        # Load YOLO as fallback
        await self._load_yolo_model()
        
        # Set active model
        if "custom" in self.models:
            self.active_model_name = "custom"
            print("✅ Using custom trained model")
        elif "yolo" in self.models:
            self.active_model_name = "yolo"
            print("✅ Using YOLO ultralytics model")
        else:
            print("❌ No models loaded!")
    
    async def _load_custom_model(self, model_path):
        """Load custom trained model"""
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            self.models["custom"] = {
                "model": model,
                "type": "custom_yolo",
                "path": str(model_path),
                "loaded": True
            }
            print(f"✅ Custom model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load custom model: {e}")
    
    async def _load_yolo_model(self):
        """Load YOLO ultralytics model as fallback"""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # Downloads automatically if not present
            self.models["yolo"] = {
                "model": model,
                "type": "yolo_ultralytics",
                "path": "yolov8n.pt",
                "loaded": True
            }
            print("✅ YOLO ultralytics model loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
    
    async def detect_humans(self, image):
        """Run human detection on image"""
        if not self.active_model_name or self.active_model_name not in self.models:
            raise Exception("No active model available")
        
        start_time = time.time()
        
        # Get active model
        model_info = self.models[self.active_model_name]
        model = model_info["model"]
        
        # Save temp image for processing
        temp_path = f"temp_detection_{int(time.time())}.jpg"
        image.save(temp_path)
        
        try:
            # Run inference
            results = model(temp_path)
            
            boxes = []
            confidences = []
            
            # Process results
            for result in results:
                for box in result.boxes:
                    # Check if detected class is 'person' (class 0 in COCO)
                    if int(box.cls) == 0:  # Person class
                        confidence = float(box.conf)
                        if confidence > 0.5:  # Confidence threshold
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            boxes.append([int(x1), int(y1), int(x2), int(y2)])
                            confidences.append(round(confidence, 3))
            
            processing_time = round(time.time() - start_time, 3)
            
            return {
                "boxes": boxes,
                "count": len(boxes),
                "confidences": confidences,
                "model_type": model_info["type"],
                "processing_time": processing_time
            }
            
        finally:
            # Clean up temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    
    async def get_health_status(self):
        """Get system health status"""
        return {
            "models_loaded": len(self.models) > 0,
            "models": {
                name: {
                    "type": info["type"],
                    "loaded": info["loaded"],
                    "active": name == self.active_model_name
                }
                for name, info in self.models.items()
            },
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(self.device)
        }
    
    async def get_models_info(self):
        """Get information about available models"""
        return [
            {
                "name": name,
                "type": info["type"],
                "path": info["path"],
                "loaded": info["loaded"],
                "active": name == self.active_model_name
            }
            for name, info in self.models.items()
        ]