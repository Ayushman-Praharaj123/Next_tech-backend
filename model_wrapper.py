import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
from PIL import Image
import asyncio

class ModelWrapper:
    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.confidence_threshold = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    async def load_models(self):
        """Load both custom and fallback models"""
        print("🔄 Loading AI models...")
        
        # Try to load custom trained model first
        custom_model_path = Path("models/best.pt")
        if custom_model_path.exists():
            try:
                self.models['custom'] = YOLO(str(custom_model_path))
                self.active_model_name = 'custom'
                print(f"✅ Custom model loaded: {custom_model_path}")
            except Exception as e:
                print(f"❌ Custom model failed: {e}")
        
        # Load fallback YOLO model
        try:
            self.models['yolo'] = YOLO('yolov8n.pt')
            if not self.active_model_name:
                self.active_model_name = 'yolo'
            print("✅ YOLO fallback model loaded")
        except Exception as e:
            print(f"❌ YOLO model failed: {e}")
            
        print(f"🎯 Active model: {self.active_model_name}")
    
    async def detect_humans(self, image, confidence=None):
        """Enhanced human detection with better accuracy"""
        print(f"🔄 Starting detection with model: {self.active_model_name}")
        
        if not self.models:
            print("❌ No models loaded!")
            raise Exception("No models loaded")
            
        if self.active_model_name not in self.models:
            print(f"❌ Active model {self.active_model_name} not found!")
            raise Exception(f"Active model {self.active_model_name} not available")
            
        conf = confidence or self.confidence_threshold
        model = self.models[self.active_model_name]
        
        print(f"🎯 Using model: {self.active_model_name}, confidence: {conf}")
        
        start_time = time.time()
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        print(f"📊 Image array shape: {img_array.shape}")
        
        # Run detection
        print("🤖 Running YOLO detection...")
        results = model(img_array, conf=conf, classes=[0])  # class 0 = person
        print(f"📋 YOLO results: {len(results)} result(s)")
        
        processing_time = time.time() - start_time
        
        # Extract results
        boxes = []
        confidences = []
        
        if len(results) > 0 and results[0].boxes is not None:
            print(f"📦 Found {len(results[0].boxes)} boxes")
            for i, box in enumerate(results[0].boxes):
                # Get coordinates (x1, y1, x2, y2)
                coords = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                print(f"   Box {i+1}: coords={coords}, conf={confidence}")
                
                boxes.append([
                    float(coords[0]), float(coords[1]), 
                    float(coords[2]), float(coords[3])
                ])
                confidences.append(float(confidence))
        else:
            print("📦 No boxes detected")
        
        result = {
            "boxes": boxes,
            "count": len(boxes),
            "confidences": confidences,
            "model_type": self.active_model_name,
            "processing_time": round(processing_time, 3),
            "confidence_threshold": conf
        }
        
        print(f"✅ Final result: {result}")
        return result
    
    async def detect_realtime_frame(self, frame):
        """Optimized detection for real-time video frames"""
        if not self.models or self.active_model_name not in self.models:
            return {"boxes": [], "count": 0, "confidences": []}
            
        model = self.models[self.active_model_name]
        
        try:
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            scale_factor = 1.0
            
            if width > 640:
                scale_factor = 640 / width
                new_width = 640
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))
                
            # Run detection with lower confidence for real-time
            results = model(frame, conf=0.3, classes=[0], verbose=False)
            
            boxes = []
            confidences = []
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Scale back to original size if resized
                    if scale_factor != 1.0:
                        coords = coords / scale_factor
                    
                    boxes.append([
                        float(coords[0]), float(coords[1]), 
                        float(coords[2]), float(coords[3])
                    ])
                    confidences.append(float(confidence))
            
            return {
                "boxes": boxes,
                "count": len(boxes),
                "confidences": confidences
            }
            
        except Exception as e:
            print(f"❌ Real-time detection error: {e}")
            return {"boxes": [], "count": 0, "confidences": []}
    
    async def get_health_status(self):
        """Get model health status"""
        return {
            "models_loaded": len(self.models) > 0,
            "active_model": self.active_model_name,
            "available_models": list(self.models.keys()),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "models": {
                name: {
                    "loaded": True,
                    "type": "YOLO" if name == "yolo" else "Custom",
                    "status": "OPERATIONAL"
                } for name in self.models.keys()
            }
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




