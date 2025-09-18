from ultralytics import YOLO
import cv2
import numpy as np

class YOLOHumanDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        
    def detect_humans(self, image_path):
        # Run inference
        results = self.model(image_path)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                # Check if detected class is 'person' (class 0 in COCO)
                if int(box.cls) == 0:  # Person class
                    confidence = float(box.conf)
                    if confidence > 0.5:  # Confidence threshold
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return boxes