from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import json
import base64
import numpy as np
from auth import get_current_user
from model_wrapper import ModelWrapper

router = APIRouter()

class CameraManager:
    def __init__(self, model_wrapper: ModelWrapper):
        self.model_wrapper = model_wrapper
        self.active_connections = []
        self.camera = None
        self.is_streaming = False
        self.gps_location = None
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📱 WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"📱 WebSocket disconnected. Total connections: {len(self.active_connections)}")
            
    async def start_camera(self, camera_id=0):
        """Start camera capture"""
        try:
            print(f"📹 Starting camera {camera_id}...")
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                print("❌ Camera failed to open")
                return False
                
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.is_streaming = True
            print("✅ Camera started successfully")
            return True
        except Exception as e:
            print(f"❌ Camera start failed: {e}")
            return False
            
    async def stop_camera(self):
        """Stop camera capture"""
        print("🛑 Stopping camera...")
        self.is_streaming = False
        if self.camera:
            self.camera.release()
            self.camera = None
        print("✅ Camera stopped")
            
    async def stream_detection(self):
        """Stream camera with real-time detection"""
        print("🎥 Starting detection stream...")
        frame_count = 0
        
        while self.is_streaming and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Run detection every 3rd frame for performance
                if frame_count % 3 == 0:
                    detection_result = await self.model_wrapper.detect_realtime_frame(frame)
                else:
                    detection_result = {"boxes": [], "count": 0, "confidences": []}
                
                # Draw bounding boxes
                annotated_frame = self.draw_detections(frame, detection_result)
                
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to all connected clients
                message = {
                    "type": "detection_frame",
                    "frame": frame_base64,
                    "detections": detection_result,
                    "timestamp": asyncio.get_event_loop().time(),
                    "gps_location": self.gps_location
                }
                
                # Send to all connections
                disconnected = []
                for connection in self.active_connections:
                    try:
                        await connection.send_text(json.dumps(message))
                    except Exception as e:
                        print(f"❌ Failed to send to client: {e}")
                        disconnected.append(connection)
                
                # Remove disconnected clients
                for conn in disconnected:
                    self.disconnect(conn)
                    
                await asyncio.sleep(0.1)  # ~10 FPS
                
            except Exception as e:
                print(f"❌ Streaming error: {e}")
                break
                
        print("🛑 Detection stream ended")
                
    def draw_detections(self, frame, detection_result):
        """Draw bounding boxes on frame"""
        annotated_frame = frame.copy()
        
        for i, box in enumerate(detection_result["boxes"]):
            x1, y1, x2, y2 = map(int, box)
            confidence = detection_result["confidences"][i] if i < len(detection_result["confidences"]) else 0.0
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"HUMAN {i+1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1+label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(annotated_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # Add detection count
        count_text = f"THREATS DETECTED: {detection_result['count']}"
        cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add GPS info if available
        if self.gps_location:
            gps_text = f"GPS: {self.gps_location['latitude']:.4f}, {self.gps_location['longitude']:.4f}"
            cv2.putText(annotated_frame, gps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated_frame

# Global camera manager
camera_manager = None

def get_camera_manager():
    global camera_manager
    if not camera_manager:
        from app import model_wrapper
        camera_manager = CameraManager(model_wrapper)
    return camera_manager

@router.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """WebSocket endpoint for real-time camera detection"""
    print("🔌 New WebSocket connection attempt...")
    manager = get_camera_manager()
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(f"📨 Received message: {message}")
            
            if message["type"] == "start_camera":
                camera_id = message.get("camera_id", 0)
                gps_location = message.get("gps_location")
                
                # Store GPS location for this session
                if gps_location:
                    manager.gps_location = gps_location
                    print(f"📍 GPS Location set: {gps_location}")
                
                success = await manager.start_camera(camera_id)
                
                if success:
                    await websocket.send_text(json.dumps({
                        "type": "camera_started",
                        "status": "success",
                        "gps_enabled": gps_location is not None
                    }))
                    # Start streaming in background
                    asyncio.create_task(manager.stream_detection())
                else:
                    await websocket.send_text(json.dumps({
                        "type": "camera_error",
                        "message": "Failed to start camera"
                    }))
                    
            elif message["type"] == "stop_camera":
                await manager.stop_camera()
                await websocket.send_text(json.dumps({
                    "type": "camera_stopped",
                    "status": "success"
                }))
                
    except WebSocketDisconnect:
        print("📱 WebSocket disconnected")
        manager.disconnect(websocket)
        if len(manager.active_connections) == 0:
            await manager.stop_camera()
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

@router.get("/status")
async def camera_status():
    """Get camera status - NO AUTH REQUIRED FOR TESTING"""
    manager = get_camera_manager()
    return {
        "is_streaming": manager.is_streaming,
        "active_connections": len(manager.active_connections),
        "camera_available": manager.camera is not None,
        "status": "operational"
    }

