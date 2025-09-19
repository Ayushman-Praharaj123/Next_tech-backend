from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime, timedelta
from pathlib import Path
import io
from PIL import Image

# Import modules
from model_wrapper import ModelWrapper
from auth import (
    authenticate_army_user, create_access_token, get_current_user,
    require_admin_access, require_clearance_level, UserLogin, Token,
    ACCESS_TOKEN_EXPIRE_MINUTES, initialize_army_auth_system
)
from camera_detection import router as camera_router

app = FastAPI(
    title="Guard-X Military Surveillance API",
    description="🎖️ CLASSIFIED - Army AI Surveillance System",
    version="2.0.0-MILITARY"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include camera router - FIX THIS
app.include_router(camera_router, prefix="/camera", tags=["camera"])

# Initialize systems
model_wrapper = ModelWrapper()
initialize_army_auth_system()

@app.on_event("startup")
async def startup_event():
    """Initialize military systems on startup"""
    print("🎖️  GUARD-X MILITARY SYSTEM INITIALIZING...")
    await model_wrapper.load_models()
    print("✅ GUARD-X SYSTEM OPERATIONAL")

# MILITARY AUTH ENDPOINTS
@app.post("/api/auth/login", response_model=Token)
async def military_login(user_credentials: UserLogin):
    """Military personnel authentication"""
    user = authenticate_army_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="UNAUTHORIZED - INVALID MILITARY CREDENTIALS"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"]
        },
        "clearance_level": user["clearance_level"],
        "unit": user["unit"]
    }

@app.get("/api/auth/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current military user info"""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "clearance_level": current_user["clearance_level"],
        "unit": current_user["unit"],
        "system_access": "AUTHORIZED"
    }

# CLASSIFIED DETECTION ENDPOINT
@app.post("/api/detect")
async def military_threat_detection(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    current_user = Depends(require_clearance_level("SECRET"))
):
    """🔒 CLASSIFIED - Military threat detection endpoint"""
    try:
        print(f"🔄 DETECTION REQUEST from {current_user['username']}")
        print(f"📁 File: {file.filename}, Type: {file.content_type}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="INVALID FILE TYPE - IMAGE REQUIRED")
        
        # Read and process image
        image_bytes = await file.read()
        print(f"📊 Image size: {len(image_bytes)} bytes")
        
        image = Image.open(io.BytesIO(image_bytes))
        print(f"🖼️  Image dimensions: {image.width}x{image.height}, Mode: {image.mode}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("🔄 Converted image to RGB")
        
        # Run military-grade detection
        print("🤖 Running AI detection...")
        detection_result = await model_wrapper.detect_humans(image, confidence)
        print(f"✅ Detection complete: {detection_result}")
        
        # Classify threat level
        threat_level = "CRITICAL" if detection_result["count"] > 3 else \
                     "HIGH" if detection_result["count"] > 1 else \
                     "MEDIUM" if detection_result["count"] > 0 else "LOW"
        
        # FIXED: Return both old and new format for compatibility
        response = {
            # New military format
            "classification": "RESTRICTED",
            "operation_id": f"GUARD-X-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "operator": current_user["username"],
            "unit": current_user["unit"],
            "clearance": current_user["clearance_level"],
            "detection": {
                "targets_identified": detection_result["count"],
                "confidence_scores": detection_result["confidences"],
                "bounding_boxes": detection_result["boxes"],
                "threat_assessment": threat_level,
                "model_used": detection_result["model_type"],
                "processing_time": detection_result["processing_time"],
                "confidence_threshold": detection_result["confidence_threshold"]
            },
            "image_metadata": {
                "filename": file.filename,
                "dimensions": f"{image.width}x{image.height}",
                "format": image.mode
            },
            "timestamp": datetime.now().isoformat(),
            "status": "MISSION_COMPLETE" if detection_result["count"] == 0 else "THREATS_DETECTED",
            
            # OLD FORMAT FOR COMPATIBILITY (Frontend expects this)
            "success": True,
            "boxes": detection_result["boxes"],
            "count": detection_result["count"],
            "confidence_scores": detection_result["confidences"],
            "model_used": detection_result["model_type"],
            "processing_time": detection_result["processing_time"],
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ MILITARY DETECTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"SYSTEM FAILURE: {str(e)}")

# ADMIN ONLY ENDPOINTS
@app.get("/api/admin/system-status")
async def admin_system_status(current_user = Depends(require_admin_access)):
    """🎖️ ADMIN ONLY - Complete system status"""
    health_status = await model_wrapper.get_health_status()
    
    return {
        "classification": "TOP_SECRET",
        "system_name": "GUARD-X MILITARY SURVEILLANCE",
        "version": "2.0.0-MILITARY",
        "status": "OPERATIONAL" if health_status.get("models_loaded") else "DEGRADED",
        "admin": current_user["username"],
        "timestamp": datetime.now().isoformat(),
        "capabilities": {
            "image_threat_detection": True,
            "realtime_surveillance": True,
            "military_authentication": True,
            "custom_ai_model": "best.pt" in str(Path("models/best.pt")) and Path("models/best.pt").exists(),
            "clearance_levels": ["SECRET", "TOP_SECRET"],
            "active_units": ["CYBER_WARFARE_DIVISION", "SURVEILLANCE_OPERATIONS"]
        },
        "models": health_status.get("models", {}),
        "security_status": "MAXIMUM"
    }

@app.get("/api/camera/test")
async def test_camera():
    """Test camera availability"""
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
            camera.release()
            return {"status": "success", "camera_available": ret}
        else:
            return {"status": "error", "camera_available": False, "message": "Camera not accessible"}
    except Exception as e:
        return {"status": "error", "camera_available": False, "message": str(e)}

@app.get("/api/health")
async def health_check():
    """System health check"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_wrapper.models) > 0,
        "active_model": model_wrapper.active_model_name
    }

if __name__ == "__main__":
    print("🎖️  STARTING GUARD-X MILITARY SERVER v2.0...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)









