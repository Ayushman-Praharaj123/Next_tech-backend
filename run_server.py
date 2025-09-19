#!/usr/bin/env python3
"""
Guard-X AI Surveillance Server
Run this file to start the FastAPI server
"""

import uvicorn
import os
from pathlib import Path

def main():
    print("Guard-X AI Surveillance System")
    print("=" * 50)
    
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print("Created models/ directory")
    
    # Check for custom model
    custom_model = models_dir / "best.pt"
    if custom_model.exists():
        print(f"Custom model found: {custom_model}")
    else:
        print(" Custom model not found, will use YOLO fallback")
        print(f" Place your trained model at: {custom_model}")
    
    print("\nStarting FastAPI server...")
    print(" Backend URL: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print(" Health Check: http://localhost:8000/api/health")
    print("\n" + "=" * 50)
    
    # Start server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()