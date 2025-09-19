#!/usr/bin/env python3
"""
Setup script for Guard-X Backend
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command):
    """Run shell command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("Guard-X Backend Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"Python {sys.version}")
    
    # Create virtual environment
    print("\n Setting up virtual environment...")
    if not run_command("python -m venv venv"):
        return False
    
    # Activate virtual environment and install packages
    print("\nInstalling dependencies...")
    
    # Windows activation
    if os.name == 'nt':
        activate_cmd = "venv\\Scripts\\activate && "
    else:
        activate_cmd = "source venv/bin/activate && "
    
    if not run_command(f"{activate_cmd}pip install --upgrade pip"):
        return False
    
    if not run_command(f"{activate_cmd}pip install -r requirements.txt"):
        return False
    
    # Create models directory
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir()
        print(" Created models/ directory")
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\n Next steps:")
    print("1. Put your trained model in: models/best.pt")
    print("2. Run server: python run_server.py")
    print("3. Or activate venv and run: python app.py")
    
    return True

if __name__ == "__main__":
    main()