#!/usr/bin/env python3
"""
Setup script for EasyOCR Text Detection and Recognition Pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "images",
        "annotated_images", 
        "checkpoints",
        "results",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install PyTorch first
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch==2.1.0", "torchvision==0.16.0",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        print("✅ PyTorch installed")
        
        # Install other dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ All dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test bidi import fix
        import bidi.algorithm
        import bidi
        bidi.get_display = bidi.algorithm.get_display
        print("✅ Bidi import fix applied")
        
        # Test EasyOCR import
        import easyocr
        print("✅ EasyOCR imported successfully")
        
        # Test OCR pipeline
        from ocr_pipeline import OCRPipeline
        pipeline = OCRPipeline(languages=['en'], gpu=False)
        print("✅ OCR Pipeline initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 EasyOCR Text Detection and Recognition Pipeline Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Add images to the 'images/' directory")
    print("   2. Run: python main.py --input images/ --output annotated_images/")
    print("   3. Or use: python ocr_pipeline.py --input images/ --output annotated_images/")
    print("   4. Check the demo notebook: jupyter notebook notebooks/demo.ipynb")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 