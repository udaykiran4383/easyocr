#!/usr/bin/env python3
"""
Helper script to prepare files for cloud upload
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_cloud_package():
    """Create a zip file with all necessary files for cloud training"""
    
    # Files to include
    files_to_include = [
        'train_cloud_gpu.py',
        'cloud_requirements.txt',
        'CLOUD_SETUP_GUIDE.md'
    ]
    
    # Create cloud package directory
    cloud_dir = Path('cloud_package')
    cloud_dir.mkdir(exist_ok=True)
    
    # Copy files
    for file in files_to_include:
        if os.path.exists(file):
            shutil.copy2(file, cloud_dir / file)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️  Warning: {file} not found")
    
    # Create zip file
    zip_name = 'ocr_cloud_training.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in cloud_dir.glob('*'):
            zipf.write(file, file.name)
    
    print(f"\n🎉 Cloud package created: {zip_name}")
    print(f"📁 Package size: {os.path.getsize(zip_name) / 1024:.1f} KB")
    print("\n📋 Files included:")
    for file in files_to_include:
        if os.path.exists(file):
            print(f"   - {file}")
    
    print(f"\n🚀 Ready to upload to cloud platform!")
    print(f"💡 Upload {zip_name} to your chosen cloud platform")

def print_cloud_instructions():
    """Print quick instructions for different cloud platforms"""
    
    print("\n" + "="*60)
    print("🌩️  CLOUD TRAINING INSTRUCTIONS")
    print("="*60)
    
    print("\n📱 Google Colab (Recommended for beginners):")
    print("1. Go to colab.research.google.com")
    print("2. Create new notebook")
    print("3. Upload train_cloud_gpu.py and cloud_requirements.txt")
    print("4. Run: !pip install -r cloud_requirements.txt")
    print("5. Run: !python train_cloud_gpu.py")
    print("6. Download results when complete")
    
    print("\n☁️  AWS EC2:")
    print("1. Launch GPU instance (g4dn.xlarge)")
    print("2. SSH into instance")
    print("3. Install dependencies: pip install -r cloud_requirements.txt")
    print("4. Run: python3 train_cloud_gpu.py")
    
    print("\n🔧 Google Cloud Platform:")
    print("1. Create VM with GPU")
    print("2. Install NVIDIA drivers")
    print("3. Install dependencies and run training")
    
    print("\n💻 Paperspace Gradient:")
    print("1. Create notebook with GPU")
    print("2. Upload files and run training")
    
    print("\n💰 Cost Estimates (1 hour training):")
    print("- Google Colab: FREE")
    print("- AWS EC2 (T4): ~$0.50")
    print("- GCP (T4): ~$0.35")
    print("- Paperspace: ~$0.60")

if __name__ == "__main__":
    print("🚀 Preparing OCR Cloud Training Package")
    print("="*50)
    
    create_cloud_package()
    print_cloud_instructions()
    
    print(f"\n✅ Ready to train your OCR model in the cloud!")
    print(f"💡 Choose a platform above and follow the instructions") 