#!/usr/bin/env python3
"""
Create POC Package Script
Packages the EasyOCR POC into a ZIP file for easy distribution.
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_poc_package():
    """Create a ZIP package of the POC"""
    
    # POC files to include
    poc_files = [
        'simple_poc_demo.py',      # Simple working version
        'poc_easyocr_demo.py',     # Full version
        'poc_requirements.txt',
        'POC_README.md'
    ]
    
    # Directories to include
    poc_dirs = [
        'checkpoints',
        'images'
    ]
    
    # Create POC directory
    poc_dir = 'easyocr_poc'
    if os.path.exists(poc_dir):
        shutil.rmtree(poc_dir)
    os.makedirs(poc_dir)
    
    print(f"📁 Creating POC package in: {poc_dir}")
    
    # Copy POC files
    for file in poc_files:
        if os.path.exists(file):
            shutil.copy2(file, poc_dir)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Copy directories
    for dir_name in poc_dirs:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(poc_dir, dir_name))
            print(f"✅ Copied directory: {dir_name}")
        else:
            print(f"⚠️  Directory not found: {dir_name}")
    
    # Create ZIP file
    zip_filename = 'easyocr_poc_package.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(poc_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, poc_dir)
                zipf.write(file_path, arcname)
                print(f"📦 Added to ZIP: {arcname}")
    
    print(f"\n🎉 POC package created: {zip_filename}")
    print(f"📁 Package size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    
    # List package contents
    print(f"\n📋 Package contents:")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for file in zipf.namelist():
            print(f"  📄 {file}")
    
    return zip_filename

def main():
    """Main function"""
    print("🚀 Creating EasyOCR POC Package")
    print("=" * 50)
    
    try:
        zip_file = create_poc_package()
        print(f"\n✅ Success! POC package ready: {zip_file}")
        print("\n📝 Instructions for your manager:")
        print("1. Extract the ZIP file")
        print("2. Install dependencies: pip install -r poc_requirements.txt")
        print("3. Run the simple demo: python simple_poc_demo.py")
        print("4. Or run the full demo: python poc_easyocr_demo.py")
        print("\n💡 Recommended: Start with simple_poc_demo.py for quick testing")
        
    except Exception as e:
        print(f"❌ Error creating package: {e}")

if __name__ == "__main__":
    main() 