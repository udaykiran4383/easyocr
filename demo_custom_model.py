#!/usr/bin/env python3
"""
Demo: Custom Trained OCR Model

This script demonstrates how to use your custom trained OCR model
and compares it with EasyOCR for basic ASCII text recognition.
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path

# Import our custom OCR
from custom_ocr_inference import CustomOCRInference

# Fix EasyOCR import
from fix_bidi_import import *
import easyocr


def compare_ocr_methods(image_path: str, custom_model_path: str):
    """
    Compare custom OCR model with EasyOCR
    
    Args:
        image_path: Path to test image
        custom_model_path: Path to custom trained model
    """
    print(f"\n🔍 Testing image: {os.path.basename(image_path)}")
    print("=" * 50)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"📏 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Method 1: Custom OCR Model
    print("\n🤖 Custom OCR Model (Trained on IIIT5K):")
    print("-" * 30)
    
    try:
        custom_ocr = CustomOCRInference(model_path=custom_model_path)
        
        start_time = time.time()
        result = custom_ocr.recognize_text(image)
        custom_time = time.time() - start_time
        
        if result['success']:
            print(f"✅ Text: '{result['text']}'")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"⏱️  Time: {custom_time:.3f}s")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Custom OCR error: {e}")
    
    # Method 2: EasyOCR
    print("\n📚 EasyOCR (Pre-trained):")
    print("-" * 30)
    
    try:
        reader = easyocr.Reader(['en'])
        
        start_time = time.time()
        results = reader.readtext(image)
        easyocr_time = time.time() - start_time
        
        if results:
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"✅ Region {i+1}: '{text}'")
                print(f"📊 Confidence: {confidence:.3f}")
                print(f"📍 BBox: {bbox}")
        else:
            print("❌ No text detected")
        
        print(f"⏱️  Time: {easyocr_time:.3f}s")
        
    except Exception as e:
        print(f"❌ EasyOCR error: {e}")


def main():
    """Main demonstration function"""
    print("🚀 Custom OCR Model Demonstration")
    print("=" * 50)
    
    # Paths
    custom_model_path = "checkpoints/best_model_epoch_19.pth"
    images_dir = "images"
    
    # Check if custom model exists
    if not os.path.exists(custom_model_path):
        print(f"❌ Custom model not found: {custom_model_path}")
        print("Please train the model first using: python train_custom_model.py")
        return
    
    print(f"✅ Custom model found: {custom_model_path}")
    
    # Get test images
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ No test images found in {images_dir}")
        return
    
    print(f"📁 Found {len(image_files)} test images")
    
    # Test on first few images
    for i, image_file in enumerate(image_files[:5]):  # Test first 5 images
        image_path = os.path.join(images_dir, image_file)
        compare_ocr_methods(image_path, custom_model_path)
        
        if i < len(image_files[:5]) - 1:
            print("\n" + "="*60 + "\n")
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 30)
    print("🎯 Custom Model (20 epochs):")
    print("   • Trained on IIIT5K dataset (basic ASCII)")
    print("   • 20 epochs training completed")
    print("   • Loss improved from 0.47 to 0.26 (44% better)")
    print("   • Fast inference for specific use cases")
    print("   • Can be further trained on your own data")
    
    print("\n📚 EasyOCR:")
    print("   • Pre-trained on diverse datasets")
    print("   • Handles complex fonts and layouts")
    print("   • Multi-language support")
    print("   • More robust for general use")
    
    print("\n💡 Recommendation:")
    print("   • Use custom model for specific, consistent text types")
    print("   • Use EasyOCR for general-purpose text recognition")
    print("   • Consider hybrid approach: EasyOCR detection + custom recognition")


if __name__ == "__main__":
    main() 