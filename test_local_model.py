#!/usr/bin/env python3
"""
Test script for Local Model Weights vs EasyOCR
Compares performance between custom trained model and default EasyOCR
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import easyocr

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from custom_ocr_inference import CustomOCRInference
    print("✅ Custom OCR Inference imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure custom_ocr_inference.py is in the same directory")
    sys.exit(1)

def test_model_comparison():
    """Compare custom model vs EasyOCR performance"""
    
    # Configuration
    MODEL_PATH = "checkpoints/best_model_epoch_7.pth"
    CHAR_SET_PATH = "char_set.txt"
    
    print("🔧 Setting up models...")
    
    # Create character set if needed
    if not os.path.exists(CHAR_SET_PATH):
        char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        with open(CHAR_SET_PATH, 'w') as f:
            f.write(char_set)
        print(f"📝 Created character set file: {CHAR_SET_PATH}")
    
    # Initialize custom OCR
    print(f"🤖 Loading custom model from: {MODEL_PATH}")
    custom_ocr = CustomOCRInference(
        model_path=MODEL_PATH,
        char_set_path=CHAR_SET_PATH
    )
    
    # Initialize EasyOCR
    print("📚 Initializing EasyOCR...")
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    print("✅ Both models initialized successfully!")
    
    # Test images
    test_images = []
    
    # Create synthetic test image
    synthetic_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(synthetic_img, "HELLO WORLD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    test_images.append(("Synthetic", synthetic_img))
    
    # Add real images if available
    images_dir = "images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in image_files[:3]:  # Test first 3 images
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                test_images.append((img_file, img))
    
    # Run comparison tests
    print(f"\n🧪 Running comparison tests on {len(test_images)} images...")
    print("=" * 80)
    
    for i, (img_name, img) in enumerate(test_images, 1):
        print(f"\n📸 Test {i}: {img_name}")
        print("-" * 40)
        
        # Test Custom Model
        print("🤖 Custom Model:")
        start_time = time.time()
        custom_result = custom_ocr.recognize_text(img)
        custom_time = time.time() - start_time
        
        print(f"  Text: '{custom_result['text']}'")
        print(f"  Confidence: {custom_result['confidence']:.3f}")
        print(f"  Time: {custom_time:.3f}s")
        print(f"  Success: {custom_result['success']}")
        
        # Test EasyOCR
        print("\n📚 EasyOCR:")
        start_time = time.time()
        easyocr_results = easyocr_reader.readtext(img)
        easyocr_time = time.time() - start_time
        
        print(f"  Found {len(easyocr_results)} text regions")
        if easyocr_results:
            best_result = max(easyocr_results, key=lambda x: x[2])
            print(f"  Best text: '{best_result[1]}'")
            print(f"  Confidence: {best_result[2]:.3f}")
        print(f"  Time: {easyocr_time:.3f}s")
        
        # Performance comparison
        print(f"\n⚡ Performance:")
        if custom_time > 0 and easyocr_time > 0:
            speedup = easyocr_time / custom_time
            print(f"  Custom model is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than EasyOCR")
    
    print("\n" + "=" * 80)
    print("🎉 Model comparison completed!")
    
    # Summary
    print("\n📊 Summary:")
    print("✅ Custom model uses local weights (no internet required)")
    print("✅ EasyOCR uses pre-trained models (requires download)")
    print("✅ Both models can be used for text recognition")
    print("✅ Custom model can be fine-tuned for specific datasets")

def test_model_loading():
    """Test different model weights"""
    
    print("\n🔍 Testing different model weights...")
    
    # Find all available model files
    checkpoint_dir = "checkpoints"
    model_files = []
    
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth') and 'best_model' in file:
                model_files.append(file)
    
    if not model_files:
        print("❌ No model files found in checkpoints directory")
        return
    
    print(f"📁 Found {len(model_files)} model files:")
    for model_file in sorted(model_files):
        file_path = os.path.join(checkpoint_dir, model_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  - {model_file} ({file_size:.1f} MB)")
    
    # Test loading the best model
    best_model = sorted(model_files)[-1]  # Latest epoch
    model_path = os.path.join(checkpoint_dir, best_model)
    
    print(f"\n🧪 Testing model: {best_model}")
    
    try:
        custom_ocr = CustomOCRInference(
            model_path=model_path,
            char_set_path="char_set.txt"
        )
        print("✅ Model loaded successfully!")
        
        # Quick test
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        result = custom_ocr.recognize_text(test_img)
        print(f"✅ Test inference successful: '{result['text']}'")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    print("🚀 Starting Local Model vs EasyOCR Comparison")
    print("=" * 80)
    
    try:
        test_model_loading()
        test_model_comparison()
        
        print("\n🎯 Key Points:")
        print("1. Your custom model uses local weights from checkpoints/")
        print("2. No internet connection required for inference")
        print("3. Model can be fine-tuned for specific use cases")
        print("4. EasyOCR provides pre-trained models with broader language support")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 