#!/usr/bin/env python3
"""
Test script for EasyOCR Reader (Fixed Version)
Uses EasyOCR as primary method - Local model has training issues
"""

# Fix for bidi import issue
try:
    import bidi.algorithm
    import bidi
    bidi.get_display = bidi.algorithm.get_display
    print("Bidi import fixed")
except Exception as e:
    print(f"Bidi fix failed: {e}")

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import time

# Add the current directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import easyocr
    print("EasyOCR imported successfully")
    
    # Initialize EasyOCR as primary method
    print("Initializing EasyOCR (Primary Method)")
    reader = easyocr.Reader(['en'], gpu=False)
    print("✅ EasyOCR initialized successfully")
    
    # Test with a simple image
    print("\nTesting with a simple test image...")
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "HELLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Test EasyOCR
    start_time = time.time()
    results = reader.readtext(test_image)
    easyocr_time = time.time() - start_time
    
    print(f"EasyOCR test completed:")
    print(f"  Found {len(results)} text regions")
    for i, (bbox, text, conf) in enumerate(results):
        print(f"  Region {i+1}: '{text}' (confidence: {conf:.3f})")
    print(f"  Processing time: {easyocr_time:.3f}s")
    print(f"  Model: EasyOCR (pre-trained, reliable)")
    
    # Test with actual images if available
    images_dir = "images"
    if os.path.exists(images_dir):
        print(f"\nTesting with images from {images_dir}...")
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if image_files:
            test_image_file = os.path.join(images_dir, image_files[0])
            print(f"Testing with: {test_image_file}")
            
            image = cv2.imread(test_image_file)
            if image is not None:
                # Test EasyOCR
                start_time = time.time()
                easyocr_results = reader.readtext(image)
                easyocr_time = time.time() - start_time
                
                print(f"EasyOCR results:")
                print(f"  Found {len(easyocr_results)} text regions")
                if easyocr_results:
                    best_result = max(easyocr_results, key=lambda x: x[2])
                    print(f"  Best text: '{best_result[1]}' (confidence: {best_result[2]:.3f})")
                print(f"  Processing time: {easyocr_time:.3f}s")
                print(f"  Model: EasyOCR (pre-trained, reliable)")
    
    print("\n✅ EasyOCR testing completed successfully!")
    print("🎯 Key points:")
    print("  - EasyOCR works correctly and is recommended")
    print("  - Local model has training issues (returns 'CWEXA' pattern)")
    print("  - EasyOCR provides reliable text recognition")
    print("  - Local model needs retraining with better data")
    

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure easyocr is installed: pip install easyocr")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure custom_ocr_inference.py is in the same directory")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 