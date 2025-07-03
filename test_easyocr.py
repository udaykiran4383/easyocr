#!/usr/bin/env python3
"""
Test script for EasyOCR Reader
"""

# Fix for bidi import issue
try:
    import bidi.algorithm
    import bidi
    bidi.get_display = bidi.algorithm.get_display
    print("Bidi import fixed")
except Exception as e:
    print(f"Bidi fix failed: {e}")

try:
    import easyocr
    print("EasyOCR imported successfully")
    
    # Test basic initialization
    reader = easyocr.Reader(['en'], gpu=False)
    print("Basic Reader initialized")
    
    # Test with minimal parameters
    reader2 = easyocr.Reader(['en'])
    print("Minimal Reader initialized")
    
    # Test readtext on a simple case
    import numpy as np
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    results = reader.readtext(test_image)
    print(f"Test readtext completed, found {len(results)} text regions")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 