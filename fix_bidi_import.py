#!/usr/bin/env python3
"""
Fix for bidi import issue in EasyOCR
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch the bidi import before importing EasyOCR
try:
    import bidi.algorithm
    # Create a get_display function in the bidi module
    import bidi
    bidi.get_display = bidi.algorithm.get_display
    print("Successfully patched bidi import")
except Exception as e:
    print(f"Error patching bidi import: {e}")

# Now import and test EasyOCR
try:
    import easyocr
    print("EasyOCR imported successfully!")
    
    # Test initialization
    reader = easyocr.Reader(['en'])
    print("EasyOCR reader initialized successfully!")
    
except Exception as e:
    print(f"Error importing EasyOCR: {e}") 