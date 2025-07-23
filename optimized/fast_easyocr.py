#!/usr/bin/env python3
"""
Fast EasyOCR Implementation

Optimized EasyOCR for faster detection with reduced parameters.
"""

import os
import cv2
import time
import logging
from pathlib import Path

# Fix EasyOCR import
from fix_bidi_import import *
import easyocr

logger = logging.getLogger(__name__)


class FastEasyOCR:
    """Fast EasyOCR implementation with optimizations"""
    
    def __init__(self, languages=['en'], gpu=True):
        """Initialize Fast EasyOCR"""
        self.languages = languages
        self.gpu = gpu
        
        # Fast detection parameters
        self.fast_params = {
            'width_ths': 0.7,      # Lower width threshold
            'height_ths': 0.7,     # Lower height threshold  
            'text_threshold': 0.6, # Lower text confidence
            'link_threshold': 0.4, # Lower link confidence
            'canvas_size': 1280,   # Smaller canvas (faster)
            'mag_ratio': 1.0,      # No magnification
            'slope_ths': 0.1,      # Less strict slope
            'ycenter_ths': 0.5,    # Less strict center
            'add_margin': 0.1      # Smaller margin
        }
        
        # Initialize reader
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            quantize=True  # Enable quantization for speed
        )
        
        print(f"✅ Fast EasyOCR initialized (GPU: {self.gpu})")
    
    def detect_fast(self, image_path, confidence_threshold=0.5):
        """Fast text detection"""
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        # Optimize image size if too large
        height, width = image.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"📏 Resized from {width}x{height} to {new_width}x{new_height}")
        
        # Fast detection
        results = self.reader.readtext(image, **self.fast_params)
        
        # Filter by confidence
        filtered_results = []
        for bbox, text, confidence in results:
            if confidence >= confidence_threshold:
                filtered_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'results': filtered_results,
            'processing_time': processing_time,
            'num_regions': len(filtered_results)
        }
    
    def process_directory(self, input_dir, output_dir=None):
        """Process all images in directory"""
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"📁 Processing {len(image_files)} images...")
        
        total_time = 0
        total_regions = 0
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(input_dir, image_file)
            result = self.detect_fast(image_path)
            
            if result['success']:
                total_time += result['processing_time']
                total_regions += result['num_regions']
                
                print(f"✅ {image_file}: {result['num_regions']} regions, {result['processing_time']:.3f}s")
                for r in result['results']:
                    print(f"   '{r['text']}' ({r['confidence']:.3f})")
            else:
                print(f"❌ {image_file}: {result.get('error', 'Failed')}")
        
        avg_time = total_time / len(image_files) if image_files else 0
        print(f"\n📊 Summary:")
        print(f"   Total images: {len(image_files)}")
        print(f"   Total regions: {total_regions}")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average time per image: {avg_time:.3f}s")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast EasyOCR')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    args = parser.parse_args()
    
    # Initialize Fast EasyOCR
    fast_ocr = FastEasyOCR(gpu=args.gpu)
    
    # Process input
    if os.path.isfile(args.input):
        result = fast_ocr.detect_fast(args.input, args.confidence)
        if result['success']:
            print(f"✅ Detected {result['num_regions']} regions in {result['processing_time']:.3f}s")
            for r in result['results']:
                print(f"   '{r['text']}' ({r['confidence']:.3f})")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
    
    elif os.path.isdir(args.input):
        fast_ocr.process_directory(args.input)
    
    else:
        print(f"❌ Input does not exist: {args.input}")


if __name__ == "__main__":
    main() 