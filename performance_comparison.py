#!/usr/bin/env python3
"""
Performance Comparison: Standard vs Optimized EasyOCR

This script compares the performance of standard EasyOCR vs optimized parameters.
"""

import os
import cv2
import time
from fix_bidi_import import *
import easyocr


def test_standard_easyocr(image_path):
    """Test standard EasyOCR with default parameters"""
    reader = easyocr.Reader(['en'], gpu=True)
    
    image = cv2.imread(image_path)
    if image is None:
        return {'error': 'Failed to load image'}
    
    start_time = time.time()
    results = reader.readtext(image)
    processing_time = time.time() - start_time
    
    return {
        'method': 'Standard',
        'processing_time': processing_time,
        'num_regions': len(results),
        'results': results
    }


def test_optimized_easyocr(image_path):
    """Test optimized EasyOCR with fast parameters"""
    reader = easyocr.Reader(['en'], gpu=True, quantize=True)
    
    image = cv2.imread(image_path)
    if image is None:
        return {'error': 'Failed to load image'}
    
    # Optimize image size
    height, width = image.shape[:2]
    if max(height, width) > 1280:
        scale = 1280 / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Fast parameters
    fast_params = {
        'width_ths': 0.7,
        'height_ths': 0.7,
        'text_threshold': 0.6,
        'link_threshold': 0.4,
        'canvas_size': 1280,
        'mag_ratio': 1.0,
        'slope_ths': 0.1,
        'ycenter_ths': 0.5,
        'add_margin': 0.1
    }
    
    start_time = time.time()
    results = reader.readtext(image, **fast_params)
    processing_time = time.time() - start_time
    
    return {
        'method': 'Optimized',
        'processing_time': processing_time,
        'num_regions': len(results),
        'results': results
    }


def main():
    """Main comparison function"""
    print("🚀 EasyOCR Performance Comparison")
    print("=" * 50)
    
    # Test images
    test_images = [
        'images/13_2.png',
        'images/30_6.png', 
        'images/34_6.png',
        'images/lhhif3wjhmhzrbw2onof.png'
    ]
    
    # Filter existing images
    test_images = [img for img in test_images if os.path.exists(img)]
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📁 Testing {len(test_images)} images")
    print()
    
    total_standard_time = 0
    total_optimized_time = 0
    
    for image_path in test_images:
        print(f"🔍 Testing: {os.path.basename(image_path)}")
        print("-" * 40)
        
        # Test standard
        standard_result = test_standard_easyocr(image_path)
        if 'error' not in standard_result:
            total_standard_time += standard_result['processing_time']
            print(f"📚 Standard: {standard_result['processing_time']:.3f}s, {standard_result['num_regions']} regions")
            for bbox, text, conf in standard_result['results']:
                print(f"   '{text}' ({conf:.3f})")
        
        # Test optimized
        optimized_result = test_optimized_easyocr(image_path)
        if 'error' not in optimized_result:
            total_optimized_time += optimized_result['processing_time']
            print(f"⚡ Optimized: {optimized_result['processing_time']:.3f}s, {optimized_result['num_regions']} regions")
            for bbox, text, conf in optimized_result['results']:
                print(f"   '{text}' ({conf:.3f})")
        
        print()
    
    # Summary
    print("📊 Performance Summary")
    print("=" * 30)
    print(f"📚 Standard EasyOCR:")
    print(f"   Total time: {total_standard_time:.3f}s")
    print(f"   Average per image: {total_standard_time/len(test_images):.3f}s")
    
    print(f"\n⚡ Optimized EasyOCR:")
    print(f"   Total time: {total_optimized_time:.3f}s")
    print(f"   Average per image: {total_optimized_time/len(test_images):.3f}s")
    
    if total_standard_time > 0:
        speedup = total_standard_time / total_optimized_time
        time_saved = total_standard_time - total_optimized_time
        print(f"\n🎯 Performance Improvement:")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Time saved: {time_saved:.3f}s")
        print(f"   Percentage improvement: {((speedup-1)*100):.1f}%")
    
    print(f"\n💡 Optimization Techniques Used:")
    print(f"   • Reduced canvas size (1280px max)")
    print(f"   • Lower confidence thresholds")
    print(f"   • Less strict detection parameters")
    print(f"   • Image resizing for large images")
    print(f"   • GPU quantization enabled")


if __name__ == "__main__":
    main() 