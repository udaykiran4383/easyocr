#!/usr/bin/env python3
"""
Generate CSV Comparison: Custom Model vs EasyOCR

This script creates a comprehensive CSV comparison between your custom trained model
and EasyOCR with detailed performance metrics.
"""

import os
import cv2
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import our custom OCR
from custom_ocr_inference import CustomOCRInference

# Fix EasyOCR import
from fix_bidi_import import *
import easyocr


def test_custom_model(image_path, model_path):
    """Test custom trained model"""
    try:
        custom_ocr = CustomOCRInference(model_path=model_path)
        
        start_time = time.time()
        result = custom_ocr.recognize_text(cv2.imread(image_path))
        custom_time = time.time() - start_time
        
        return {
            'text': result['text'],
            'confidence': result['confidence'],
            'processing_time': custom_time,
            'success': result['success']
        }
    except Exception as e:
        return {
            'text': 'ERROR',
            'confidence': 0.0,
            'processing_time': 0.0,
            'success': False,
            'error': str(e)
        }


def test_easyocr_standard(image_path):
    """Test standard EasyOCR"""
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        
        start_time = time.time()
        results = reader.readtext(cv2.imread(image_path))
        easyocr_time = time.time() - start_time
        
        if results:
            # Get the first result
            bbox, text, confidence = results[0]
            return {
                'text': text,
                'confidence': confidence,
                'processing_time': easyocr_time,
                'success': True,
                'num_regions': len(results)
            }
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'processing_time': easyocr_time,
                'success': True,
                'num_regions': 0
            }
    except Exception as e:
        return {
            'text': 'ERROR',
            'confidence': 0.0,
            'processing_time': 0.0,
            'success': False,
            'error': str(e),
            'num_regions': 0
        }


def test_easyocr_optimized(image_path):
    """Test optimized EasyOCR"""
    try:
        reader = easyocr.Reader(['en'], gpu=True, quantize=True)
        
        # Optimize image
        image = cv2.imread(image_path)
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
        easyocr_time = time.time() - start_time
        
        if results:
            bbox, text, confidence = results[0]
            return {
                'text': text,
                'confidence': confidence,
                'processing_time': easyocr_time,
                'success': True,
                'num_regions': len(results)
            }
        else:
            return {
                'text': '',
                'confidence': 0.0,
                'processing_time': easyocr_time,
                'success': True,
                'num_regions': 0
            }
    except Exception as e:
        return {
            'text': 'ERROR',
            'confidence': 0.0,
            'processing_time': 0.0,
            'success': False,
            'error': str(e),
            'num_regions': 0
        }


def generate_comparison_csv():
    """Generate comprehensive CSV comparison"""
    print("🚀 Generating CSV Comparison: Custom Model vs EasyOCR")
    print("=" * 60)
    
    # Model paths
    custom_model_path = "checkpoints/best_model_epoch_19.pth"
    
    if not os.path.exists(custom_model_path):
        print(f"❌ Custom model not found: {custom_model_path}")
        return
    
    print(f"✅ Using improved custom model: {custom_model_path}")
    
    # Test images
    test_images = [
        'images/13_2.png',
        'images/27_1.png',
        'images/30_6.png',
        'images/32_2.png',
        'images/34_6.png',
        'images/34_13.png',
        'images/34_20.png',
        'images/37_6.png',
        'images/39_1.png',
        'images/6_7.png',
        'images/lhhif3wjhmhzrbw2onof.png',
        'images/sacutvgklqpwpmghr2or.png',
        'images/Sample-handwritten-text-input-for-OCR.png',
        'images/tz6eptuq2vgo3sivx7fi.png'
    ]
    
    # Filter existing images
    test_images = [img for img in test_images if os.path.exists(img)]
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📁 Testing {len(test_images)} images")
    print()
    
    # Results storage
    comparison_data = []
    
    for i, image_path in enumerate(test_images):
        print(f"Processing {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        
        # Get image info
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        image_size = height * width
        
        # Test all methods
        custom_result = test_custom_model(image_path, custom_model_path)
        easyocr_std_result = test_easyocr_standard(image_path)
        easyocr_opt_result = test_easyocr_optimized(image_path)
        
        # Create row data
        row_data = {
            'image_name': os.path.basename(image_path),
            'image_width': width,
            'image_height': height,
            'image_size_pixels': image_size,
            'image_size_category': 'Small' if image_size < 10000 else 'Medium' if image_size < 100000 else 'Large',
            
            # Custom Model Results
            'custom_text': custom_result['text'],
            'custom_confidence': custom_result['confidence'],
            'custom_time_seconds': custom_result['processing_time'],
            'custom_success': custom_result['success'],
            
            # EasyOCR Standard Results
            'easyocr_std_text': easyocr_std_result['text'],
            'easyocr_std_confidence': easyocr_std_result['confidence'],
            'easyocr_std_time_seconds': easyocr_std_result['processing_time'],
            'easyocr_std_success': easyocr_std_result['success'],
            'easyocr_std_regions': easyocr_std_result.get('num_regions', 0),
            
            # EasyOCR Optimized Results
            'easyocr_opt_text': easyocr_opt_result['text'],
            'easyocr_opt_confidence': easyocr_opt_result['confidence'],
            'easyocr_opt_time_seconds': easyocr_opt_result['processing_time'],
            'easyocr_opt_success': easyocr_opt_result['success'],
            'easyocr_opt_regions': easyocr_opt_result.get('num_regions', 0),
            
            # Performance Comparisons
            'speedup_vs_easyocr_std': easyocr_std_result['processing_time'] / custom_result['processing_time'] if custom_result['processing_time'] > 0 else 0,
            'speedup_vs_easyocr_opt': easyocr_opt_result['processing_time'] / custom_result['processing_time'] if custom_result['processing_time'] > 0 else 0,
            'easyocr_opt_vs_std_speedup': easyocr_std_result['processing_time'] / easyocr_opt_result['processing_time'] if easyocr_opt_result['processing_time'] > 0 else 0,
        }
        
        comparison_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Add summary statistics
    summary_stats = {
        'metric': [
            'Total Images',
            'Custom Model - Avg Time (s)',
            'Custom Model - Avg Confidence',
            'Custom Model - Success Rate (%)',
            'EasyOCR Standard - Avg Time (s)',
            'EasyOCR Standard - Avg Confidence',
            'EasyOCR Standard - Success Rate (%)',
            'EasyOCR Optimized - Avg Time (s)',
            'EasyOCR Optimized - Avg Confidence',
            'EasyOCR Optimized - Success Rate (%)',
            'Custom vs EasyOCR Std - Avg Speedup',
            'Custom vs EasyOCR Opt - Avg Speedup',
            'EasyOCR Opt vs Std - Avg Speedup'
        ],
        'value': [
            len(test_images),
            df['custom_time_seconds'].mean(),
            df['custom_confidence'].mean(),
            (df['custom_success'].sum() / len(df)) * 100,
            df['easyocr_std_time_seconds'].mean(),
            df['easyocr_std_confidence'].mean(),
            (df['easyocr_std_success'].sum() / len(df)) * 100,
            df['easyocr_opt_time_seconds'].mean(),
            df['easyocr_opt_confidence'].mean(),
            (df['easyocr_opt_success'].sum() / len(df)) * 100,
            df['speedup_vs_easyocr_std'].mean(),
            df['speedup_vs_easyocr_opt'].mean(),
            df['easyocr_opt_vs_std_speedup'].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed comparison CSV
    detailed_csv_path = f"ocr_comparison_detailed_{timestamp}.csv"
    df.to_csv(detailed_csv_path, index=False)
    
    # Summary statistics CSV
    summary_csv_path = f"ocr_comparison_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Performance comparison CSV
    performance_data = []
    for _, row in df.iterrows():
        performance_data.append({
            'image_name': row['image_name'],
            'image_size_category': row['image_size_category'],
            'custom_time_ms': row['custom_time_seconds'] * 1000,
            'easyocr_std_time_ms': row['easyocr_std_time_seconds'] * 1000,
            'easyocr_opt_time_ms': row['easyocr_opt_time_seconds'] * 1000,
            'custom_confidence': row['custom_confidence'],
            'easyocr_std_confidence': row['easyocr_std_confidence'],
            'easyocr_opt_confidence': row['easyocr_opt_confidence'],
            'custom_speedup_vs_std': row['speedup_vs_easyocr_std'],
            'custom_speedup_vs_opt': row['speedup_vs_easyocr_opt']
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_csv_path = f"ocr_performance_comparison_{timestamp}.csv"
    performance_df.to_csv(performance_csv_path, index=False)
    
    print(f"\n✅ CSV files generated successfully!")
    print(f"📊 Detailed comparison: {detailed_csv_path}")
    print(f"📈 Summary statistics: {summary_csv_path}")
    print(f"⚡ Performance comparison: {performance_csv_path}")
    
    # Print summary
    print(f"\n📋 Summary:")
    print(f"   Custom Model: {df['custom_time_seconds'].mean():.3f}s avg, {df['custom_confidence'].mean():.3f} confidence")
    print(f"   EasyOCR Std: {df['easyocr_std_time_seconds'].mean():.3f}s avg, {df['easyocr_std_confidence'].mean():.3f} confidence")
    print(f"   EasyOCR Opt: {df['easyocr_opt_time_seconds'].mean():.3f}s avg, {df['easyocr_opt_confidence'].mean():.3f} confidence")
    print(f"   Custom vs EasyOCR Std: {df['speedup_vs_easyocr_std'].mean():.1f}x faster")
    print(f"   Custom vs EasyOCR Opt: {df['speedup_vs_easyocr_opt'].mean():.1f}x faster")
    
    return detailed_csv_path, summary_csv_path, performance_csv_path


if __name__ == "__main__":
    generate_comparison_csv() 