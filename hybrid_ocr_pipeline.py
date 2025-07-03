#!/usr/bin/env python3
"""
Hybrid OCR Pipeline

This script combines EasyOCR for text detection with custom trained model for recognition.
It provides the best of both worlds: EasyOCR's robust detection and custom model's
specialized recognition for your specific dataset.
"""

import os
import cv2
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

# Fix for bidi import issue in EasyOCR
try:
    import bidi.algorithm
    import bidi
    bidi.get_display = bidi.algorithm.get_display
except Exception as e:
    logging.warning(f"Could not patch bidi import: {e}")

import easyocr
import torch

from custom_ocr_inference import CustomOCRInference
from config import OCR_CONFIG, IMAGE_CONFIG, OUTPUT_CONFIG, PATHS
from utils import (
    setup_logging, get_supported_image_files, load_image, save_image,
    preprocess_image, draw_bounding_boxes, calculate_processing_metrics,
    format_bbox_for_csv, save_results_to_csv, save_results_to_json,
    create_summary_report, validate_image_path, create_output_filename,
    check_gpu_availability, get_memory_usage, create_progress_bar, update_progress
)

logger = setup_logging()


class HybridOCRPipeline:
    """
    Hybrid OCR Pipeline combining EasyOCR detection with custom recognition
    """
    
    def __init__(self, custom_model_path: str, languages: List[str] = None, 
                 gpu: bool = None, config: Dict = None):
        """
        Initialize the hybrid OCR pipeline
        
        Args:
            custom_model_path: Path to custom trained model
            languages: List of language codes for EasyOCR detection
            gpu: Whether to use GPU acceleration
            config: Configuration dictionary
        """
        self.config = config or OCR_CONFIG.copy()
        self.languages = languages or self.config.get('languages', ['en'])
        self.gpu = gpu if gpu is not None else self.config.get('gpu', True)
        
        # Initialize EasyOCR reader for detection
        self.easyocr_reader = None
        self._initialize_easyocr()
        
        # Initialize custom OCR for recognition
        self.custom_ocr = None
        self._initialize_custom_ocr(custom_model_path)
        
        # Processing statistics
        self.stats = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'total_text_regions': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0,
            'easyocr_detections': 0,
            'custom_recognitions': 0
        }
        
        logger.info(f"Hybrid OCR Pipeline initialized")
        logger.info(f"EasyOCR languages: {self.languages}")
        logger.info(f"Custom model: {custom_model_path}")
        logger.info(f"GPU acceleration: {'Enabled' if self.gpu else 'Disabled'}")
    
    def _initialize_easyocr(self):
        """Initialize EasyOCR reader for text detection"""
        try:
            # Check GPU availability
            if self.gpu and not check_gpu_availability():
                logger.warning("GPU requested but not available. Falling back to CPU.")
                self.gpu = False
            
            # Initialize EasyOCR reader with basic parameters only
            self.easyocr_reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu
            )
            
            logger.info("EasyOCR reader initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
    
    def _initialize_custom_ocr(self, model_path: str):
        """Initialize custom OCR for text recognition"""
        try:
            self.custom_ocr = CustomOCRInference(model_path=model_path)
            logger.info("Custom OCR initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize custom OCR: {e}")
            raise
    
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple]:
        """
        Detect text regions using EasyOCR
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes
        """
        try:
            # Use EasyOCR for detection only
            results = self.easyocr_reader.detect(image)
            
            # Extract bounding boxes
            boxes = []
            if results[0] is not None:
                for box in results[0]:
                    # Convert to tuple format
                    box_tuple = tuple(map(tuple, box))
                    boxes.append(box_tuple)
            
            return boxes
            
        except Exception as e:
            logger.error(f"Error in text detection: {e}")
            return []
    
    def recognize_text_in_region(self, image: np.ndarray, bbox: Tuple) -> Dict:
        """
        Recognize text in a specific region using custom model
        
        Args:
            image: Input image
            bbox: Bounding box coordinates
            
        Returns:
            Recognition result dictionary
        """
        try:
            # Extract region from image
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add some padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            # Extract region
            region = image[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'success': False,
                    'error': 'Empty region'
                }
            
            # Recognize text using custom model
            result = self.custom_ocr.recognize_text(region)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text recognition: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     save_annotated: bool = True, confidence_threshold: float = 0.5,
                     preprocess: bool = True) -> Dict:
        """
        Process a single image using hybrid OCR
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            save_annotated: Whether to save annotated image
            confidence_threshold: Confidence threshold for results
            preprocess: Whether to preprocess image
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Validate image path
        if not validate_image_path(image_path):
            return {
                'success': False,
                'error': 'Invalid image path',
                'image_path': image_path
            }
        
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'image_path': image_path
                }
            
            # Preprocess image if requested
            if preprocess:
                image = preprocess_image(image, IMAGE_CONFIG.get('preprocessing', {}))
            
            # Step 1: Detect text regions using EasyOCR
            logger.info(f"Detecting text regions in {os.path.basename(image_path)}")
            bboxes = self.detect_text_regions(image)
            self.stats['easyocr_detections'] += len(bboxes)
            
            # Step 2: Recognize text in each region using custom model
            results = []
            for i, bbox in enumerate(bboxes):
                logger.info(f"Recognizing text in region {i+1}/{len(bboxes)}")
                
                # Recognize text using custom model
                recognition_result = self.recognize_text_in_region(image, bbox)
                self.stats['custom_recognitions'] += 1
                
                # Filter by confidence
                if recognition_result['confidence'] >= confidence_threshold:
                    results.append({
                        'bbox': format_bbox_for_csv(bbox),
                        'text': recognition_result['text'],
                        'confidence': recognition_result['confidence'],
                        'region_index': i
                    })
            
            # Calculate processing metrics
            end_time = time.time()
            processing_metrics = calculate_processing_metrics(
                start_time, end_time, image.shape[:2]
            )
            
            # Prepare result data
            result_data = {
                'image_name': os.path.basename(image_path),
                'image_path': image_path,
                'text': [r['text'] for r in results],
                'confidence': [r['confidence'] for r in results],
                'bbox': [r['bbox'] for r in results],
                'processing_time': processing_metrics['processing_time'],
                'image_width': processing_metrics['image_width'],
                'image_height': processing_metrics['image_height'],
                'num_text_regions': len(results),
                'num_detected_regions': len(bboxes),
                'success': True
            }
            
            # Save annotated image if requested
            if save_annotated and results:
                annotated_image = image.copy()
                for result in results:
                    bbox = result['bbox']
                    text = result['text']
                    confidence = result['confidence']
                    
                    # Draw bounding box and text
                    points = np.array(eval(bbox), dtype=np.int32)
                    cv2.polylines(annotated_image, [points], True, (0, 255, 0), 2)
                    
                    # Add text label
                    label = f"{text} ({confidence:.2f})"
                    cv2.putText(annotated_image, label, 
                              (points[0][0], points[0][1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save annotated image
                if output_dir is None:
                    output_dir = PATHS['annotated_images']
                
                output_filename = create_output_filename(
                    os.path.basename(image_path), 
                    suffix='_hybrid_annotated'
                )
                output_path = os.path.join(output_dir, output_filename)
                save_image(annotated_image, output_path)
                result_data['annotated_image_path'] = output_path
            
            # Update statistics
            self.stats['total_images'] += 1
            self.stats['successful_images'] += 1
            self.stats['total_text_regions'] += len(results)
            self.stats['total_processing_time'] += processing_metrics['processing_time']
            
            if results:
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                self.stats['average_confidence'] = (
                    (self.stats['average_confidence'] * (self.stats['total_text_regions'] - len(results)) + 
                     sum(r['confidence'] for r in results)) / self.stats['total_text_regions']
                )
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            self.stats['total_images'] += 1
            self.stats['failed_images'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
    
    def process_directory(self, input_dir: str, output_dir: str = None,
                         save_annotated: bool = True, confidence_threshold: float = 0.5,
                         preprocess: bool = True) -> List[Dict]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            save_annotated: Whether to save annotated images
            confidence_threshold: Confidence threshold
            preprocess: Whether to preprocess images
            
        Returns:
            List of processing results
        """
        # Get supported image files
        image_files = get_supported_image_files(input_dir)
        
        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = PATHS['annotated_images']
        os.makedirs(output_dir, exist_ok=True)
        
        # Process images
        results = []
        progress_bar = create_progress_bar(len(image_files), "Processing images")
        
        for i, image_file in enumerate(image_files):
            result = self.process_image(
                image_file, output_dir, save_annotated, 
                confidence_threshold, preprocess
            )
            results.append(result)
            
            update_progress(progress_bar, i + 1)
        
        progress_bar.close()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"hybrid_ocr_results_{timestamp}.csv")
        save_results_to_csv(results, csv_path)
        
        # Save to JSON
        json_path = os.path.join(output_dir, f"hybrid_ocr_results_{timestamp}.json")
        save_results_to_json(results, json_path)
        
        # Create summary report
        summary = create_summary_report(results, self.stats)
        summary_path = os.path.join(output_dir, f"hybrid_ocr_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Processing completed. Results saved to {output_dir}")
        logger.info(f"Summary: {summary}")
        
        return results


def main():
    """Main function for hybrid OCR pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid OCR Pipeline')
    parser.add_argument('--input', '-i', required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', '-o',
                       help='Output directory for results')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to custom trained model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-annotated', action='store_true',
                       help='Do not save annotated images')
    parser.add_argument('--languages', '-l', nargs='+', default=['en'],
                       help='Languages for EasyOCR detection (default: en)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    
    args = parser.parse_args()
    
    # Initialize hybrid OCR pipeline
    hybrid_ocr = HybridOCRPipeline(
        custom_model_path=args.model,
        languages=args.languages,
        gpu=args.gpu
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        result = hybrid_ocr.process_image(
            args.input, args.output, 
            save_annotated=not args.no_annotated,
            confidence_threshold=args.confidence
        )
        
        if result['success']:
            print(f"✅ Successfully processed {args.input}")
            print(f"📝 Detected {result['num_text_regions']} text regions")
            for i, (text, conf) in enumerate(zip(result['text'], result['confidence'])):
                print(f"   Region {i+1}: '{text}' (confidence: {conf:.3f})")
        else:
            print(f"❌ Failed to process {args.input}: {result.get('error', 'Unknown error')}")
    
    elif os.path.isdir(args.input):
        # Directory
        results = hybrid_ocr.process_directory(
            args.input, args.output,
            save_annotated=not args.no_annotated,
            confidence_threshold=args.confidence
        )
        
        successful = sum(1 for r in results if r['success'])
        print(f"✅ Processed {successful}/{len(results)} images successfully")
    
    else:
        print(f"❌ Input path does not exist: {args.input}")


if __name__ == "__main__":
    main() 