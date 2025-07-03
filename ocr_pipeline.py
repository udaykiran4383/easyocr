"""
EasyOCR Text Detection and Recognition Pipeline

This module implements a comprehensive OCR pipeline using EasyOCR for text detection
and recognition with CRAFT + CRNN architecture.
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

from config import OCR_CONFIG, IMAGE_CONFIG, OUTPUT_CONFIG, PATHS
from utils import (
    setup_logging, get_supported_image_files, load_image, save_image,
    preprocess_image, draw_bounding_boxes, calculate_processing_metrics,
    format_bbox_for_csv, save_results_to_csv, save_results_to_json,
    create_summary_report, validate_image_path, create_output_filename,
    check_gpu_availability, get_memory_usage, create_progress_bar, update_progress
)

logger = setup_logging()


class OCRPipeline:
    """
    Main OCR Pipeline class for text detection and recognition using EasyOCR
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = None, 
                 config: Dict = None):
        """
        Initialize the OCR pipeline
        
        Args:
            languages: List of language codes for OCR
            gpu: Whether to use GPU acceleration
            config: Configuration dictionary
        """
        self.config = config or OCR_CONFIG.copy()
        self.languages = languages or self.config.get('languages', ['en'])
        self.gpu = gpu if gpu is not None else self.config.get('gpu', True)
        
        # Initialize EasyOCR reader
        self.reader = None
        self._initialize_reader()
        
        # Processing statistics
        self.stats = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'total_text_regions': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        logger.info(f"OCR Pipeline initialized with languages: {self.languages}")
        logger.info(f"GPU acceleration: {'Enabled' if self.gpu else 'Disabled'}")
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader with specified configuration"""
        try:
            # Check GPU availability
            if self.gpu and not check_gpu_availability():
                logger.warning("GPU requested but not available. Falling back to CPU.")
                self.gpu = False
            
            # Initialize EasyOCR reader with basic parameters only
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu
            )
            
            logger.info("EasyOCR reader initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     save_annotated: bool = True, confidence_threshold: float = 0.5,
                     preprocess: bool = True) -> Dict:
        """
        Process a single image for text detection and recognition
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save annotated images
            save_annotated: Whether to save annotated image
            confidence_threshold: Minimum confidence threshold
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary containing processing results
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
            
            # Perform OCR
            results = self.reader.readtext(image)
            
            # Filter results by confidence
            if confidence_threshold > 0:
                results = [r for r in results if r[2] >= confidence_threshold]
            
            # Calculate processing metrics
            end_time = time.time()
            processing_metrics = calculate_processing_metrics(
                start_time, end_time, image.shape[:2]
            )
            
            # Prepare result data
            result_data = {
                'image_name': os.path.basename(image_path),
                'image_path': image_path,
                'text': [r[1] for r in results],
                'confidence': [r[2] for r in results],
                'bbox': [format_bbox_for_csv(r[0]) for r in results],
                'processing_time': processing_metrics['processing_time'],
                'image_width': processing_metrics['image_width'],
                'image_height': processing_metrics['image_height'],
                'num_text_regions': len(results),
                'success': True
            }
            
            # Save annotated image if requested
            if save_annotated and results:
                annotated_image = draw_bounding_boxes(image, results)
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = create_output_filename(
                        image_path, "annotated", ".png"
                    )
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if save_image(annotated_image, output_path):
                        result_data['annotated_image_path'] = output_path
                        logger.info(f"Annotated image saved to {output_path}")
            
            # Update statistics
            self._update_stats(result_data, processing_metrics)
            
            logger.info(f"Processed {image_path}: {len(results)} text regions found")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'processing_time': time.time() - start_time
            }
    
    def process_folder(self, input_dir: str, output_dir: str = None,
                      confidence_threshold: float = 0.5, save_annotated: bool = True,
                      save_results: bool = True, output_format: str = 'csv') -> List[Dict]:
        """
        Process all images in a folder
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for results
            confidence_threshold: Minimum confidence threshold
            save_annotated: Whether to save annotated images
            save_results: Whether to save results to file
            output_format: Output format ('csv' or 'json')
            
        Returns:
            List of processing results
        """
        # Get list of supported image files
        image_files = get_supported_image_files(input_dir)
        
        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process images
        results = []
        progress_bar = create_progress_bar(len(image_files), "Processing images")
        
        for image_file in image_files:
            result = self.process_image(
                image_file, output_dir, save_annotated, confidence_threshold
            )
            results.append(result)
            update_progress(progress_bar)
        
        # Save results if requested
        if save_results and results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == 'csv':
                output_path = os.path.join(
                    output_dir or PATHS['results'], 
                    f"ocr_results_{timestamp}.csv"
                )
                save_results_to_csv(results, output_path)
            else:
                output_path = os.path.join(
                    output_dir or PATHS['results'], 
                    f"ocr_results_{timestamp}.json"
                )
                save_results_to_json(results, output_path)
            
            # Create summary report
            summary_path = os.path.join(
                output_dir or PATHS['results'], 
                f"summary_report_{timestamp}.json"
            )
            create_summary_report(results, summary_path)
        
        # Log final statistics
        self._log_final_stats()
        
        return results
    
    def _update_stats(self, result_data: Dict, processing_metrics: Dict):
        """Update processing statistics"""
        self.stats['total_images'] += 1
        
        if result_data.get('success', False):
            self.stats['successful_images'] += 1
            self.stats['total_text_regions'] += result_data.get('num_text_regions', 0)
            self.stats['total_processing_time'] += processing_metrics['processing_time']
            
            # Update average confidence
            confidences = result_data.get('confidence', [])
            if confidences:
                current_avg = self.stats['average_confidence']
                total_regions = self.stats['total_text_regions']
                new_avg = (current_avg * (total_regions - len(confidences)) + 
                          sum(confidences)) / total_regions
                self.stats['average_confidence'] = new_avg
        else:
            self.stats['failed_images'] += 1
    
    def _log_final_stats(self):
        """Log final processing statistics"""
        logger.info("=== Processing Statistics ===")
        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Successful: {self.stats['successful_images']}")
        logger.info(f"Failed: {self.stats['failed_images']}")
        logger.info(f"Total text regions: {self.stats['total_text_regions']}")
        logger.info(f"Average confidence: {self.stats['average_confidence']:.3f}")
        logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f}s")
        
        if self.stats['successful_images'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['successful_images']
            logger.info(f"Average time per image: {avg_time:.2f}s")
    
    def get_stats(self) -> Dict:
        """Get current processing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'total_text_regions': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        return get_memory_usage()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'reader') and self.reader is not None:
            del self.reader


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EasyOCR Text Detection and Recognition Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--languages', '-l', nargs='+', default=['en'], help='Languages for OCR')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-annotated', action='store_true', help='Do not save annotated images')
    parser.add_argument('--format', '-f', choices=['csv', 'json'], default='csv', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = OCRPipeline(languages=args.languages, gpu=args.gpu)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        result = pipeline.process_image(
            args.input, args.output, not args.no_annotated, args.confidence
        )
        print(f"Processed: {result}")
    else:
        # Directory
        results = pipeline.process_folder(
            args.input, args.output, args.confidence, 
            not args.no_annotated, True, args.format
        )
        print(f"Processed {len(results)} images")


if __name__ == "__main__":
    main() 