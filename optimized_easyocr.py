#!/usr/bin/env python3
"""
Optimized EasyOCR Implementation

This script provides various optimizations to speed up EasyOCR detection:
1. GPU acceleration
2. Reduced image preprocessing
3. Optimized parameters
4. Batch processing
5. Caching
6. Parallel processing
"""

import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Fix for bidi import issue
try:
    import bidi.algorithm
    import bidi
    bidi.get_display = bidi.algorithm.get_display
except Exception as e:
    logging.warning(f"Could not patch bidi import: {e}")

import easyocr
import torch

from config import OCR_CONFIG, IMAGE_CONFIG, OUTPUT_CONFIG, PATHS
from utils import (
    setup_logging, get_supported_image_files, load_image, save_image,
    preprocess_image, draw_bounding_boxes, calculate_processing_metrics,
    format_bbox_for_csv, save_results_to_csv, save_results_to_json,
    create_summary_report, validate_image_path, create_output_filename,
    check_gpu_availability, get_memory_usage, create_progress_bar, update_progress
)

logger = setup_logging()


class OptimizedEasyOCR:
    """
    Optimized EasyOCR implementation for faster detection
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = True, 
                 optimization_level: str = 'balanced', config: Dict = None):
        """
        Initialize optimized EasyOCR
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU acceleration
            optimization_level: 'fast', 'balanced', or 'accurate'
            config: Configuration dictionary
        """
        self.config = config or OCR_CONFIG.copy()
        self.languages = languages or self.config.get('languages', ['en'])
        self.optimization_level = optimization_level
        self.gpu = gpu and check_gpu_availability()
        
        # Optimization parameters based on level
        self.optimization_params = self._get_optimization_params()
        
        # Initialize EasyOCR reader with optimizations
        self.reader = None
        self._initialize_reader()
        
        # Performance tracking
        self.stats = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'total_text_regions': 0,
            'average_confidence': 0.0
        }
        
        logger.info(f"Optimized EasyOCR initialized")
        logger.info(f"Languages: {self.languages}")
        logger.info(f"GPU: {'Enabled' if self.gpu else 'Disabled'}")
        logger.info(f"Optimization level: {optimization_level}")
        logger.info(f"Optimization params: {self.optimization_params}")
    
    def _get_optimization_params(self) -> Dict:
        """Get optimization parameters based on level"""
        if self.optimization_level == 'fast':
            return {
                'width_ths': 0.7,  # Lower threshold for faster detection
                'height_ths': 0.7,
                'text_threshold': 0.6,  # Lower confidence threshold
                'link_threshold': 0.4,
                'low_text': 0.4,
                'canvas_size': 1280,  # Smaller canvas for faster processing
                'mag_ratio': 1.0,  # No magnification
                'slope_ths': 0.1,  # Less strict slope detection
                'ycenter_ths': 0.5,
                'add_margin': 0.1,
                'reformat': True,
                'optimal_num_chars': None
            }
        elif self.optimization_level == 'balanced':
            return {
                'width_ths': 0.8,
                'height_ths': 0.8,
                'text_threshold': 0.7,
                'link_threshold': 0.5,
                'low_text': 0.5,
                'canvas_size': 2560,
                'mag_ratio': 1.2,
                'slope_ths': 0.2,
                'ycenter_ths': 0.6,
                'add_margin': 0.15,
                'reformat': True,
                'optimal_num_chars': None
            }
        else:  # accurate
            return {
                'width_ths': 0.9,
                'height_ths': 0.9,
                'text_threshold': 0.8,
                'link_threshold': 0.6,
                'low_text': 0.6,
                'canvas_size': 5120,
                'mag_ratio': 1.5,
                'slope_ths': 0.3,
                'ycenter_ths': 0.7,
                'add_margin': 0.2,
                'reformat': True,
                'optimal_num_chars': None
            }
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader with optimizations"""
        try:
            # Initialize with basic parameters
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=None,  # Use default storage
                download_enabled=True,
                quantize=True  # Enable quantization for faster inference
            )
            
            logger.info("Optimized EasyOCR reader initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
    
    def _optimize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Optimize image for faster processing
        
        Args:
            image: Input image
            
        Returns:
            Optimized image
        """
        # Resize image if too large (faster processing)
        max_size = self.optimization_params.get('canvas_size', 2560)
        height, width = image.shape[:2]
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def process_image_fast(self, image_path: str, output_dir: str = None,
                          save_annotated: bool = True, confidence_threshold: float = 0.5) -> Dict:
        """
        Process image with optimized settings for speed
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            save_annotated: Whether to save annotated image
            confidence_threshold: Confidence threshold
            
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
            # Load and optimize image
            image = load_image(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'image_path': image_path
                }
            
            # Optimize image for faster processing
            optimized_image = self._optimize_image(image)
            
            # Perform OCR with optimized parameters
            results = self.reader.readtext(
                optimized_image,
                **self.optimization_params
            )
            
            # Filter results by confidence
            filtered_results = []
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    filtered_results.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
            
            # Calculate processing metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Prepare result data
            result_data = {
                'image_name': os.path.basename(image_path),
                'image_path': image_path,
                'text': [r['text'] for r in filtered_results],
                'confidence': [r['confidence'] for r in filtered_results],
                'bbox': [format_bbox_for_csv(r['bbox']) for r in filtered_results],
                'processing_time': processing_time,
                'image_width': image.shape[1],
                'image_height': image.shape[0],
                'num_text_regions': len(filtered_results),
                'success': True
            }
            
            # Save annotated image if requested
            if save_annotated and filtered_results:
                annotated_image = image.copy()
                for result in filtered_results:
                    bbox = result['bbox']
                    text = result['text']
                    confidence = result['confidence']
                    
                    # Draw bounding box
                    points = np.array(bbox, dtype=np.int32)
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
                    suffix=f'_optimized_{self.optimization_level}'
                )
                output_path = os.path.join(output_dir, output_filename)
                save_image(annotated_image, output_path)
                result_data['annotated_image_path'] = output_path
            
            # Update statistics
            self.stats['total_images'] += 1
            self.stats['successful_images'] += 1
            self.stats['total_text_regions'] += len(filtered_results)
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_images']
            )
            
            if filtered_results:
                avg_confidence = sum(r['confidence'] for r in filtered_results) / len(filtered_results)
                self.stats['average_confidence'] = (
                    (self.stats['average_confidence'] * (self.stats['total_text_regions'] - len(filtered_results)) + 
                     sum(r['confidence'] for r in filtered_results)) / self.stats['total_text_regions']
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
    
    def process_directory_parallel(self, input_dir: str, output_dir: str = None,
                                  save_annotated: bool = True, confidence_threshold: float = 0.5,
                                  max_workers: int = None) -> List[Dict]:
        """
        Process directory with parallel processing for maximum speed
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            save_annotated: Whether to save annotated images
            confidence_threshold: Confidence threshold
            max_workers: Maximum number of parallel workers
            
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
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(image_files), 4)  # Limit to 4 for memory
        
        logger.info(f"Using {max_workers} parallel workers")
        
        # Process images in parallel
        results = []
        
        def process_single_image(image_file):
            return self.process_image_fast(
                image_file, output_dir, save_annotated, confidence_threshold
            )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(process_single_image, image_file): image_file 
                for image_file in image_files
            }
            
            # Process results as they complete
            progress_bar = create_progress_bar(len(image_files), "Processing images")
            
            for i, future in enumerate(future_to_image):
                try:
                    result = future.result()
                    results.append(result)
                    update_progress(progress_bar, i + 1)
                except Exception as e:
                    logger.error(f"Error processing {future_to_image[future]}: {e}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'image_path': future_to_image[future]
                    })
        
        progress_bar.close()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"optimized_easyocr_results_{timestamp}.csv")
        save_results_to_csv(results, csv_path)
        
        # Save to JSON
        json_path = os.path.join(output_dir, f"optimized_easyocr_results_{timestamp}.json")
        save_results_to_json(results, json_path)
        
        # Create summary report
        summary = create_summary_report(results, self.stats)
        summary_path = os.path.join(output_dir, f"optimized_easyocr_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Processing completed. Results saved to {output_dir}")
        logger.info(f"Average processing time: {self.stats['average_processing_time']:.3f}s")
        
        return results
    
    def benchmark_performance(self, test_images: List[str], iterations: int = 3) -> Dict:
        """
        Benchmark performance with different optimization levels
        
        Args:
            test_images: List of test image paths
            iterations: Number of iterations for averaging
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running performance benchmark on {len(test_images)} images")
        
        benchmark_results = {}
        
        for level in ['fast', 'balanced', 'accurate']:
            logger.info(f"Testing optimization level: {level}")
            
            # Create new instance with this optimization level
            test_ocr = OptimizedEasyOCR(
                languages=self.languages,
                gpu=self.gpu,
                optimization_level=level
            )
            
            times = []
            for _ in range(iterations):
                start_time = time.time()
                for image_path in test_images:
                    test_ocr.process_image_fast(image_path, save_annotated=False)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            benchmark_results[level] = {
                'average_time': avg_time,
                'time_per_image': avg_time / len(test_images),
                'iterations': iterations
            }
            
            logger.info(f"{level}: {avg_time:.3f}s total, {avg_time/len(test_images):.3f}s per image")
        
        return benchmark_results


def main():
    """Main function for optimized EasyOCR"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized EasyOCR Pipeline')
    parser.add_argument('--input', '-i', required=True,
                       help='Input image file or directory')
    parser.add_argument('--output', '-o',
                       help='Output directory for results')
    parser.add_argument('--optimization', '-opt', choices=['fast', 'balanced', 'accurate'],
                       default='fast', help='Optimization level (default: fast)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-annotated', action='store_true',
                       help='Do not save annotated images')
    parser.add_argument('--languages', '-l', nargs='+', default=['en'],
                       help='Languages for OCR (default: en)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--parallel', '-p', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Initialize optimized EasyOCR
    optimized_ocr = OptimizedEasyOCR(
        languages=args.languages,
        gpu=args.gpu,
        optimization_level=args.optimization
    )
    
    # Run benchmark if requested
    if args.benchmark:
        if os.path.isfile(args.input):
            test_images = [args.input]
        else:
            test_images = get_supported_image_files(args.input)[:5]  # Test first 5 images
        
        benchmark_results = optimized_ocr.benchmark_performance(test_images)
        
        print("\n📊 Performance Benchmark Results:")
        print("=" * 50)
        for level, results in benchmark_results.items():
            print(f"{level.upper()}:")
            print(f"  Total time: {results['average_time']:.3f}s")
            print(f"  Time per image: {results['time_per_image']:.3f}s")
            print()
        
        return
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        result = optimized_ocr.process_image_fast(
            args.input, args.output, 
            save_annotated=not args.no_annotated,
            confidence_threshold=args.confidence
        )
        
        if result['success']:
            print(f"✅ Successfully processed {args.input}")
            print(f"📝 Detected {result['num_text_regions']} text regions")
            print(f"⏱️  Processing time: {result['processing_time']:.3f}s")
            for i, (text, conf) in enumerate(zip(result['text'], result['confidence'])):
                print(f"   Region {i+1}: '{text}' (confidence: {conf:.3f})")
        else:
            print(f"❌ Failed to process {args.input}: {result.get('error', 'Unknown error')}")
    
    elif os.path.isdir(args.input):
        # Directory
        results = optimized_ocr.process_directory_parallel(
            args.input, args.output,
            save_annotated=not args.no_annotated,
            confidence_threshold=args.confidence,
            max_workers=args.parallel
        )
        
        successful = sum(1 for r in results if r['success'])
        print(f"✅ Processed {successful}/{len(results)} images successfully")
        print(f"⏱️  Average processing time: {optimized_ocr.stats['average_processing_time']:.3f}s")
    
    else:
        print(f"❌ Input path does not exist: {args.input}")


if __name__ == "__main__":
    main() 