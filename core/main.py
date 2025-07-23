#!/usr/bin/env python3
"""
Main execution script for EasyOCR Text Detection and Recognition Pipeline

This script provides a command-line interface for the OCR pipeline with various modes:
- inference: Process images for text detection and recognition
- training: Train custom models (future implementation)
- evaluation: Evaluate trained models (future implementation)
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix for bidi import issue
try:
    import bidi.algorithm
    import bidi
    bidi.get_display = bidi.algorithm.get_display
except Exception as e:
    print(f"Warning: Could not patch bidi import: {e}")

from ocr_pipeline import OCRPipeline
from config import PATHS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inference_mode(args):
    """Run inference mode - process images for OCR"""
    logger.info("Starting inference mode...")
    
    # Initialize pipeline
    pipeline = OCRPipeline(
        languages=args.languages,
        gpu=args.gpu
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        logger.info(f"Processing single image: {args.input}")
        result = pipeline.process_image(
            args.input,
            args.output,
            save_annotated=not args.no_annotated,
            confidence_threshold=args.confidence
        )
        
        if result['success']:
            print(f"✓ Successfully processed {args.input}")
            print(f"  Text regions found: {result['num_text_regions']}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
            if result['text']:
                print("  Detected text:")
                for i, (text, conf) in enumerate(zip(result['text'], result['confidence'])):
                    print(f"    {i+1}. '{text}' (confidence: {conf:.3f})")
        else:
            print(f"✗ Failed to process {args.input}: {result.get('error', 'Unknown error')}")
            
    else:
        # Directory
        logger.info(f"Processing directory: {args.input}")
        results = pipeline.process_folder(
            input_dir=args.input,
            output_dir=args.output,
            confidence_threshold=args.confidence,
            save_annotated=not args.no_annotated,
            save_results=True,
            output_format=args.format
        )
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        total_regions = sum(r.get('num_text_regions', 0) for r in results)
        
        print(f"\n=== Processing Summary ===")
        print(f"✓ Successfully processed: {successful}/{len(results)} images")
        print(f"  Total text regions found: {total_regions}")
        print(f"  Average confidence: {pipeline.stats['average_confidence']:.3f}")
        print(f"  Total processing time: {pipeline.stats['total_processing_time']:.2f}s")
        
        if args.output:
            print(f"\nResults saved to: {args.output}")


def training_mode(args):
    """Run training mode - train custom models"""
    logger.info("Training mode not yet implemented")
    print("Training mode is planned for future implementation.")
    print("For now, please use the pre-trained models provided by EasyOCR.")


def evaluation_mode(args):
    """Run evaluation mode - evaluate trained models"""
    logger.info("Evaluation mode not yet implemented")
    print("Evaluation mode is planned for future implementation.")
    print("For now, please use the inference mode to test the pipeline.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='EasyOCR Text Detection and Recognition Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python main.py --mode inference --input images/sample.png --output results/
  
  # Process all images in a folder
  python main.py --mode inference --input images/ --output annotated_images/
  
  # Process with custom confidence threshold
  python main.py --mode inference --input images/ --confidence 0.7
  
  # Process without saving annotated images
  python main.py --mode inference --input images/ --no-annotated
        """
    )
    
    # Main mode argument
    parser.add_argument(
        '--mode', '-m',
        choices=['inference', 'training', 'evaluation'],
        default='inference',
        help='Operation mode (default: inference)'
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input image file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for results'
    )
    
    # OCR parameters
    parser.add_argument(
        '--languages', '-l',
        nargs='+',
        default=['en'],
        help='Languages for OCR (default: en)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (if available)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-annotated',
        action='store_true',
        help='Do not save annotated images'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'json'],
        default='csv',
        help='Output format for results (default: csv)'
    )
    
    # Training parameters (for future use)
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (training mode only)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (training mode only)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Created output directory: {args.output}")
    
    # Run appropriate mode
    try:
        if args.mode == 'inference':
            inference_mode(args)
        elif args.mode == 'training':
            training_mode(args)
        elif args.mode == 'evaluation':
            evaluation_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 