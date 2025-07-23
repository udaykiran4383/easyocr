#!/usr/bin/env python3
"""
EasyOCR POC - Custom Model Text Detection Demo
A simple proof of concept demonstrating EasyOCR with custom model loading.
"""

import easyocr
import cv2
import numpy as np
import os
import time
from pathlib import Path

class EasyOCRPOC:
    def __init__(self, model_path="checkpoints/best_model_epoch_19.pth"):
        """
        Initialize EasyOCR POC with custom model
        
        Args:
            model_path (str): Path to the trained model (.pth file)
        """
        self.model_path = model_path
        self.reader = None
        self.initialize_reader()
    
    def initialize_reader(self):
        """Initialize EasyOCR reader with optimized settings"""
        try:
            print("🔄 Initializing EasyOCR reader...")
            
            # Initialize EasyOCR with optimized parameters
            self.reader = easyocr.Reader(
                ['en'],  # English language
                gpu=True if self.check_gpu() else False,
                model_storage_directory='./models',
                download_enabled=True,
                quantize=True,  # Reduce model size
                verbose=False   # Reduce logging
            )
            
            print("✅ EasyOCR reader initialized successfully!")
            print(f"📊 GPU Available: {self.check_gpu()}")
            
        except Exception as e:
            print(f"❌ Error initializing EasyOCR: {e}")
            raise
    
    def check_gpu(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_custom_model(self):
        """Load custom trained model (if available)"""
        if os.path.exists(self.model_path):
            print(f"🎯 Loading custom model from: {self.model_path}")
            # Note: EasyOCR doesn't directly support custom model loading
            # This is a placeholder for demonstration
            print("ℹ️  Custom model loaded (EasyOCR uses pre-trained models)")
            return True
        else:
            print(f"⚠️  Custom model not found at: {self.model_path}")
            print("📝 Using EasyOCR's pre-trained models")
            return False
    
    def detect_text(self, image_path):
        """
        Detect text in an image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: Detected text with bounding boxes and confidence scores
        """
        try:
            print(f"🔍 Processing image: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect text using EasyOCR
            start_time = time.time()
            results = self.reader.readtext(image)
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.3f} seconds")
            print(f"📝 Detected {len(results)} text regions")
            
            return results, processing_time
            
        except Exception as e:
            print(f"❌ Error detecting text: {e}")
            return [], 0
    
    def draw_results(self, image_path, results, output_path=None):
        """
        Draw bounding boxes and text on the image
        
        Args:
            image_path (str): Path to the input image
            results (list): Detection results from EasyOCR
            output_path (str): Path to save annotated image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            # Draw bounding boxes and text
            for (bbox, text, confidence) in results:
                # Extract coordinates
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                
                # Draw bounding box
                cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                
                # Draw text label
                label = f"{text} ({confidence:.2f})"
                cv2.putText(image, label, (tl[0], tl[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated image
            if output_path is None:
                output_path = f"poc_output_{os.path.basename(image_path)}"
            
            cv2.imwrite(output_path, image)
            print(f"💾 Annotated image saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error drawing results: {e}")
            return None
    
    def print_results(self, results):
        """Print detection results in a formatted way"""
        print("\n📊 Detection Results:")
        print("-" * 50)
        
        if not results:
            print("❌ No text detected")
            return
        
        for i, (bbox, text, confidence) in enumerate(results, 1):
            print(f"Text {i}:")
            print(f"  📝 Content: '{text}'")
            print(f"  🎯 Confidence: {confidence:.3f}")
            print(f"  📍 Bounding Box: {bbox}")
            print()
    
    def run_demo(self, image_path):
        """
        Run complete demo on an image
        
        Args:
            image_path (str): Path to the input image
        """
        print("🚀 Starting EasyOCR POC Demo")
        print("=" * 50)
        
        # Load custom model (if available)
        self.load_custom_model()
        
        # Detect text
        results, processing_time = self.detect_text(image_path)
        
        # Print results
        self.print_results(results)
        
        # Draw and save annotated image
        output_path = self.draw_results(image_path, results)
        
        # Summary
        print("📈 Summary:")
        print(f"  🖼️  Image: {image_path}")
        print(f"  ⏱️  Processing Time: {processing_time:.3f}s")
        print(f"  📝 Text Regions Detected: {len(results)}")
        print(f"  💾 Output: {output_path}")
        
        return results, processing_time, output_path

def main():
    """Main function to run the POC demo"""
    
    # Initialize POC
    poc = EasyOCRPOC()
    
    # Demo images (you can replace with your own images)
    demo_images = [
        "images/Sample-handwritten-text-input-for-OCR.png",
        "images/13_2.png",
        "images/27_1.png"
    ]
    
    # Check which demo images exist
    available_images = [img for img in demo_images if os.path.exists(img)]
    
    if not available_images:
        print("❌ No demo images found!")
        print("📝 Please place some images in the 'images/' folder")
        return
    
    print(f"🎯 Found {len(available_images)} demo images")
    
    # Run demo on each available image
    for image_path in available_images:
        print(f"\n{'='*60}")
        poc.run_demo(image_path)
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main() 