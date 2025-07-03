#!/usr/bin/env python3
"""
Simple EasyOCR POC Demo
A minimal proof of concept for EasyOCR text detection.
"""

import cv2
import os
import time

def simple_ocr_demo():
    """Simple OCR demo using EasyOCR"""
    
    print("🚀 Simple EasyOCR POC Demo")
    print("=" * 40)
    
    try:
        # Import EasyOCR (with error handling)
        print("📦 Importing EasyOCR...")
        import easyocr
        print("✅ EasyOCR imported successfully!")
        
        # Initialize reader
        print("🔄 Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], verbose=False)
        print("✅ EasyOCR reader ready!")
        
        # Find demo images
        demo_images = [
            "images/Sample-handwritten-text-input-for-OCR.png",
            "images/13_2.png",
            "images/27_1.png"
        ]
        
        available_images = [img for img in demo_images if os.path.exists(img)]
        
        if not available_images:
            print("❌ No demo images found!")
            print("📝 Please place some images in the 'images/' folder")
            return
        
        print(f"🎯 Found {len(available_images)} demo images")
        
        # Process each image
        for image_path in available_images:
            print(f"\n{'='*50}")
            print(f"🔍 Processing: {image_path}")
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                continue
            
            # Detect text
            start_time = time.time()
            results = reader.readtext(image)
            processing_time = time.time() - start_time
            
            print(f"⏱️  Processing time: {processing_time:.3f} seconds")
            print(f"📝 Detected {len(results)} text regions")
            
            # Display results
            if results:
                print("\n📊 Detection Results:")
                print("-" * 30)
                for i, (bbox, text, confidence) in enumerate(results, 1):
                    print(f"Text {i}: '{text}' (confidence: {confidence:.3f})")
                
                # Draw results on image
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
                output_path = f"simple_poc_output_{os.path.basename(image_path)}"
                cv2.imwrite(output_path, image)
                print(f"💾 Annotated image saved: {output_path}")
            else:
                print("❌ No text detected")
            
            print(f"{'='*50}")
        
        print("\n🎉 Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📝 Please install dependencies: pip install -r poc_requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📝 Please check your setup and try again")

if __name__ == "__main__":
    simple_ocr_demo() 