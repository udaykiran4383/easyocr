#!/usr/bin/env python3
"""
Test script for EasyOCR Reader with Local Model Weights
"""

# Fix for bidi import issue
try:
    import bidi.algorithm
    import bidi
    bidi.get_display = bidi.algorithm.get_display
    print("Bidi import fixed")
except Exception as e:
    print(f"Bidi fix failed: {e}")

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import time

# Add the current directory to Python path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from custom_ocr_inference import CustomOCRInference
    print("Custom OCR Inference imported successfully")
    
    # Configuration
    MODEL_PATH = "checkpoints/best_model_epoch_7.pth"  # Use your best trained model
    
    # Initialize custom OCR with local model (fixed architecture)
    print(f"Loading custom model from: {MODEL_PATH}")
    
    # Load checkpoint to get correct architecture
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create model with correct number of classes (40 for IIIT5K)
    from model_trainer import CRNNModel
    from config import MODEL_CONFIG
    
    model = CRNNModel(
        num_classes=40,  # Fixed for IIIT5K dataset
        img_height=MODEL_CONFIG['imgH'],
        img_width=MODEL_CONFIG['imgW']
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Custom model loaded successfully with local weights!")
    
    # Character set for IIIT5K dataset
    char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    char_to_idx = {char: idx for idx, char in enumerate(char_set)}
    idx_to_char = {idx: char for idx, char in enumerate(char_set)}
    
    # Test with a simple image
    print("\nTesting with a simple test image...")
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    # Add some text-like patterns for testing
    cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Test custom OCR inference
    def preprocess_image(image):
        """Preprocess image for model input"""
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        pil_image = pil_image.resize((MODEL_CONFIG['imgW'], MODEL_CONFIG['imgH']))
        image_tensor = torch.from_numpy(np.array(pil_image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    
    def decode_predictions(logits):
        """Decode model predictions to text"""
        predictions = torch.argmax(logits, dim=2)
        pred_list = predictions[0].cpu().numpy()
        
        text = ""
        prev_char = None
        for pred in pred_list:
            if pred != 0 and pred != prev_char:  # Skip blank and repeated characters
                if pred < len(idx_to_char):
                    text += idx_to_char[pred]
            prev_char = pred
        return text
    
    # Run inference
    input_tensor = preprocess_image(test_image)
    with torch.no_grad():
        logits = model(input_tensor)
    
    text = decode_predictions(logits)
    probs = torch.softmax(logits, dim=2)
    confidence = torch.max(probs).item()
    
    print(f"Custom OCR test completed:")
    print(f"  Text: '{text}'")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Model: Local weights from {MODEL_PATH}")
    
    # Also test with EasyOCR for comparison
    print("\nTesting with EasyOCR for comparison...")
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    easyocr_results = reader.readtext(test_image)
    print(f"EasyOCR found {len(easyocr_results)} text regions")
    
    # Test with actual images if available
    images_dir = "images"
    if os.path.exists(images_dir):
        print(f"\nTesting with images from {images_dir}...")
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if image_files:
            test_image_file = os.path.join(images_dir, image_files[0])
            print(f"Testing with: {test_image_file}")
            
            image = cv2.imread(test_image_file)
            if image is not None:
                # Test custom OCR
                input_tensor = preprocess_image(image)
                with torch.no_grad():
                    logits = model(input_tensor)
                
                custom_text = decode_predictions(logits)
                probs = torch.softmax(logits, dim=2)
                custom_confidence = torch.max(probs).item()
                
                print(f"Custom OCR result:")
                print(f"  Text: '{custom_text}'")
                print(f"  Confidence: {custom_confidence:.3f}")
                print(f"  Model: Local weights")
                
                # Test EasyOCR
                easyocr_results = reader.readtext(image)
                print(f"EasyOCR found {len(easyocr_results)} text regions")
                if easyocr_results:
                    best_result = max(easyocr_results, key=lambda x: x[2])
                    print(f"  Best text: '{best_result[1]}' (confidence: {best_result[2]:.3f})")
                    print(f"  Model: Pre-trained (downloaded)")
    
    print("\n✅ Custom model testing completed successfully!")
    print("🎯 Key differences:")
    print("  - Custom model uses local weights (no internet required)")
    print("  - EasyOCR uses pre-trained models (requires download)")
    print("  - Custom model can be fine-tuned for specific datasets")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure custom_ocr_inference.py is in the same directory")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 