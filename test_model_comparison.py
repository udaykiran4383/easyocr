#!/usr/bin/env python3
"""
Test script to compare different model checkpoints and find the best one
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import CRNNModel
from config import MODEL_CONFIG

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

def decode_predictions(logits, idx_to_char):
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

def test_model(model_path, test_image, char_set):
    """Test a specific model checkpoint"""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model
        model = CRNNModel(
            num_classes=len(char_set) + 1,  # +1 for CTC blank
            img_height=MODEL_CONFIG['imgH'],
            img_width=MODEL_CONFIG['imgW']
        )
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create character mappings
        char_to_idx = {char: idx + 1 for idx, char in enumerate(char_set)}  # +1 for CTC blank
        idx_to_char = {idx + 1: char for idx, char in enumerate(char_set)}  # +1 for CTC blank
        
        # Run inference
        input_tensor = preprocess_image(test_image)
        with torch.no_grad():
            logits = model(input_tensor)
        
        text = decode_predictions(logits, idx_to_char)
        probs = torch.softmax(logits, dim=2)
        confidence = torch.max(probs).item()
        
        return text, confidence
        
    except Exception as e:
        return f"Error: {e}", 0.0

def main():
    # Character set
    char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    # Create a simple test image
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "HELLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Available model checkpoints
    checkpoints = [
        "checkpoints/best_model_epoch_1.pth",
        "checkpoints/best_model_epoch_2.pth", 
        "checkpoints/best_model_epoch_4.pth",
        "checkpoints/best_model_epoch_5.pth",
        "checkpoints/best_model_epoch_7.pth",
        "checkpoints/best_model_epoch_10.pth",
        "checkpoints/best_model_epoch_13.pth",
        "checkpoints/best_model_epoch_17.pth",
        "checkpoints/best_model_epoch_19.pth",
    ]
    
    print("Testing different model checkpoints...")
    print("=" * 60)
    
    results = []
    
    for checkpoint in checkpoints:
        if os.path.exists(checkpoint):
            print(f"\nTesting: {checkpoint}")
            text, confidence = test_model(checkpoint, test_image, char_set)
            results.append((checkpoint, text, confidence))
            print(f"  Output: '{text}'")
            print(f"  Confidence: {confidence:.3f}")
        else:
            print(f"\nSkipping: {checkpoint} (not found)")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    # Sort by confidence
    results.sort(key=lambda x: x[2], reverse=True)
    
    for i, (checkpoint, text, confidence) in enumerate(results):
        print(f"{i+1}. {os.path.basename(checkpoint)}")
        print(f"   Output: '{text}'")
        print(f"   Confidence: {confidence:.3f}")
    
    # Test with actual images if available
    images_dir = "images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if image_files:
            print(f"\n" + "=" * 60)
            print("TESTING WITH ACTUAL IMAGES:")
            print("=" * 60)
            
            for img_file in image_files[:3]:  # Test first 3 images
                img_path = os.path.join(images_dir, img_file)
                print(f"\nTesting with: {img_file}")
                
                image = cv2.imread(img_path)
                if image is not None:
                    # Test with best model
                    best_checkpoint = results[0][0] if results else checkpoints[-1]
                    text, confidence = test_model(best_checkpoint, image, char_set)
                    print(f"  Best model output: '{text}'")
                    print(f"  Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main() 