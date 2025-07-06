#!/usr/bin/env python3
"""
Debug script to investigate the model issue
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import easyocr

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import CRNNModel
from config import MODEL_CONFIG

def debug_model_inference():
    """Debug the model inference step by step"""
    print("🔍 Debugging Model Inference")
    print("=" * 50)
    
    # Create a simple test image
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "A", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    print(f"📸 Test image created with text 'A'")
    
    # Load model
    model_path = "checkpoints/best_model_epoch_19.pth"
    print(f"🤖 Loading model: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model = CRNNModel(
        num_classes=40,
        img_height=MODEL_CONFIG['imgH'],
        img_width=MODEL_CONFIG['imgW']
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Preprocess image
    print(f"🖼️  Preprocessing image to size: {MODEL_CONFIG['imgW']}x{MODEL_CONFIG['imgH']}")
    
    if len(test_image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(test_image).convert('RGB')
    
    pil_image = pil_image.resize((MODEL_CONFIG['imgW'], MODEL_CONFIG['imgH']))
    image_tensor = torch.from_numpy(np.array(pil_image)).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    print(f"📊 Input tensor shape: {image_tensor.shape}")
    print(f"📊 Input tensor range: {image_tensor.min():.3f} to {image_tensor.max():.3f}")
    
    # Run inference
    print("🚀 Running inference...")
    with torch.no_grad():
        logits = model(image_tensor)
    
    print(f"📊 Output logits shape: {logits.shape}")
    print(f"📊 Logits range: {logits.min():.3f} to {logits.max():.3f}")
    
    # Get predictions
    predictions = torch.argmax(logits, dim=2)
    pred_list = predictions[0].cpu().numpy()
    
    print(f"📊 Predictions shape: {predictions.shape}")
    print(f"📊 First 10 predictions: {pred_list[:10]}")
    
    # Character set
    char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    idx_to_char = {idx: char for idx, char in enumerate(char_set)}
    
    print(f"📝 Character set length: {len(char_set)}")
    print(f"📝 Character set: {char_set}")
    
    # Decode to text
    text = ""
    prev_char = None
    for i, pred in enumerate(pred_list):
        if pred != 0 and pred != prev_char:
            if pred < len(idx_to_char):
                char = idx_to_char[pred]
                text += char
                print(f"  Position {i}: pred={pred} -> char='{char}'")
        prev_char = pred
    
    print(f"📝 Final text: '{text}'")
    
    # Calculate confidence
    probs = torch.softmax(logits, dim=2)
    confidence = torch.max(probs).item()
    print(f"🎯 Confidence: {confidence:.3f}")
    
    return text, confidence

def test_easyocr_comparison():
    """Test EasyOCR for comparison"""
    print("\n" + "=" * 50)
    print("📚 Testing EasyOCR for Comparison")
    print("=" * 50)
    
    # Create test image
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "A", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Test EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(test_image)
    
    print(f"📊 EasyOCR found {len(results)} text regions")
    for i, (bbox, text, conf) in enumerate(results):
        print(f"  Region {i+1}: '{text}' (confidence: {conf:.3f})")
    
    return results

def check_model_architecture():
    """Check the model architecture"""
    print("\n" + "=" * 50)
    print("🏗️  Checking Model Architecture")
    print("=" * 50)
    
    from model_trainer import CRNNModel
    
    model = CRNNModel(
        num_classes=40,
        img_height=MODEL_CONFIG['imgH'],
        img_width=MODEL_CONFIG['imgW']
    )
    
    print(f"📊 Model config:")
    print(f"  - num_classes: {40}")
    print(f"  - img_height: {MODEL_CONFIG['imgH']}")
    print(f"  - img_width: {MODEL_CONFIG['imgW']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # Print model structure
    print(f"📊 Model structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            print(f"  {name}: {module}")

def main():
    """Main debug function"""
    print("🚀 Starting Model Debug")
    print("=" * 50)
    
    # Check model architecture
    check_model_architecture()
    
    # Debug model inference
    text, confidence = debug_model_inference()
    
    # Test EasyOCR
    easyocr_results = test_easyocr_comparison()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEBUG SUMMARY")
    print("=" * 50)
    print(f"🤖 Custom model output: '{text}' (conf: {confidence:.3f})")
    print(f"📚 EasyOCR output: {[r[1] for r in easyocr_results]}")
    
    if text and text != "A":
        print("❌ Issue identified: Model is not recognizing simple text correctly")
        print("🔧 Possible causes:")
        print("  1. Training data issue")
        print("  2. Character set mismatch")
        print("  3. Model architecture problem")
        print("  4. Preprocessing issue")
        print("💡 Recommendation: Use EasyOCR for now, investigate training process")

if __name__ == "__main__":
    main() 