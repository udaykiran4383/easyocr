#!/usr/bin/env python3
"""
Test all available models to find one that works properly
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import CRNNModel
from config import MODEL_CONFIG

def test_model(model_path, test_image):
    """Test a specific model"""
    try:
        print(f"\n🧪 Testing: {os.path.basename(model_path)}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create model
        model = CRNNModel(
            num_classes=40,
            img_height=MODEL_CONFIG['imgH'],
            img_width=MODEL_CONFIG['imgW']
        )
        
        # Load weights
        model.load_state_dict(state_dict)
        model.eval()
        
        # Preprocess image
        if len(test_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(test_image).convert('RGB')
        
        pil_image = pil_image.resize((MODEL_CONFIG['imgW'], MODEL_CONFIG['imgH']))
        image_tensor = torch.from_numpy(np.array(pil_image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            logits = model(image_tensor)
        
        # Decode predictions
        predictions = torch.argmax(logits, dim=2)
        pred_list = predictions[0].cpu().numpy()
        
        # Character set
        char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        idx_to_char = {idx: char for idx, char in enumerate(char_set)}
        
        # Decode to text
        text = ""
        prev_char = None
        for pred in pred_list:
            if pred != 0 and pred != prev_char:
                if pred < len(idx_to_char):
                    text += idx_to_char[pred]
            prev_char = pred
        
        # Calculate confidence
        probs = torch.softmax(logits, dim=2)
        confidence = torch.max(probs).item()
        
        print(f"  Text: '{text}'")
        print(f"  Confidence: {confidence:.3f}")
        
        return text, confidence
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, 0.0

def main():
    """Test all available models"""
    print("🔍 Testing All Available Models")
    print("=" * 50)
    
    # Create test image
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "HELLO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Find all model files
    checkpoint_dir = "checkpoints"
    model_files = []
    
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth') and 'best_model' in file:
                model_files.append(os.path.join(checkpoint_dir, file))
    
    if not model_files:
        print("❌ No model files found!")
        return
    
    print(f"📁 Found {len(model_files)} model files")
    
    # Test each model
    results = []
    for model_path in sorted(model_files):
        text, confidence = test_model(model_path, test_image)
        results.append((os.path.basename(model_path), text, confidence))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    for model_name, text, confidence in results:
        status = "✅" if text and text != "CWIA" else "❌"
        print(f"{status} {model_name}: '{text}' (conf: {confidence:.3f})")
    
    # Find best working model
    working_models = [(name, text, conf) for name, text, conf in results if text and text != "CWIA"]
    
    if working_models:
        print(f"\n🎉 Found {len(working_models)} working model(s)!")
        best_model = max(working_models, key=lambda x: x[2])
        print(f"🏆 Best model: {best_model[0]} - '{best_model[1]}' (conf: {best_model[2]:.3f})")
    else:
        print("\n❌ No working models found. All models return 'CWIA' or errors.")
        print("🔧 This suggests a training or architecture issue.")
        print("💡 Recommendation: Use EasyOCR for now, or retrain the model.")

if __name__ == "__main__":
    main() 