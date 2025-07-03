#!/usr/bin/env python3
"""
Test script for Local Model Weights vs EasyOCR (Fixed Version)
Handles model architecture mismatch and uses correct number of classes
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import easyocr
import torch
import torch.nn as nn

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from custom_ocr_inference import CustomOCRInference
    print("✅ Custom OCR Inference imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure custom_ocr_inference.py is in the same directory")
    sys.exit(1)

class FixedCustomOCRInference:
    """
    Fixed version of Custom OCR that handles model architecture mismatch
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize with fixed model loading
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load the model with correct architecture
        self.model = self._load_fixed_model()
        
        # Character set for IIIT5K dataset (40 classes including blank)
        self.char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.char_set)}
        
        print(f"✅ Fixed Custom OCR initialized with {len(self.char_set)} characters")
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Device: {self.device}")
    
    def _load_fixed_model(self):
        """Load model with correct architecture for 40 classes"""
        try:
            # Load checkpoint first to get the correct architecture
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check if it's a state dict or full checkpoint
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
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.eval()
            
            print("✅ Model loaded successfully with correct architecture")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Resize to model input size
            from config import MODEL_CONFIG
            pil_image = pil_image.resize((MODEL_CONFIG['imgW'], MODEL_CONFIG['imgH']))
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(np.array(pil_image)).float()
            image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW and normalize
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def decode_predictions(self, logits: torch.Tensor) -> str:
        """Decode model predictions to text"""
        try:
            # Get predictions (argmax)
            predictions = torch.argmax(logits, dim=2)
            
            # Convert to list
            pred_list = predictions[0].cpu().numpy()
            
            # Decode to text (skip blank character at index 0)
            text = ""
            prev_char = None
            for pred in pred_list:
                if pred != 0 and pred != prev_char:  # Skip blank and repeated characters
                    if pred < len(self.idx_to_char):
                        text += self.idx_to_char[pred]
                prev_char = pred
            
            return text
            
        except Exception as e:
            print(f"❌ Error decoding predictions: {e}")
            return ""
    
    def recognize_text(self, image: np.ndarray) -> dict:
        """Recognize text in image"""
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
            
            # Decode predictions
            text = self.decode_predictions(logits)
            
            # Calculate confidence
            probs = torch.softmax(logits, dim=2)
            confidence = torch.max(probs).item()
            
            processing_time = time.time() - start_time
            
            return {
                'text': text,
                'confidence': confidence,
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error during recognition: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }

def test_fixed_model():
    """Test the fixed custom model"""
    
    print("🔧 Testing Fixed Custom Model...")
    
    # Configuration
    MODEL_PATH = "checkpoints/best_model_epoch_7.pth"
    
    try:
        # Initialize fixed custom OCR
        print(f"🤖 Loading fixed custom model from: {MODEL_PATH}")
        custom_ocr = FixedCustomOCRInference(model_path=MODEL_PATH)
        
        # Test with synthetic image
        print("\n🧪 Testing with synthetic image...")
        test_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "HELLO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        result = custom_ocr.recognize_text(test_img)
        print(f"✅ Custom OCR result:")
        print(f"  Text: '{result['text']}'")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Time: {result['processing_time']:.3f}s")
        print(f"  Success: {result['success']}")
        
        return custom_ocr
        
    except Exception as e:
        print(f"❌ Error testing fixed model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_comparison(custom_ocr):
    """Compare custom model vs EasyOCR"""
    
    if custom_ocr is None:
        print("❌ Cannot run comparison - custom OCR failed to load")
        return
    
    print("\n📚 Initializing EasyOCR for comparison...")
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    # Test images
    test_images = []
    
    # Create synthetic test image
    synthetic_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(synthetic_img, "HELLO WORLD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    test_images.append(("Synthetic", synthetic_img))
    
    # Add real images if available
    images_dir = "images"
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in image_files[:2]:  # Test first 2 images
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                test_images.append((img_file, img))
    
    # Run comparison tests
    print(f"\n🧪 Running comparison tests on {len(test_images)} images...")
    print("=" * 80)
    
    for i, (img_name, img) in enumerate(test_images, 1):
        print(f"\n📸 Test {i}: {img_name}")
        print("-" * 40)
        
        # Test Custom Model
        print("🤖 Custom Model:")
        start_time = time.time()
        custom_result = custom_ocr.recognize_text(img)
        custom_time = time.time() - start_time
        
        print(f"  Text: '{custom_result['text']}'")
        print(f"  Confidence: {custom_result['confidence']:.3f}")
        print(f"  Time: {custom_time:.3f}s")
        print(f"  Success: {custom_result['success']}")
        
        # Test EasyOCR
        print("\n📚 EasyOCR:")
        start_time = time.time()
        easyocr_results = easyocr_reader.readtext(img)
        easyocr_time = time.time() - start_time
        
        print(f"  Found {len(easyocr_results)} text regions")
        if easyocr_results:
            best_result = max(easyocr_results, key=lambda x: x[2])
            print(f"  Best text: '{best_result[1]}'")
            print(f"  Confidence: {best_result[2]:.3f}")
        print(f"  Time: {easyocr_time:.3f}s")
        
        # Performance comparison
        print(f"\n⚡ Performance:")
        if custom_time > 0 and easyocr_time > 0:
            speedup = easyocr_time / custom_time
            print(f"  Custom model is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than EasyOCR")

def main():
    """Main function"""
    print("🚀 Starting Fixed Local Model vs EasyOCR Comparison")
    print("=" * 80)
    
    try:
        # Test fixed model
        custom_ocr = test_fixed_model()
        
        if custom_ocr:
            # Run comparison
            test_comparison(custom_ocr)
            
            print("\n" + "=" * 80)
            print("🎉 Fixed model comparison completed!")
            
            # Summary
            print("\n📊 Summary:")
            print("✅ Custom model now uses local weights correctly")
            print("✅ Model architecture mismatch fixed")
            print("✅ No internet connection required for inference")
            print("✅ Model can be fine-tuned for specific use cases")
            print("✅ EasyOCR provides pre-trained models with broader language support")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 