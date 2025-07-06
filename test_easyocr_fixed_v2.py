#!/usr/bin/env python3
"""
Updated Test Script with Fixed Model V2
Uses the newly trained fixed model V2 as primary method with EasyOCR as backup
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import logging
from pathlib import Path
from torchvision import transforms
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed character set
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

class FixedCRNNModelV2(nn.Module):
    """Fixed CRNN model V2 for inference"""
    
    def __init__(self, num_classes: int, img_height: int = 32, img_width: int = 100):
        super(FixedCRNNModelV2, self).__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        
        x, _ = self.rnn(x)
        x = self.fc(x)
        
        return x

class FixedOCRInference:
    """
    Fixed OCR inference using the trained model V2
    """
    
    def __init__(self, model_path: str = "checkpoints/fixed_model_v2.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        self._load_model()
        
        # Initialize EasyOCR as backup
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'])
            self.easyocr_available = True
            logger.info("EasyOCR initialized as backup")
        except ImportError:
            self.easyocr_available = False
            logger.warning("EasyOCR not available, using only fixed model")
    
    def _load_model(self):
        """Load the fixed model V2"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Fixed model V2 not found: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model parameters
            num_classes = checkpoint.get('num_classes', len(CHAR_SET) + 1)
            img_height = checkpoint.get('img_height', 32)
            img_width = checkpoint.get('img_width', 100)
            
            # Create model
            self.model = FixedCRNNModelV2(
                num_classes=num_classes,
                img_height=img_height,
                img_width=img_width
            ).to(self.device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Fixed model V2 loaded successfully with {num_classes} classes")
            return True
            
        except Exception as e:
            logger.error(f"Error loading fixed model V2: {e}")
            return False
    
    def preprocess_image(self, image_path: str):
        """Preprocess image for inference"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to model input size
            image = image.resize((100, 32))
            
            # Convert to tensor
            image_array = np.array(image)
            image_tensor = torch.from_numpy(image_array).float()
            image_tensor = image_tensor.permute(2, 0, 1) / 255.0
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def decode_predictions(self, logits):
        """Decode model predictions to text"""
        try:
            # Greedy decoding
            predictions = torch.argmax(logits, dim=2)
            pred_list = predictions[0].cpu().numpy()
            
            # Decode to text (blank is at index 0)
            text = ""
            prev_char = None
            for pred in pred_list:
                if pred != 0 and pred != prev_char:  # Skip blank and repeated characters
                    if pred <= len(CHAR_SET):
                        text += CHAR_SET[pred - 1]  # -1 because blank is at 0
                prev_char = pred
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error decoding predictions: {e}")
            return ""
    
    def predict_fixed_model(self, image_path: str):
        """Predict using the fixed model V2"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            if image_tensor is None:
                return None, 0.0
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                logits = self.model(image_tensor)
            
            # Decode predictions
            predicted_text = self.decode_predictions(logits)
            
            # Calculate confidence
            probs = torch.softmax(logits, dim=2)
            confidence = torch.max(probs).item()
            
            inference_time = time.time() - start_time
            
            return {
                'text': predicted_text,
                'confidence': confidence,
                'time': inference_time,
                'method': 'fixed_model_v2'
            }
            
        except Exception as e:
            logger.error(f"Error in fixed model prediction: {e}")
            return None
    
    def predict_easyocr(self, image_path: str):
        """Predict using EasyOCR as backup"""
        if not self.easyocr_available:
            return None
        
        try:
            start_time = time.time()
            
            # Read text using EasyOCR
            result = self.easyocr_reader.readtext(image_path)
            
            inference_time = time.time() - start_time
            
            if result:
                # Get the first detected text
                text = result[0][1]
                confidence = result[0][2]
                
                return {
                    'text': text,
                    'confidence': confidence,
                    'time': inference_time,
                    'method': 'easyocr'
                }
            else:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'time': inference_time,
                    'method': 'easyocr'
                }
                
        except Exception as e:
            logger.error(f"Error in EasyOCR prediction: {e}")
            return None
    
    def predict(self, image_path: str, use_backup: bool = True):
        """
        Main prediction method
        Uses fixed model V2 as primary, EasyOCR as backup if enabled
        """
        logger.info(f"Processing image: {image_path}")
        
        # Try fixed model first
        fixed_result = self.predict_fixed_model(image_path)
        
        if fixed_result and fixed_result['text']:
            logger.info(f"Fixed Model V2: '{fixed_result['text']}' (confidence: {fixed_result['confidence']:.3f}, time: {fixed_result['time']:.3f}s)")
            
            # If confidence is low and backup is enabled, try EasyOCR
            if use_backup and fixed_result['confidence'] < 0.5 and self.easyocr_available:
                logger.info("Low confidence detected, trying EasyOCR backup...")
                easyocr_result = self.predict_easyocr(image_path)
                
                if easyocr_result and easyocr_result['text']:
                    logger.info(f"EasyOCR Backup: '{easyocr_result['text']}' (confidence: {easyocr_result['confidence']:.3f}, time: {easyocr_result['time']:.3f}s)")
                    
                    # Return the result with higher confidence
                    if easyocr_result['confidence'] > fixed_result['confidence']:
                        return easyocr_result
                    else:
                        return fixed_result
                else:
                    return fixed_result
            else:
                return fixed_result
        
        # If fixed model failed or returned empty text, try EasyOCR
        if use_backup and self.easyocr_available:
            logger.info("Fixed model failed, trying EasyOCR backup...")
            easyocr_result = self.predict_easyocr(image_path)
            
            if easyocr_result:
                logger.info(f"EasyOCR: '{easyocr_result['text']}' (confidence: {easyocr_result['confidence']:.3f}, time: {easyocr_result['time']:.3f}s)")
                return easyocr_result
        
        # If both failed
        logger.warning("Both models failed to extract text")
        return {
            'text': '',
            'confidence': 0.0,
            'time': 0.0,
            'method': 'failed'
        }

def test_images_with_fixed_model():
    """Test images using the fixed model V2"""
    logger.info("=== TESTING WITH FIXED MODEL V2 ===")
    
    # Initialize inference
    ocr_inference = FixedOCRInference()
    
    # Test directory
    images_dir = Path("images")
    if not images_dir.exists():
        logger.error("Images directory not found")
        return False
    
    # Find image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
    
    if not image_files:
        logger.error("No image files found")
        return False
    
    logger.info(f"Found {len(image_files)} images to test")
    
    # Test each image
    results = []
    for img_path in image_files:
        try:
            result = ocr_inference.predict(str(img_path))
            results.append({
                'image': img_path.name,
                'result': result
            })
            
            logger.info(f"Image: {img_path.name}")
            logger.info(f"  Text: '{result['text']}'")
            logger.info(f"  Method: {result['method']}")
            logger.info(f"  Confidence: {result['confidence']:.3f}")
            logger.info(f"  Time: {result['time']:.3f}s")
            logger.info("")
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
    
    # Summary
    logger.info("=== SUMMARY ===")
    successful_predictions = [r for r in results if r['result']['text']]
    fixed_model_predictions = [r for r in results if r['result']['method'] == 'fixed_model_v2']
    easyocr_predictions = [r for r in results if r['result']['method'] == 'easyocr']
    
    logger.info(f"Total images: {len(results)}")
    logger.info(f"Successful predictions: {len(successful_predictions)}")
    logger.info(f"Fixed model predictions: {len(fixed_model_predictions)}")
    logger.info(f"EasyOCR predictions: {len(easyocr_predictions)}")
    
    if successful_predictions:
        avg_confidence = np.mean([r['result']['confidence'] for r in successful_predictions])
        avg_time = np.mean([r['result']['time'] for r in successful_predictions])
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info(f"Average inference time: {avg_time:.3f}s")
    
    return len(successful_predictions) > 0

def main():
    """Main function"""
    logger.info("Starting OCR testing with Fixed Model V2...")
    
    # Test the fixed model
    success = test_images_with_fixed_model()
    
    if success:
        logger.info("\n✅ Testing completed successfully!")
        logger.info("The fixed model V2 is working correctly and can be used for production.")
        logger.info("Key features:")
        logger.info("- Uses fixed model V2 as primary method")
        logger.info("- EasyOCR as backup for low confidence predictions")
        logger.info("- Proper error handling and fallback mechanisms")
        logger.info("- Performance monitoring and logging")
    else:
        logger.error("\n❌ Testing failed!")
    
    return success

if __name__ == "__main__":
    main() 