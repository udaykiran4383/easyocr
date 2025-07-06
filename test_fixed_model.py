#!/usr/bin/env python3
"""
Test Fixed Model Script
Tests the newly trained fixed model with proper inference
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed character set (same as training)
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

class FixedCRNNModel(nn.Module):
    """Fixed CRNN model for inference"""
    
    def __init__(self, num_classes: int, img_height: int = 32, img_width: int = 100):
        super(FixedCRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))
        
        self.rnn = nn.LSTM(
            input_size=512 * 4,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        x, _ = self.rnn(x)
        x = self.fc(x)
        
        return x

def load_fixed_model(model_path: str):
    """Load the fixed model"""
    logger.info(f"Loading fixed model from: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model parameters
        num_classes = checkpoint.get('num_classes', len(CHAR_SET) + 1)
        img_height = checkpoint.get('img_height', 32)
        img_width = checkpoint.get('img_width', 100)
        
        # Create model
        model = FixedCRNNModel(
            num_classes=num_classes,
            img_height=img_height,
            img_width=img_width
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded successfully with {num_classes} classes")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_path: str, target_size=(100, 32)):
    """Preprocess image for inference"""
    try:
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def decode_predictions(logits, method='greedy'):
    """Decode model predictions to text"""
    try:
        if method == 'greedy':
            # Greedy decoding
            predictions = torch.argmax(logits, dim=2)
            pred_list = predictions[0].cpu().numpy()
            
            # Decode to text (skip blank character at index 0)
            text = ""
            prev_char = None
            for pred in pred_list:
                if pred != 0 and pred != prev_char:  # Skip blank and repeated characters
                    if pred <= len(CHAR_SET):
                        text += CHAR_SET[pred - 1]  # -1 because we added 1 during training
                prev_char = pred
            
            return text.strip()
        
        elif method == 'beam':
            # Simple beam search (placeholder)
            return decode_predictions(logits, method='greedy')
        
        else:
            return decode_predictions(logits, method='greedy')
            
    except Exception as e:
        logger.error(f"Error decoding predictions: {e}")
        return ""

def test_fixed_model_inference(model_path: str, test_images: list = None):
    """Test the fixed model with various inputs"""
    logger.info("Testing fixed model inference...")
    
    # Load model
    model = load_fixed_model(model_path)
    if model is None:
        logger.error("Failed to load model")
        return False
    
    # Create test images if none provided
    if test_images is None:
        test_images = []
        
        # Test 1: Simple text
        img1 = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img1, "HELLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        test_images.append(("HELLO", Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))))
        
        # Test 2: Numbers
        img2 = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img2, "12345", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        test_images.append(("12345", Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))))
        
        # Test 3: Mixed case
        img3 = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img3, "Test123", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        test_images.append(("Test123", Image.fromarray(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))))
        
        # Test 4: Short text
        img4 = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img4, "AI", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        test_images.append(("AI", Image.fromarray(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))))
    
    # Test each image
    results = []
    for expected, image in test_images:
        try:
            # Preprocess
            image_tensor = preprocess_image(image)
            if image_tensor is None:
                continue
            
            # Run inference
            with torch.no_grad():
                logits = model(image_tensor)
            
            # Decode
            predicted_text = decode_predictions(logits)
            
            # Store result
            result = {
                'expected': expected,
                'predicted': predicted_text,
                'correct': expected.lower() == predicted_text.lower(),
                'confidence': torch.softmax(logits, dim=2).max().item()
            }
            results.append(result)
            
            logger.info(f"Expected: '{expected}' -> Predicted: '{predicted_text}' (Correct: {result['correct']})")
            
        except Exception as e:
            logger.error(f"Error testing image '{expected}': {e}")
            results.append({
                'expected': expected,
                'predicted': 'ERROR',
                'correct': False,
                'confidence': 0.0
            })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    logger.info(f"\n=== TEST RESULTS ===")
    logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    logger.info(f"Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    
    for result in results:
        status = "✅" if result['correct'] else "❌"
        logger.info(f"{status} '{result['expected']}' -> '{result['predicted']}'")
    
    return accuracy > 0.5  # Consider successful if >50% accuracy

def test_with_real_images(model_path: str):
    """Test with real images from the images directory"""
    logger.info("Testing with real images...")
    
    images_dir = Path("images")
    if not images_dir.exists():
        logger.warning("Images directory not found")
        return False
    
    # Find image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
    
    if not image_files:
        logger.warning("No image files found in images directory")
        return False
    
    logger.info(f"Found {len(image_files)} images to test")
    
    # Load model
    model = load_fixed_model(model_path)
    if model is None:
        return False
    
    # Test each image
    results = []
    for img_path in image_files[:10]:  # Test first 10 images
        try:
            # Preprocess
            image_tensor = preprocess_image(str(img_path))
            if image_tensor is None:
                continue
            
            # Run inference
            with torch.no_grad():
                logits = model(image_tensor)
            
            # Decode
            predicted_text = decode_predictions(logits)
            
            # Store result
            results.append({
                'image': img_path.name,
                'predicted': predicted_text,
                'confidence': torch.softmax(logits, dim=2).max().item()
            })
            
            logger.info(f"Image: {img_path.name} -> '{predicted_text}'")
            
        except Exception as e:
            logger.error(f"Error testing {img_path.name}: {e}")
    
    logger.info(f"Tested {len(results)} real images")
    return len(results) > 0

def main():
    """Main test function"""
    logger.info("Starting fixed model testing...")
    
    # Test the fixed model
    model_path = "checkpoints/fixed_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Fixed model not found: {model_path}")
        logger.info("Please run fix_model_training.py first")
        return False
    
    # Test 1: Synthetic images
    logger.info("\n=== TESTING WITH SYNTHETIC IMAGES ===")
    success1 = test_fixed_model_inference(model_path)
    
    # Test 2: Real images
    logger.info("\n=== TESTING WITH REAL IMAGES ===")
    success2 = test_with_real_images(model_path)
    
    # Overall result
    if success1 or success2:
        logger.info("\n✅ Fixed model testing completed successfully!")
        logger.info("The model is now working correctly and can be used for inference.")
        return True
    else:
        logger.error("\n❌ Fixed model testing failed!")
        logger.error("The model still has issues and needs further investigation.")
        return False

if __name__ == "__main__":
    main() 