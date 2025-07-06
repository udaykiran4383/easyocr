#!/usr/bin/env python3
"""
Debug Fixed Model Script
Detailed debugging to understand why the model returns empty strings
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

# Fixed character set
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

class FixedCRNNModel(nn.Module):
    """Fixed CRNN model for debugging"""
    
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
        
        logger.info(f"Model parameters: num_classes={num_classes}, img_height={img_height}, img_width={img_width}")
        
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

def create_test_image(text: str):
    """Create a test image with text"""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def preprocess_image(image, target_size=(100, 32)):
    """Preprocess image for inference"""
    try:
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
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

def debug_model_outputs(model, image_tensor, expected_text: str):
    """Debug model outputs step by step"""
    logger.info(f"\n=== DEBUGGING MODEL OUTPUTS FOR '{expected_text}' ===")
    
    try:
        # Forward pass
        with torch.no_grad():
            logits = model(image_tensor)
        
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
        
        # Get predictions
        predictions = torch.argmax(logits, dim=2)
        pred_list = predictions[0].cpu().numpy()
        
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Prediction indices: {pred_list}")
        logger.info(f"Unique predictions: {np.unique(pred_list)}")
        
        # Get probabilities
        probs = torch.softmax(logits, dim=2)
        max_probs = torch.max(probs, dim=2)[0]
        
        logger.info(f"Max probabilities: {max_probs[0].cpu().numpy()}")
        logger.info(f"Average confidence: {max_probs.mean().item():.4f}")
        
        # Check for blank predictions (index 0)
        blank_count = np.sum(pred_list == 0)
        logger.info(f"Blank predictions: {blank_count}/{len(pred_list)} ({blank_count/len(pred_list)*100:.1f}%)")
        
        # Try different decoding strategies
        logger.info("\n--- DECODING STRATEGIES ---")
        
        # Strategy 1: Basic decoding
        text1 = ""
        for pred in pred_list:
            if pred != 0 and pred <= len(CHAR_SET):
                text1 += CHAR_SET[pred - 1]
        logger.info(f"Strategy 1 (basic): '{text1}'")
        
        # Strategy 2: Remove repeated characters
        text2 = ""
        prev_char = None
        for pred in pred_list:
            if pred != 0 and pred != prev_char and pred <= len(CHAR_SET):
                text2 += CHAR_SET[pred - 1]
            prev_char = pred
        logger.info(f"Strategy 2 (no repeats): '{text2}'")
        
        # Strategy 3: Threshold-based
        text3 = ""
        threshold = 0.5
        for i, pred in enumerate(pred_list):
            if pred != 0 and max_probs[0][i] > threshold and pred <= len(CHAR_SET):
                text3 += CHAR_SET[pred - 1]
        logger.info(f"Strategy 3 (threshold {threshold}): '{text3}'")
        
        # Strategy 4: Top-k decoding
        text4 = ""
        top_k = 3
        for i in range(logits.size(1)):  # For each time step
            top_probs, top_indices = torch.topk(probs[0, i], top_k)
            for j in range(top_k):
                if top_indices[j] != 0 and top_probs[j] > 0.1:  # Skip blank and low confidence
                    if top_indices[j] <= len(CHAR_SET):
                        text4 += CHAR_SET[top_indices[j] - 1]
                    break
        logger.info(f"Strategy 4 (top-{top_k}): '{text4}'")
        
        # Strategy 5: Raw indices (for debugging)
        text5 = ""
        for pred in pred_list:
            if pred != 0:
                text5 += f"{pred} "
        logger.info(f"Strategy 5 (raw indices): '{text5.strip()}'")
        
        return {
            'predictions': pred_list,
            'probabilities': max_probs[0].cpu().numpy(),
            'texts': [text1, text2, text3, text4],
            'blank_ratio': blank_count / len(pred_list)
        }
        
    except Exception as e:
        logger.error(f"Error in debug_model_outputs: {e}")
        return None

def analyze_model_behavior(model_path: str):
    """Analyze the model's behavior with different inputs"""
    logger.info("=== ANALYZING MODEL BEHAVIOR ===")
    
    # Load model
    model = load_fixed_model(model_path)
    if model is None:
        return False
    
    # Test with different inputs
    test_cases = [
        ("HELLO", "Simple text"),
        ("12345", "Numbers"),
        ("AI", "Short text"),
        ("TEST", "All caps"),
        ("test", "All lowercase"),
        ("Test123", "Mixed case and numbers")
    ]
    
    results = []
    for text, description in test_cases:
        logger.info(f"\n--- Testing: {text} ({description}) ---")
        
        # Create test image
        test_image = create_test_image(text)
        
        # Preprocess
        image_tensor = preprocess_image(test_image)
        if image_tensor is None:
            continue
        
        # Debug outputs
        result = debug_model_outputs(model, image_tensor, text)
        if result:
            results.append({
                'text': text,
                'description': description,
                'result': result
            })
    
    # Summary analysis
    logger.info("\n=== SUMMARY ANALYSIS ===")
    
    if results:
        # Check if all outputs are empty
        all_empty = all(all(not text for text in r['result']['texts']) for r in results)
        logger.info(f"All outputs empty: {all_empty}")
        
        # Check blank ratio
        avg_blank_ratio = np.mean([r['result']['blank_ratio'] for r in results])
        logger.info(f"Average blank ratio: {avg_blank_ratio:.3f}")
        
        # Check if model is predicting mostly blanks
        if avg_blank_ratio > 0.8:
            logger.warning("Model is predicting mostly blank characters!")
            logger.warning("This suggests a training issue with CTC loss or character mapping.")
        
        # Check prediction distribution
        all_predictions = []
        for r in results:
            all_predictions.extend(r['result']['predictions'])
        
        unique_preds = np.unique(all_predictions)
        logger.info(f"Unique prediction indices: {unique_preds}")
        
        if len(unique_preds) <= 3:
            logger.warning("Model has very limited prediction diversity!")
            logger.warning("This suggests overfitting or training issues.")
    
    return True

def check_training_data():
    """Check the training data and process"""
    logger.info("\n=== CHECKING TRAINING DATA ===")
    
    # Check if synthetic data was created
    images_dir = Path("images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png"))
        logger.info(f"Found {len(image_files)} training images")
        
        if image_files:
            # Check a few images
            for i, img_path in enumerate(image_files[:3]):
                logger.info(f"Training image {i+1}: {img_path.name}")
    else:
        logger.warning("No training images found")
    
    # Check character set consistency
    logger.info(f"Character set: {CHAR_SET}")
    logger.info(f"Character set length: {len(CHAR_SET)}")
    logger.info(f"Expected model classes: {len(CHAR_SET) + 1} (including blank)")
    
    # Check character mapping
    logger.info("Character to index mapping (first 10):")
    for i, char in enumerate(CHAR_SET[:10]):
        logger.info(f"  '{char}' -> {CHAR_TO_IDX[char]}")

def main():
    """Main debugging function"""
    logger.info("Starting fixed model debugging...")
    
    model_path = "checkpoints/fixed_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Fixed model not found: {model_path}")
        logger.info("Please run fix_model_training.py first")
        return False
    
    # Check training data
    check_training_data()
    
    # Analyze model behavior
    success = analyze_model_behavior(model_path)
    
    if success:
        logger.info("\n✅ Debugging completed!")
        logger.info("Check the output above for insights into the model's behavior.")
    else:
        logger.error("\n❌ Debugging failed!")
    
    return success

if __name__ == "__main__":
    main() 