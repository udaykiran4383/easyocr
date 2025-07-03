#!/usr/bin/env python3
"""
Custom OCR Inference Script

This script uses the trained custom CRNN model for text recognition,
independent of EasyOCR. It can be used as a replacement or complement
to EasyOCR for specific use cases.
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

from model_trainer import CRNNModel
from config import MODEL_CONFIG, PATHS

logger = logging.getLogger(__name__)


class CustomOCRInference:
    """
    Custom OCR inference using trained CRNN model
    """
    
    def __init__(self, model_path: str, char_set_path: str = None, device: str = None):
        """
        Initialize custom OCR inference
        
        Args:
            model_path: Path to trained model checkpoint
            char_set_path: Path to character set file
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load character set
        if char_set_path is None:
            char_set_path = Path(PATHS['results']) / 'character_set.txt'
        
        self.char_set = self._load_char_set(char_set_path)
        
        # Add padding characters to match the trained model (40 classes)
        # The model was trained with 40 classes, so we need to pad the character set
        while len(self.char_set) < 40:
            self.char_set += '_'  # Add padding character
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.char_set)}
        
        # Load model
        self.model = self._load_model(model_path)
        
        logger.info(f"Custom OCR initialized with {len(self.char_set)} characters")
        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Device: {self.device}")
    
    def _load_char_set(self, char_set_path: str) -> str:
        """Load character set from file"""
        try:
            with open(char_set_path, 'r', encoding='utf-8') as f:
                char_set = f.read().strip()
            logger.info(f"Loaded character set: {char_set}")
            return char_set
        except Exception as e:
            logger.error(f"Error loading character set: {e}")
            # Fallback to basic ASCII
            return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    def _load_model(self, model_path: str) -> CRNNModel:
        """Load trained model"""
        try:
            # Initialize model
            model = CRNNModel(
                num_classes=len(self.char_set),
                img_height=MODEL_CONFIG['imgH'],
                img_width=MODEL_CONFIG['imgW']
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image).convert('RGB')
        
        # Resize
        pil_image = pil_image.resize((MODEL_CONFIG['imgW'], MODEL_CONFIG['imgH']))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(pil_image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW and normalize
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def decode_predictions(self, logits: torch.Tensor) -> str:
        """
        Decode model predictions to text
        
        Args:
            logits: Model output logits
            
        Returns:
            Decoded text string
        """
        # Get predictions (argmax)
        predictions = torch.argmax(logits, dim=2)
        
        # Convert to list
        pred_list = predictions[0].cpu().numpy()
        
        # Decode to text
        text = ""
        prev_char = None
        for pred in pred_list:
            if pred != 0 and pred != prev_char:  # Skip blank and repeated characters
                if pred < len(self.idx_to_char):
                    text += self.idx_to_char[pred]
            prev_char = pred
        
        return text
    
    def recognize_text(self, image: np.ndarray) -> Dict:
        """
        Recognize text in image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with recognition results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(input_tensor)
            
            # Decode predictions
            text = self.decode_predictions(logits)
            
            # Calculate confidence (simplified)
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
            logger.error(f"Error during recognition: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def recognize_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Recognize text in multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of recognition results
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.recognize_text(image)
            results.append(result)
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom OCR Inference')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--char-set', help='Path to character set file')
    
    args = parser.parse_args()
    
    # Initialize custom OCR
    custom_ocr = CustomOCRInference(
        model_path=args.model,
        char_set_path=args.char_set
    )
    
    # Load and process image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    # Recognize text
    result = custom_ocr.recognize_text(image)
    
    # Print results
    print(f"Recognized text: {result['text']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Processing time: {result['processing_time']:.3f}s")
    print(f"Success: {result['success']}")


if __name__ == "__main__":
    main() 