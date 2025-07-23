#!/usr/bin/env python3
"""
Simple Custom Model Training Script
Uses actual images with correct labels from EasyOCR as ground truth
Simplified approach for small datasets
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import logging
from pathlib import Path
import time
import easyocr
import random
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Character set for OCR
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHAR_SET)}  # +1 because 0 is blank
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHAR_SET)}
BLANK_IDX = 0
NUM_CLASSES = len(CHAR_SET) + 1

class SimpleCRNNModel(nn.Module):
    """
    Simplified CRNN model for small datasets
    """
    
    def __init__(self, num_classes: int, img_height: int = 32, img_width: int = 128):
        super(SimpleCRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # Simplified CNN backbone
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Simple LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.classifier = nn.Linear(256, num_classes)  # 256 = 128*2 (bidirectional)
        
        logger.info(f"Simple CRNN Model initialized:")
        logger.info(f"  Input size: {img_height}x{img_width}")
        logger.info(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        # CNN feature extraction
        x = self.conv_blocks(x)  # (batch, 128, height, width)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # (batch, 128, 1, width)
        
        # Reshape for RNN
        batch_size, channels, height, width = x.size()
        x = x.squeeze(2)  # (batch, 128, width)
        x = x.permute(0, 2, 1)  # (batch, width, 128)
        
        # LSTM sequence modeling
        x, _ = self.lstm(x)  # (batch, width, 256)
        
        # Classification
        output = self.classifier(x)  # (batch, width, num_classes)
        
        return output

class SimpleImageDataset(Dataset):
    """
    Simple dataset using real images with EasyOCR labels
    """
    
    def __init__(self, image_dir: str, easyocr_reader, transform=None, 
                 img_height: int = 32, img_width: int = 128):
        self.image_dir = Path(image_dir)
        self.easyocr_reader = easyocr_reader
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # Find all image files
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
        
        # Get labels from EasyOCR
        self.data = self._prepare_data()
        
        logger.info(f"Simple Image Dataset: {len(self.data)} samples prepared")
    
    def _prepare_data(self) -> List[Dict]:
        """Prepare data by getting correct labels from EasyOCR"""
        data = []
        
        logger.info("Getting ground truth labels from EasyOCR...")
        for img_path in self.image_files:
            try:
                # Get EasyOCR prediction as ground truth
                results = self.easyocr_reader.readtext(str(img_path))
                
                if results:
                    # Use the highest confidence result
                    best_result = max(results, key=lambda x: x[2])
                    text = best_result[1].strip()
                    confidence = best_result[2]
                    
                    # Filter valid characters and ensure non-empty
                    filtered_text = ''.join([c for c in text if c in CHAR_SET])
                    
                    if filtered_text and confidence > 0.5:  # Only use high-confidence predictions
                        data.append({
                            'image_path': img_path,
                            'text': filtered_text,
                            'confidence': confidence
                        })
                        logger.info(f"  {img_path.name}: '{filtered_text}' (conf: {confidence:.3f})")
                    else:
                        logger.warning(f"  {img_path.name}: Skipped (low confidence or no valid text)")
                else:
                    logger.warning(f"  {img_path.name}: No text detected")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        if not data:
            raise ValueError("No valid training data found! Check image directory and EasyOCR setup.")
        
        # Simple augmentation by creating multiple versions with different transforms
        augmented_data = []
        for item in data:
            # Original
            augmented_data.append(item)
            
            # Create simple augmented versions
            for i in range(10):  # 10 augmented versions per original for more data
                augmented_data.append({
                    'image_path': item['image_path'],
                    'text': item['text'],
                    'confidence': item['confidence'],
                    'augment_id': i
                })
        
        logger.info(f"Dataset augmented: {len(data)} -> {len(augmented_data)} samples")
        return augmented_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            # Load image
            image = cv2.imread(str(item['image_path']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simple augmentation
            if 'augment_id' in item:
                # Apply simple random augmentations
                if random.random() > 0.5:
                    # Random brightness
                    brightness = random.uniform(0.8, 1.2)
                    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
                
                if random.random() > 0.5:
                    # Random contrast
                    contrast = random.uniform(0.8, 1.2)
                    image = np.clip((image - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
                
                if random.random() > 0.7:
                    # Add noise
                    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
                    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Resize to target size
            image = cv2.resize(image, (self.img_width, self.img_height))
            
            # Convert to PIL for transforms
            image = Image.fromarray(image)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Convert text to indices
            text = item['text']
            label_indices = [CHAR_TO_IDX.get(char, BLANK_IDX) for char in text]
            
            return {
                'image': image,
                'label': torch.tensor(label_indices, dtype=torch.long),
                'text': text,
                'length': len(label_indices),
                'confidence': item['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            dummy_image = torch.zeros(3, self.img_height, self.img_width)
            return {
                'image': dummy_image,
                'label': torch.tensor([1, 2, 3], dtype=torch.long),  # "123"
                'text': '123',
                'length': 3,
                'confidence': 1.0
            }

def simple_collate_fn(batch):
    """Simple collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    confidences = [item['confidence'] for item in batch]
    
    # Handle variable length labels
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Concatenate all labels for CTC loss
    concatenated_labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return {
        'images': images,
        'labels': concatenated_labels,
        'lengths': lengths,
        'texts': texts,
        'confidences': confidences
    }

class SimpleOCRTrainer:
    """
    Simple OCR trainer using real data
    """
    
    def __init__(self, img_height: int = 32, img_width: int = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_height = img_height
        self.img_width = img_width
        
        # Initialize EasyOCR for ground truth labels
        logger.info("Initializing EasyOCR for ground truth labeling...")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'))
        
        logger.info(f"Simple OCR Trainer initialized on {self.device}")
    
    def prepare_data(self, image_dir: str, batch_size: int = 8, val_split: float = 0.15):
        """Prepare training and validation data"""
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create full dataset
        full_dataset = SimpleImageDataset(
            image_dir=image_dir,
            easyocr_reader=self.easyocr_reader,
            transform=transform,
            img_height=self.img_height,
            img_width=self.img_width
        )
        
        # Split into train/val
        total_size = len(full_dataset)
        val_size = max(1, int(total_size * val_split))  # At least 1 validation sample
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=simple_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=simple_collate_fn
        )
        
        # Initialize model
        self.model = SimpleCRNNModel(
            num_classes=NUM_CLASSES,
            img_height=self.img_height,
            img_width=self.img_width
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        
        # Loss function
        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        
        logger.info(f"Data prepared: {train_size} train, {val_size} val samples")
        return True
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move to device
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                label_lengths = batch['lengths'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Prepare CTC inputs
                batch_size, seq_len, num_classes = outputs.shape
                
                # CTC expects (seq_len, batch, num_classes)
                outputs = outputs.log_softmax(2).transpose(0, 1)
                
                # Input lengths (all sequences have same length)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                
                # Calculate loss
                loss = self.criterion(outputs, labels, input_lengths, label_lengths)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                              f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Move to device
                    images = batch['images'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    label_lengths = batch['lengths'].to(self.device)
                    texts = batch['texts']
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate loss
                    batch_size, seq_len, num_classes = outputs.shape
                    outputs_for_loss = outputs.log_softmax(2).transpose(0, 1)
                    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                    
                    loss = self.criterion(outputs_for_loss, labels, input_lengths, label_lengths)
                    total_loss += loss.item()
                    
                    # Decode predictions for accuracy
                    predictions = torch.argmax(outputs, dim=2)
                    
                    for i, (pred_seq, gt_text) in enumerate(zip(predictions, texts)):
                        # Decode prediction
                        pred_text = self._decode_prediction(pred_seq)
                        
                        # Compare with ground truth (case insensitive)
                        if pred_text.lower() == gt_text.lower():
                            correct_predictions += 1
                        total_predictions += 1
                        
                        logger.info(f"Val example: '{pred_text}' vs '{gt_text}' -> {'✅' if pred_text.lower() == gt_text.lower() else '❌'}")
                    
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _decode_prediction(self, pred_seq):
        """Decode prediction sequence to text"""
        text = ""
        prev_char = None
        
        for pred_idx in pred_seq:
            pred_idx = pred_idx.item()
            if pred_idx != BLANK_IDX and pred_idx != prev_char:
                if pred_idx in IDX_TO_CHAR:
                    text += IDX_TO_CHAR[pred_idx]
            prev_char = pred_idx
        
        return text.strip()
    
    def save_model(self, save_path: str, epoch: int, loss: float, accuracy: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'char_set': CHAR_SET,
            'num_classes': NUM_CLASSES,
            'img_height': self.img_height,
            'img_width': self.img_width
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved: {save_path} (epoch {epoch+1}, loss: {loss:.4f}, acc: {accuracy:.3f})")
    
    def train(self, image_dir: str, epochs: int = 50, save_dir: str = "checkpoints"):
        """Main training function"""
        logger.info(f"Starting simple OCR training for {epochs} epochs...")
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Prepare data
        if not self.prepare_data(image_dir):
            logger.error("Failed to prepare data")
            return False
        
        best_accuracy = 0.0
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.3f}")
            
            # Save best model based on accuracy (or loss if accuracy is 0)
            is_best = False
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_loss = val_loss
                patience_counter = 0
                is_best = True
            elif val_accuracy == best_accuracy and val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
            
            if is_best:
                # Save best model
                best_model_path = save_dir / "best_simple_model.pth"
                self.save_model(best_model_path, epoch, val_loss, val_accuracy)
                logger.info(f"🎉 New best model! Accuracy: {val_accuracy:.3f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_dir / f"simple_model_epoch_{epoch+1}.pth"
                self.save_model(checkpoint_path, epoch, val_loss, val_accuracy)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best accuracy: {best_accuracy:.3f}")
        logger.info(f"Best model saved: {save_dir}/best_simple_model.pth")
        
        return True

def test_simple_model(model_path: str, test_image_dir: str):
    """Test the simple trained model"""
    logger.info(f"Testing simple model: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleCRNNModel(
        num_classes=checkpoint['num_classes'],
        img_height=checkpoint['img_height'],
        img_width=checkpoint['img_width']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize EasyOCR for comparison
    easyocr_reader = easyocr.Reader(['en'])
    
    # Test images
    test_dir = Path(test_image_dir)
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(list(test_dir.glob(f'*{ext}')))
    
    logger.info(f"Testing on {len(image_files)} images...")
    
    correct = 0
    total = 0
    
    # Test transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for img_path in image_files:
        try:
            # Load and preprocess image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (checkpoint['img_width'], checkpoint['img_height']))
            
            pil_image = Image.fromarray(image)
            image_tensor = transform(pil_image).unsqueeze(0)
            
            # Model prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                predictions = torch.argmax(outputs, dim=2)
                pred_text = ""
                prev_char = None
                
                for pred_idx in predictions[0]:
                    pred_idx = pred_idx.item()
                    if pred_idx != BLANK_IDX and pred_idx != prev_char:
                        if pred_idx in IDX_TO_CHAR:
                            pred_text += IDX_TO_CHAR[pred_idx]
                    prev_char = pred_idx
            
            # EasyOCR ground truth
            easyocr_results = easyocr_reader.readtext(str(img_path))
            gt_text = easyocr_results[0][1] if easyocr_results else ""
            gt_text = ''.join([c for c in gt_text if c in CHAR_SET])
            
            # Compare
            is_correct = pred_text.lower() == gt_text.lower()
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            logger.info(f"{status} {img_path.name}: '{pred_text}' vs '{gt_text}'")
            
        except Exception as e:
            logger.error(f"Error testing {img_path}: {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"\nTest Results: {correct}/{total} = {accuracy:.3f} accuracy")
    return accuracy

def main():
    """Main function"""
    logger.info("🚀 Starting Simple Custom Model Training")
    logger.info("=" * 60)
    
    # Configuration
    IMAGE_DIR = "images"
    EPOCHS = 50
    SAVE_DIR = "checkpoints"
    
    # Check if images directory exists
    if not Path(IMAGE_DIR).exists():
        logger.error(f"Image directory not found: {IMAGE_DIR}")
        return
    
    # Train the model
    trainer = SimpleOCRTrainer(img_height=32, img_width=128)
    
    success = trainer.train(
        image_dir=IMAGE_DIR,
        epochs=EPOCHS,
        save_dir=SAVE_DIR
    )
    
    if success:
        # Test the model
        best_model_path = Path(SAVE_DIR) / "best_simple_model.pth"
        if best_model_path.exists():
            accuracy = test_simple_model(str(best_model_path), IMAGE_DIR)
            
            if accuracy > 0.5:
                logger.info("🎉 SUCCESS! Model achieves >50% accuracy!")
                logger.info("✅ The custom model is now working correctly!")
            else:
                logger.warning(f"⚠️ Model accuracy is {accuracy:.3f} - improving but needs more training")
        else:
            logger.error("Best model not found for testing")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main() 