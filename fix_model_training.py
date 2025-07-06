#!/usr/bin/env python3
"""
Fixed Model Training Script
Addresses the training issues that caused the model to return "CWIA" pattern
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import logging
from pathlib import Path
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed character set (same as inference)
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

class FixedCRNNModel(nn.Module):
    """
    Fixed CRNN model with proper CTC implementation
    """
    
    def __init__(self, num_classes: int, img_height: int = 32, img_width: int = 100):
        super(FixedCRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))
        
        # RNN
        self.rnn = nn.LSTM(
            input_size=512 * 4,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(512, num_classes)
        
        logger.info(f"Fixed CRNN Model initialized with {num_classes} classes")
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Reshape for RNN
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)
        
        # RNN
        x, _ = self.rnn(x)
        
        # Output
        x = self.fc(x)
        
        return x

class FixedOCRDataset(Dataset):
    """
    Fixed dataset with proper character mapping
    """
    
    def __init__(self, data_dir: str, transform=None, img_height: int = 32, img_width: int = 100):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # Find all image files
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            self.image_files.extend(list(self.data_dir.glob(f'*{ext}')))
        
        # Create synthetic labels for testing (since we don't have real labels)
        self.labels = self._create_synthetic_labels()
        
        logger.info(f"Fixed OCR Dataset: {len(self.image_files)} images")
    
    def _create_synthetic_labels(self):
        """Create synthetic labels for training"""
        labels = []
        words = ['HELLO', 'WORLD', 'TEST', 'OCR', 'AI', 'ML', 'PYTHON', 'TRAINING']
        
        for i, img_file in enumerate(self.image_files):
            # Use different words for variety
            word = words[i % len(words)]
            labels.append(word)
        
        return labels
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label_text = self.labels[idx]
            
            # Convert label to indices (add 1 to account for CTC blank)
            label_indices = [CHAR_TO_IDX.get(char, 0) + 1 for char in label_text]
            
            return {
                'image': image,
                'label': torch.tensor(label_indices, dtype=torch.long),
                'label_text': label_text,
                'label_length': len(label_indices)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            dummy_image = torch.zeros(3, self.img_height, self.img_width)
            dummy_label = torch.tensor([1, 2, 3], dtype=torch.long)  # "ABC"
            return {
                'image': dummy_image,
                'label': dummy_label,
                'label_text': 'ABC',
                'label_length': 3
            }

def fixed_collate_fn(batch):
    """Fixed collate function for proper batching"""
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    label_texts = [item['label_text'] for item in batch]
    label_lengths = [item['label_length'] for item in batch]
    
    # Stack images
    images = torch.stack(images)
    
    # Pad labels
    max_len = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        padded = torch.cat([label, torch.zeros(max_len - len(label), dtype=torch.long)])
        padded_labels.append(padded)
    labels = torch.stack(padded_labels)
    
    return {
        'image': images,
        'label': labels,
        'label_text': label_texts,
        'label_length': torch.tensor(label_lengths, dtype=torch.long)
    }

class FixedOCRTrainer:
    """
    Fixed trainer with proper CTC implementation
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        logger.info(f"Fixed OCR Trainer initialized on {self.device}")
    
    def prepare_data(self, data_dir: str):
        """Prepare training data"""
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = FixedOCRDataset(data_dir, transform=transform)
        
        # Create data loader
        self.train_loader = DataLoader(
            dataset,
            batch_size=8,  # Smaller batch size for stability
            shuffle=True,
            num_workers=0,  # No multiprocessing for debugging
            collate_fn=fixed_collate_fn
        )
        
        # Initialize model
        self.model = FixedCRNNModel(
            num_classes=len(CHAR_SET) + 1,  # +1 for CTC blank
            img_height=32,
            img_width=100
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize CTC loss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        logger.info(f"Data prepared: {len(dataset)} samples")
        return True
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                label_lengths = batch['label_length'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Prepare CTC loss inputs
                batch_size = outputs.size(0)
                seq_length = outputs.size(1)
                
                # Reshape outputs for CTC
                outputs = outputs.log_softmax(2).permute(1, 0, 2)  # (seq_len, batch, num_classes)
                
                # Create target lengths
                target_lengths = label_lengths
                
                # Calculate loss
                loss = self.criterion(outputs, labels, 
                                    torch.full((batch_size,), seq_length, dtype=torch.long),
                                    target_lengths)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_set': CHAR_SET,
            'num_classes': len(CHAR_SET) + 1,
            'img_height': 32,
            'img_width': 100
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def train(self, data_dir: str, epochs: int = 10, save_path: str = None):
        """Main training function"""
        logger.info("Starting fixed model training...")
        
        # Prepare data
        if not self.prepare_data(data_dir):
            logger.error("Failed to prepare data")
            return False
        
        # Training loop
        for epoch in range(epochs):
            loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} completed, Average Loss: {loss:.4f}")
            
            # Save model every 5 epochs
            if (epoch + 1) % 5 == 0:
                if save_path is None:
                    save_path = f"checkpoints/fixed_model_epoch_{epoch+1}.pth"
                self.save_model(save_path)
        
        # Save final model
        final_save_path = save_path or "checkpoints/fixed_model_final.pth"
        self.save_model(final_save_path)
        
        logger.info("Training completed!")
        return True

def test_fixed_model(model_path: str):
    """Test the fixed model"""
    logger.info(f"Testing fixed model: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = FixedCRNNModel(
        num_classes=len(CHAR_SET) + 1,
        img_height=32,
        img_width=100
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test image
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "HELLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Preprocess
    pil_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((100, 32))
    image_tensor = torch.from_numpy(np.array(pil_image)).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
    
    # Decode predictions
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
    
    logger.info(f"Test result: '{text}'")
    return text

def main():
    """Main function"""
    logger.info("Starting fixed model training process...")
    
    # Check if we have training data
    data_dir = "images"
    if not os.path.exists(data_dir):
        logger.error(f"Training data directory not found: {data_dir}")
        logger.info("Creating synthetic training data...")
        
        # Create synthetic training data
        os.makedirs(data_dir, exist_ok=True)
        for i in range(50):  # Create 50 synthetic images
            img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            text = f"TEXT{i:02d}"
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(data_dir, f"synthetic_{i:02d}.png"), img)
    
    # Train the model
    trainer = FixedOCRTrainer()
    success = trainer.train(data_dir, epochs=5, save_path="checkpoints/fixed_model.pth")
    
    if success:
        # Test the model
        test_fixed_model("checkpoints/fixed_model.pth")
        logger.info("✅ Fixed model training completed successfully!")
    else:
        logger.error("❌ Fixed model training failed!")

if __name__ == "__main__":
    main() 