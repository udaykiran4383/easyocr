#!/usr/bin/env python3
"""
Mac-Friendly OCR Model Training Script
Optimized for MacBook to prevent overheating while training a working OCR model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import string
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Mac-friendly settings
BATCH_SIZE = 8  # Smaller batch size
LEARNING_RATE = 0.001
EPOCHS = 20  # Fewer epochs
IMG_HEIGHT = 32
IMG_WIDTH = 100
NUM_CLASSES = 37  # A-Z, 0-9, space

class SimpleOCRDataset(Dataset):
    """Simple dataset with basic text generation"""
    
    def __init__(self, num_samples=1000, img_height=32, img_width=100):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        
        # Simple character set: A-Z, 0-9, space
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # Generate simple training data
        self.data = []
        for _ in range(num_samples):
            # Generate random text (1-8 characters)
            length = random.randint(1, 8)
            text = ''.join(random.choices(self.chars, k=length)).strip()
            if text:  # Only add non-empty text
                self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Create simple synthetic image
        img = Image.new('L', (self.img_width, self.img_height), 255)
        draw = ImageDraw.Draw(img)
        
        # Use a simple font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((5, 5), text, fill=0, font=font)
        
        # Convert to tensor
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        
        # Create target sequence
        target = [self.char_to_idx[char] for char in text]
        target = torch.tensor(target, dtype=torch.long)
        
        return img_tensor, target, text

class MacFriendlyOCRModel(nn.Module):
    """Simplified OCR model optimized for Mac training"""
    
    def __init__(self, num_classes, img_height=32, img_width=100):
        super(MacFriendlyOCRModel, self).__init__()
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # Simplified CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, img_width // 4))
        )
        
        # RNN for sequence modeling
        self.rnn = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.1)
        
        # Output layer
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)  # (batch, 128, 1, width//4)
        x = x.squeeze(2)  # (batch, 128, width//4)
        x = x.permute(0, 2, 1)  # (batch, width//4, 128)
        
        # RNN processing
        x, _ = self.rnn(x)
        
        # Classification
        x = self.classifier(x)  # (batch, width//4, num_classes)
        
        return x

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    texts = [item[2] for item in batch]
    
    # Stack images
    images = torch.stack(images)
    
    return images, targets, texts

def train_model():
    """Train the Mac-friendly OCR model"""
    logger.info("🚀 Starting Mac-friendly OCR training...")
    
    # Create datasets
    train_dataset = SimpleOCRDataset(num_samples=500, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    val_dataset = SimpleOCRDataset(num_samples=100, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = MacFriendlyOCRModel(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH).to(device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        logger.info(f"📚 Epoch {epoch+1}/{EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, (images, targets, texts) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Prepare CTC inputs
            batch_size = outputs.size(0)
            seq_len = outputs.size(1)
            
            # Reshape outputs for CTC
            outputs = outputs.log_softmax(2).permute(1, 0, 2)  # (seq_len, batch, num_classes)
            
            # Prepare targets
            target_lengths = torch.tensor([len(target) for target in targets])
            target_concat = torch.cat(targets)
            
            # CTC loss
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            
            loss = criterion(outputs, target_concat, input_lengths, target_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Add small delay to prevent overheating
            if batch_idx % 10 == 0:
                time.sleep(0.1)
                logger.info(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets, texts in val_loader:
                images = images.to(device)
                
                outputs = model(images)
                outputs = outputs.log_softmax(2).permute(1, 0, 2)
                
                batch_size = outputs.size(1)
                seq_len = outputs.size(0)
                
                target_lengths = torch.tensor([len(target) for target in targets])
                target_concat = torch.cat(targets)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
                
                loss = criterion(outputs, target_concat, input_lengths, target_lengths)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        logger.info(f"📊 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'num_classes': NUM_CLASSES,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH
            }, 'checkpoints/mac_friendly_model.pth')
            logger.info(f"💾 Saved best model (loss: {best_loss:.4f})")
        
        # Add delay between epochs to cool down
        time.sleep(1)
    
    logger.info("✅ Training completed!")
    return model

def test_model(model_path='checkpoints/mac_friendly_model.pth'):
    """Test the trained model"""
    logger.info("🧪 Testing Mac-friendly model...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MacFriendlyOCRModel(
        checkpoint['num_classes'], 
        checkpoint['img_height'], 
        checkpoint['img_width']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test with simple examples
    test_texts = ['HELLO', '12345', 'AI', 'TEST', 'PYTHON']
    
    for text in test_texts:
        # Create test image
        img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 255)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((5, 5), text, fill=0, font=font)
        
        # Convert to tensor
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = outputs.log_softmax(2)
            
            # Decode
            _, predicted = torch.max(outputs, 2)
            predicted = predicted.squeeze().cpu().numpy()
            
            # Convert to text
            chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            predicted_text = ''.join([chars[idx] for idx in predicted if idx < len(chars)])
            predicted_text = predicted_text.strip()
            
            logger.info(f"Expected: '{text}' -> Predicted: '{predicted_text}'")

if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train model
    model = train_model()
    
    # Test model
    test_model()
    
    logger.info("🎉 Mac-friendly training completed successfully!")
    logger.info("💡 This model is optimized for MacBook performance and should not cause overheating.") 