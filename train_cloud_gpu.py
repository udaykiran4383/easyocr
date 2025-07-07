#!/usr/bin/env python3
"""
Cloud GPU OCR Model Training Script
Optimized for cloud GPU platforms (Google Colab, AWS, etc.)
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
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cloud-optimized settings
BATCH_SIZE = 64  # Larger batch size for GPU
LEARNING_RATE = 0.001
EPOCHS = 100  # More epochs for better accuracy
IMG_HEIGHT = 32
IMG_WIDTH = 128
NUM_CLASSES = 63  # A-Z, a-z, 0-9, space, special chars

class CloudOCRDataset(Dataset):
    """Advanced dataset with extensive data augmentation"""
    
    def __init__(self, num_samples=10000, img_height=32, img_width=128, is_training=True):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.is_training = is_training
        
        # Extended character set
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # Generate diverse training data
        self.data = []
        
        # Common words and patterns
        common_words = [
            'HELLO', 'WORLD', 'PYTHON', 'OCR', 'AI', 'ML', 'TEST', 'DEMO',
            '12345', '67890', 'ABCDE', 'FGHIJ', 'KLMNO', 'PQRST', 'UVWXY', 'Z',
            'hello', 'world', 'python', 'test', 'demo', 'ai', 'ml', 'ocr',
            'Hello', 'World', 'Python', 'Test', 'Demo', 'AI', 'ML', 'OCR'
        ]
        
        # Add common words
        for word in common_words:
            self.data.append(word)
        
        # Generate random text
        for _ in range(num_samples - len(common_words)):
            length = random.randint(1, 12)
            text = ''.join(random.choices(self.chars, k=length)).strip()
            if text and len(text) > 0:
                self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def augment_image(self, img):
        """Apply data augmentation"""
        if not self.is_training:
            return img
        
        # Convert PIL to numpy for OpenCV operations
        img_np = np.array(img)
        
        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            height, width = img_np.shape
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_np = cv2.warpAffine(img_np, rotation_matrix, (width, height), 
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, img_np.shape).astype(np.uint8)
            img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random brightness/contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-10, 10)    # Brightness
            img_np = np.clip(alpha * img_np + beta, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Create synthetic image
        img = Image.new('L', (self.img_width, self.img_height), 255)
        draw = ImageDraw.Draw(img)
        
        # Try different fonts
        font_size = random.randint(16, 24) if self.is_training else 20
        try:
            # Try to use a system font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (self.img_width - text_width) // 2
        y = (self.img_height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill=0, font=font)
        
        # Apply augmentation
        img = self.augment_image(img)
        
        # Convert to tensor
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        
        # Create target sequence
        target = [self.char_to_idx[char] for char in text]
        target = torch.tensor(target, dtype=torch.long)
        
        return img_tensor, target, text

class CloudOCRModel(nn.Module):
    """Advanced OCR model for cloud GPU training"""
    
    def __init__(self, num_classes, img_height=32, img_width=128):
        super(CloudOCRModel, self).__init__()
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # Enhanced CNN backbone
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, img_width // 8))
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, 
                           bidirectional=True, dropout=0.1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, num_heads=8, dropout=0.1)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = self.features(x)  # (batch, 256, 1, width//8)
        x = x.squeeze(2)  # (batch, 256, width//8)
        x = x.permute(0, 2, 1)  # (batch, width//8, 256)
        
        # LSTM processing
        x, _ = self.lstm(x)  # (batch, width//8, 512)
        
        # Self-attention
        x = x.permute(1, 0, 2)  # (width//8, batch, 512)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.permute(1, 0, 2)  # (batch, width//8, 512)
        
        # Classification
        x = self.classifier(x)  # (batch, width//8, num_classes)
        
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
    """Train the cloud OCR model"""
    logger.info("🚀 Starting Cloud GPU OCR training...")
    
    # Create datasets
    train_dataset = CloudOCRDataset(num_samples=20000, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, is_training=True)
    val_dataset = CloudOCRDataset(num_samples=2000, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Create model
    model = CloudOCRModel(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH).to(device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        logger.info(f"📚 Epoch {epoch+1}/{EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (images, targets, texts) in enumerate(progress_bar):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Prepare CTC inputs
            batch_size = outputs.size(0)
            seq_len = outputs.size(1)
            
            # Reshape outputs for CTC
            outputs = outputs.log_softmax(2).permute(1, 0, 2)  # (seq_len, batch, num_classes)
            
            # Prepare targets
            target_lengths = torch.tensor([len(target) for target in targets], device=device)
            target_concat = torch.cat(targets).to(device)
            
            # CTC loss
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
            
            loss = criterion(outputs, target_concat, input_lengths, target_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets, texts in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                
                outputs = model(images)
                outputs = outputs.log_softmax(2).permute(1, 0, 2)
                
                batch_size = outputs.size(1)
                seq_len = outputs.size(0)
                
                target_lengths = torch.tensor([len(target) for target in targets], device=device)
                target_concat = torch.cat(targets).to(device)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
                
                loss = criterion(outputs, target_concat, input_lengths, target_lengths)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
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
                'img_width': IMG_WIDTH,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'checkpoints/cloud_gpu_model.pth')
            logger.info(f"💾 Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_val_loss,
                'num_classes': NUM_CLASSES,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, f'checkpoints/cloud_gpu_model_epoch_{epoch+1}.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-50:], label='Train Loss (last 50)')
    plt.plot(val_losses[-50:], label='Val Loss (last 50)')
    plt.title('Recent Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    logger.info("✅ Training completed!")
    return model

def test_model(model_path='checkpoints/cloud_gpu_model.pth'):
    """Test the trained model"""
    logger.info("🧪 Testing Cloud GPU model...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = CloudOCRModel(
        checkpoint['num_classes'], 
        checkpoint['img_height'], 
        checkpoint['img_width']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test with various examples
    test_texts = [
        'HELLO', 'WORLD', 'PYTHON', 'OCR', 'AI', 'ML', 'TEST', 'DEMO',
        '12345', '67890', 'ABCDE', 'FGHIJ', 'KLMNO', 'PQRST', 'UVWXY', 'Z',
        'hello', 'world', 'python', 'test', 'demo', 'ai', 'ml', 'ocr',
        'Hello', 'World', 'Python', 'Test', 'Demo', 'AI', 'ML', 'OCR'
    ]
    
    correct = 0
    total = len(test_texts)
    
    for text in test_texts:
        # Create test image
        img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 255)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (IMG_WIDTH - text_width) // 2
        y = (IMG_HEIGHT - text_height) // 2
        draw.text((x, y), text, fill=0, font=font)
        
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
            chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
            predicted_text = ''.join([chars[idx] for idx in predicted if idx < len(chars)])
            predicted_text = predicted_text.strip()
            
            is_correct = predicted_text == text
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            logger.info(f"{status} Expected: '{text}' -> Predicted: '{predicted_text}'")
    
    accuracy = (correct / total) * 100
    logger.info(f"📊 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train model
    model = train_model()
    
    # Test model
    accuracy = test_model()
    
    logger.info("🎉 Cloud GPU training completed successfully!")
    logger.info(f"🏆 Final Model Accuracy: {accuracy:.2f}%") 