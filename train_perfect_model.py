#!/usr/bin/env python3
"""
Perfect Model Training Script
Comprehensive training with extensive data, longer epochs, and optimizations
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
from PIL import Image, ImageFont, ImageDraw
import logging
from pathlib import Path
import time
import random
import string
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed character set
CHAR_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_SET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_SET)}

class PerfectCRNNModel(nn.Module):
    """
    Perfect CRNN model with enhanced architecture
    """
    
    def __init__(self, num_classes: int, img_height: int = 32, img_width: int = 100):
        super(PerfectCRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # Enhanced CNN feature extraction
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout2d(0.1),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Dropout2d(0.1),
        )
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Enhanced RNN
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # Output layer with attention
        self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
        
        logger.info(f"Perfect CRNN Model initialized:")
        logger.info(f"  Input size: {img_height}x{img_width}")
        logger.info(f"  RNN input size: {512}")
        logger.info(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Reshape for RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        
        # RNN sequence modeling
        x, _ = self.rnn(x)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        
        # Output layer
        x = self.fc(x)
        
        return x

class PerfectOCRDataset(Dataset):
    """
    Perfect dataset with extensive data generation and augmentation
    """
    
    def __init__(self, num_samples: int = 10000, transform=None, img_height: int = 32, img_width: int = 100):
        self.num_samples = num_samples
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # Generate extensive training data
        self.data = self._generate_perfect_data()
        
        logger.info(f"Perfect OCR Dataset: {len(self.data)} samples")
    
    def _generate_perfect_data(self):
        """Generate extensive and diverse training data"""
        data = []
        
        # Extensive word list
        words = [
            # Common words
            'HELLO', 'WORLD', 'TEST', 'OCR', 'AI', 'ML', 'PYTHON', 'TRAINING',
            '12345', '67890', 'ABC', 'XYZ', '123', '456', '789', '000',
            'HELP', 'CODE', 'DATA', 'MODEL', 'FIX', 'BUG', 'GOOD', 'BAD',
            
            # Programming terms
            'FUNCTION', 'VARIABLE', 'CLASS', 'METHOD', 'OBJECT', 'ARRAY',
            'STRING', 'INTEGER', 'BOOLEAN', 'FLOAT', 'DOUBLE', 'VOID',
            'PUBLIC', 'PRIVATE', 'PROTECTED', 'STATIC', 'FINAL', 'ABSTRACT',
            
            # Numbers and codes
            '2024', '2025', '1990', '2000', '3000', '5000', '10000',
            'A1B2C3', 'X9Y8Z7', 'PASS123', 'SECRET', 'ACCESS', 'LOGIN',
            
            # Short words
            'HI', 'BYE', 'YES', 'NO', 'OK', 'UP', 'DOWN', 'LEFT', 'RIGHT',
            'TOP', 'BOTTOM', 'SIDE', 'CENTER', 'MIDDLE', 'FRONT', 'BACK',
            
            # Long words
            'SUPERCALIFRAGILISTIC', 'ANTIDISESTABLISHMENTARIANISM',
            'PNEUMONOULTRAMICROSCOPICSILICOVOLCANOCONIOSIS',
            
            # Mixed case
            'Hello', 'World', 'Test', 'Ocr', 'Ai', 'Ml', 'Python', 'Training',
            'Test123', 'Hello123', 'World456', 'Python789', 'AI2024',
            
            # Special patterns
            'AAAA', 'BBBB', 'CCCC', 'DDDD', 'EEEE', 'FFFF',
            '1111', '2222', '3333', '4444', '5555', '6666',
            'ABCD', 'EFGH', 'IJKL', 'MNOP', 'QRST', 'UVWX', 'YZ',
        ]
        
        # Generate synthetic data with variations
        for i in range(self.num_samples):
            # Select word with some randomness
            if random.random() < 0.7:
                word = random.choice(words)
            else:
                # Generate random word
                word_length = random.randint(2, 8)
                word = ''.join(random.choices(string.ascii_uppercase + string.digits, k=word_length))
            
            # Create multiple variations of each word
            variations = self._create_word_variations(word)
            
            for variation in variations:
                if len(data) >= self.num_samples:
                    break
                    
                # Create synthetic image with different styles
                img = self._create_perfect_image(variation)
                
                data.append({
                    'image': img,
                    'text': variation,
                    'length': len(variation)
                })
        
        # Shuffle data
        random.shuffle(data)
        return data[:self.num_samples]
    
    def _create_word_variations(self, word):
        """Create variations of a word"""
        variations = [word]
        
        # Uppercase
        if word != word.upper():
            variations.append(word.upper())
        
        # Lowercase
        if word != word.lower():
            variations.append(word.lower())
        
        # Title case
        if word != word.title():
            variations.append(word.title())
        
        # Add numbers
        if not any(c.isdigit() for c in word):
            variations.append(word + str(random.randint(0, 999)))
        
        return variations
    
    def _create_perfect_image(self, text):
        """Create high-quality synthetic image"""
        # Create base image
        img = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 255
        
        # Add some background variation
        if random.random() < 0.3:
            # Add subtle noise
            noise = np.random.randint(240, 255, (self.img_height, self.img_width, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
        
        # Choose font style
        font_choices = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        ]
        font = random.choice(font_choices)
        
        # Vary font scale
        font_scale = random.uniform(0.4, 0.8)
        thickness = random.randint(1, 3)
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Center the text
        x = (self.img_width - text_width) // 2
        y = (self.img_height + text_height) // 2
        
        # Add slight random offset for robustness
        x += random.randint(-5, 5)
        y += random.randint(-3, 3)
        
        # Ensure text stays within bounds
        x = max(5, min(x, self.img_width - text_width - 5))
        y = max(text_height + 5, min(y, self.img_height - 5))
        
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)
        
        # Add some augmentation
        if random.random() < 0.2:
            # Add slight rotation
            angle = random.uniform(-5, 5)
            center = (self.img_width // 2, self.img_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (self.img_width, self.img_height), 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            # Get data
            item = self.data[idx]
            image = item['image']
            text = item['text']
            
            # Convert to PIL
            pil_image = Image.fromarray(image)
            
            # Apply transformations
            if self.transform:
                pil_image = self.transform(pil_image)
            
            # Convert text to indices
            label_indices = [CHAR_TO_IDX.get(char, 0) for char in text]
            
            return {
                'image': pil_image,
                'label': torch.tensor(label_indices, dtype=torch.long),
                'label_text': text,
                'label_length': len(label_indices)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            dummy_image = torch.zeros(3, self.img_height, self.img_width)
            dummy_label = torch.tensor([0, 1, 2], dtype=torch.long)
            return {
                'image': dummy_image,
                'label': dummy_label,
                'label_text': '012',
                'label_length': 3
            }

def perfect_collate_fn(batch):
    """Perfect collate function for proper batching"""
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

class PerfectOCRTrainer:
    """
    Perfect trainer with extensive training and optimizations
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        
        logger.info(f"Perfect OCR Trainer initialized on {self.device}")
    
    def prepare_data(self, num_samples: int = 10000):
        """Prepare extensive training data"""
        # Define transformations with augmentation
        transform = transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Add some augmentation
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            ], p=0.3),
        ])
        
        # Create dataset
        dataset = PerfectOCRDataset(num_samples=num_samples, transform=transform)
        
        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Larger batch size
            shuffle=True,
            num_workers=2,  # Use multiple workers
            pin_memory=True,
            collate_fn=perfect_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=perfect_collate_fn
        )
        
        # Initialize model
        self.model = PerfectCRNNModel(
            num_classes=len(CHAR_SET) + 1,
            img_height=32,
            img_width=100
        ).to(self.device)
        
        # Initialize optimizer with better parameters
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        
        # Initialize CTC loss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
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
                outputs = outputs.log_softmax(2).permute(1, 0, 2)
                
                # Create target lengths
                target_lengths = label_lengths
                
                # Calculate loss
                loss = self.criterion(outputs, labels, 
                                    torch.full((batch_size,), seq_length, dtype=torch.long),
                                    target_lengths)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
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
                    outputs = outputs.log_softmax(2).permute(1, 0, 2)
                    
                    # Create target lengths
                    target_lengths = label_lengths
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels, 
                                        torch.full((batch_size,), seq_length, dtype=torch.long),
                                        target_lengths)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_model(self, save_path: str, epoch: int, train_loss: float, val_loss: float):
        """Save the trained model with metadata"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_set': CHAR_SET,
            'num_classes': len(CHAR_SET) + 1,
            'img_height': 32,
            'img_width': 100,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def train(self, num_samples: int = 10000, epochs: int = 100, save_path: str = None):
        """Main training function with extensive training"""
        logger.info("Starting perfect model training...")
        
        # Prepare data
        if not self.prepare_data(num_samples):
            logger.error("Failed to prepare data")
            return False
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path is None:
                    save_path = "checkpoints/perfect_model.pth"
                self.save_model(save_path, epoch, train_loss, val_loss)
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoints/perfect_model_epoch_{epoch+1}.pth"
                self.save_model(checkpoint_path, epoch, train_loss, val_loss)
            
            # Early stopping check
            if epoch > 20 and val_loss > best_val_loss * 1.1:
                logger.info("Early stopping triggered")
                break
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses)
        
        logger.info("Perfect training completed!")
        return True
    
    def _plot_training_curves(self, train_losses, val_losses):
        """Plot training curves"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss', color='blue')
            plt.plot(val_losses, label='Val Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_curves.png')
            plt.close()
            logger.info("Training curves saved to training_curves.png")
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")

def test_perfect_model(model_path: str):
    """Test the perfect model"""
    logger.info(f"Testing perfect model: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PerfectCRNNModel(
        num_classes=len(CHAR_SET) + 1,
        img_height=32,
        img_width=100
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test cases
    test_cases = [
        "HELLO", "WORLD", "TEST", "OCR", "AI", "ML", "PYTHON",
        "12345", "67890", "ABC", "XYZ", "123", "456", "789",
        "HELP", "CODE", "DATA", "MODEL", "FIX", "BUG", "GOOD"
    ]
    
    results = []
    for text in test_cases:
        # Create test image
        img = np.ones((32, 100, 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Preprocess
        pil_image = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(pil_image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            logits = model(image_tensor)
        
        # Decode predictions
        predictions = torch.argmax(logits, dim=2)
        pred_list = predictions[0].cpu().numpy()
        
        # Decode to text
        predicted_text = ""
        prev_char = None
        for pred in pred_list:
            if pred != 0 and pred != prev_char:
                if pred <= len(CHAR_SET):
                    predicted_text += CHAR_SET[pred - 1]
            prev_char = pred
        
        # Check accuracy
        correct = text.lower() == predicted_text.lower()
        results.append({
            'expected': text,
            'predicted': predicted_text,
            'correct': correct
        })
        
        logger.info(f"Expected: '{text}' -> Predicted: '{predicted_text}' (Correct: {correct})")
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    logger.info(f"Overall Accuracy: {accuracy:.2%}")
    
    return accuracy

def main():
    """Main function"""
    logger.info("Starting perfect model training...")
    
    # Train the model extensively
    trainer = PerfectOCRTrainer()
    success = trainer.train(num_samples=15000, epochs=150, save_path="checkpoints/perfect_model.pth")
    
    if success:
        # Test the model
        accuracy = test_perfect_model("checkpoints/perfect_model.pth")
        if accuracy > 0.8:
            logger.info("✅ Perfect model training completed successfully!")
            logger.info(f"Model achieved {accuracy:.2%} accuracy - Ready for production!")
        else:
            logger.warning(f"⚠️ Model accuracy is {accuracy:.2%} - May need more training")
    else:
        logger.error("❌ Perfect model training failed!")

if __name__ == "__main__":
    main() 