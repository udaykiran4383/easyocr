"""
Custom Model Trainer for EasyOCR

This module implements custom model training for EasyOCR using the IIIT5K dataset
and other custom datasets. It provides a simplified training interface that works
with EasyOCR's architecture.
"""

import os
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
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence

from config import TRAINING_CONFIG, MODEL_CONFIG, PATHS
from data_preprocessor import IIIT5KPreprocessor

logger = logging.getLogger(__name__)


class OCRDataset(Dataset):
    """
    Custom dataset for OCR training
    """
    
    def __init__(self, csv_file: str, transform=None, img_height: int = 32, img_width: int = 100):
        """
        Initialize the dataset
        
        Args:
            csv_file: Path to CSV file with image paths and labels
            transform: Image transformations
            img_height: Target image height
            img_width: Target image width
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        # Create character to index mapping
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocabulary()
        
        logger.info(f"Dataset loaded: {len(self.data)} samples")
        logger.info(f"Vocabulary size: {len(self.char_to_idx)} characters")
    
    def _build_vocabulary(self):
        """Build character vocabulary from all labels"""
        all_chars = set()
        for label in self.data['label']:
            all_chars.update(label)
        
        # Add special tokens
        all_chars.add('<PAD>')  # Padding
        all_chars.add('<SOS>')  # Start of sequence
        all_chars.add('<EOS>')  # End of sequence
        all_chars.add('<UNK>')  # Unknown character
        
        # Create mappings
        for i, char in enumerate(sorted(all_chars)):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.data.iloc[idx]['image_path']
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label = self.data.iloc[idx]['label']
            
            # Convert label to indices
            label_indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in label]
            label_indices = [self.char_to_idx['<SOS>']] + label_indices + [self.char_to_idx['<EOS>']]
            
            return {
                'image': image,
                'label': torch.tensor(label_indices, dtype=torch.long),
                'label_text': label,
                'label_length': len(label_indices)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, self.img_height, self.img_width)
            dummy_label = torch.tensor([self.char_to_idx['<SOS>'], self.char_to_idx['<EOS>']], dtype=torch.long)
            return {
                'image': dummy_image,
                'label': dummy_label,
                'label_text': '',
                'label_length': 2
            }


class CRNNModel(nn.Module):
    """
    Simple and robust CRNN model for OCR training
    """
    
    def __init__(self, num_classes: int, img_height: int = 32, img_width: int = 100):
        """
        Initialize the CRNN model
        
        Args:
            num_classes: Number of character classes
            img_height: Input image height
            img_width: Input image width
        """
        super(CRNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN feature extraction - simplified and robust
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
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))  # Fixed height, variable width
        
        # RNN sequence modeling
        self.rnn = nn.LSTM(
            input_size=512 * 4,  # 512 channels * 4 height
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(512, num_classes)  # 512 = 256*2 (bidirectional)
        
        logger.info(f"CRNN Model initialized:")
        logger.info(f"  Input size: {img_height}x{img_width}")
        logger.info(f"  RNN input size: {512 * 4}")
        logger.info(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, channels, height, width)
        
        # Adaptive pooling to standardize height
        x = self.adaptive_pool(x)  # (batch, 512, 4, width)
        
        # Reshape for RNN
        # Convert from (batch, channels, height, width) to (batch, width, channels*height)
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, width, channels*height)
        
        # RNN sequence modeling
        x, _ = self.rnn(x)  # (batch, width, hidden_size*2)
        
        # Output layer
        x = self.fc(x)  # (batch, width, num_classes)
        
        return x


def ocr_collate_fn(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    label_texts = [item['label_text'] for item in batch]
    label_lengths = [item['label_length'] for item in batch]
    # Pad labels
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return {
        'image': torch.stack(images),
        'label': labels_padded,
        'label_text': label_texts,
        'label_length': torch.tensor(label_lengths, dtype=torch.long)
    }


class OCRTrainer:
    """
    Trainer for custom OCR models
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration
        """
        self.config = config or TRAINING_CONFIG.copy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        
        logger.info(f"OCR Trainer initialized on device: {self.device}")
    
    def prepare_dataset(self, train_csv: str, val_csv: str, char_set_file: str = None) -> bool:
        """
        Prepare the dataset for training
        
        Args:
            train_csv: Path to training CSV file
            val_csv: Path to validation CSV file
            char_set_file: Path to character set file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Define transformations
            transform = transforms.Compose([
                transforms.Resize((self.config['imgH'], self.config['imgW'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Create datasets
            train_dataset = OCRDataset(
                train_csv, 
                transform=transform,
                img_height=self.config['imgH'],
                img_width=self.config['imgW']
            )
            
            val_dataset = OCRDataset(
                val_csv,
                transform=transform,
                img_height=self.config['imgH'],
                img_width=self.config['imgW']
            )
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                collate_fn=ocr_collate_fn
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                collate_fn=ocr_collate_fn
            )
            
            # Initialize model
            num_classes = len(train_dataset.char_to_idx)
            self.model = CRNNModel(
                num_classes=num_classes,
                img_height=self.config['imgH'],
                img_width=self.config['imgW']
            ).to(self.device)
            
            # Initialize optimizer
            if self.config['optimizer'] == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config['learning_rate'],
                    weight_decay=self.config['weight_decay']
                )
            else:
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config['learning_rate'],
                    momentum=self.config['momentum'],
                    weight_decay=self.config['weight_decay']
                )
            
            # Initialize loss function
            self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
            
            logger.info(f"Dataset prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
            logger.info(f"Model initialized with {num_classes} classes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            return False
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
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
                
                # Create target lengths (all sequences have the same length in this simplified version)
                target_lengths = torch.full((batch_size,), labels.size(1), dtype=torch.long)
                
                # Calculate loss
                loss = self.criterion(outputs, labels, target_lengths, target_lengths)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if batch_idx % self.config['log_interval'] == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def validate(self) -> Dict:
        """
        Validate the model
        
        Args:
            Dictionary with validation metrics
        """
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
                    outputs = outputs.log_softmax(2).permute(1, 0, 2)
                    target_lengths = torch.full((batch_size,), labels.size(1), dtype=torch.long)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels, target_lengths, target_lengths)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def train(self, train_csv: str, val_csv: str, epochs: int = None, 
              save_dir: str = None) -> bool:
        """
        Train the model
        
        Args:
            train_csv: Path to training CSV file
            val_csv: Path to validation CSV file
            epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
            
        Returns:
            True if successful, False otherwise
        """
        if epochs is None:
            epochs = self.config['epochs']
        
        if save_dir is None:
            save_dir = PATHS['checkpoints']
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Prepare dataset
        if not self.prepare_dataset(train_csv, val_csv):
            return False
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Log results
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save model
                model_path = save_dir / f'best_model_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'config': self.config
                }, model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config['model_save_interval'] == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'config': self.config
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")
        return True
    
    def save_model(self, save_path: str):
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
        logger.info(f"Model saved to {save_path}")


def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom OCR Model Trainer')
    parser.add_argument('--train-csv', required=True, help='Path to training CSV file')
    parser.add_argument('--val-csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = OCRTrainer()
    
    # Train model
    success = trainer.train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")


if __name__ == "__main__":
    main() 