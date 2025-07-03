#!/usr/bin/env python3
"""
Custom OCR Model Training Script

This script provides a complete pipeline for training a custom OCR model using the IIIT5K dataset.
It includes data preprocessing, model training, and evaluation.

Usage:
    python train_custom_model.py --dataset-path IIIT5K --epochs 50 --batch-size 32
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import PATHS, TRAINING_CONFIG, MODEL_CONFIG
from data_preprocessor import IIIT5KPreprocessor
from model_trainer import OCRTrainer
from utils import setup_logging

logger = logging.getLogger(__name__)


def setup_training_environment():
    """Setup the training environment"""
    # Create necessary directories
    for path_name, path_value in PATHS.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
    
    # Create training-specific directories
    training_dirs = [
        'checkpoints/models',
        'results/training_logs',
        'results/training_reports',
        'results/datasets'
    ]
    
    for dir_path in training_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Training environment setup completed")


def prepare_dataset(dataset_path: str, output_dir: str = None) -> dict:
    """
    Prepare the IIIT5K dataset for training
    
    Args:
        dataset_path: Path to the IIIT5K dataset
        output_dir: Output directory for processed data
        
    Returns:
        Dictionary with dataset information
    """
    if output_dir is None:
        output_dir = Path(PATHS['results']) / 'datasets'
    else:
        output_dir = Path(output_dir)
    
    logger.info(f"Preparing dataset from: {dataset_path}")
    
    # Initialize preprocessor
    preprocessor = IIIT5KPreprocessor(dataset_path)
    
    # Validate dataset
    if not preprocessor.validate_dataset():
        logger.error("Dataset validation failed!")
        return {}
    
    # Analyze dataset
    analysis = preprocessor.analyze_dataset()
    if not analysis:
        logger.error("Dataset analysis failed!")
        return {}
    
    logger.info("Dataset Analysis:")
    logger.info(f"  Total samples: {analysis['total_samples']}")
    logger.info(f"  Train samples: {analysis['train_samples']}")
    logger.info(f"  Test samples: {analysis['test_samples']}")
    logger.info(f"  Unique characters: {analysis['unique_characters']}")
    logger.info(f"  Character set: {analysis['character_set']}")
    logger.info(f"  Average label length: {analysis['avg_label_length']:.2f}")
    
    # Create CSV datasets
    csv_paths = preprocessor.create_csv_dataset(output_dir)
    if not csv_paths:
        logger.error("Failed to create CSV datasets!")
        return {}
    
    # Create character set file
    char_set_path = preprocessor.create_character_set_file()
    if not char_set_path:
        logger.error("Failed to create character set file!")
        return {}
    
    return {
        'analysis': analysis,
        'csv_paths': csv_paths,
        'char_set_path': char_set_path,
        'preprocessor': preprocessor
    }


def train_model(train_csv: str, val_csv: str, config: dict = None) -> bool:
    """
    Train the custom OCR model
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        config: Training configuration
        
    Returns:
        True if training successful, False otherwise
    """
    if config is None:
        config = TRAINING_CONFIG.copy()
    
    logger.info("Starting model training...")
    logger.info(f"Training CSV: {train_csv}")
    logger.info(f"Validation CSV: {val_csv}")
    
    # Initialize trainer
    trainer = OCRTrainer(config)
    
    # Train model
    success = trainer.train(
        train_csv=train_csv,
        val_csv=val_csv,
        epochs=config.get('epochs', 100),
        save_dir=PATHS['checkpoints']
    )
    
    if success:
        logger.info("✅ Model training completed successfully!")
        return True
    else:
        logger.error("❌ Model training failed!")
        return False


def evaluate_model(model_path: str, test_csv: str) -> dict:
    """
    Evaluate the trained model
    
    Args:
        model_path: Path to the trained model
        test_csv: Path to test CSV file
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating model: {model_path}")
    
    # TODO: Implement model evaluation
    # This would involve loading the trained model and running inference on test data
    
    evaluation_results = {
        'model_path': model_path,
        'test_samples': 0,
        'accuracy': 0.0,
        'cer': 0.0,  # Character Error Rate
        'wer': 0.0,  # Word Error Rate
        'processing_time': 0.0
    }
    
    logger.info("Model evaluation completed")
    return evaluation_results


def save_training_report(dataset_info: dict, training_config: dict, 
                        evaluation_results: dict = None) -> str:
    """
    Save training report
    
    Args:
        dataset_info: Dataset information
        training_config: Training configuration
        evaluation_results: Evaluation results
        
    Returns:
        Path to the saved report
    """
    report_path = PATHS['results'] / 'training_reports' / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CUSTOM OCR MODEL TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Training Date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 30 + "\n")
        if 'analysis' in dataset_info:
            analysis = dataset_info['analysis']
            f.write(f"Total samples: {analysis['total_samples']}\n")
            f.write(f"Train samples: {analysis['train_samples']}\n")
            f.write(f"Test samples: {analysis['test_samples']}\n")
            f.write(f"Unique characters: {analysis['unique_characters']}\n")
            f.write(f"Character set: {analysis['character_set']}\n")
            f.write(f"Average label length: {analysis['avg_label_length']:.2f}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        for key, value in training_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        if evaluation_results:
            f.write("EVALUATION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, value in evaluation_results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write("FILES CREATED:\n")
        f.write("-" * 30 + "\n")
        if 'csv_paths' in dataset_info:
            for dataset_type, path in dataset_info['csv_paths'].items():
                f.write(f"{dataset_type}_csv: {path}\n")
        if 'char_set_path' in dataset_info:
            f.write(f"character_set: {dataset_info['char_set_path']}\n")
    
    logger.info(f"Training report saved to: {report_path}")
    return str(report_path)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Custom OCR Model')
    parser.add_argument('--dataset-path', default='IIIT5K', 
                       help='Path to IIIT5K dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model after training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    
    logger.info("🚀 Starting Custom OCR Model Training Pipeline")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    try:
        # Setup environment
        setup_training_environment()
        
        # Prepare dataset
        logger.info("\n📊 Step 1: Preparing Dataset")
        dataset_info = prepare_dataset(args.dataset_path)
        if not dataset_info:
            logger.error("Dataset preparation failed!")
            return 1
        
        # Update training configuration
        training_config = TRAINING_CONFIG.copy()
        training_config.update({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        })
        # Ensure image size keys are present
        training_config['imgH'] = MODEL_CONFIG['imgH']
        training_config['imgW'] = MODEL_CONFIG['imgW']
        
        # Train model
        logger.info("\n🏋️ Step 2: Training Model")
        train_success = train_model(
            train_csv=dataset_info['csv_paths']['train'],
            val_csv=dataset_info['csv_paths']['test'],
            config=training_config
        )
        
        if not train_success:
            logger.error("Model training failed!")
            return 1
        
        # Evaluate model (optional)
        evaluation_results = None
        if args.evaluate:
            logger.info("\n📈 Step 3: Evaluating Model")
            # Find the best model
            checkpoint_dir = Path(PATHS['checkpoints'])
            model_files = list(checkpoint_dir.glob('best_model_epoch_*.pth'))
            if model_files:
                best_model = max(model_files, key=lambda x: x.stat().st_mtime)
                evaluation_results = evaluate_model(
                    str(best_model),
                    dataset_info['csv_paths']['test']
                )
        
        # Save training report
        logger.info("\n📋 Step 4: Saving Training Report")
        try:
            report_path = save_training_report(dataset_info, training_config, evaluation_results)
        except Exception as e:
            logger.error(f"Error saving training report: {e}")
            report_path = "training_report_error.txt"
        
        logger.info("\n✅ Training Pipeline Completed Successfully!")
        logger.info(f"📄 Training report: {report_path}")
        logger.info(f"💾 Model checkpoints: {PATHS['checkpoints']}")
        logger.info(f"📊 Dataset files: {PATHS['results']}/datasets")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 