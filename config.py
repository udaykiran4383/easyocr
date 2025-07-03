"""
Configuration settings for the EasyOCR Text Detection and Recognition Pipeline
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
IMAGES_DIR = PROJECT_ROOT / "images"
ANNOTATED_IMAGES_DIR = PROJECT_ROOT / "annotated_images"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
IIIT5K_DIR = PROJECT_ROOT / "IIIT5K"

# Create directories if they don't exist
for directory in [IMAGES_DIR, ANNOTATED_IMAGES_DIR, CHECKPOINTS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# OCR Configuration
OCR_CONFIG = {
    'languages': ['en'],  # Supported languages
    'gpu': True,  # Use GPU if available
    'model_storage_directory': str(CHECKPOINTS_DIR),
    'download_enabled': True,
    'recognition_network': 'standard',
    'detection_network': 'craft',
    'paragraph': False,
    'min_size': 10,
    'rotation_info': [90, 180, 270],
    'canvas_size': 2560,
    'mag_ratio': 1.5,
    'slope_ths': 0.2,
    'ycenter_ths': 0.5,
    'height_ths': 0.5,
    'width_ths': 0.5,
    'add_margin': 0.1,
    'text_threshold': 0.6,
    'link_threshold': 0.4,
    'low_text': 0.4,
    'poly': False,
    'cuda': True,
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'supported_formats': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'],
    'max_image_size': (4096, 4096),  # Maximum image dimensions
    'resize_factor': 1.0,  # Resize factor for processing
    'preprocessing': {
        'denoise': True,
        'sharpen': False,
        'contrast_enhancement': True,
        'deskew': True,
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'model_save_interval': 5,
    'log_interval': 100,
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'weight_decay': 1e-4,
    'gradient_clip': 5.0,
    'mixed_precision': True,
    'num_workers': 4,
    'pin_memory': True,
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'imgH': 32,
    'imgW': 100,
    'input_channel': 3,  # RGB images
    'output_channel': 512,
    'hidden_size': 256,
    'num_fiducial': 20,
    'Transformation': 'TPS',
    'FeatureExtraction': 'ResNet',
    'SequenceModeling': 'BiLSTM',
    'Prediction': 'Attn',
    'character_set': '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'num_class': 63,  # 62 characters + 1 for CTC blank
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'cer', 'wer', 'ned'],
    'test_batch_size': 64,
    'save_predictions': True,
    'visualization': True,
    'confidence_threshold': 0.5,
}

# Output Configuration
OUTPUT_CONFIG = {
    'csv_columns': ['image_name', 'text', 'confidence', 'bbox', 'processing_time'],
    'save_annotated_images': True,
    'save_detection_maps': False,
    'save_recognition_results': True,
    'output_format': 'csv',
    'include_metadata': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(RESULTS_DIR / 'ocr_pipeline.log'),
    'console': True,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'num_threads': 4,
    'memory_fraction': 0.8,
    'enable_optimization': True,
    'cache_models': True,
}

# Dataset Configuration
DATASET_CONFIG = {
    'IIIT5K': {
        'train_data': str(IIIT5K_DIR / 'train'),
        'test_data': str(IIIT5K_DIR / 'test'),
        'train_labels': str(IIIT5K_DIR / 'trainCharBound.mat'),
        'test_labels': str(IIIT5K_DIR / 'testCharBound.mat'),
        'lexicon': str(IIIT5K_DIR / 'lexicon.txt'),
    }
}

# File paths
PATHS = {
    'images': str(IMAGES_DIR),
    'annotated_images': str(ANNOTATED_IMAGES_DIR),
    'checkpoints': str(CHECKPOINTS_DIR),
    'results': str(RESULTS_DIR),
    'iiit5k': str(IIIT5K_DIR),
    'logs': str(RESULTS_DIR / 'logs'),
    'models': str(CHECKPOINTS_DIR / 'models'),
    'reports': str(RESULTS_DIR / 'reports'),
}

# Create additional directories
for path in PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True) 