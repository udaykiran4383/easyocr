"""
Data Preprocessor for IIIT5K Dataset

This module handles the preparation of the IIIT5K dataset for EasyOCR training,
including loading MATLAB files, creating CSV datasets, and preparing image-label pairs.
"""

import os
import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from config import DATASET_CONFIG, PATHS

logger = logging.getLogger(__name__)


class IIIT5KPreprocessor:
    """
    Preprocessor for IIIT5K dataset
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the preprocessor
        
        Args:
            dataset_path: Path to the IIIT5K dataset
        """
        self.dataset_path = Path(dataset_path or PATHS['iiit5k'])
        self.train_data = None
        self.test_data = None
        self.train_char_data = None
        self.test_char_data = None
        
        logger.info(f"IIIT5K Preprocessor initialized with dataset path: {self.dataset_path}")
    
    def load_matlab_data(self) -> bool:
        """
        Load MATLAB data files from the IIIT5K dataset
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load training data
            train_mat_path = self.dataset_path / 'traindata.mat'
            if train_mat_path.exists():
                self.train_data = scipy.io.loadmat(str(train_mat_path))
                logger.info(f"Loaded training data: {len(self.train_data['traindata'][0])} samples")
            
            # Load test data
            test_mat_path = self.dataset_path / 'testdata.mat'
            if test_mat_path.exists():
                self.test_data = scipy.io.loadmat(str(test_mat_path))
                logger.info(f"Loaded test data: {len(self.test_data['testdata'][0])} samples")
            
            # Load character bounding box data
            train_char_path = self.dataset_path / 'trainCharBound.mat'
            if train_char_path.exists():
                self.train_char_data = scipy.io.loadmat(str(train_char_path))
                logger.info(f"Loaded training character data: {len(self.train_char_data['trainCharBound'][0])} samples")
            
            test_char_path = self.dataset_path / 'testCharBound.mat'
            if test_char_path.exists():
                self.test_char_data = scipy.io.loadmat(str(test_char_path))
                logger.info(f"Loaded test character data: {len(self.test_char_data['testCharBound'][0])} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading MATLAB data: {e}")
            return False
    
    def extract_image_label_pairs(self, data_type: str = 'train') -> List[Dict]:
        """
        Extract image-label pairs from the loaded data
        
        Args:
            data_type: 'train' or 'test'
            
        Returns:
            List of dictionaries with image paths and labels
        """
        if data_type == 'train':
            data = self.train_data
            char_data = self.train_char_data
            image_dir = 'train'
        else:
            data = self.test_data
            char_data = self.test_char_data
            image_dir = 'test'
        
        if data is None:
            logger.error(f"No {data_type} data loaded")
            return []
        
        pairs = []
        data_array = data[f'{data_type}data'][0]
        
        for i, entry in enumerate(data_array):
            try:
                # Extract image name and ground truth
                img_name = entry['ImgName'][0]
                ground_truth = entry['GroundTruth'][0]
                
                # Create full image path
                # The img_name already contains the subdirectory (e.g., "test/1002_1.png")
                img_path = str(self.dataset_path / img_name)
                
                # Verify image exists
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}")
                    continue
                
                # Create pair
                pair = {
                    'image_path': img_path,
                    'label': ground_truth,
                    'image_name': img_name,
                    'dataset': data_type
                }
                
                # Add character bounding box info if available
                if char_data is not None and i < len(char_data[f'{data_type}CharBound'][0]):
                    char_entry = char_data[f'{data_type}CharBound'][0][i]
                    pair['characters'] = char_entry['chars'][0]
                    pair['char_bbox'] = char_entry['charBB']
                
                pairs.append(pair)
                
            except Exception as e:
                logger.error(f"Error processing entry {i}: {e}")
                continue
        
        logger.info(f"Extracted {len(pairs)} {data_type} image-label pairs")
        return pairs
    
    def create_csv_dataset(self, output_dir: str = None) -> Dict[str, str]:
        """
        Create CSV datasets for training and testing
        
        Args:
            output_dir: Output directory for CSV files
            
        Returns:
            Dictionary with paths to created CSV files
        """
        if output_dir is None:
            output_dir = PATHS['results']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load data if not already loaded
        if self.train_data is None or self.test_data is None:
            if not self.load_matlab_data():
                return {}
        
        # Extract pairs
        train_pairs = self.extract_image_label_pairs('train')
        test_pairs = self.extract_image_label_pairs('test')
        
        # Create DataFrames
        train_df = pd.DataFrame(train_pairs)
        test_df = pd.DataFrame(test_pairs)
        
        # Save CSV files
        train_csv_path = output_dir / 'iiit5k_train.csv'
        test_csv_path = output_dir / 'iiit5k_test.csv'
        
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        
        logger.info(f"Created training CSV: {train_csv_path} with {len(train_df)} samples")
        logger.info(f"Created test CSV: {test_csv_path} with {len(test_df)} samples")
        
        return {
            'train': str(train_csv_path),
            'test': str(test_csv_path)
        }
    
    def analyze_dataset(self) -> Dict:
        """
        Analyze the dataset characteristics
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.train_data is None or self.test_data is None:
            if not self.load_matlab_data():
                return {}
        
        train_pairs = self.extract_image_label_pairs('train')
        test_pairs = self.extract_image_label_pairs('test')
        
        # Collect all labels
        all_labels = [pair['label'] for pair in train_pairs + test_pairs]
        
        # Analyze character set
        all_chars = set()
        label_lengths = []
        
        for label in all_labels:
            all_chars.update(label)
            label_lengths.append(len(label))
        
        # Character frequency
        char_freq = {}
        for label in all_labels:
            for char in label:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        analysis = {
            'total_samples': len(train_pairs) + len(test_pairs),
            'train_samples': len(train_pairs),
            'test_samples': len(test_pairs),
            'unique_characters': len(all_chars),
            'character_set': sorted(list(all_chars)),
            'avg_label_length': np.mean(label_lengths),
            'min_label_length': min(label_lengths),
            'max_label_length': max(label_lengths),
            'character_frequency': char_freq,
            'most_common_chars': sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        }
        
        logger.info(f"Dataset analysis: {analysis['total_samples']} total samples")
        logger.info(f"Character set: {len(analysis['character_set'])} unique characters")
        logger.info(f"Average label length: {analysis['avg_label_length']:.2f}")
        
        return analysis
    
    def create_character_set_file(self, output_path: str = None) -> str:
        """
        Create a character set file for EasyOCR training
        
        Args:
            output_path: Output path for character set file
            
        Returns:
            Path to the created character set file
        """
        if output_path is None:
            output_path = Path(PATHS['results']) / 'character_set.txt'
        
        analysis = self.analyze_dataset()
        if not analysis:
            return ""
        
        # Create character set string
        char_set = ''.join(analysis['character_set'])
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(char_set)
        
        logger.info(f"Created character set file: {output_path}")
        logger.info(f"Character set: {char_set}")
        
        return str(output_path)
    
    def validate_dataset(self) -> bool:
        """
        Validate the dataset integrity
        
        Returns:
            True if dataset is valid, False otherwise
        """
        if not self.load_matlab_data():
            return False
        
        train_pairs = self.extract_image_label_pairs('train')
        test_pairs = self.extract_image_label_pairs('test')
        
        # Check if we have data
        if len(train_pairs) == 0 and len(test_pairs) == 0:
            logger.error("No valid image-label pairs found")
            return False
        
        # Check if images exist
        missing_images = 0
        for pair in train_pairs + test_pairs:
            if not os.path.exists(pair['image_path']):
                missing_images += 1
        
        if missing_images > 0:
            logger.warning(f"{missing_images} images are missing")
        
        logger.info(f"Dataset validation: {len(train_pairs)} train, {len(test_pairs)} test pairs")
        return True


def main():
    """Main function for testing the preprocessor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IIIT5K Dataset Preprocessor')
    parser.add_argument('--dataset-path', default='IIIT5K', help='Path to IIIT5K dataset')
    parser.add_argument('--output-dir', default='results', help='Output directory for CSV files')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset characteristics')
    parser.add_argument('--validate', action='store_true', help='Validate dataset integrity')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = IIIT5KPreprocessor(args.dataset_path)
    
    # Validate dataset
    if args.validate:
        if preprocessor.validate_dataset():
            print("✅ Dataset validation passed")
        else:
            print("❌ Dataset validation failed")
            return
    
    # Analyze dataset
    if args.analyze:
        analysis = preprocessor.analyze_dataset()
        if analysis:
            print("\n📊 Dataset Analysis:")
            print(f"Total samples: {analysis['total_samples']}")
            print(f"Train samples: {analysis['train_samples']}")
            print(f"Test samples: {analysis['test_samples']}")
            print(f"Unique characters: {analysis['unique_characters']}")
            print(f"Character set: {analysis['character_set']}")
            print(f"Average label length: {analysis['avg_label_length']:.2f}")
    
    # Create CSV datasets
    csv_paths = preprocessor.create_csv_dataset(args.output_dir)
    if csv_paths:
        print(f"\n✅ Created CSV datasets:")
        for dataset_type, path in csv_paths.items():
            print(f"  {dataset_type}: {path}")
    
    # Create character set file
    char_set_path = preprocessor.create_character_set_file()
    if char_set_path:
        print(f"✅ Created character set file: {char_set_path}")


if __name__ == "__main__":
    main() 