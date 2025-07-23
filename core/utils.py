"""
Utility functions for the EasyOCR Text Detection and Recognition Pipeline
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('easyocr_pipeline')
    logger.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_supported_image_files(directory: str) -> List[str]:
    """Get list of supported image files from directory"""
    supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    image_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from path with error handling"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """Save image to path with error handling"""
    try:
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def preprocess_image(image: np.ndarray, config: Dict) -> np.ndarray:
    """Apply preprocessing to image"""
    processed = image.copy()
    
    # Convert to grayscale if needed
    if len(processed.shape) == 3 and config.get('grayscale', False):
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Denoise
    if config.get('denoise', False):
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    # Contrast enhancement
    if config.get('contrast_enhancement', False):
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sharpen
    if config.get('sharpen', False):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    return processed


def draw_bounding_boxes(image: np.ndarray, results: List[Tuple], 
                       color: Tuple[int, int, int] = (0, 255, 0), 
                       thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on image"""
    annotated_image = image.copy()
    
    for bbox, text, confidence in results:
        # Convert bbox to integer coordinates
        bbox = np.array(bbox, dtype=np.int32)
        
        # Draw bounding box
        cv2.polylines(annotated_image, [bbox], True, color, thickness)
        
        # Add text label
        if text:
            # Get top-left corner for text placement
            x, y = bbox[0]
            cv2.putText(annotated_image, f"{text} ({confidence:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated_image


def calculate_bbox_area(bbox: List[Tuple[float, float]]) -> float:
    """Calculate area of bounding box"""
    bbox = np.array(bbox)
    x_coords = bbox[:, 0]
    y_coords = bbox[:, 1]
    return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))


def filter_results_by_confidence(results: List[Tuple], 
                                confidence_threshold: float = 0.5) -> List[Tuple]:
    """Filter OCR results by confidence threshold"""
    return [result for result in results if result[2] >= confidence_threshold]


def sort_results_by_confidence(results: List[Tuple], 
                              reverse: bool = True) -> List[Tuple]:
    """Sort OCR results by confidence score"""
    return sorted(results, key=lambda x: x[2], reverse=reverse)


def create_results_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create pandas DataFrame from OCR results"""
    df = pd.DataFrame(results)
    return df


def save_results_to_csv(results: List[Dict], output_path: str) -> bool:
    """Save OCR results to CSV file"""
    try:
        df = create_results_dataframe(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")
        return False


def save_results_to_json(results: List[Dict], output_path: str) -> bool:
    """Save OCR results to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        return False


def visualize_results(image: np.ndarray, results: List[Tuple], 
                     save_path: str = None, show: bool = False) -> None:
    """Visualize OCR results with matplotlib"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw bounding boxes
    for bbox, text, confidence in results:
        bbox = np.array(bbox)
        polygon = patches.Polygon(bbox, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(polygon)
        
        # Add text annotation
        x, y = bbox[0]
        ax.text(x, y - 5, f"{text} ({confidence:.2f})", 
                fontsize=8, color='red', weight='bold')
    
    ax.set_title('OCR Results')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def calculate_processing_metrics(start_time: float, end_time: float, 
                                image_size: Tuple[int, int]) -> Dict:
    """Calculate processing metrics"""
    processing_time = end_time - start_time
    image_area = image_size[0] * image_size[1]
    
    return {
        'processing_time': processing_time,
        'image_width': image_size[1],
        'image_height': image_size[0],
        'image_area': image_area,
        'pixels_per_second': image_area / processing_time if processing_time > 0 else 0
    }


def format_bbox_for_csv(bbox: List[Tuple[float, float]]) -> str:
    """Format bounding box for CSV output"""
    bbox_str = []
    for point in bbox:
        bbox_str.append(f"{point[0]:.2f},{point[1]:.2f}")
    return "|".join(bbox_str)


def parse_bbox_from_csv(bbox_str: str) -> List[Tuple[float, float]]:
    """Parse bounding box from CSV string"""
    bbox = []
    for point_str in bbox_str.split("|"):
        x, y = map(float, point_str.split(","))
        bbox.append((x, y))
    return bbox


def get_image_info(image_path: str) -> Dict:
    """Get basic information about an image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {}
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # Get file size
        file_size = os.path.getsize(image_path)
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'file_size': file_size,
            'aspect_ratio': width / height if height > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return {}


def create_summary_report(results: List[Dict], output_path: str) -> bool:
    """Create a summary report of OCR processing"""
    try:
        if not results:
            logger.warning("No results to create summary report")
            return False
        
        df = pd.DataFrame(results)
        
        summary = {
            'total_images': len(df),
            'total_text_regions': len(df),
            'average_confidence': df['confidence'].mean() if 'confidence' in df else 0,
            'min_confidence': df['confidence'].min() if 'confidence' in df else 0,
            'max_confidence': df['confidence'].max() if 'confidence' in df else 0,
            'average_processing_time': df['processing_time'].mean() if 'processing_time' in df else 0,
            'total_processing_time': df['processing_time'].sum() if 'processing_time' in df else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating summary report: {e}")
        return False


def validate_image_path(image_path: str) -> bool:
    """Validate if image path exists and is readable"""
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist: {image_path}")
        return False
    
    if not os.path.isfile(image_path):
        logger.error(f"Path is not a file: {image_path}")
        return False
    
    # Try to read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Cannot read image: {image_path}")
        return False
    
    return True


def create_output_filename(input_path: str, suffix: str = "", 
                          extension: str = ".png") -> str:
    """Create output filename from input path"""
    input_name = Path(input_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if suffix:
        return f"{input_name}_{suffix}_{timestamp}{extension}"
    else:
        return f"{input_name}_{timestamp}{extension}"


def check_gpu_availability() -> bool:
    """Check if GPU is available for processing"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_memory_usage() -> Dict:
    """Get current memory usage information"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil not available'}


def cleanup_temp_files(temp_dir: str) -> None:
    """Clean up temporary files"""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")


def create_progress_bar(total: int, description: str = "Processing") -> None:
    """Create a simple progress bar"""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=description)
    except ImportError:
        # Fallback to simple progress tracking
        logger.info(f"Starting {description} for {total} items")
        return None


def update_progress(progress_bar, increment: int = 1) -> None:
    """Update progress bar"""
    if progress_bar is not None:
        try:
            progress_bar.update(increment)
        except:
            pass 