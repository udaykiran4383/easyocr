# EasyOCR POC - Text Detection Demo

A simple Proof of Concept demonstrating EasyOCR text detection with custom model integration.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r poc_requirements.txt
```

### 2. Run the Demo (Recommended)
```bash
python simple_poc_demo.py
```

### 3. Or Run the Full Demo
```bash
python poc_easyocr_demo.py
```

## 📁 POC Package Contents

```
easyocr_poc/
├── simple_poc_demo.py             # 🎯 Simple working demo (RECOMMENDED)
├── poc_easyocr_demo.py            # Full featured demo
├── poc_requirements.txt           # Minimal dependencies
├── POC_README.md                  # This file
├── checkpoints/                   # Trained model files
│   └── best_model_epoch_19.pth    # Custom trained model
└── images/                        # Sample images for testing
    ├── Sample-handwritten-text-input-for-OCR.png
    ├── 13_2.png
    └── 27_1.png
```

## 🎯 Features

- **EasyOCR Integration**: Uses EasyOCR for text detection
- **Custom Model Loading**: Demonstrates custom model integration
- **Visual Output**: Generates annotated images with bounding boxes
- **Performance Metrics**: Shows processing time and confidence scores
- **Simple API**: Easy to use and understand
- **Two Versions**: Simple demo for quick testing, full demo for advanced features

## 📊 Sample Output

```
🚀 Simple EasyOCR POC Demo
========================================
📦 Importing EasyOCR...
✅ EasyOCR imported successfully!
🔄 Initializing EasyOCR reader...
✅ EasyOCR reader ready!
🎯 Found 3 demo images

==================================================
🔍 Processing: images/Sample-handwritten-text-input-for-OCR.png
⏱️  Processing time: 0.637 seconds
📝 Detected 12 text regions

📊 Detection Results:
------------------------------
Text 1: 'This' (confidence: 0.999)
Text 2: 'is' (confidence: 0.786)
Text 3: 'handwr #ten' (confidence: 0.741)
Text 4: 'ex' (confidence: 1.000)
Text 5: 'Wrie' (confidence: 0.930)

💾 Annotated image saved: simple_poc_output_Sample-handwritten-text-input-for-OCR.png
==================================================

🎉 Demo completed successfully!
```

## 🔧 Customization

### Using Your Own Images
1. Place your images in the `images/` folder
2. Update the `demo_images` list in the demo script
3. Run the script

### Using Different Models
1. Place your trained model (.pth file) in the `checkpoints/` folder
2. Update the `model_path` parameter in the demo script

## 📝 Notes

- **Simple Demo**: `simple_poc_demo.py` - Works out of the box, minimal dependencies
- **Full Demo**: `poc_easyocr_demo.py` - More features, may require additional setup
- This POC uses EasyOCR's pre-trained models for text detection
- GPU acceleration is automatically detected and used if available
- Annotated images are saved with bounding boxes and confidence scores

## 🎯 Use Cases

- Document text extraction
- Image text detection
- OCR performance evaluation
- Custom model integration demonstration
- Quick proof of concept for stakeholders

## 🚨 Troubleshooting

### Import Errors
If you get import errors, try:
```bash
pip install --force-reinstall python-bidi
```

### No Images Found
Make sure you have images in the `images/` folder or update the demo_images list in the script.

### Performance Issues
- Use GPU if available for faster processing
- Reduce image size for faster processing
- Close other applications to free up memory 