# Cloud GPU OCR Training Setup Guide

This guide will help you run the OCR model training on cloud GPU platforms to achieve much better accuracy without overheating your local machine.

## 🚀 Quick Start Options

### Option 1: Google Colab (Free/Paid)
**Best for: Quick testing, free GPU access**

1. **Open Google Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Create a new notebook

2. **Upload Files**
   ```python
   # Upload the training script
   from google.colab import files
   files.upload()  # Upload train_cloud_gpu.py
   ```

3. **Install Dependencies**
   ```python
   !pip install -r cloud_requirements.txt
   ```

4. **Run Training**
   ```python
   !python train_cloud_gpu.py
   ```

5. **Download Results**
   ```python
   from google.colab import files
   files.download('checkpoints/cloud_gpu_model.pth')
   files.download('training_history.png')
   ```

### Option 2: AWS EC2 (Paid)
**Best for: Long training, high performance**

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - Select GPU instance (g4dn.xlarge or p3.2xlarge)
   - Configure security group to allow SSH

2. **Connect and Setup**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and dependencies
   sudo apt install python3 python3-pip python3-venv -y
   
   # Create virtual environment
   python3 -m venv ocr_env
   source ocr_env/bin/activate
   
   # Install PyTorch with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Install other dependencies
   pip install -r cloud_requirements.txt
   ```

3. **Upload and Run**
   ```bash
   # Upload your files (use scp or git clone)
   scp -i your-key.pem train_cloud_gpu.py ubuntu@your-instance-ip:~/
   scp -i your-key.pem cloud_requirements.txt ubuntu@your-instance-ip:~/
   
   # Run training
   python3 train_cloud_gpu.py
   ```

### Option 3: Google Cloud Platform (Paid)
**Best for: Enterprise, managed ML**

1. **Create VM Instance**
   ```bash
   gcloud compute instances create ocr-training \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator="type=nvidia-tesla-t4,count=1" \
     --image-family=debian-11 \
     --image-project=debian-cloud \
     --maintenance-policy=TERMINATE \
     --restart-on-failure
   ```

2. **Setup GPU Drivers**
   ```bash
   # Install NVIDIA drivers
   curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
   sudo python3 install_gpu_driver.py
   ```

3. **Install Dependencies and Run**
   ```bash
   # Similar to AWS setup
   sudo apt update
   sudo apt install python3 python3-pip -y
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install -r cloud_requirements.txt
   python3 train_cloud_gpu.py
   ```

### Option 4: Paperspace Gradient (Free/Paid)
**Best for: ML-focused platform**

1. **Create Notebook**
   - Go to [gradient.paperspace.com](https://gradient.paperspace.com)
   - Create new notebook with GPU

2. **Upload and Run**
   ```bash
   # Upload files through web interface
   pip install -r cloud_requirements.txt
   python train_cloud_gpu.py
   ```

## 📊 Expected Results

With GPU training, you should achieve:
- **Training Time**: 30-60 minutes (vs 6+ hours on CPU)
- **Accuracy**: 85-95% on synthetic data
- **Model Size**: ~50-100MB
- **Inference Speed**: <100ms per image

## 🔧 Customization Options

### Adjust Training Parameters
Edit `train_cloud_gpu.py` to modify:

```python
# Training settings
EPOCHS = 100          # Increase for better accuracy
BATCH_SIZE = 64       # Adjust based on GPU memory
LEARNING_RATE = 0.001 # Adjust if training is unstable

# Dataset settings
num_samples = 20000   # Increase for more data
IMG_WIDTH = 128       # Adjust image size
```

### Add Real Data
Replace synthetic data with real images:

```python
class RealOCRDataset(Dataset):
    def __init__(self, image_dir, labels_file):
        # Load real images and labels
        pass
```

## 📁 File Structure

```
easyosr/
├── train_cloud_gpu.py          # Main training script
├── cloud_requirements.txt      # Dependencies
├── CLOUD_SETUP_GUIDE.md       # This guide
├── checkpoints/               # Model checkpoints
│   ├── cloud_gpu_model.pth    # Best model
│   └── cloud_gpu_model_epoch_*.pth  # Checkpoints
└── training_history.png       # Training plots
```

## 🚨 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   BATCH_SIZE = 32  # or 16
   ```

2. **Training Too Slow**
   ```python
   # Increase batch size if memory allows
   BATCH_SIZE = 128
   # Use mixed precision
   from torch.cuda.amp import autocast, GradScaler
   ```

3. **Model Not Learning**
   ```python
   # Check learning rate
   LEARNING_RATE = 0.0001  # Try lower
   # Increase training data
   num_samples = 50000
   ```

### GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 💰 Cost Estimation

| Platform | GPU Type | Cost/Hour | Est. Total Cost |
|----------|----------|-----------|-----------------|
| Google Colab | Tesla T4 | Free | $0 |
| AWS EC2 | Tesla T4 | $0.526 | ~$0.50 |
| AWS EC2 | Tesla V100 | $3.06 | ~$3.00 |
| GCP | Tesla T4 | $0.35 | ~$0.35 |
| Paperspace | RTX 4000 | $0.59 | ~$0.60 |

*Estimates based on 1-hour training time*

## 🎯 Next Steps

1. **Choose a platform** from the options above
2. **Upload the training script** to your chosen platform
3. **Install dependencies** using the provided requirements
4. **Run training** and monitor progress
5. **Download the trained model** when complete
6. **Test locally** using the existing test scripts

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify GPU drivers are properly installed
3. Ensure all dependencies are installed
4. Monitor GPU memory usage during training

The cloud training should produce a much more accurate model that can actually recognize text properly! 