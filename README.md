# Human Pose Estimation Training System

A comprehensive Python-based system for training human pose estimation models using deep learning. This system supports both HRNet and SimpleBaseline architectures and can work with MPII dataset annotations.

## Features

- **Multiple Model Architectures**: HRNet and SimpleBaseline models
- **MediaPipe Integration**: Fast, real-time pose estimation
- **Hybrid Approach**: Combine MediaPipe speed with custom model accuracy
- **Comprehensive Dataset Support**: MPII dataset with custom image collections
- **Advanced Training Pipeline**: Data augmentation, learning rate scheduling, early stopping
- **Evaluation Metrics**: PCK, PCKh, AP, and joint-wise accuracy
- **Inference Tools**: Support for images, videos, and real-time webcam processing
- **Model Comparison**: Side-by-side comparison of different approaches
- **Performance Benchmarking**: Speed and accuracy comparisons
- **Visualization**: Pose visualization and training curve plotting

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
   ```

## Project Structure

```
PoseEstimation/
├── dataset.py              # Dataset loader for MPII annotations
├── model.py                # HRNet and SimpleBaseline model architectures
├── train.py                # Training pipeline with comprehensive utilities
├── evaluate.py             # Evaluation metrics and model assessment
├── inference.py            # Inference script for trained models
├── mediapipe_integration.py # MediaPipe integration and hybrid approach
├── enhanced_inference.py   # Enhanced inference with MediaPipe + custom models
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── test_setup.py          # Setup verification script
├── quick_start.py         # Quick start guide
├── images/                # Your image dataset (24,984 images)
├── mpii_human_pose_v1_u12_2/  # MPII dataset annotations
│   ├── mpii_human_pose_v1_u12_1.mat
│   └── README.md
└── outputs/               # Training outputs (created during training)
```

## Quick Start

### 1. Training a Model

```bash
python train.py
```

This will:
- Load your images from the `images/` folder
- Use MPII annotations from `mpii_human_pose_v1_u12_2/`
- Train a SimpleBaseline model (default)
- Save checkpoints and training curves to `outputs/`

### 2. Training Configuration

Edit the `create_config()` function in `train.py` to customize:

```python
config = {
    'model_type': 'simplebaseline',  # or 'hrnet'
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'backbone': 'resnet50',  # for SimpleBaseline
    # ... more options
}
```

### 3. Running Inference

**Option A: MediaPipe Only (Fast, No Training Required)**
```bash
# Process a single image with MediaPipe
python enhanced_inference.py --mode mediapipe --input image.jpg --output result.jpg

# Process a video with MediaPipe
python enhanced_inference.py --mode mediapipe --input video.mp4 --output result.mp4

# Real-time webcam with MediaPipe
python enhanced_inference.py --mode webcam --input webcam
```

**Option B: Custom Trained Model**
```bash
# Process a single image
python inference.py --model_path outputs/checkpoint_best.pth --input image.jpg --output result.jpg

# Process a video
python inference.py --model_path outputs/checkpoint_best.pth --input video.mp4 --output result.mp4

# Real-time webcam processing
python inference.py --model_path outputs/checkpoint_best.pth --input webcam
```

**Option C: Hybrid Approach (MediaPipe + Custom Model)**
```bash
# Compare both models on an image
python enhanced_inference.py --mode hybrid --input image.jpg --custom_model outputs/checkpoint_best.pth --output comparison.jpg

# Side-by-side comparison
python enhanced_inference.py --mode compare --input image.jpg --custom_model outputs/checkpoint_best.pth

# Benchmark performance
python enhanced_inference.py --mode benchmark --input image.jpg --custom_model outputs/checkpoint_best.pth
```

### 4. Evaluating a Model

```bash
python evaluate.py
```

This will evaluate your trained model and generate comprehensive metrics.

### 5. MediaPipe Integration

The system now includes comprehensive MediaPipe integration:

**MediaPipe Advantages:**
- ⚡ **Fast**: Real-time processing without GPU
- 🎯 **Accurate**: State-of-the-art pose detection
- 📱 **Mobile Optimized**: Works on mobile devices
- 🔧 **No Training**: Ready to use out of the box

**Hybrid Approach Benefits:**
- 🚀 **Speed**: Use MediaPipe for real-time applications
- 🎯 **Accuracy**: Use custom models for highest precision
- 🔍 **Comparison**: Side-by-side evaluation of both approaches
- 📊 **Benchmarking**: Performance analysis and optimization

**When to Use Each Approach:**
- **MediaPipe**: Real-time applications, mobile deployment, quick prototyping
- **Custom Models**: Highest accuracy requirements, specific domain adaptation
- **Hybrid**: Best of both worlds, comprehensive evaluation

## Detailed Usage

### Training Options

**Model Types**:
- `simplebaseline`: Faster training, good performance
- `hrnet`: State-of-the-art performance, longer training time

**Key Training Parameters**:
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-3)
- `optimizer`: 'adam', 'adamw', or 'sgd'
- `scheduler`: 'step', 'cosine', or 'plateau'

### Dataset Configuration

The system automatically:
- Loads images from the `images/` folder
- Uses MPII annotations for ground truth
- Creates train/validation splits
- Applies data augmentation (rotation, scaling, flipping)

### Model Architectures

**SimpleBaseline**:
- Uses ResNet backbone (ResNet50/101)
- Deconvolution layers for upsampling
- Faster training and inference
- Good for getting started

**HRNet**:
- High-Resolution Network architecture
- Maintains high-resolution features throughout
- State-of-the-art performance
- More computationally intensive

### Evaluation Metrics

The system provides comprehensive evaluation:

- **PCK@0.1-0.5**: Percentage of Correct Keypoints at different thresholds
- **PCKh**: PCK normalized by head size
- **AP**: Average Precision metrics
- **Joint-wise Accuracy**: Per-joint performance analysis

### Inference Features

**Image Processing**:
- Single image pose estimation
- Batch processing support
- Confidence threshold filtering

**Video Processing**:
- Frame-by-frame pose estimation
- Real-time processing capability
- Frame rate optimization

**Webcam Processing**:
- Live pose estimation
- Real-time visualization
- Frame saving capability

## Advanced Configuration

### Custom Dataset

To use your own dataset:

1. Place images in the `images/` folder
2. Create annotations in MPII format or modify `dataset.py`
3. Update the dataset loading logic

### Model Customization

**Adding New Architectures**:
1. Implement in `model.py`
2. Add to `create_model()` function
3. Update training configuration

**Custom Loss Functions**:
1. Modify the loss function in `train.py`
2. Add custom evaluation metrics in `evaluate.py`

### Training Monitoring

The training system provides:
- Real-time loss and accuracy logging
- Training curve visualization
- Model checkpointing
- Early stopping support

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

**Slow Training**:
- Increase batch size (if memory allows)
- Use more workers for data loading
- Enable pin_memory

**Poor Performance**:
- Check data quality and annotations
- Adjust learning rate
- Try different model architectures
- Increase training epochs

### Performance Tips

1. **Use GPU**: Training is much faster with CUDA
2. **Batch Size**: Larger batches generally improve performance
3. **Data Augmentation**: Helps with generalization
4. **Learning Rate**: Start with 1e-3, adjust based on convergence
5. **Early Stopping**: Prevents overfitting

## File Descriptions

- **`dataset.py`**: Handles MPII dataset loading, data augmentation, and preprocessing
- **`model.py`**: Implements HRNet and SimpleBaseline architectures
- **`train.py`**: Complete training pipeline with monitoring and checkpointing
- **`evaluate.py`**: Comprehensive evaluation metrics and visualization
- **`inference.py`**: Inference tools for images, videos, and webcam

## Output Files

Training creates:
- `outputs/training_YYYYMMDD_HHMMSS/`: Timestamped training directory
- `checkpoint_best.pth`: Best model checkpoint
- `checkpoint_latest.pth`: Latest model checkpoint
- `config.json`: Training configuration
- `training.log`: Training logs
- `training_curves.png`: Loss and accuracy plots

## Next Steps

1. **Start with SimpleBaseline**: Good for initial experiments
2. **Monitor Training**: Watch loss curves and validation metrics
3. **Experiment with Parameters**: Try different learning rates, batch sizes
4. **Evaluate Thoroughly**: Use comprehensive evaluation metrics
5. **Optimize for Your Use Case**: Adjust confidence thresholds, visualization

## Support

For issues or questions:
1. Check the training logs in `outputs/`
2. Verify your data format and annotations
3. Ensure all dependencies are installed correctly
4. Check GPU memory usage and batch size settings

## License

This project uses the MPII Human Pose Dataset. Please refer to the MPII dataset license for usage terms.
