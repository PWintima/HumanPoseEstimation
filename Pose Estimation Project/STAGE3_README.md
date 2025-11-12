# Stage 3: Dataset Preparation and Analysis

This directory contains the complete implementation of Stage 3 for the Human Pose Recognition project, focusing on classical image processing techniques using OpenCV.

## Overview

Stage 3 prepares, analyzes, and curates the dataset before model training. The emphasis is on demonstrating understanding of image quality improvement using classical image processing methods, not deep learning complexity.

## Module Structure

### 1. `data_explorer.py` - Exploratory Data Analysis
- **Functions:**
  - `compute_brightness()` - Calculates mean brightness
  - `compute_contrast()` - Computes contrast as standard deviation
  - `compute_variance_of_laplacian()` - Blur detection metric
  - `compute_noise_level()` - Estimates noise using MAD
  - `compute_activity_measure()` - Measures image texture/activity
  - `analyze_dataset()` - Batch analysis of all images
  - `generate_eda_plots()` - Creates histogram visualizations
  - `save_eda_results()` - Saves CSV and summary statistics

### 2. `data_cleaner.py` - Data Validation & Cleaning
- **Functions:**
  - `detect_blur()` - Detects blurry images using variance of Laplacian
  - `remove_noisy_images()` - Filters images based on noise threshold
  - `apply_clahe()` - Contrast Limited Adaptive Histogram Equalization
  - `gamma_correction()` - Brightness adjustment using gamma transform
  - `apply_denoising()` - Multiple denoising methods (bilateral, gaussian, median, NL-means)
  - `normalize_image()` - Image normalization (minmax, zscore)
  - `apply_all_enhancements()` - Complete enhancement pipeline
  - `compare_pre_post_metrics()` - Quantifies improvement after enhancement

### 3. `augmentor.py` - Data Augmentation & Curation
- **Functions:**
  - `rotate_image()` - Image rotation by specified angle
  - `flip_image()` - Horizontal/vertical flipping
  - `adjust_brightness_contrast()` - Brightness and contrast adjustment
  - `apply_zoom()` - Zoom in/out with cropping and resizing
  - `augment_image()` - Random augmentation pipeline
  - `split_dataset()` - Reproducible train/val/test split (70/20/10)
  - `augment_dataset()` - Batch augmentation of image directory

### 4. `visualize.py` - Visualization Tools
- **Functions:**
  - `create_before_after_comparison()` - Side-by-side comparison images
  - `create_metrics_overlay()` - Metrics display on images
  - `create_block_diagram()` - Pipeline flow diagram
  - `generate_quality_report()` - Batch quality report generation

### 5. `stage3_main.py` - Main Orchestration Script
- Orchestrates all Stage 3 operations
- Generates all required outputs
- Creates `dataset_stats.json` for Stage 4

## Usage

### Quick Start

```bash
# Run complete Stage 3 pipeline
python stage3_main.py
```

### Individual Module Usage

```python
# EDA Example
from data_explorer import analyze_dataset, generate_eda_plots
df = analyze_dataset("images", max_images=100)
generate_eda_plots(df, "outputs/eda")

# Cleaning Example
from data_cleaner import apply_all_enhancements
import cv2
image = cv2.imread("image.jpg")
enhanced = apply_all_enhancements(image)
cv2.imwrite("enhanced.jpg", enhanced)

# Augmentation Example
from augmentor import split_dataset, augment_image
splits = split_dataset("images", "outputs/splits")
augmented = augment_image(image)
```

## Output Structure

After running `stage3_main.py`, the following directory structure will be created:

```
outputs/
├── eda/
│   ├── eda_results.csv              # Full EDA metrics for all images
│   ├── eda_summary_statistics.csv   # Statistical summary
│   ├── blur_score_distribution.png  # Blur score histogram
│   ├── brightness_distribution.png  # Brightness histogram
│   ├── activity_distribution.png    # Activity histogram
│   └── eda_summary.png              # Combined summary plot
│
├── validation/
│   ├── pipeline_diagram.png         # Processing pipeline diagram
│   ├── *_comparison.png             # Before/after comparisons
│   └── *_metrics.png                # Metrics overlay images
│
├── splits/
│   ├── train/                       # 70% of dataset
│   ├── val/                         # 20% of dataset
│   └── test/                        # 10% of dataset
│
└── augmented/
    └── train/                       # Augmented training images

dataset_stats.json                   # Summary statistics for Stage 4
```

## Configuration

### Key Parameters

**EDA:**
- `MAX_IMAGES_FOR_EDA`: Set to `None` to process all images, or a number for testing

**Blur Detection:**
- `threshold=100.0`: Blur threshold (lower = more sensitive)

**Noise Filtering:**
- `noise_threshold=10.0`: Maximum acceptable noise level (MAD)

**Enhancement:**
- `clip_limit=2.0`: CLAHE contrast limit
- `gamma_value=1.2`: Gamma correction value

**Augmentation:**
- `augmentation_factor=2`: Number of augmented versions per image
- Default probabilities: rotation (50%), flip (50%), brightness (50%), contrast (50%), zoom (30%)

**Dataset Split:**
- `train_ratio=0.7`
- `val_ratio=0.2`
- `test_ratio=0.1`
- `random_seed=42` (for reproducibility)

## Image Processing Techniques Used

1. **Blur Detection**: Variance of Laplacian operator
2. **Noise Estimation**: Median Absolute Deviation (MAD)
3. **Denoising**: Bilateral filter, Gaussian blur, Median filter, Non-local means
4. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. **Brightness Adjustment**: Gamma correction
6. **Activity Measurement**: Gradient energy (Sobel operators)

## Code Standards

- **PEP-8 compliant**: All code follows Python PEP-8 style guidelines
- **Docstrings**: Every function includes comprehensive docstrings
- **Inline Comments**: Image processing operations are explained
- **Modular Design**: Each module is independent and reusable
- **Type Hints**: Functions include type annotations

## Dependencies

All required packages are listed in `requirements.txt`:
- opencv-python >= 4.5.0
- numpy >= 1.21.0
- matplotlib >= 3.3.0
- pandas (for data analysis)
- tqdm (for progress bars)

## Notes

- MediaPipe Pose is NOT used in Stage 3 (per requirements)
- Focus is entirely on classical image processing techniques
- All enhancements use OpenCV operations
- Results are reproducible with fixed random seeds
- Processing large datasets may take time (progress bars provided)

## Next Steps

After Stage 3 completion:
1. Review `dataset_stats.json` for dataset characteristics
2. Examine EDA plots to understand data distribution
3. Check validation outputs for enhancement quality
4. Use split datasets for Stage 4 model training

