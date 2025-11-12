# Stage 3 Quick Start Guide

## Running Stage 3

### Complete Pipeline
```bash
python stage3_main.py
```

This will:
1. Analyze all images in the `images/` directory
2. Generate EDA reports and plots
3. Validate and clean sample images
4. Split dataset (70/20/10)
5. Augment training set
6. Generate visualizations
7. Create `dataset_stats.json`

### Testing with Limited Images

For faster testing, edit `stage3_main.py` and change:
```python
MAX_IMAGES_FOR_EDA = 100  # Process only 100 images
```

## Output Files

After running, check:
- `outputs/eda/` - All analysis results
- `outputs/validation/` - Before/after comparisons
- `outputs/splits/` - Train/val/test directories
- `dataset_stats.json` - Summary for Stage 4

## Key Modules

1. **data_explorer.py** - Statistics and EDA
2. **data_cleaner.py** - Image enhancement
3. **augmentor.py** - Augmentation and splitting
4. **visualize.py** - Visual comparisons
5. **stage3_main.py** - Main script

All modules follow PEP-8 and include comprehensive docstrings.

