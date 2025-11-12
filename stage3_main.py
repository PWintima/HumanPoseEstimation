"""
Stage 3 Main Script: Dataset Preparation and Analysis

This script orchestrates the complete Stage 3 pipeline:
1. Exploratory Data Analysis (EDA)
2. Data Validation & Cleaning
3. Data Augmentation & Curation
4. Visualization and Reporting

Follows PEP-8 standards and generates all required deliverables.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import our modules
from data_explorer import (
    analyze_dataset, save_eda_results, generate_eda_plots
)
from data_cleaner import (
    detect_blur, remove_noisy_images, apply_all_enhancements,
    compare_pre_post_metrics
)
from augmentor import split_dataset, augment_dataset
from visualize import (
    create_before_after_comparison, create_metrics_overlay,
    create_block_diagram, generate_quality_report
)


def create_output_directories():
    """Create necessary output directory structure."""
    directories = [
        "outputs",
        "outputs/eda",
        "outputs/validation",
        "outputs/augmented",
        "outputs/splits",
        "outputs/splits/train",
        "outputs/splits/val",
        "outputs/splits/test",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Output directories created.")


def run_eda(images_dir: str, max_images: int = None) -> Dict:
    """
    Run Exploratory Data Analysis on the dataset.

    Args:
        images_dir: Directory containing images.
        max_images: Maximum number of images to process (None for all).

    Returns:
        Dictionary with EDA summary statistics.
    """
    print("\n" + "="*60)
    print("STAGE 3.1: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Analyze dataset
    df = analyze_dataset(images_dir, max_images=max_images)

    # Save results
    save_eda_results(df, "outputs/eda")
    generate_eda_plots(df, "outputs/eda")

    # Compute summary statistics for JSON output
    summary = {
        'total_images': len(df),
        'mean_brightness': float(df['brightness'].mean()),
        'std_brightness': float(df['brightness'].std()),
        'mean_contrast': float(df['contrast'].mean()),
        'std_contrast': float(df['contrast'].std()),
        'mean_blur_score': float(df['blur_score'].mean()),
        'std_blur_score': float(df['blur_score'].std()),
        'mean_noise_level': float(df['noise_level'].mean()),
        'std_noise_level': float(df['noise_level'].std()),
        'mean_activity': float(df['activity'].mean()),
        'std_activity': float(df['activity'].std()),
        'blurry_count': int((df['blur_score'] < 100.0).sum()),
        'blurry_percentage': float((df['blur_score'] < 100.0).mean() * 100),
    }

    return summary, df


def run_data_cleaning(images_dir: str, sample_size: int = 20) -> Dict:
    """
    Run data validation and cleaning operations.

    Args:
        images_dir: Directory containing images.
        sample_size: Number of sample images for validation.

    Returns:
        Dictionary with cleaning statistics.
    """
    print("\n" + "="*60)
    print("STAGE 3.2: DATA VALIDATION & CLEANING")
    print("="*60)

    # Get sample images for validation
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        str(p) for p in Path(images_dir).glob('*')
        if p.suffix.lower() in image_extensions
    ][:sample_size]

    blurry_count = 0
    enhancement_improvements = []

    print(f"Processing {len(image_paths)} sample images for validation...")

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Detect blur
        is_blurry, blur_score = detect_blur(image, threshold=100.0)
        if is_blurry:
            blurry_count += 1

        # Apply enhancements
        enhanced = apply_all_enhancements(image)

        # Compare metrics
        metrics = compare_pre_post_metrics(image, enhanced)
        enhancement_improvements.append(metrics['improvements'])

        # Create comparison visualizations (first 5 only)
        if len([p for p in Path("outputs/validation").glob("*_comparison.png")]) < 5:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            create_before_after_comparison(
                image, enhanced,
                title=f"Validation: {os.path.basename(image_path)}",
                save_path=f"outputs/validation/{base_name}_comparison.png"
            )
            create_metrics_overlay(
                image, enhanced,
                save_path=f"outputs/validation/{base_name}_metrics.png"
            )

    # Generate quality report
    generate_quality_report(image_paths, "outputs/validation", sample_size=min(10, len(image_paths)))

    # Compute average improvements
    avg_improvements = {}
    if enhancement_improvements:
        for key in enhancement_improvements[0].keys():
            avg_improvements[key] = float(np.mean([imp[key] for imp in enhancement_improvements]))

    cleaning_stats = {
        'samples_processed': len(image_paths),
        'blurry_images_detected': blurry_count,
        'blurry_percentage': float(blurry_count / len(image_paths) * 100) if image_paths else 0,
        'average_improvements': avg_improvements,
    }

    return cleaning_stats


def run_augmentation_and_splitting(images_dir: str) -> Dict:
    """
    Run data augmentation and dataset splitting.

    Args:
        images_dir: Directory containing images.

    Returns:
        Dictionary with augmentation and splitting statistics.
    """
    print("\n" + "="*60)
    print("STAGE 3.3: DATA AUGMENTATION & CURATION")
    print("="*60)

    # Split dataset
    print("Splitting dataset into train/val/test...")
    splits = split_dataset(
        images_dir,
        "outputs/splits",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=42
    )

    # Augment training set
    print("\nAugmenting training set...")
    augment_dataset(
        "outputs/splits/train",
        "outputs/augmented/train",
        augmentation_factor=2
    )

    # Count augmented images
    train_aug_count = len(list(Path("outputs/augmented/train").glob("*")))

    augmentation_stats = {
        'train_count': len(splits['train']),
        'val_count': len(splits['val']),
        'test_count': len(splits['test']),
        'train_after_augmentation': train_aug_count,
        'augmentation_factor': 2,
        'total_after_augmentation': train_aug_count + len(splits['val']) + len(splits['test']),
    }

    return augmentation_stats


def generate_dataset_stats_json(eda_summary: Dict,
                                cleaning_stats: Dict,
                                augmentation_stats: Dict,
                                output_path: str = "dataset_stats.json"):
    """
    Generate comprehensive dataset statistics JSON file.

    This JSON file will be used by Stage 4 for model training.

    Args:
        eda_summary: EDA summary statistics.
        cleaning_stats: Data cleaning statistics.
        augmentation_stats: Augmentation and splitting statistics.
        output_path: Path to save JSON file.
    """
    stats = {
        'stage': 3,
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_images': eda_summary['total_images'],
            'image_quality': {
                'mean_brightness': eda_summary['mean_brightness'],
                'mean_contrast': eda_summary['mean_contrast'],
                'mean_blur_score': eda_summary['mean_blur_score'],
                'mean_noise_level': eda_summary['mean_noise_level'],
                'blurry_percentage': eda_summary['blurry_percentage'],
            }
        },
        'preprocessing': {
            'cleaning_applied': True,
            'enhancements': {
                'denoising': True,
                'clahe': True,
                'gamma_correction': False,
            },
            'average_improvements': cleaning_stats.get('average_improvements', {}),
        },
        'augmentation': {
            'augmentation_factor': augmentation_stats['augmentation_factor'],
            'train_count_original': augmentation_stats['train_count'],
            'train_count_augmented': augmentation_stats['train_after_augmentation'],
        },
        'splits': {
            'train': {
                'count': augmentation_stats['train_count'],
                'percentage': 70.0,
            },
            'val': {
                'count': augmentation_stats['val_count'],
                'percentage': 20.0,
            },
            'test': {
                'count': augmentation_stats['test_count'],
                'percentage': 10.0,
            },
        },
    }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\nDataset statistics saved to {output_path}")


def main():
    """Main execution function for Stage 3."""
    print("\n" + "="*60)
    print("HUMAN POSE RECOGNITION - STAGE 3")
    print("Dataset Preparation and Analysis")
    print("="*60)

    # Configuration
    IMAGES_DIR = "images"
    MAX_IMAGES_FOR_EDA = None  # Set to None to process all images, or a number for testing

    # Create output directories
    create_output_directories()

    # Create block diagram
    print("\nGenerating pipeline block diagram...")
    create_block_diagram("outputs/validation/pipeline_diagram.png")

    # Step 1: Exploratory Data Analysis
    eda_summary, eda_df = run_eda(IMAGES_DIR, max_images=MAX_IMAGES_FOR_EDA)

    # Step 2: Data Validation & Cleaning
    cleaning_stats = run_data_cleaning(IMAGES_DIR, sample_size=20)

    # Step 3: Data Augmentation & Splitting
    augmentation_stats = run_augmentation_and_splitting(IMAGES_DIR)

    # Generate dataset_stats.json
    generate_dataset_stats_json(eda_summary, cleaning_stats, augmentation_stats)

    # Print final summary
    print("\n" + "="*60)
    print("STAGE 3 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nTotal images analyzed: {eda_summary['total_images']}")
    print(f"Train set: {augmentation_stats['train_count']} images")
    print(f"Validation set: {augmentation_stats['val_count']} images")
    print(f"Test set: {augmentation_stats['test_count']} images")
    print(f"Train set after augmentation: {augmentation_stats['train_after_augmentation']} images")
    print(f"\nOutputs saved to:")
    print("  - outputs/eda/ (EDA results and plots)")
    print("  - outputs/validation/ (before/after comparisons)")
    print("  - outputs/splits/ (train/val/test splits)")
    print("  - outputs/augmented/ (augmented training images)")
    print("  - dataset_stats.json (summary statistics for Stage 4)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

