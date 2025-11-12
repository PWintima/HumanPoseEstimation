"""
Exploratory Data Analysis (EDA) Module for Stage 3.

This module provides functions to analyze dataset images using classical
image processing techniques. It computes quality metrics including brightness,
contrast, blur scores, and noise levels.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple


def compute_brightness(image: np.ndarray) -> float:
    """
    Compute mean brightness of an image.

    Converts image to grayscale and calculates the mean pixel intensity,
    which represents overall brightness of the image.

    Args:
        image: Input image in BGR format (OpenCV default).

    Returns:
        Mean brightness value (0-255 scale).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_contrast(image: np.ndarray) -> float:
    """
    Compute contrast as standard deviation of pixel intensities.

    Higher standard deviation indicates greater contrast between
    light and dark regions in the image.

    Args:
        image: Input image in BGR format.

    Returns:
        Contrast value (standard deviation).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def compute_variance_of_laplacian(image: np.ndarray) -> float:
    """
    Compute variance of Laplacian as a blur metric.

    The Laplacian operator highlights regions of rapid intensity change.
    Blurry images have low variance, while sharp images have high variance.
    This is a commonly used metric for blur detection.

    Args:
        image: Input image in BGR format.

    Returns:
        Variance of Laplacian (blur score).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_noise_level(image: np.ndarray) -> float:
    """
    Estimate noise level using median absolute deviation (MAD).

    Applies a median filter to estimate the noise-free image,
    then computes the median absolute deviation of residuals
    as a robust measure of noise level.

    Args:
        image: Input image in BGR format.

    Returns:
        Estimated noise level (MAD value).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply median filter to estimate noise-free image
    denoised = cv2.medianBlur(gray, 5)
    # Compute residuals
    residuals = gray.astype(np.float64) - denoised.astype(np.float64)
    # Median absolute deviation as robust noise estimate
    mad = np.median(np.abs(residuals - np.median(residuals)))
    return float(mad)


def compute_activity_measure(image: np.ndarray) -> float:
    """
    Compute image activity measure (energy of gradient).

    Calculates the sum of squared gradients, which measures
    overall image activity/texture. Higher values indicate
    more detailed or textured regions.

    Args:
        image: Input image in BGR format.

    Returns:
        Activity measure (energy of gradients).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Return mean energy
    return float(np.mean(gradient_magnitude**2))


def analyze_image(image_path: str) -> Dict[str, float]:
    """
    Analyze a single image and compute all quality metrics.

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary containing all computed metrics.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return {
        'filename': os.path.basename(image_path),
        'brightness': compute_brightness(image),
        'contrast': compute_contrast(image),
        'blur_score': compute_variance_of_laplacian(image),
        'noise_level': compute_noise_level(image),
        'activity': compute_activity_measure(image),
        'width': image.shape[1],
        'height': image.shape[0],
    }


def analyze_dataset(images_dir: str, max_images: int = None) -> pd.DataFrame:
    """
    Perform EDA on all images in the dataset directory.

    Iterates through all images, computes quality metrics,
    and returns a DataFrame with results.

    Args:
        images_dir: Directory containing images.
        max_images: Maximum number of images to process (None for all).

    Returns:
        DataFrame with columns: filename, brightness, contrast,
        blur_score, noise_level, activity, width, height.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        str(p) for p in Path(images_dir).glob('*')
        if p.suffix.lower() in image_extensions
    ]

    if max_images:
        image_paths = image_paths[:max_images]

    results = []
    print(f"Analyzing {len(image_paths)} images...")

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            metrics = analyze_image(image_path)
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    df = pd.DataFrame(results)
    return df


def generate_eda_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate and save EDA visualization plots.

    Creates three main plots:
    1. Histogram of blur scores
    2. Mean intensity (brightness) distribution
    3. Activity distribution

    Args:
        df: DataFrame with computed metrics.
        output_dir: Directory to save plot files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style for better-looking plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    fig_size = (12, 4)

    # 1. Histogram of blur scores
    plt.figure(figsize=fig_size)
    plt.hist(df['blur_score'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Blur Score (Variance of Laplacian)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Blur Scores in Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blur_score_distribution.png'), dpi=300)
    plt.close()

    # 2. Mean intensity (brightness) distribution
    plt.figure(figsize=fig_size)
    plt.hist(df['brightness'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Mean Brightness')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mean Brightness in Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'brightness_distribution.png'), dpi=300)
    plt.close()

    # 3. Activity distribution
    plt.figure(figsize=fig_size)
    plt.hist(df['activity'], bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Activity Measure (Gradient Energy)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Image Activity in Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_distribution.png'), dpi=300)
    plt.close()

    # 4. Combined comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(df['blur_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Blur Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Blur Score Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(df['brightness'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Brightness')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Brightness Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(df['activity'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Activity Measure')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Activity Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(df['noise_level'], bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Noise Level (MAD)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Noise Level Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_summary.png'), dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")


def save_eda_results(df: pd.DataFrame, output_dir: str):
    """
    Save EDA results as CSV and generate summary statistics.

    Args:
        df: DataFrame with computed metrics.
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full results to CSV
    csv_path = os.path.join(output_dir, 'eda_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"EDA results saved to {csv_path}")

    # Generate and save summary statistics
    summary = df.describe()
    summary_path = os.path.join(output_dir, 'eda_summary_statistics.csv')
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to {summary_path}")

    # Print summary to console
    print("\n=== EDA Summary Statistics ===")
    print(summary)


if __name__ == "__main__":
    # Example usage
    images_dir = "images"
    output_dir = "outputs/eda"

    df = analyze_dataset(images_dir, max_images=100)  # Test with 100 images
    save_eda_results(df, output_dir)
    generate_eda_plots(df, output_dir)

