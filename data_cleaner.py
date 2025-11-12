"""
Data Validation & Cleaning Module for Stage 3.

This module implements OpenCV-based image processing filters for denoising,
contrast enhancement, and normalization to improve image quality.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from data_explorer import compute_variance_of_laplacian, compute_noise_level


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Detect if an image is blurry using variance of Laplacian.

    The Laplacian operator highlights edges. Blurry images have
    fewer sharp edges, resulting in lower variance. A threshold
    is used to classify images as blurry or sharp.

    Args:
        image: Input image in BGR format.
        threshold: Blur threshold (lower = more sensitive to blur).

    Returns:
        Tuple of (is_blurry: bool, blur_score: float).
    """
    blur_score = compute_variance_of_laplacian(image)
    is_blurry = blur_score < threshold
    return is_blurry, blur_score


def remove_noisy_images(image_paths: List[str], 
                       noise_threshold: float = 10.0) -> Tuple[List[str], List[str]]:
    """
    Filter out images with high noise levels.

    Computes noise level for each image and separates them into
    clean and noisy sets based on a threshold.

    Args:
        image_paths: List of image file paths.
        noise_threshold: Maximum acceptable noise level (MAD).

    Returns:
        Tuple of (clean_images: List[str], noisy_images: List[str]).
    """
    clean_images = []
    noisy_images = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        noise_level = compute_noise_level(image)
        if noise_level <= noise_threshold:
            clean_images.append(image_path)
        else:
            noisy_images.append(image_path)

    return clean_images, noisy_images


def apply_clahe(image: np.ndarray, 
                clip_limit: float = 2.0, 
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE enhances contrast by applying histogram equalization
    to small regions of the image, preventing over-amplification
    of noise in uniform areas. This is particularly effective
    for improving visibility in low-contrast images.

    Args:
        image: Input image in BGR format.
        clip_limit: Threshold for contrast limiting (higher = more contrast).
        tile_grid_size: Grid size for local histogram equalization.

    Returns:
        Enhanced image in BGR format.
    """
    # Convert to LAB color space for better perceptual uniformity
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE to L channel only (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_enhanced = clahe.apply(l_channel)

    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_channel_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to adjust image brightness.

    Gamma correction uses a power-law transform to adjust
    the intensity values. Gamma < 1 brightens the image,
    gamma > 1 darkens it, and gamma = 1 has no effect.

    Args:
        image: Input image in BGR format.
        gamma: Gamma value (typically 0.5 to 2.0).

    Returns:
        Gamma-corrected image in BGR format.
    """
    # Build lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply lookup table using LUT
    corrected = cv2.LUT(image, table)
    return corrected


def apply_denoising(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Apply denoising filter to reduce image noise.

    Provides multiple denoising methods:
    - 'bilateral': Preserves edges while reducing noise
    - 'gaussian': Fast Gaussian blur
    - 'median': Effective for salt-and-pepper noise
    - 'nlmeans': Non-local means denoising (best quality, slower)

    Args:
        image: Input image in BGR format.
        method: Denoising method to use.

    Returns:
        Denoised image in BGR format.
    """
    if method == 'bilateral':
        # Bilateral filter preserves edges
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'gaussian':
        # Gaussian blur for general denoising
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        # Median filter for salt-and-pepper noise
        denoised = cv2.medianBlur(image, 5)
    elif method == 'nlmeans':
        # Non-local means denoising (best quality)
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
    else:
        raise ValueError(f"Unknown denoising method: {method}")

    return denoised


def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize image pixel values to a standard range.

    Normalization methods:
    - 'minmax': Scale to [0, 255] range
    - 'zscore': Zero-mean unit-variance normalization

    Args:
        image: Input image in BGR format.
        method: Normalization method.

    Returns:
        Normalized image (may need type conversion).
    """
    if method == 'minmax':
        # Min-max normalization to [0, 255]
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    elif method == 'zscore':
        # Z-score normalization (zero mean, unit variance)
        normalized = (image - image.mean()) / image.std()
        # Scale back to [0, 255] range
        normalized = ((normalized - normalized.min()) / 
                     (normalized.max() - normalized.min()) * 255)
        return normalized.astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_all_enhancements(image: np.ndarray,
                          use_denoising: bool = True,
                          use_clahe: bool = True,
                          use_gamma: bool = False,
                          gamma_value: float = 1.2) -> np.ndarray:
    """
    Apply a pipeline of enhancement operations to improve image quality.

    This function chains together multiple enhancement techniques
    in a logical order for optimal results.

    Args:
        image: Input image in BGR format.
        use_denoising: Whether to apply denoising.
        use_clahe: Whether to apply CLAHE contrast enhancement.
        use_gamma: Whether to apply gamma correction.
        gamma_value: Gamma value if gamma correction is enabled.

    Returns:
        Enhanced image in BGR format.
    """
    enhanced = image.copy()

    # Step 1: Denoising (apply before other enhancements)
    if use_denoising:
        enhanced = apply_denoising(enhanced, method='bilateral')

    # Step 2: CLAHE for contrast enhancement
    if use_clahe:
        enhanced = apply_clahe(enhanced, clip_limit=2.0)

    # Step 3: Gamma correction for brightness adjustment
    if use_gamma:
        enhanced = gamma_correction(enhanced, gamma=gamma_value)

    return enhanced


def compare_pre_post_metrics(original_image: np.ndarray,
                            enhanced_image: np.ndarray) -> dict:
    """
    Compare metrics between original and enhanced images.

    Computes various quality metrics for both images to
    quantify the improvement achieved by enhancement.

    Args:
        original_image: Original image in BGR format.
        enhanced_image: Enhanced image in BGR format.

    Returns:
        Dictionary with metric comparisons.
    """
    from data_explorer import (
        compute_brightness, compute_contrast,
        compute_variance_of_laplacian, compute_noise_level,
        compute_activity_measure
    )

    metrics_original = {
        'brightness': compute_brightness(original_image),
        'contrast': compute_contrast(original_image),
        'blur_score': compute_variance_of_laplacian(original_image),
        'noise_level': compute_noise_level(original_image),
        'activity': compute_activity_measure(original_image),
    }

    metrics_enhanced = {
        'brightness': compute_brightness(enhanced_image),
        'contrast': compute_contrast(enhanced_image),
        'blur_score': compute_variance_of_laplacian(enhanced_image),
        'noise_level': compute_noise_level(enhanced_image),
        'activity': compute_activity_measure(enhanced_image),
    }

    # Compute improvements (percentage change)
    improvements = {}
    for key in metrics_original:
        if metrics_original[key] != 0:
            improvement = ((metrics_enhanced[key] - metrics_original[key]) / 
                          metrics_original[key]) * 100
            improvements[f'{key}_improvement'] = improvement

    return {
        'original': metrics_original,
        'enhanced': metrics_enhanced,
        'improvements': improvements
    }


if __name__ == "__main__":
    # Example usage
    test_image_path = "images/000001163.jpg"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        if image is not None:
            enhanced = apply_all_enhancements(image)
            metrics = compare_pre_post_metrics(image, enhanced)
            print("Metric comparison:")
            print(metrics)

