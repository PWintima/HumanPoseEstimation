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
    - 'gaussian': Fast Gaussian blur → remove high-frequency noise
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
        # Gaussian blur for general denoising → remove high-frequency noise
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


def apply_adaptive_smoothing(image: np.ndarray, blur_score: float, 
                             threshold: float = 100.0) -> np.ndarray:
    """
    Apply adaptive smoothing based on blur detection.
    
    Light cleaning for already-clear images, stronger cleaning for blurry ones.
    Uses Laplacian-based blur detection to determine smoothing strength.
    
    Args:
        image: Input image in BGR format.
        blur_score: Variance of Laplacian blur score.
        threshold: Blur threshold (lower = more sensitive to blur).
    
    Returns:
        Adaptively smoothed image in BGR format.
    """
    is_blurry = blur_score < threshold
    
    if is_blurry:
        # Stronger cleaning for blurry images
        # Use Gaussian + Median for blurry samples (less noise)
        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        smoothed = cv2.medianBlur(smoothed, 5)
    else:
        # Light cleaning for already-clear images
        # Use bilateral filter to preserve structure
        smoothed = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
    
    return smoothed


def enhance_edges_laplacian(image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Enhance edges using Laplacian operator → detect blur + preserve structure.
    
    The Laplacian operator highlights edges. This function uses it to
    enhance edge definition while preserving image structure.
    
    Args:
        image: Input image in BGR format.
        alpha: Strength of edge enhancement (0.0 to 1.0).
    
    Returns:
        Edge-enhanced image in BGR format.
    """
    # Convert to grayscale for Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Normalize Laplacian to [0, 255]
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    laplacian_normalized = laplacian_normalized.astype(np.uint8)
    
    # Convert back to BGR
    laplacian_bgr = cv2.cvtColor(laplacian_normalized, cv2.COLOR_GRAY2BGR)
    
    # Blend with original image to enhance edges
    enhanced = cv2.addWeighted(image, 1.0 - alpha, laplacian_bgr, alpha, 0)
    
    return enhanced


def normalize_brightness(image: np.ndarray, target_brightness: float = 128.0) -> np.ndarray:
    """
    Normalize brightness across images for consistency.
    
    Adjusts image brightness to a target value to ensure consistent
    brightness across the dataset, reducing model confusion.
    
    Args:
        image: Input image in BGR format.
        target_brightness: Target brightness value (0-255).
    
    Returns:
        Brightness-normalized image in BGR format.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Calculate current average brightness
    current_brightness = np.mean(l_channel)
    
    # Calculate adjustment factor
    if current_brightness > 0:
        adjustment = target_brightness / current_brightness
        # Apply adjustment with clipping
        l_channel_adjusted = np.clip(l_channel * adjustment, 0, 255).astype(np.uint8)
    else:
        l_channel_adjusted = l_channel
    
    # Merge channels and convert back to BGR
    lab_normalized = cv2.merge([l_channel_adjusted, a, b])
    normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    
    return normalized


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


def apply_stage4_preprocessing(image: np.ndarray, 
                               mean_brightness: float = 120.0) -> np.ndarray:
    """
    Apply Stage 4 preprocessing pipeline (best variant from report).
    
    Implements the optimal preprocessing variant: CLAHE + Gamma + Median
    which achieved 82% Good, 13% Acceptable, 5% Poor ratings.
    
    Pipeline order (matching Stage 4 report):
    1. Intensity normalization (to [0,255])
    2. CLAHE for local contrast improvement
    3. Gamma correction (tuned using Stage 3 brightness statistics)
    4. Median filtering for salt-and-pepper noise removal
    
    Args:
        image: Input image in BGR format.
        mean_brightness: Mean brightness from Stage 3 EDA (default: 120.0).
                        Used to tune gamma correction.
    
    Returns:
        Enhanced image in BGR format.
    """
    enhanced = image.copy()
    
    # Step 1: Intensity normalization to [0,255]
    enhanced = normalize_image(enhanced, method='minmax')
    
    # Step 2: CLAHE for local contrast improvement
    enhanced = apply_clahe(enhanced, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Step 3: Gamma correction tuned using Stage 3 brightness statistics
    # Gamma < 1 → brighten dark images, Gamma > 1 → compress overly bright images
    # Tune gamma based on mean brightness from EDA
    if mean_brightness < 100:
        gamma_value = 0.8  # Brighten dark images
    elif mean_brightness > 150:
        gamma_value = 1.3  # Compress overly bright images
    else:
        gamma_value = 1.0  # No adjustment needed
    
    enhanced = gamma_correction(enhanced, gamma=gamma_value)
    
    # Step 4: Median filtering for salt-and-pepper noise removal
    enhanced = apply_denoising(enhanced, method='median')
    
    return enhanced


def apply_all_enhancements(image: np.ndarray,
                          use_denoising: bool = True,
                          use_clahe: bool = True,
                          use_gamma: bool = False,
                          gamma_value: float = 1.2,
                          use_adaptive_smoothing: bool = True,
                          use_edge_enhancement: bool = True,
                          use_brightness_normalization: bool = True,
                          use_stage4_pipeline: bool = False,
                          mean_brightness: float = 120.0) -> np.ndarray:
    """
    Apply a pipeline of enhancement operations to improve image quality.
    
    If use_stage4_pipeline=True, applies Stage 4 optimal preprocessing:
    CLAHE + Gamma + Median (82% Good rating from Stage 4 report).
    
    Otherwise, implements Stage 2: Image Cleaning techniques:
    - Gaussian filtering → remove high-frequency noise
    - Median filtering → remove salt-and-pepper noise
    - Laplacian → detect blur + preserve structure
    - Adaptive smoothing → light cleaning for already-clear images
    - Blur detection → using Variance of Laplacian
    
    Achieves improvements:
    - Less noise in blurry samples
    - Clearer outlines of limbs
    - Better edge definition
    - Consistent brightness across images
    - Reduces model confusion

    This function chains together multiple enhancement techniques
    in a logical order for optimal results.

    Args:
        image: Input image in BGR format.
        use_denoising: Whether to apply denoising.
        use_clahe: Whether to apply CLAHE contrast enhancement.
        use_gamma: Whether to apply gamma correction.
        gamma_value: Gamma value if gamma correction is enabled.
        use_adaptive_smoothing: Whether to use adaptive smoothing based on blur.
        use_edge_enhancement: Whether to enhance edges using Laplacian.
        use_brightness_normalization: Whether to normalize brightness.
        use_stage4_pipeline: If True, use Stage 4 optimal preprocessing (CLAHE+Gamma+Median).
        mean_brightness: Mean brightness from EDA for gamma tuning (default: 120.0).

    Returns:
        Enhanced image in BGR format.
    """
    # Use Stage 4 optimal preprocessing if requested
    if use_stage4_pipeline:
        return apply_stage4_preprocessing(image, mean_brightness=mean_brightness)
    
    enhanced = image.copy()
    
    # Step 0: Detect blur for adaptive processing
    blur_score = compute_variance_of_laplacian(image)
    is_blurry = blur_score < 100.0

    # Step 1: Adaptive smoothing based on blur detection
    # Light cleaning for already-clear images, stronger for blurry ones
    if use_adaptive_smoothing:
        enhanced = apply_adaptive_smoothing(enhanced, blur_score, threshold=100.0)
    elif use_denoising:
        # Fallback to standard denoising if adaptive smoothing disabled
        if is_blurry:
            # For blurry images: Gaussian + Median (less noise in blurry samples)
            enhanced = apply_denoising(enhanced, method='gaussian')
            enhanced = apply_denoising(enhanced, method='median')
        else:
            # For clear images: Bilateral (preserves edges)
            enhanced = apply_denoising(enhanced, method='bilateral')

    # Step 2: Edge enhancement using Laplacian (better edge definition, clearer outlines)
    if use_edge_enhancement:
        # Use stronger enhancement for blurry images
        alpha = 0.4 if is_blurry else 0.2
        enhanced = enhance_edges_laplacian(enhanced, alpha=alpha)

    # Step 3: CLAHE for contrast enhancement (improves visibility)
    if use_clahe:
        enhanced = apply_clahe(enhanced, clip_limit=2.0)

    # Step 4: Brightness normalization (consistent brightness across images)
    if use_brightness_normalization:
        enhanced = normalize_brightness(enhanced, target_brightness=128.0)

    # Step 5: Gamma correction for brightness adjustment (optional)
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

