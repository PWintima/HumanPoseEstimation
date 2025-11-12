"""
Data Augmentation & Curation Module for Stage 3.

This module provides augmentation functions to increase dataset diversity
and improve model generalization. Also includes dataset splitting utilities.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import shutil


def rotate_image(image: np.ndarray, angle: float, 
                border_value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Rotate image by a specified angle.

    Rotation is performed around the image center. The border
    color is used to fill empty areas created by rotation.

    Args:
        image: Input image in BGR format.
        angle: Rotation angle in degrees (positive = counter-clockwise).
        border_value: RGB tuple for border fill color.

    Returns:
        Rotated image in BGR format.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new dimensions to contain rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Apply rotation
    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_width, new_height),
        borderValue=border_value, flags=cv2.INTER_LINEAR
    )

    return rotated


def flip_image(image: np.ndarray, flip_code: int) -> np.ndarray:
    """
    Flip image horizontally or vertically.

    Args:
        image: Input image in BGR format.
        flip_code: 0 = vertical flip, 1 = horizontal flip, -1 = both.

    Returns:
        Flipped image in BGR format.
    """
    return cv2.flip(image, flip_code)


def adjust_brightness_contrast(image: np.ndarray,
                               brightness: float = 0.0,
                               contrast: float = 1.0) -> np.ndarray:
    """
    Adjust brightness and contrast of an image.

    Applies the transformation: output = alpha * input + beta
    where alpha controls contrast and beta controls brightness.

    Args:
        image: Input image in BGR format.
        brightness: Brightness adjustment (-100 to 100, typically).
        contrast: Contrast multiplier (>0, 1.0 = no change).

    Returns:
        Adjusted image in BGR format.
    """
    # Convert to float for calculation
    adjusted = image.astype(np.float32)

    # Apply contrast (multiply)
    adjusted = adjusted * contrast

    # Apply brightness (add)
    adjusted = adjusted + brightness

    # Clip to valid range and convert back to uint8
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    return adjusted


def apply_zoom(image: np.ndarray, zoom_factor: float,
              interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Apply zoom (crop and resize) to an image.

    Zooms into the center of the image by the specified factor.
    zoom_factor > 1.0 zooms in, < 1.0 zooms out.

    Args:
        image: Input image in BGR format.
        zoom_factor: Zoom multiplier (>1 = zoom in, <1 = zoom out).
        interpolation: Interpolation method for resizing.

    Returns:
        Zoomed image in BGR format.
    """
    height, width = image.shape[:2]

    # Calculate crop dimensions
    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)

    # Calculate crop start position (centered)
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    # Crop the center region
    cropped = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Resize back to original dimensions
    zoomed = cv2.resize(cropped, (width, height), interpolation=interpolation)

    return zoomed


def augment_image(image: np.ndarray,
                 augmentation_config: Optional[Dict] = None) -> np.ndarray:
    """
    Apply random augmentations to an image based on configuration.

    This function applies a combination of augmentation techniques
    based on the provided configuration dictionary. Each augmentation
    has a probability of being applied.

    Args:
        image: Input image in BGR format.
        augmentation_config: Dictionary with augmentation parameters:
            - 'rotation_range': Tuple (min_angle, max_angle) or None
            - 'rotation_prob': Probability of rotation
            - 'flip_horizontal': Probability of horizontal flip
            - 'flip_vertical': Probability of vertical flip
            - 'brightness_range': Tuple (min, max) or None
            - 'brightness_prob': Probability of brightness adjustment
            - 'contrast_range': Tuple (min, max) or None
            - 'contrast_prob': Probability of contrast adjustment
            - 'zoom_range': Tuple (min, max) or None
            - 'zoom_prob': Probability of zoom

    Returns:
        Augmented image in BGR format.
    """
    if augmentation_config is None:
        # Default augmentation configuration
        augmentation_config = {
            'rotation_range': (-15, 15),
            'rotation_prob': 0.5,
            'flip_horizontal': 0.5,
            'flip_vertical': 0.0,
            'brightness_range': (-30, 30),
            'brightness_prob': 0.5,
            'contrast_range': (0.8, 1.2),
            'contrast_prob': 0.5,
            'zoom_range': (0.9, 1.1),
            'zoom_prob': 0.3,
        }

    augmented = image.copy()

    # Apply rotation
    if (augmentation_config.get('rotation_range') and
            random.random() < augmentation_config.get('rotation_prob', 0)):
        min_angle, max_angle = augmentation_config['rotation_range']
        angle = random.uniform(min_angle, max_angle)
        augmented = rotate_image(augmented, angle)

    # Apply horizontal flip
    if random.random() < augmentation_config.get('flip_horizontal', 0):
        augmented = flip_image(augmented, 1)

    # Apply vertical flip
    if random.random() < augmentation_config.get('flip_vertical', 0):
        augmented = flip_image(augmented, 0)

    # Apply brightness adjustment
    if (augmentation_config.get('brightness_range') and
            random.random() < augmentation_config.get('brightness_prob', 0)):
        min_bright, max_bright = augmentation_config['brightness_range']
        brightness = random.uniform(min_bright, max_bright)
        contrast = 1.0
        if augmentation_config.get('contrast_range'):
            min_contrast, max_contrast = augmentation_config['contrast_range']
            if random.random() < augmentation_config.get('contrast_prob', 0):
                contrast = random.uniform(min_contrast, max_contrast)
        augmented = adjust_brightness_contrast(augmented, brightness, contrast)
    # Apply contrast only if brightness wasn't applied
    elif (augmentation_config.get('contrast_range') and
          random.random() < augmentation_config.get('contrast_prob', 0)):
        min_contrast, max_contrast = augmentation_config['contrast_range']
        contrast = random.uniform(min_contrast, max_contrast)
        augmented = adjust_brightness_contrast(augmented, 0.0, contrast)

    # Apply zoom
    if (augmentation_config.get('zoom_range') and
            random.random() < augmentation_config.get('zoom_prob', 0)):
        min_zoom, max_zoom = augmentation_config['zoom_range']
        zoom_factor = random.uniform(min_zoom, max_zoom)
        augmented = apply_zoom(augmented, zoom_factor)

    return augmented


def split_dataset(images_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 random_seed: int = 42) -> Dict[str, List[str]]:
    """
    Split dataset into train, validation, and test sets reproducibly.

    Uses a fixed random seed to ensure reproducible splits.
    Images are copied to respective subdirectories.

    Args:
        images_dir: Directory containing source images.
        output_dir: Base directory for split datasets.
        train_ratio: Proportion for training set (default 0.7).
        val_ratio: Proportion for validation set (default 0.2).
        test_ratio: Proportion for test set (default 0.1).
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping split names to lists of image paths.
    """
    # Validate ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Get all image paths
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        str(p) for p in Path(images_dir).glob('*')
        if p.suffix.lower() in image_extensions
    ]

    # Shuffle with fixed seed
    random.shuffle(image_paths)

    # Calculate split indices
    total_images = len(image_paths)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split paths
    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]

    # Create output directories
    splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }

    for split_name, paths in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Copy images to split directories
        for src_path in paths:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(split_dir, filename)
            shutil.copy2(src_path, dst_path)

        print(f"{split_name}: {len(paths)} images ({len(paths)/total_images*100:.1f}%)")

    return splits


def augment_dataset(images_dir: str,
                   output_dir: str,
                   augmentation_factor: int = 2,
                   augmentation_config: Optional[Dict] = None):
    """
    Create augmented versions of all images in a directory.

    Generates multiple augmented versions of each image and saves
    them to the output directory with modified filenames.

    Args:
        images_dir: Directory containing source images.
        output_dir: Directory to save augmented images.
        augmentation_factor: Number of augmented versions per image.
        augmentation_config: Augmentation configuration dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        str(p) for p in Path(images_dir).glob('*')
        if p.suffix.lower() in image_extensions
    ]

    print(f"Augmenting {len(image_paths)} images...")

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        base_name = Path(image_path).stem
        extension = Path(image_path).suffix

        # Save original
        original_output = os.path.join(output_dir, f"{base_name}_orig{extension}")
        cv2.imwrite(original_output, image)

        # Generate augmented versions
        for i in range(augmentation_factor):
            augmented = augment_image(image, augmentation_config)
            aug_output = os.path.join(
                output_dir, f"{base_name}_aug{i+1}{extension}"
            )
            cv2.imwrite(aug_output, augmented)

    print(f"Augmented images saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    images_dir = "images"
    output_dir = "outputs/augmented"

    # Split dataset
    splits = split_dataset(
        images_dir, "outputs/splits",
        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
        random_seed=42
    )

    # Augment training set only
    augment_dataset(
        "outputs/splits/train", output_dir,
        augmentation_factor=2
    )

