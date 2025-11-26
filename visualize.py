"""
Visualization Module for Stage 3.

This module provides functions for creating before/after visual comparisons,
quality reports, and block diagrams for the dataset processing pipeline.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from typing import List, Tuple, Optional
from data_cleaner import compare_pre_post_metrics


def create_before_after_comparison(original: np.ndarray,
                                  enhanced: np.ndarray,
                                  title: str = "Before/After Comparison",
                                  save_path: Optional[str] = None) -> np.ndarray:
    """
    Create a side-by-side before/after comparison image matching the reference style.
    
    Creates a clean before/after comparison with labels in bottom corners,
    matching the style shown in validation images.

    Args:
        original: Original image in BGR format.
        enhanced: Enhanced image in BGR format.
        title: Title for the comparison (e.g., "Validation: filename.jpg").
        save_path: Optional path to save the comparison image.

    Returns:
        Comparison image as numpy array (BGR format).
    """
    # Ensure images have the same height
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]

    if h1 != h2:
        # Resize to match the smaller height
        target_height = min(h1, h2)
        scale1 = target_height / h1
        scale2 = target_height / h2
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        original = cv2.resize(original, (new_w1, target_height))
        enhanced = cv2.resize(enhanced, (new_w2, target_height))
        h1, w1 = original.shape[:2]
        h2, w2 = enhanced.shape[:2]

    # Concatenate horizontally with a small gap
    gap = 0  # No gap for cleaner look
    comparison = np.zeros((h1, w1 + w2 + gap, 3), dtype=np.uint8)
    comparison[:, :w1] = original
    comparison[:, w1 + gap:] = enhanced

    h, w = comparison.shape[:2]

    # Add BEFORE label (bottom right of left image)
    before_label = "BEFORE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(before_label, font, font_scale, thickness)
    
    # Position: bottom right corner of left image
    label_x = w1 - text_width - 15
    label_y = h1 - 15
    
    # Draw black rectangle background
    cv2.rectangle(comparison,
                 (label_x - 10, label_y - text_height - 10),
                 (label_x + text_width + 10, label_y + baseline + 5),
                 (0, 0, 0), -1)
    
    # Draw white text
    cv2.putText(comparison, before_label, (label_x, label_y),
               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Add AFTER label (bottom right of right image)
    after_label = "AFTER"
    (text_width, text_height), baseline = cv2.getTextSize(after_label, font, font_scale, thickness)
    
    # Position: bottom right corner of right image
    label_x = w - text_width - 15
    label_y = h - 15
    
    # Draw green rectangle background
    cv2.rectangle(comparison,
                 (label_x - 10, label_y - text_height - 10),
                 (label_x + text_width + 10, label_y + baseline + 5),
                 (0, 128, 0), -1)  # Green color
    
    # Draw white text
    cv2.putText(comparison, after_label, (label_x, label_y),
               font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Add title at the top center (if provided)
    if title:
        title_font_scale = 0.7
        title_thickness = 2
        (title_width, title_height), title_baseline = cv2.getTextSize(title, font, title_font_scale, title_thickness)
        title_x = (w - title_width) // 2
        title_y = 30
        
        # Draw semi-transparent background for title
        overlay = comparison.copy()
        cv2.rectangle(overlay,
                     (title_x - 15, title_y - title_height - 10),
                     (title_x + title_width + 15, title_y + title_baseline + 5),
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, comparison, 0.3, 0, comparison)
        
        # Draw title text
        cv2.putText(comparison, title, (title_x, title_y),
                   font, title_font_scale, (0, 0, 0), title_thickness, cv2.LINE_AA)

    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, comparison)
        print(f"Comparison saved to {save_path}")

    return comparison


def create_metrics_overlay(original: np.ndarray,
                          enhanced: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Create a visualization with metrics overlay comparing original and enhanced images.

    Displays images side-by-side with metric values displayed on each image.

    Args:
        original: Original image in BGR format.
        enhanced: Enhanced image in BGR format.
        save_path: Optional path to save the visualization.
    """
    metrics = compare_pre_post_metrics(original, enhanced)

    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Display original image with metrics
    axes[0].imshow(original_rgb)
    axes[0].axis('off')
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold')

    # Add metrics text overlay
    orig_metrics = metrics['original']
    metrics_text = (
        f"Brightness: {orig_metrics['brightness']:.1f}\n"
        f"Contrast: {orig_metrics['contrast']:.1f}\n"
        f"Blur Score: {orig_metrics['blur_score']:.1f}\n"
        f"Noise Level: {orig_metrics['noise_level']:.2f}\n"
        f"Activity: {orig_metrics['activity']:.1f}"
    )
    axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Display enhanced image with metrics
    axes[1].imshow(enhanced_rgb)
    axes[1].axis('off')
    axes[1].set_title('Enhanced Image', fontsize=16, fontweight='bold')

    # Add metrics text overlay
    enh_metrics = metrics['enhanced']
    improvements = metrics['improvements']
    metrics_text = (
        f"Brightness: {enh_metrics['brightness']:.1f} "
        f"({improvements['brightness_improvement']:+.1f}%)\n"
        f"Contrast: {enh_metrics['contrast']:.1f} "
        f"({improvements['contrast_improvement']:+.1f}%)\n"
        f"Blur Score: {enh_metrics['blur_score']:.1f} "
        f"({improvements['blur_score_improvement']:+.1f}%)\n"
        f"Noise Level: {enh_metrics['noise_level']:.2f} "
        f"({improvements['noise_level_improvement']:+.1f}%)\n"
        f"Activity: {enh_metrics['activity']:.1f} "
        f"({improvements['activity_improvement']:+.1f}%)"
    )
    axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics overlay saved to {save_path}")

    plt.close()


def create_block_diagram(save_path: Optional[str] = None):
    """
    Create a block diagram figure showing the Stage 3 processing pipeline.

    Visualizes the flow from raw images through EDA, cleaning,
    augmentation, and dataset splitting.

    Args:
        save_path: Optional path to save the block diagram.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define box positions and sizes
    boxes = [
        # Row 1: Input and EDA
        {'pos': (1, 6.5), 'size': (1.5, 0.8), 'label': 'Raw Images\nDataset', 'color': 'lightblue'},
        {'pos': (3.5, 6.5), 'size': (1.5, 0.8), 'label': 'EDA\nAnalysis', 'color': 'lightyellow'},
        {'pos': (6, 6.5), 'size': (1.5, 0.8), 'label': 'Quality\nMetrics', 'color': 'lightcoral'},
        
        # Row 2: Cleaning
        {'pos': (1, 4.5), 'size': (1.5, 0.8), 'label': 'Blur\nDetection', 'color': 'lightgreen'},
        {'pos': (3.5, 4.5), 'size': (1.5, 0.8), 'label': 'Denoising\nFilter', 'color': 'lightgreen'},
        {'pos': (6, 4.5), 'size': (1.5, 0.8), 'label': 'CLAHE\nEnhancement', 'color': 'lightgreen'},
        {'pos': (8.5, 4.5), 'size': (1.5, 0.8), 'label': 'Gamma\nCorrection', 'color': 'lightgreen'},
        
        # Row 3: Augmentation
        {'pos': (1, 2.5), 'size': (1.5, 0.8), 'label': 'Rotation\nFlip', 'color': 'plum'},
        {'pos': (3.5, 2.5), 'size': (1.5, 0.8), 'label': 'Brightness\nContrast', 'color': 'plum'},
        {'pos': (6, 2.5), 'size': (1.5, 0.8), 'label': 'Zoom\nAugmentation', 'color': 'plum'},
        
        # Row 4: Output
        {'pos': (3.5, 0.5), 'size': (1.5, 0.8), 'label': 'Train\n70%', 'color': 'wheat'},
        {'pos': (5.5, 0.5), 'size': (1.5, 0.8), 'label': 'Val\n20%', 'color': 'wheat'},
        {'pos': (7.5, 0.5), 'size': (1.5, 0.8), 'label': 'Test\n10%', 'color': 'wheat'},
    ]

    # Draw boxes
    for box in boxes:
        x, y = box['pos']
        w, h = box['size']
        fancy_box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(fancy_box)
        ax.text(x, y, box['label'], ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        # Row 1 flow
        ((1.75, 6.5), (2.75, 6.5)),
        ((5, 6.5), (5.75, 6.5)),
        
        # EDA to Cleaning (downward)
        ((3.5, 6.1), (2.5, 4.9)),
        ((3.5, 6.1), (4.25, 4.9)),
        
        # Cleaning flow
        ((2.25, 4.5), (2.75, 4.5)),
        ((5, 4.5), (5.75, 4.5)),
        ((7.5, 4.5), (8, 4.5)),
        
        # Cleaning to Augmentation (downward)
        ((2.5, 4.1), (1.75, 2.9)),
        ((4.25, 4.1), (3.5, 2.9)),
        ((6.75, 4.1), (6, 2.9)),
        
        # Augmentation flow
        ((2.25, 2.5), (2.75, 2.5)),
        ((5, 2.5), (5.75, 2.5)),
        
        # Augmentation to Splits (downward)
        ((3.5, 2.1), (4.25, 0.9)),
        ((5.5, 2.1), (4.5, 0.9)),
        ((6, 2.1), (6.75, 0.9)),
    ]

    for (start, end) in arrows:
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='->', mutation_scale=20,
            linewidth=2, color='black', alpha=0.6
        )
        ax.add_patch(arrow)

    # Add title
    ax.text(5, 7.5, 'Stage 3: Dataset Preparation Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold')

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='lightblue', label='Input'),
        patches.Patch(facecolor='lightyellow', label='EDA'),
        patches.Patch(facecolor='lightcoral', label='Analysis'),
        patches.Patch(facecolor='lightgreen', label='Cleaning'),
        patches.Patch(facecolor='plum', label='Augmentation'),
        patches.Patch(facecolor='wheat', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Block diagram saved to {save_path}")

    plt.close()


def generate_quality_report(image_paths: List[str],
                           output_dir: str,
                           sample_size: int = 10):
    """
    Generate a quality report with before/after comparisons for sample images.

    Processes a sample of images, applies enhancements, and creates
    comparison visualizations with metrics.

    Args:
        image_paths: List of image file paths to process.
        output_dir: Directory to save quality report images.
        sample_size: Number of sample images to process.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select sample images
    sample_paths = image_paths[:sample_size] if len(image_paths) > sample_size else image_paths

    from data_cleaner import apply_all_enhancements

    print(f"Generating quality report for {len(sample_paths)} sample images...")

    for i, image_path in enumerate(sample_paths):
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Apply enhancements
        enhanced = apply_all_enhancements(image)

        # Create comparison
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        create_before_after_comparison(
            image, enhanced,
            title=f"Sample {i+1}: {os.path.basename(image_path)}",
            save_path=comparison_path
        )

        # Create metrics overlay
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.png")
        create_metrics_overlay(image, enhanced, save_path=metrics_path)

    print(f"Quality report saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    output_dir = "outputs/validation"

    # Create block diagram
    create_block_diagram(os.path.join(output_dir, "pipeline_diagram.png"))

    # Example: Create comparison for a test image
    test_image_path = "images/000001163.jpg"
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        if image is not None:
            from data_cleaner import apply_all_enhancements
            enhanced = apply_all_enhancements(image)
            create_before_after_comparison(
                image, enhanced,
                save_path=os.path.join(output_dir, "test_comparison.png")
            )
            create_metrics_overlay(
                image, enhanced,
                save_path=os.path.join(output_dir, "test_metrics.png")
            )

