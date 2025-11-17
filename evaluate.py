"""
Evaluation Module for Human Pose Estimation Models

This module provides comprehensive evaluation metrics:
- PCK (Percentage of Correct Keypoints) at multiple thresholds
- PCKh (PCK normalized by head size)
- Average Precision (AP) metrics
- Joint-wise accuracy analysis
- Prediction visualization and comparison

These metrics are standard for evaluating pose estimation model performance.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os
from sklearn.metrics import average_precision_score
import json

from model import create_model
from dataset import MPIIPoseDataset


class PoseEvaluator:
    """
    Comprehensive evaluation metrics for pose estimation models.
    
    Implements standard pose estimation evaluation metrics including PCK,
    PCKh, AP, and joint-wise accuracy. Provides visualization tools for
    comparing predictions with ground truth.
    """
    
    def __init__(self, num_joints: int = 16):
        self.num_joints = num_joints
        
        # MPII joint names
        self.joint_names = [
            'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
            'pelvis', 'thorax', 'upper_neck', 'head_top',
            'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
        ]
        
        # MPII skeleton connections
        self.skeleton = [
            [0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],  # legs
            [6, 7], [7, 8], [8, 9],  # torso and head
            [7, 12], [12, 11], [11, 10],  # right arm
            [7, 13], [13, 14], [14, 15]  # left arm
        ]
    
    def compute_pck_accuracy(self, 
                           predicted_heatmaps: torch.Tensor, 
                           ground_truth_keypoints: torch.Tensor,
                           thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict[str, float]:
        """
        Compute PCK (Percentage of Correct Keypoints) accuracy
        
        Args:
            predicted_heatmaps: Predicted heatmaps [B, num_joints, H, W]
            ground_truth_keypoints: Ground truth keypoints [B, num_joints, 3] (x, y, visibility)
            thresholds: List of PCK thresholds
        
        Returns:
            Dictionary with PCK accuracies for each threshold
        """
        batch_size, num_joints, height, width = predicted_heatmaps.shape
        
        # Get predicted keypoint locations
        predicted_heatmaps_flat = predicted_heatmaps.view(batch_size, num_joints, -1)
        _, max_indices = torch.max(predicted_heatmaps_flat, dim=2)
        pred_y = max_indices // width
        pred_x = max_indices % width
        
        # Scale ground truth keypoints to heatmap size
        scale_x = width / 256.0  # Assuming input image size is 256x256
        scale_y = height / 256.0
        
        gt_x = ground_truth_keypoints[:, :, 0] * scale_x
        gt_y = ground_truth_keypoints[:, :, 1] * scale_y
        
        # Compute distances
        distances = torch.sqrt((pred_x.float() - gt_x) ** 2 + (pred_y.float() - gt_y) ** 2)
        
        # Check visibility
        visible = ground_truth_keypoints[:, :, 2] > 0
        
        # Compute PCK for each threshold
        pck_scores = {}
        for threshold in thresholds:
            correct = (distances < threshold) & visible
            pck = correct.sum().float() / visible.sum().float()
            pck_scores[f'PCK@{threshold}'] = pck.item()
        
        return pck_scores
    
    def compute_pckh_accuracy(self, 
                            predicted_heatmaps: torch.Tensor, 
                            ground_truth_keypoints: torch.Tensor,
                            head_size: torch.Tensor,
                            thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict[str, float]:
        """
        Compute PCKh (Percentage of Correct Keypoints with respect to head size) accuracy
        
        Args:
            predicted_heatmaps: Predicted heatmaps [B, num_joints, H, W]
            ground_truth_keypoints: Ground truth keypoints [B, num_joints, 3]
            head_size: Head size for normalization [B]
            thresholds: List of PCKh thresholds
        
        Returns:
            Dictionary with PCKh accuracies for each threshold
        """
        batch_size, num_joints, height, width = predicted_heatmaps.shape
        
        # Get predicted keypoint locations
        predicted_heatmaps_flat = predicted_heatmaps.view(batch_size, num_joints, -1)
        _, max_indices = torch.max(predicted_heatmaps_flat, dim=2)
        pred_y = max_indices // width
        pred_x = max_indices % width
        
        # Scale ground truth keypoints to heatmap size
        scale_x = width / 256.0
        scale_y = height / 256.0
        
        gt_x = ground_truth_keypoints[:, :, 0] * scale_x
        gt_y = ground_truth_keypoints[:, :, 1] * scale_y
        
        # Compute distances
        distances = torch.sqrt((pred_x.float() - gt_x) ** 2 + (pred_y.float() - gt_y) ** 2)
        
        # Normalize by head size
        head_size_scaled = head_size * min(scale_x, scale_y)
        normalized_distances = distances / head_size_scaled.unsqueeze(1)
        
        # Check visibility
        visible = ground_truth_keypoints[:, :, 2] > 0
        
        # Compute PCKh for each threshold
        pckh_scores = {}
        for threshold in thresholds:
            correct = (normalized_distances < threshold) & visible
            pckh = correct.sum().float() / visible.sum().float()
            pckh_scores[f'PCKh@{threshold}'] = pckh.item()
        
        return pckh_scores
    
    def compute_ap_metrics(self, 
                         predicted_heatmaps: torch.Tensor, 
                         ground_truth_keypoints: torch.Tensor,
                         thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]) -> Dict[str, float]:
        """
        Compute Average Precision (AP) metrics
        
        Args:
            predicted_heatmaps: Predicted heatmaps [B, num_joints, H, W]
            ground_truth_keypoints: Ground truth keypoints [B, num_joints, 3]
            thresholds: List of distance thresholds
        
        Returns:
            Dictionary with AP metrics
        """
        batch_size, num_joints, height, width = predicted_heatmaps.shape
        
        # Get predicted keypoint locations
        predicted_heatmaps_flat = predicted_heatmaps.view(batch_size, num_joints, -1)
        _, max_indices = torch.max(predicted_heatmaps_flat, dim=2)
        pred_y = max_indices // width
        pred_x = max_indices % width
        
        # Scale ground truth keypoints to heatmap size
        scale_x = width / 256.0
        scale_y = height / 256.0
        
        gt_x = ground_truth_keypoints[:, :, 0] * scale_x
        gt_y = ground_truth_keypoints[:, :, 1] * scale_y
        
        # Compute distances
        distances = torch.sqrt((pred_x.float() - gt_x) ** 2 + (pred_y.float() - gt_y) ** 2)
        
        # Check visibility
        visible = ground_truth_keypoints[:, :, 2] > 0
        
        # Compute AP for each threshold
        ap_scores = {}
        for threshold in thresholds:
            correct = (distances < threshold) & visible
            ap = correct.sum().float() / visible.sum().float()
            ap_scores[f'AP@{threshold}'] = ap.item()
        
        # Compute mAP (mean AP across all thresholds)
        ap_values = list(ap_scores.values())
        ap_scores['mAP'] = np.mean(ap_values)
        
        return ap_scores
    
    def compute_joint_wise_accuracy(self, 
                                  predicted_heatmaps: torch.Tensor, 
                                  ground_truth_keypoints: torch.Tensor,
                                  threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute joint-wise accuracy
        
        Args:
            predicted_heatmaps: Predicted heatmaps [B, num_joints, H, W]
            ground_truth_keypoints: Ground truth keypoints [B, num_joints, 3]
            threshold: Distance threshold
        
        Returns:
            Dictionary with accuracy for each joint
        """
        batch_size, num_joints, height, width = predicted_heatmaps.shape
        
        # Get predicted keypoint locations
        predicted_heatmaps_flat = predicted_heatmaps.view(batch_size, num_joints, -1)
        _, max_indices = torch.max(predicted_heatmaps_flat, dim=2)
        pred_y = max_indices // width
        pred_x = max_indices % width
        
        # Scale ground truth keypoints to heatmap size
        scale_x = width / 256.0
        scale_y = height / 256.0
        
        gt_x = ground_truth_keypoints[:, :, 0] * scale_x
        gt_y = ground_truth_keypoints[:, :, 1] * scale_y
        
        # Compute distances
        distances = torch.sqrt((pred_x.float() - gt_x) ** 2 + (pred_y.float() - gt_y) ** 2)
        
        # Check visibility
        visible = ground_truth_keypoints[:, :, 2] > 0
        
        # Compute accuracy for each joint
        joint_accuracies = {}
        for i, joint_name in enumerate(self.joint_names):
            joint_visible = visible[:, i]
            if joint_visible.sum() > 0:
                joint_distances = distances[:, i]
                joint_correct = (joint_distances < threshold) & joint_visible
                accuracy = joint_correct.sum().float() / joint_visible.sum().float()
                joint_accuracies[joint_name] = accuracy.item()
            else:
                joint_accuracies[joint_name] = 0.0
        
        return joint_accuracies
    
    def visualize_predictions(self, 
                             images: torch.Tensor,
                             predicted_heatmaps: torch.Tensor,
                             ground_truth_keypoints: torch.Tensor,
                             num_samples: int = 4,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize pose predictions
        
        Args:
            images: Input images [B, 3, H, W]
            predicted_heatmaps: Predicted heatmaps [B, num_joints, H, W]
            ground_truth_keypoints: Ground truth keypoints [B, num_joints, 3]
            num_samples: Number of samples to visualize
            save_path: Path to save visualization
        """
        batch_size = min(images.shape[0], num_samples)
        fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
        
        if batch_size == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(batch_size):
            # Original image
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Sample {i+1}')
            axes[0, i].axis('off')
            
            # Predicted pose
            pred_pose = self._heatmaps_to_keypoints(predicted_heatmaps[i:i+1])
            gt_pose = ground_truth_keypoints[i].cpu().numpy()
            
            axes[1, i].imshow(img)
            self._draw_skeleton(axes[1, i], pred_pose, color='red', alpha=0.8)
            self._draw_skeleton(axes[1, i], gt_pose, color='green', alpha=0.6)
            axes[1, i].set_title(f'Predicted (Red) vs GT (Green)')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def _heatmaps_to_keypoints(self, heatmaps: torch.Tensor) -> np.ndarray:
        """Convert heatmaps to keypoint coordinates"""
        batch_size, num_joints, height, width = heatmaps.shape
        keypoints = np.zeros((batch_size, num_joints, 2))
        
        for b in range(batch_size):
            for j in range(num_joints):
                heatmap = heatmaps[b, j].cpu().numpy()
                max_idx = np.argmax(heatmap)
                y, x = np.unravel_index(max_idx, heatmap.shape)
                keypoints[b, j, 0] = x
                keypoints[b, j, 1] = y
        
        return keypoints[0]  # Return first batch
    
    def _draw_skeleton(self, ax, keypoints: np.ndarray, color: str = 'red', alpha: float = 0.8):
        """Draw skeleton on image"""
        for connection in self.skeleton:
            start_joint = connection[0]
            end_joint = connection[1]
            
            if (keypoints[start_joint, 2] > 0 and keypoints[end_joint, 2] > 0):
                x_coords = [keypoints[start_joint, 0], keypoints[end_joint, 0]]
                y_coords = [keypoints[start_joint, 1], keypoints[end_joint, 1]]
                ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=alpha)
        
        # Draw joints
        visible_joints = keypoints[keypoints[:, 2] > 0]
        if len(visible_joints) > 0:
            ax.scatter(visible_joints[:, 0], visible_joints[:, 1], 
                      c=color, s=20, alpha=alpha)
    
    def evaluate_model(self, 
                      model: nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      save_visualizations: bool = True,
                      output_dir: str = 'evaluation_results') -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained pose estimation model
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            save_visualizations: Whether to save prediction visualizations
            output_dir: Directory to save results
        
        Returns:
            Dictionary with all evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_ground_truths = []
        all_images = []
        
        print("Running evaluation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(device)
                heatmaps = batch['heatmaps'].to(device)
                keypoints = batch['keypoints'].to(device)
                
                # Forward pass
                predicted_heatmaps = model(images)
                
                all_predictions.append(predicted_heatmaps.cpu())
                all_ground_truths.append(keypoints.cpu())
                all_images.append(images.cpu())
                
                if batch_idx % 100 == 0:
                    print(f"Processed {batch_idx * len(images)} samples...")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truths = torch.cat(all_ground_truths, dim=0)
        all_images = torch.cat(all_images, dim=0)
        
        print(f"Evaluating {len(all_predictions)} samples...")
        
        # Compute all metrics
        results = {}
        
        # PCK metrics
        pck_scores = self.compute_pck_accuracy(all_predictions, all_ground_truths)
        results.update(pck_scores)
        
        # AP metrics
        ap_scores = self.compute_ap_metrics(all_predictions, all_ground_truths)
        results.update(ap_scores)
        
        # Joint-wise accuracy
        joint_accuracies = self.compute_joint_wise_accuracy(all_predictions, all_ground_truths)
        results.update(joint_accuracies)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nPCK Accuracy:")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"  PCK@{threshold}: {results[f'PCK@{threshold}']:.4f}")
        
        print(f"\nMean AP: {results['mAP']:.4f}")
        
        print("\nJoint-wise Accuracy:")
        for joint_name, accuracy in joint_accuracies.items():
            print(f"  {joint_name}: {accuracy:.4f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save visualizations
        if save_visualizations:
            print("\nSaving visualizations...")
            self.visualize_predictions(
                all_images[:8],  # First 8 samples
                all_predictions[:8],
                all_ground_truths[:8],
                save_path=os.path.join(output_dir, 'predictions_visualization.png')
            )
        
        return results


def evaluate_trained_model(model_path: str, 
                          images_dir: str,
                          annotations_file: str,
                          model_type: str = 'simplebaseline',
                          device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to trained model checkpoint
        images_dir: Directory containing images
        annotations_file: Path to annotations file
        model_type: Type of model ('hrnet' or 'simplebaseline')
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(
        model_type=model_type,
        num_joints=config['num_joints'],
        backbone=config.get('backbone', 'resnet50')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create evaluation dataset
    eval_dataset = MPIIPoseDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        is_training=False
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Create evaluator
    evaluator = PoseEvaluator(num_joints=config['num_joints'])
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model=model,
        data_loader=eval_loader,
        device=device,
        output_dir='evaluation_results'
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate a trained model
    results = evaluate_trained_model(
        model_path='outputs/checkpoint_best.pth',
        images_dir='images',
        annotations_file='mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat',
        model_type='simplebaseline',
        device=device
    )
    
    print("Evaluation completed!")

