"""
Inference Module for Human Pose Estimation

This module provides inference capabilities for trained pose estimation models:
- Single image pose estimation
- Batch processing
- Video processing (frame-by-frame)
- Real-time webcam processing
- Pose visualization with skeleton drawing

Supports both HRNet and SimpleBaseline models. Can process images, videos,
or live webcam feeds.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
from typing import List, Tuple, Optional, Dict
import json
import time

from model import create_model
from dataset import MPIIPoseDataset


class PoseInference:
    """
    Inference class for pose estimation models.
    
    Provides methods to run inference on images, videos, or webcam feeds
    using trained pose estimation models. Includes preprocessing, postprocessing,
    and visualization utilities.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'simplebaseline',
                 device: torch.device = None,
                 input_size: Tuple[int, int] = (256, 256)):
        """
        Initialize pose inference
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('hrnet' or 'simplebaseline')
            device: Device to run inference on
            input_size: Input image size (height, width)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.input_size = input_size
        
        # Load model
        self.model, self.config = self._load_model(model_path, model_type)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Joint names and skeleton
        self.joint_names = [
            'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
            'pelvis', 'thorax', 'upper_neck', 'head_top',
            'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
        ]
        
        self.skeleton = [
            [0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],  # legs
            [6, 7], [7, 8], [8, 9],  # torso and head
            [7, 12], [12, 11], [11, 10],  # right arm
            [7, 13], [13, 14], [14, 15]  # left arm
        ]
        
        print(f"Model loaded on {self.device}")
        print(f"Model type: {model_type}")
        print(f"Number of joints: {self.config['num_joints']}")
    
    def _load_model(self, model_path: str, model_type: str) -> Tuple[nn.Module, Dict]:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        model = create_model(
            model_type=model_type,
            num_joints=config['num_joints'],
            backbone=config.get('backbone', 'resnet50')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)
        
        return tensor_image.to(self.device)
    
    def postprocess_heatmaps(self, heatmaps: torch.Tensor) -> np.ndarray:
        """Convert heatmaps to keypoint coordinates"""
        batch_size, num_joints, height, width = heatmaps.shape
        
        keypoints = np.zeros((num_joints, 3))  # x, y, confidence
        
        for j in range(num_joints):
            heatmap = heatmaps[0, j].cpu().numpy()
            
            # Find peak
            max_val = np.max(heatmap)
            if max_val > 0.1:  # Threshold for confidence
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                keypoints[j, 0] = x
                keypoints[j, 1] = y
                keypoints[j, 2] = max_val
            else:
                keypoints[j, 2] = 0  # Not detected
        
        return keypoints
    
    def predict_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Predict pose for a single image
        
        Args:
            image: Input image as numpy array (BGR or RGB)
        
        Returns:
            Keypoints array [num_joints, 3] (x, y, confidence)
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            heatmaps = self.model(input_tensor)
        
        # Postprocess
        keypoints = self.postprocess_heatmaps(heatmaps)
        
        return keypoints
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict pose for a batch of images
        
        Args:
            images: List of input images
        
        Returns:
            List of keypoints arrays
        """
        results = []
        
        for image in images:
            keypoints = self.predict_single_image(image)
            results.append(keypoints)
        
        return results
    
    def visualize_pose(self, 
                      image: np.ndarray, 
                      keypoints: np.ndarray,
                      confidence_threshold: float = 0.3,
                      save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize pose on image
        
        Args:
            image: Original image
            keypoints: Predicted keypoints [num_joints, 3]
            confidence_threshold: Minimum confidence for drawing
            save_path: Path to save visualization
        
        Returns:
            Image with pose visualization
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        # Scale keypoints to image size
        scale_x = image.shape[1] / self.input_size[1]
        scale_y = image.shape[0] / self.input_size[0]
        
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] *= scale_x
        scaled_keypoints[:, 1] *= scale_y
        
        # Draw skeleton
        for connection in self.skeleton:
            start_joint = connection[0]
            end_joint = connection[1]
            
            if (scaled_keypoints[start_joint, 2] > confidence_threshold and 
                scaled_keypoints[end_joint, 2] > confidence_threshold):
                
                start_point = (int(scaled_keypoints[start_joint, 0]), 
                             int(scaled_keypoints[start_joint, 1]))
                end_point = (int(scaled_keypoints[end_joint, 0]), 
                           int(scaled_keypoints[end_joint, 1]))
                
                cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw joints
        for i, (x, y, conf) in enumerate(scaled_keypoints):
            if conf > confidence_threshold:
                cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                # Add joint label
                cv2.putText(vis_image, str(i), (int(x) + 5, int(y) - 5), 
                           cv2.FONT_HERSHEY_SMALL, 0.5, (255, 255, 255), 1)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def process_video(self, 
                     video_path: str, 
                     output_path: str,
                     confidence_threshold: float = 0.3,
                     max_frames: Optional[int] = None) -> None:
        """
        Process video for pose estimation
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            confidence_threshold: Minimum confidence for drawing
            max_frames: Maximum number of frames to process
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Predict pose
            keypoints = self.predict_single_image(frame)
            
            # Visualize pose
            vis_frame = self.visualize_pose(frame, keypoints, confidence_threshold)
            
            # Write frame
            out.write(vis_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time
                print(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video processing completed. Output saved to: {output_path}")
    
    def process_webcam(self, confidence_threshold: float = 0.3) -> None:
        """
        Process webcam feed for real-time pose estimation
        
        Args:
            confidence_threshold: Minimum confidence for drawing
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict pose
            keypoints = self.predict_single_image(frame)
            
            # Visualize pose
            vis_frame = self.visualize_pose(frame, keypoints, confidence_threshold)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
            
            cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Pose Estimation', vis_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f'webcam_frame_{int(time.time())}.jpg'
                cv2.imwrite(save_path, vis_frame)
                print(f"Frame saved to: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Pose Estimation Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='simplebaseline',
                       choices=['hrnet', 'simplebaseline'],
                       help='Type of model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image/video path or "webcam" for live feed')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Output path for image or video')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for visualization')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process for video')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize inference
    inference = PoseInference(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device
    )
    
    # Process input
    if args.input.lower() == 'webcam':
        # Live webcam processing
        inference.process_webcam(confidence_threshold=args.confidence)
    
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video processing
        inference.process_video(
            video_path=args.input,
            output_path=args.output,
            confidence_threshold=args.confidence,
            max_frames=args.max_frames
        )
    
    else:
        # Single image processing
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return
        
        # Load image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image {args.input}")
            return
        
        print(f"Processing image: {args.input}")
        
        # Predict pose
        keypoints = inference.predict_single_image(image)
        
        # Visualize pose
        vis_image = inference.visualize_pose(image, keypoints, args.confidence)
        
        # Save result
        cv2.imwrite(args.output, vis_image)
        print(f"Result saved to: {args.output}")
        
        # Print keypoint information
        print("\nDetected keypoints:")
        for i, (name, (x, y, conf)) in enumerate(zip(inference.joint_names, keypoints)):
            if conf > args.confidence:
                print(f"  {name}: ({x:.1f}, {y:.1f}) - confidence: {conf:.3f}")


if __name__ == "__main__":
    main()

