"""
Integrated Human Pose Estimation System

This module unifies all project components:
- Trained custom model inference (HRNet or SimpleBaseline)
- MediaPipe as validation/fallback
- Real-time webcam processing with visualization
- Hybrid approach combining both methods
- Skeleton visualization with confidence thresholds
- Performance metrics (FPS, inference time)

Usage:
    # Run webcam with trained model
    python integrated_pose_system.py --mode webcam --model_path path/to/checkpoint.pth
    
    # Run with MediaPipe for fast inference
    python integrated_pose_system.py --mode webcam --use_mediapipe
    
    # Hybrid mode: compare trained model vs MediaPipe
    python integrated_pose_system.py --mode webcam --model_path path/to/checkpoint.pth --hybrid
"""

import cv2
import torch
import numpy as np
import time
import argparse
from typing import Tuple, Optional, Dict
import os

from inference import PoseInference
from mediapipe_integration import MediaPipePoseEstimator


class IntegratedPoseSystem:
    """
    Unified pose estimation system combining trained models and MediaPipe.
    
    Supports:
    - Custom trained model inference
    - MediaPipe fast inference
    - Hybrid mode comparing both
    - Real-time webcam processing
    - Performance monitoring
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = 'simplebaseline',
                 use_mediapipe: bool = True,
                 hybrid_mode: bool = False,
                 device: Optional[torch.device] = None,
                 input_size: Tuple[int, int] = (256, 256),
                 confidence_threshold: float = 0.3):
        """
        Initialize integrated pose system.
        
        Args:
            model_path: Path to trained model checkpoint (optional)
            model_type: 'hrnet' or 'simplebaseline'
            use_mediapipe: Whether to use MediaPipe
            hybrid_mode: Compare both methods when True (requires model_path)
            device: Device to run inference on (auto-detect if None)
            input_size: Input image size for trained model
            confidence_threshold: Confidence threshold for visualization
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.hybrid_mode = hybrid_mode
        
        # Load trained model if provided
        self.trained_model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            self.trained_model = PoseInference(
                model_path=model_path,
                model_type=model_type,
                device=self.device,
                input_size=input_size
            )
            print("✓ Trained model loaded")
        elif model_path:
            print(f"⚠ Model path not found: {model_path}")
        
        # Load MediaPipe if requested
        self.mediapipe_model = None
        if use_mediapipe:
            print("Loading MediaPipe pose estimator...")
            self.mediapipe_model = MediaPipePoseEstimator(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ MediaPipe loaded")
        
        # Performance metrics
        self.frame_count = 0
        self.total_time = 0.0
        self.inference_times = []
        
        # Joint names and skeleton from dataset
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
    
    def predict_trained_model(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference with trained model.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Keypoints array [num_joints, 3] or None if model not loaded
        """
        if self.trained_model is None:
            return None
        
        try:
            keypoints = self.trained_model.predict_single_image(frame)
            return keypoints
        except Exception as e:
            print(f"Error in trained model inference: {e}")
            return None
    
    def predict_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference with MediaPipe.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Keypoints array [16, 3] (x, y, confidence) or None
        """
        if self.mediapipe_model is None:
            return None
        
        try:
            _, results = self.mediapipe_model.process_image(frame)
            if results.pose_landmarks is None:
                return None
            
            keypoints = self.mediapipe_model.extract_keypoints(results, frame.shape[:2])
            return keypoints
        except Exception as e:
            print(f"Error in MediaPipe inference: {e}")
            return None
    
    def visualize_pose(self,
                      frame: np.ndarray,
                      keypoints: np.ndarray,
                      label: str = "Pose",
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw skeleton on frame.
        
        Args:
            frame: Input frame
            keypoints: Keypoints array [num_joints, 3]
            label: Label to display (e.g., "Trained" or "MediaPipe")
            color: Color for skeleton (BGR)
        
        Returns:
            Frame with skeleton drawn
        """
        vis_frame = frame.copy()
        
        # Scale keypoints to image size
        scale_x = frame.shape[1] / self.input_size[1]
        scale_y = frame.shape[0] / self.input_size[0]
        
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] *= scale_x
        scaled_keypoints[:, 1] *= scale_y
        
        # Draw skeleton connections
        for connection in self.skeleton:
            start_joint = connection[0]
            end_joint = connection[1]
            
            if (scaled_keypoints[start_joint, 2] > self.confidence_threshold and
                scaled_keypoints[end_joint, 2] > self.confidence_threshold):
                
                start_point = (int(scaled_keypoints[start_joint, 0]),
                             int(scaled_keypoints[start_joint, 1]))
                end_point = (int(scaled_keypoints[end_joint, 0]),
                           int(scaled_keypoints[end_joint, 1]))
                
                cv2.line(vis_frame, start_point, end_point, color, 2)
        
        # Draw joints
        for i, (x, y, conf) in enumerate(scaled_keypoints):
            if conf > self.confidence_threshold:
                cv2.circle(vis_frame, (int(x), int(y)), 4, color, -1)
                cv2.putText(vis_frame, str(i), (int(x) + 5, int(y) - 5),
                           cv2.FONT_HERSHEY_SMALL, 0.4, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_webcam(self, camera_index: int = 0) -> None:
        """
        Process webcam feed with pose estimation.
        
        Args:
            camera_index: Webcam index (default: 0)
        
        Controls:
            q: quit
            s: save frame
            m: toggle between methods (in hybrid mode)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("=" * 60)
        print("INTEGRATED POSE ESTIMATION SYSTEM")
        print("=" * 60)
        
        if self.trained_model:
            print("✓ Trained Model: ACTIVE")
        else:
            print("✗ Trained Model: Not loaded")
        
        if self.mediapipe_model:
            print("✓ MediaPipe: ACTIVE")
        else:
            print("✗ MediaPipe: Not loaded")
        
        print("\nControls:")
        print("  q: Quit")
        print("  s: Save frame")
        if self.hybrid_mode:
            print("  m: Toggle method")
        print("=" * 60)
        
        show_trained = True  # For hybrid mode toggle
        frame_count = 0
        start_time = time.time()
        fps = 0.0  # Initialize fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            inference_start = time.time()
            
            # Get predictions
            trained_kpts = None
            mediapipe_kpts = None
            
            if self.hybrid_mode:
                if show_trained and self.trained_model:
                    trained_kpts = self.predict_trained_model(frame)
                if not show_trained and self.mediapipe_model:
                    mediapipe_kpts = self.predict_mediapipe(frame)
            else:
                if self.trained_model:
                    trained_kpts = self.predict_trained_model(frame)
                elif self.mediapipe_model:
                    mediapipe_kpts = self.predict_mediapipe(frame)
            
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # Visualize
            vis_frame = frame.copy()
            
            if trained_kpts is not None:
                vis_frame = self.visualize_pose(vis_frame, trained_kpts, "Trained", (0, 255, 0))
            
            if mediapipe_kpts is not None:
                vis_frame = self.visualize_pose(vis_frame, mediapipe_kpts, "MediaPipe", (255, 0, 0))
            
            # Add metrics
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
            
            cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, f'Inference: {inference_time*1000:.1f}ms', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            if self.hybrid_mode:
                method = "Trained" if show_trained else "MediaPipe"
                cv2.putText(vis_frame, f'Method: {method} (press m to toggle)', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            
            # Display
            cv2.imshow('Integrated Pose Estimation', vis_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f'pose_frame_{int(time.time())}.jpg'
                cv2.imwrite(save_path, vis_frame)
                print(f"Frame saved: {save_path}")
            elif key == ord('m') and self.hybrid_mode:
                show_trained = not show_trained
                print(f"Switched to: {'Trained' if show_trained else 'MediaPipe'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")
        print(f"Average inference time: {np.mean(self.inference_times)*1000:.1f}ms")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Integrated Pose Estimation System")
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='simplebaseline',
                       choices=['hrnet', 'simplebaseline'],
                       help='Type of trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--use_mediapipe', action='store_true', default=True,
                       help='Use MediaPipe for pose estimation')
    parser.add_argument('--hybrid', action='store_true',
                       help='Compare trained model vs MediaPipe (requires --model_path)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for visualization')
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
    
    # Initialize system
    system = IntegratedPoseSystem(
        model_path=args.model_path,
        model_type=args.model_type,
        use_mediapipe=args.use_mediapipe,
        hybrid_mode=args.hybrid,
        device=device,
        confidence_threshold=args.confidence
    )
    
    # Run webcam processing
    system.process_webcam(camera_index=args.camera)


if __name__ == '__main__':
    main()
