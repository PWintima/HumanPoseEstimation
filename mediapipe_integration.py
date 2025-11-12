import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from PIL import Image
import json

from model import create_model
from dataset import MPIIPoseDataset


class MediaPipePoseEstimator:
    """MediaPipe pose estimation wrapper"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 2,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose estimator
        
        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks
            enable_segmentation: Whether to enable segmentation
            smooth_segmentation: Whether to smooth segmentation
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe to MPII joint mapping
        self.mpii_mapping = {
            'nose': 9,           # head_top
            'left_eye_inner': 9,  # head_top
            'left_eye': 9,        # head_top
            'left_eye_outer': 9,   # head_top
            'right_eye_inner': 9,  # head_top
            'right_eye': 9,        # head_top
            'right_eye_outer': 9,  # head_top
            'left_ear': 9,         # head_top
            'right_ear': 9,        # head_top
            'mouth_left': 9,       # head_top
            'mouth_right': 9,      # head_top
            'left_shoulder': 13,   # l_shoulder
            'right_shoulder': 12,  # r_shoulder
            'left_elbow': 14,      # l_elbow
            'right_elbow': 11,     # r_elbow
            'left_wrist': 15,      # l_wrist
            'right_wrist': 10,     # r_wrist
            'left_pinky': 15,      # l_wrist
            'right_pinky': 10,     # r_wrist
            'left_index': 15,      # l_wrist
            'right_index': 10,     # r_wrist
            'left_thumb': 15,       # l_wrist
            'right_thumb': 10,     # r_wrist
            'left_hip': 3,          # l_hip
            'right_hip': 2,        # r_hip
            'left_knee': 4,         # l_knee
            'right_knee': 1,       # r_knee
            'left_ankle': 5,       # l_ankle
            'right_ankle': 0,      # r_ankle
            'left_heel': 5,        # l_ankle
            'right_heel': 0,       # r_ankle
            'left_foot_index': 5,   # l_ankle
            'right_foot_index': 0   # r_ankle
        }
        
        # MPII joint names
        self.mpii_joint_names = [
            'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
            'pelvis', 'thorax', 'upper_neck', 'head_top',
            'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
        ]
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process image with MediaPipe
        
        Args:
            image: Input image (BGR or RGB)
        
        Returns:
            Tuple of (annotated_image, results_dict)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Process image
        results = self.pose.process(rgb_image)
        
        # Draw pose landmarks
        annotated_image = rgb_image.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Convert back to BGR
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        return annotated_image, results
    
    def extract_keypoints(self, results, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract keypoints in MPII format
        
        Args:
            results: MediaPipe results
            image_shape: (height, width) of image
        
        Returns:
            Keypoints array [16, 3] (x, y, visibility)
        """
        keypoints = np.zeros((16, 3))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Map MediaPipe landmarks to MPII format
            for mp_landmark_name, mpii_idx in self.mpii_mapping.items():
                if hasattr(landmarks, mp_landmark_name):
                    landmark = getattr(landmarks, mp_landmark_name)
                    keypoints[mpii_idx, 0] = landmark.x * image_shape[1]  # x
                    keypoints[mpii_idx, 1] = landmark.y * image_shape[0]  # y
                    keypoints[mpii_idx, 2] = landmark.visibility  # visibility
        
            # Calculate pelvis (midpoint between hips)
            if keypoints[2, 2] > 0 and keypoints[3, 2] > 0:  # Both hips visible
                keypoints[6, 0] = (keypoints[2, 0] + keypoints[3, 0]) / 2
                keypoints[6, 1] = (keypoints[2, 1] + keypoints[3, 1]) / 2
                keypoints[6, 2] = min(keypoints[2, 2], keypoints[3, 2])
            
            # Calculate thorax (midpoint between shoulders)
            if keypoints[12, 2] > 0 and keypoints[13, 2] > 0:  # Both shoulders visible
                keypoints[7, 0] = (keypoints[12, 0] + keypoints[13, 0]) / 2
                keypoints[7, 1] = (keypoints[12, 1] + keypoints[13, 1]) / 2
                keypoints[7, 2] = min(keypoints[12, 2], keypoints[13, 2])
            
            # Calculate upper neck (midpoint between thorax and head)
            if keypoints[7, 2] > 0 and keypoints[9, 2] > 0:  # Thorax and head visible
                keypoints[8, 0] = (keypoints[7, 0] + keypoints[9, 0]) / 2
                keypoints[8, 1] = (keypoints[7, 1] + keypoints[9, 1]) / 2
                keypoints[8, 2] = min(keypoints[7, 2], keypoints[9, 2])
        
        return keypoints
    
    def process_video(self, 
                     video_path: str, 
                     output_path: str,
                     max_frames: Optional[int] = None) -> None:
        """
        Process video with MediaPipe
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Processing video with MediaPipe: {total_frames} frames at {fps} FPS")
        
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
            
            # Process frame
            annotated_frame, results = self.process_image(frame)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time
                print(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"MediaPipe video processing completed. Output saved to: {output_path}")
    
    def process_webcam(self) -> None:
        """Process webcam feed with MediaPipe"""
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
            
            # Process frame
            annotated_frame, results = self.process_image(frame)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
            
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('MediaPipe Pose Estimation', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f'mediapipe_frame_{int(time.time())}.jpg'
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame saved to: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()


class HybridPoseEstimator:
    """Hybrid pose estimator combining MediaPipe and custom models"""
    
    def __init__(self, 
                 custom_model_path: Optional[str] = None,
                 model_type: str = 'simplebaseline',
                 device: torch.device = None,
                 use_mediapipe: bool = True,
                 use_custom_model: bool = True):
        """
        Initialize hybrid pose estimator
        
        Args:
            custom_model_path: Path to trained custom model
            model_type: Type of custom model
            device: Device for custom model
            use_mediapipe: Whether to use MediaPipe
            use_custom_model: Whether to use custom model
        """
        self.use_mediapipe = use_mediapipe
        self.use_custom_model = use_custom_model
        
        # Initialize MediaPipe
        if self.use_mediapipe:
            self.mediapipe_estimator = MediaPipePoseEstimator()
            print("✓ MediaPipe initialized")
        
        # Initialize custom model
        if self.use_custom_model and custom_model_path:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.device = device
            self.custom_model, self.config = self._load_custom_model(custom_model_path, model_type)
            print(f"✓ Custom model loaded on {device}")
        else:
            self.custom_model = None
    
    def _load_custom_model(self, model_path: str, model_type: str):
        """Load custom trained model"""
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
    
    def predict_hybrid(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict pose using both MediaPipe and custom model
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with predictions from both models
        """
        results = {}
        
        # MediaPipe prediction
        if self.use_mediapipe:
            annotated_image, mp_results = self.mediapipe_estimator.process_image(image)
            mp_keypoints = self.mediapipe_estimator.extract_keypoints(mp_results, image.shape[:2])
            results['mediapipe'] = {
                'keypoints': mp_keypoints,
                'annotated_image': annotated_image,
                'results': mp_results
            }
        
        # Custom model prediction
        if self.use_custom_model and self.custom_model is not None:
            # Preprocess for custom model
            input_tensor = self._preprocess_for_custom_model(image)
            
            with torch.no_grad():
                heatmaps = self.custom_model(input_tensor)
            
            custom_keypoints = self._postprocess_heatmaps(heatmaps)
            results['custom_model'] = {
                'keypoints': custom_keypoints,
                'heatmaps': heatmaps.cpu().numpy()
            }
        
        return results
    
    def _preprocess_for_custom_model(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for custom model"""
        from torchvision import transforms
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor_image = transform(pil_image).unsqueeze(0)
        return tensor_image.to(self.device)
    
    def _postprocess_heatmaps(self, heatmaps: torch.Tensor) -> np.ndarray:
        """Convert heatmaps to keypoints"""
        batch_size, num_joints, height, width = heatmaps.shape
        
        keypoints = np.zeros((num_joints, 3))
        
        for j in range(num_joints):
            heatmap = heatmaps[0, j].cpu().numpy()
            max_val = np.max(heatmap)
            if max_val > 0.1:
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                keypoints[j, 0] = x
                keypoints[j, 1] = y
                keypoints[j, 2] = max_val
            else:
                keypoints[j, 2] = 0
        
        return keypoints
    
    def visualize_comparison(self, 
                           image: np.ndarray, 
                           results: Dict[str, np.ndarray],
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize comparison between MediaPipe and custom model
        
        Args:
            image: Original image
            results: Results from predict_hybrid
            save_path: Path to save visualization
        
        Returns:
            Comparison visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # MediaPipe result
        if 'mediapipe' in results:
            mp_image = results['mediapipe']['annotated_image']
            axes[1].imshow(cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title('MediaPipe')
            axes[1].axis('off')
        
        # Custom model result
        if 'custom_model' in results:
            custom_image = image.copy()
            custom_keypoints = results['custom_model']['keypoints']
            
            # Draw custom model keypoints
            scale_x = image.shape[1] / 256.0
            scale_y = image.shape[0] / 256.0
            
            for i, (x, y, conf) in enumerate(custom_keypoints):
                if conf > 0.3:
                    scaled_x = x * scale_x
                    scaled_y = y * scale_y
                    cv2.circle(custom_image, (int(scaled_x), int(scaled_y)), 3, (0, 0, 255), -1)
                    cv2.putText(custom_image, str(i), (int(scaled_x) + 5, int(scaled_y) - 5), 
                               cv2.FONT_HERSHEY_SMALL, 0.5, (255, 255, 255), 1)
            
            axes[2].imshow(cv2.cvtColor(custom_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Custom Model')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return custom_image


def compare_models(image_path: str, 
                  custom_model_path: Optional[str] = None,
                  model_type: str = 'simplebaseline') -> None:
    """
    Compare MediaPipe and custom model on a single image
    
    Args:
        image_path: Path to input image
        custom_model_path: Path to custom model (optional)
        model_type: Type of custom model
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Initialize hybrid estimator
    estimator = HybridPoseEstimator(
        custom_model_path=custom_model_path,
        model_type=model_type,
        use_mediapipe=True,
        use_custom_model=custom_model_path is not None
    )
    
    # Predict with both models
    results = estimator.predict_hybrid(image)
    
    # Visualize comparison
    estimator.visualize_comparison(image, results, save_path='model_comparison.png')
    
    # Print keypoint comparison
    print("\nKeypoint Comparison:")
    print("=" * 50)
    
    if 'mediapipe' in results and 'custom_model' in results:
        mp_keypoints = results['mediapipe']['keypoints']
        custom_keypoints = results['custom_model']['keypoints']
        
        joint_names = estimator.mediapipe_estimator.mpii_joint_names
        
        for i, name in enumerate(joint_names):
            mp_conf = mp_keypoints[i, 2]
            custom_conf = custom_keypoints[i, 2]
            
            if mp_conf > 0 or custom_conf > 0:
                print(f"{name:12}: MediaPipe={mp_conf:.3f}, Custom={custom_conf:.3f}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='MediaPipe + Custom Model Comparison')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--custom_model', type=str, default=None, help='Custom model path')
    parser.add_argument('--model_type', type=str, default='simplebaseline', help='Custom model type')
    
    args = parser.parse_args()
    
    compare_models(args.image, args.custom_model, args.model_type)

