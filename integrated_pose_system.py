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
from typing import Tuple, Optional, Dict, List
import os

from inference import PoseInference
from mediapipe_integration import MediaPipePoseEstimator


class MovementTracker:
    """
    Tracks movement direction, speed, and velocity of body parts.
    """
    
    def __init__(self, history_size: int = 5):
        """Initialize movement tracker."""
        self.history_size = history_size
        self.position_history = []  # List of keypoint positions over time
        
    def add_frame(self, keypoints: np.ndarray):
        """Add current frame positions."""
        if keypoints is not None:
            self.position_history.append(keypoints.copy())
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)
    
    def get_movement_velocity(self, joint_idx: int) -> Tuple[float, float]:
        """
        Calculate movement velocity for a specific joint.
        
        Returns:
            (velocity_x, velocity_y) in pixels per frame
        """
        if len(self.position_history) < 2:
            return (0.0, 0.0)
        
        current = self.position_history[-1]
        previous = self.position_history[-2]
        
        if (current[joint_idx, 2] > 0.1 and previous[joint_idx, 2] > 0.1):
            vel_x = current[joint_idx, 0] - previous[joint_idx, 0]
            vel_y = current[joint_idx, 1] - previous[joint_idx, 1]
            return (vel_x, vel_y)
        
        return (0.0, 0.0)
    
    def get_movement_direction(self, joint_idx: int) -> str:
        """Get movement direction for a joint (up, down, left, right, still)."""
        vel_x, vel_y = self.get_movement_velocity(joint_idx)
        speed = np.sqrt(vel_x**2 + vel_y**2)
        
        if speed < 5:  # Threshold for "still"
            return "still"
        
        # Determine primary direction
        if abs(vel_y) > abs(vel_x):
            return "up" if vel_y < 0 else "down"
        else:
            return "right" if vel_x > 0 else "left"
    
    def get_movement_speed(self, joint_idx: int) -> float:
        """Get movement speed in pixels per frame."""
        vel_x, vel_y = self.get_movement_velocity(joint_idx)
        return np.sqrt(vel_x**2 + vel_y**2)


class PoseQualityAnalyzer:
    """
    Analyzes pose quality and provides detailed feedback.
    """
    
    def __init__(self):
        """Initialize pose quality analyzer."""
        pass
    
    def analyze_body_alignment(self, keypoints: np.ndarray, num_keypoints: int) -> Dict[str, str]:
        """
        Analyze body alignment and posture quality.
        
        Returns:
            Dictionary with alignment feedback
        """
        feedback = {}
        
        def get_point(idx):
            if idx < num_keypoints and keypoints[idx, 2] > 0.2:
                return np.array([keypoints[idx, 0], keypoints[idx, 1]])
            return None
        
        if num_keypoints == 33:
            r_shoulder = get_point(12)
            l_shoulder = get_point(11)
            r_hip = get_point(24)
            l_hip = get_point(23)
            head = get_point(0)
        else:
            r_shoulder = get_point(12)
            l_shoulder = get_point(13)
            r_hip = get_point(2)
            l_hip = get_point(3)
            head = get_point(9)
        
        # Check shoulder alignment
        if r_shoulder is not None and l_shoulder is not None:
            shoulder_diff = abs(r_shoulder[1] - l_shoulder[1])
            if shoulder_diff > 20:
                feedback['shoulders'] = f"Shoulders uneven (difference: {shoulder_diff:.0f}px)"
            else:
                feedback['shoulders'] = "Shoulders aligned ✓"
        
        # Check hip alignment
        if r_hip is not None and l_hip is not None:
            hip_diff = abs(r_hip[1] - l_hip[1])
            if hip_diff > 15:
                feedback['hips'] = f"Hips uneven (difference: {hip_diff:.0f}px)"
            else:
                feedback['hips'] = "Hips aligned ✓"
        
        # Check body vertical alignment
        if (r_shoulder is not None and l_shoulder is not None and
            r_hip is not None and l_hip is not None):
            shoulder_center = (r_shoulder[0] + l_shoulder[0]) / 2
            hip_center = (r_hip[0] + l_hip[0]) / 2
            alignment_diff = abs(shoulder_center - hip_center)
            
            if alignment_diff > 30:
                feedback['body'] = f"Body leaning (offset: {alignment_diff:.0f}px)"
            else:
                feedback['body'] = "Body vertically aligned ✓"
        
        # Check head position
        if head is not None and r_shoulder is not None and l_shoulder is not None:
            shoulder_center_y = (r_shoulder[1] + l_shoulder[1]) / 2
            head_offset = head[1] - shoulder_center_y
            
            if head_offset < -50:
                feedback['head'] = "Head positioned well above shoulders ✓"
            elif head_offset > 30:
                feedback['head'] = "Head slightly low"
            else:
                feedback['head'] = "Head position good ✓"
        
        return feedback
    
    def get_detailed_body_description(self, keypoints: np.ndarray, num_keypoints: int) -> List[str]:
        """
        Get detailed description of body part positions.
        
        Returns:
            List of detailed descriptions
        """
        details = []
        
        def get_point(idx):
            if idx < num_keypoints and keypoints[idx, 2] > 0.2:
                return np.array([keypoints[idx, 0], keypoints[idx, 1]])
            return None
        
        if num_keypoints == 33:
            r_shoulder = get_point(12)
            l_shoulder = get_point(11)
            r_elbow = get_point(14)
            l_elbow = get_point(13)
            r_wrist = get_point(16)
            l_wrist = get_point(15)
            r_hip = get_point(24)
            l_hip = get_point(23)
            r_knee = get_point(26)
            l_knee = get_point(25)
            head = get_point(0)
        else:
            r_shoulder = get_point(12)
            l_shoulder = get_point(13)
            r_elbow = get_point(11)
            l_elbow = get_point(14)
            r_wrist = get_point(10)
            l_wrist = get_point(15)
            r_hip = get_point(2)
            l_hip = get_point(3)
            r_knee = get_point(1)
            l_knee = get_point(4)
            head = get_point(9)
        
        # Arm positions
        if r_shoulder is not None and r_wrist is not None:
            if r_wrist[1] < r_shoulder[1] - 30:
                details.append("Right arm: raised above shoulder")
            elif r_wrist[1] > r_shoulder[1] + 50:
                details.append("Right arm: hanging down")
            else:
                details.append("Right arm: at shoulder level")
        
        if l_shoulder is not None and l_wrist is not None:
            if l_wrist[1] < l_shoulder[1] - 30:
                details.append("Left arm: raised above shoulder")
            elif l_wrist[1] > l_shoulder[1] + 50:
                details.append("Left arm: hanging down")
            else:
                details.append("Left arm: at shoulder level")
        
        # Leg positions
        if r_hip is not None and r_knee is not None:
            if r_knee[1] > r_hip[1] + 80:
                details.append("Right leg: extended")
            elif abs(r_knee[1] - r_hip[1]) < 60:
                details.append("Right leg: bent (sitting/squatting)")
            else:
                details.append("Right leg: normal standing")
        
        if l_hip is not None and l_knee is not None:
            if l_knee[1] > l_hip[1] + 80:
                details.append("Left leg: extended")
            elif abs(l_knee[1] - l_hip[1]) < 60:
                details.append("Left leg: bent (sitting/squatting)")
            else:
                details.append("Left leg: normal standing")
        
        return details


class PoseHistoryTracker:
    """
    Tracks pose history over time for temporal smoothing and activity recognition.
    """
    
    def __init__(self, history_size: int = 10, smoothing_alpha: float = 0.7):
        """
        Initialize pose history tracker.
        
        Args:
            history_size: Number of frames to keep in history
            smoothing_alpha: Exponential smoothing factor (0-1, higher = more smoothing)
        """
        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha
        self.pose_history = []  # List of pose names
        self.keypoint_history = []  # List of keypoint arrays
        self.confidence_history = []  # List of confidence scores
        self.movement_tracker = MovementTracker(history_size=5)
        self.quality_analyzer = PoseQualityAnalyzer()
        
    def add_frame(self, pose_name: str, keypoints: np.ndarray, confidence: float):
        """Add a new frame to history."""
        self.pose_history.append(pose_name)
        self.keypoint_history.append(keypoints.copy() if keypoints is not None else None)
        self.confidence_history.append(confidence)
        
        # Track movement
        if keypoints is not None:
            self.movement_tracker.add_frame(keypoints)
        
        # Keep only recent history
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
            self.keypoint_history.pop(0)
            self.confidence_history.pop(0)
    
    def get_smoothed_pose(self) -> str:
        """Get the most common pose in recent history."""
        if not self.pose_history:
            return "No Pose Detected"
        
        # Count pose occurrences in recent history
        recent_poses = self.pose_history[-5:] if len(self.pose_history) >= 5 else self.pose_history
        pose_counts = {}
        for pose in recent_poses:
            # Extract base pose name (remove details)
            base_pose = pose.split(',')[0] if ',' in pose else pose
            base_pose = base_pose.split('with')[0].strip() if 'with' in base_pose else base_pose
            pose_counts[base_pose] = pose_counts.get(base_pose, 0) + 1
        
        # Return most common pose
        if pose_counts:
            return max(pose_counts, key=pose_counts.get)
        return self.pose_history[-1]
    
    def smooth_keypoints(self, current_keypoints: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to keypoints."""
        if current_keypoints is None:
            return None
        
        if not self.keypoint_history or self.keypoint_history[-1] is None:
            return current_keypoints
        
        # Get last smoothed keypoints
        last_keypoints = self.keypoint_history[-1]
        
        # Apply exponential smoothing
        smoothed = (self.smoothing_alpha * last_keypoints + 
                   (1 - self.smoothing_alpha) * current_keypoints)
        
        return smoothed
    
    def detect_activity(self) -> str:
        """Detect activity based on pose sequence."""
        if len(self.pose_history) < 3:
            return None
        
        recent_poses = self.pose_history[-5:]
        
        # Detect jumping (rapid up/down movement)
        if len(recent_poses) >= 3:
            up_poses = sum(1 for p in recent_poses if 'raising' in p.lower() or 'arms' in p.lower())
            down_poses = sum(1 for p in recent_poses if 'standing' in p.lower() or 'neutral' in p.lower())
            if up_poses >= 2 and down_poses >= 1:
                return "Jumping detected"
        
        # Detect dancing (rapid pose changes)
        unique_poses = len(set(recent_poses))
        if unique_poses >= 4:
            return "Dancing detected"
        
        return None
    
    def get_average_confidence(self) -> float:
        """Get average confidence over recent history."""
        if not self.confidence_history:
            return 0.0
        recent = self.confidence_history[-5:] if len(self.confidence_history) >= 5 else self.confidence_history
        return np.mean(recent)
    
    def get_movement_info(self, joint_idx: int) -> Dict[str, any]:
        """Get movement information for a joint."""
        direction = self.movement_tracker.get_movement_direction(joint_idx)
        speed = self.movement_tracker.get_movement_speed(joint_idx)
        return {
            'direction': direction,
            'speed': speed,
            'is_moving': speed > 5
        }
    
    def get_quality_feedback(self, keypoints: np.ndarray, num_keypoints: int) -> Dict[str, str]:
        """Get pose quality feedback."""
        if keypoints is None:
            return {}
        return self.quality_analyzer.analyze_body_alignment(keypoints, num_keypoints)
    
    def get_detailed_description(self, keypoints: np.ndarray, num_keypoints: int) -> List[str]:
        """Get detailed body part descriptions."""
        if keypoints is None:
            return []
        return self.quality_analyzer.get_detailed_body_description(keypoints, num_keypoints)


class PoseClassifier:
    """
    Classifies human poses based on keypoint positions and angles.
    
    Analyzes the spatial relationships between keypoints to identify
    common poses and gestures like standing, T-pose, arms raised, etc.
    """
    
    def __init__(self):
        """Initialize pose classifier with joint indices."""
        # MPII joint indices for reference
        self.joint_indices = {
            'r_ankle': 0, 'r_knee': 1, 'r_hip': 2, 'l_hip': 3, 'l_knee': 4, 'l_ankle': 5,
            'pelvis': 6, 'thorax': 7, 'upper_neck': 8, 'head_top': 9,
            'r_wrist': 10, 'r_elbow': 11, 'r_shoulder': 12, 'l_shoulder': 13, 'l_elbow': 14, 'l_wrist': 15
        }
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle between three points (point2 is the vertex).
        
        Args:
            point1: First point [x, y]
            point2: Vertex point [x, y]
            point3: Third point [x, y]
            
        Returns:
            Angle in degrees
        """
        # Convert to vectors
        v1 = point1 - point2
        v2 = point3 - point2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle
    
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)
    
    def is_visible(self, keypoints: np.ndarray, joint_idx: int, threshold: float = 0.2) -> bool:
        """
        Check if a joint is visible (confidence > threshold).
        
        Lowered threshold from 0.3 to 0.2 to detect more keypoints
        and allow pose classification with partial visibility.
        """
        return keypoints[joint_idx, 2] > threshold
    
    def calculate_pose_confidence(self, keypoints: np.ndarray) -> float:
        """
        Calculate overall confidence score for a pose detection.
        
        Args:
            keypoints: Keypoints array [num_joints, 3]
            
        Returns:
            Confidence score between 0 and 1
        """
        if keypoints is None:
            return 0.0
        
        # Count visible keypoints
        visible_kpts = keypoints[keypoints[:, 2] > 0.1]
        if len(visible_kpts) == 0:
            return 0.0
        
        # Average confidence of visible keypoints
        avg_confidence = np.mean(visible_kpts[:, 2])
        
        # Weight by number of visible keypoints (more keypoints = higher confidence)
        visibility_ratio = len(visible_kpts) / len(keypoints)
        
        # Combined confidence score
        confidence = avg_confidence * (0.5 + 0.5 * visibility_ratio)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def classify_pose(self, keypoints: np.ndarray) -> str:
        """
        Classify the pose based on keypoint positions.
        
        Args:
            keypoints: Keypoints array [num_joints, 3] (x, y, confidence)
                      Can be 16 (MPII) or 33 (MediaPipe) keypoints
            
        Returns:
            Pose name as string
        """
        if keypoints is None:
            return "No Pose Detected"
        
        num_keypoints = len(keypoints)
        if num_keypoints not in [16, 33]:
            return "No Pose Detected"
        
        # Check minimum visibility requirements (lowered for better detection)
        min_visible_joints = 5 if num_keypoints == 33 else 3
        visible_count = sum(1 for i in range(num_keypoints) if self.is_visible(keypoints, i))
        
        # If we have very few keypoints, still try to classify but show status
        if visible_count < min_visible_joints:
            if visible_count == 0:
                return "No Person Detected"
            return f"Detecting... ({visible_count}/{num_keypoints} keypoints visible)"
        
        # Extract key joint positions (only if visible)
        def get_point(idx):
            if idx < num_keypoints and self.is_visible(keypoints, idx):
                return np.array([keypoints[idx, 0], keypoints[idx, 1]])
            return None
        
        # Map keypoints based on format (16 MPII or 33 MediaPipe)
        if num_keypoints == 33:
            # MediaPipe 33 keypoint indices
            r_shoulder = get_point(12)
            l_shoulder = get_point(11)
            r_elbow = get_point(14)
            l_elbow = get_point(13)
            r_wrist = get_point(16)
            l_wrist = get_point(15)
            r_hip = get_point(24)
            l_hip = get_point(23)
            r_knee = get_point(26)
            l_knee = get_point(25)
            head = get_point(0)  # nose
            thorax = None  # Calculate from shoulders
            
            # Finger keypoints (MediaPipe 33 keypoints include fingers)
            r_pinky = get_point(18)
            l_pinky = get_point(17)
            r_index = get_point(20)
            l_index = get_point(19)
            r_thumb = get_point(22)
            l_thumb = get_point(21)
        else:
            # MPII 16 keypoint indices (legacy)
            r_shoulder = get_point(12)
            l_shoulder = get_point(13)
            r_elbow = get_point(11)
            l_elbow = get_point(14)
            r_wrist = get_point(10)
            l_wrist = get_point(15)
            r_hip = get_point(2)
            l_hip = get_point(3)
            r_knee = get_point(1)
            l_knee = get_point(4)
            head = get_point(9)
            thorax = get_point(7)
        
        # Calculate thorax from shoulders if using 33 keypoints
        if num_keypoints == 33 and r_shoulder is not None and l_shoulder is not None:
            thorax = np.array([(r_shoulder[0] + l_shoulder[0]) / 2, 
                             (r_shoulder[1] + l_shoulder[1]) / 2])
        
        # Check for T-Pose (arms horizontal, straight)
        if (r_shoulder is not None and l_shoulder is not None and
            r_elbow is not None and l_elbow is not None and
            r_wrist is not None and l_wrist is not None):
            
            # Check if arms are horizontal (shoulders and wrists at similar height)
            shoulder_y = (r_shoulder[1] + l_shoulder[1]) / 2
            wrist_y_avg = (r_wrist[1] + l_wrist[1]) / 2
            
            # Check if arms are extended (elbows and wrists far from shoulders)
            r_arm_extended = self.calculate_distance(r_shoulder, r_wrist) > self.calculate_distance(r_shoulder, r_elbow) * 1.5
            l_arm_extended = self.calculate_distance(l_shoulder, l_wrist) > self.calculate_distance(l_shoulder, l_elbow) * 1.5
            
            if (abs(shoulder_y - wrist_y_avg) < 50 and r_arm_extended and l_arm_extended):
                # Add finger details if available
                details = ["both arms extended straight out to your sides", "forming a perfect cross shape with your body", "shoulders and wrists aligned horizontally"]
                if num_keypoints == 33:
                    visible_fingers = []
                    if r_pinky is not None: visible_fingers.append("right pinky")
                    if l_pinky is not None: visible_fingers.append("left pinky")
                    if r_index is not None: visible_fingers.append("right index")
                    if l_index is not None: visible_fingers.append("left index")
                    if visible_fingers:
                        details.append(f"fingers visible: {', '.join(visible_fingers)}")
                return f"You're in a T-pose with {', '.join(details)}"
        
        # Check for Arms Raised (both wrists above shoulders)
        if (r_shoulder is not None and l_shoulder is not None and
            r_wrist is not None and l_wrist is not None):
            
            shoulder_y_avg = (r_shoulder[1] + l_shoulder[1]) / 2
            if (r_wrist[1] < shoulder_y_avg - 30 and l_wrist[1] < shoulder_y_avg - 30):
                # Check if arms are straight up
                if (r_elbow is not None and l_elbow is not None):
                    r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                    l_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
                    if r_angle > 150 and l_angle > 150:
                        details = ["both arms fully extended straight up", "reaching toward the ceiling", "hands positioned high above your head"]
                        if num_keypoints == 33:
                            if (r_index is not None and r_index[1] < r_wrist[1]) or (l_index is not None and l_index[1] < l_wrist[1]):
                                details.append("fingers extended upward")
                            if r_thumb is not None or l_thumb is not None:
                                details.append("thumbs visible and extended")
                        return f"You're raising both arms straight up above your head, {', '.join(details)}"
                # Check finger positions for more detail
                details = ["both hands elevated above your shoulders", "upper body actively engaged", "arms raised in an upward position"]
                if num_keypoints == 33:
                    if (r_index is not None and r_index[1] < r_wrist[1]) or (l_index is not None and l_index[1] < l_wrist[1]):
                        details.append("fingers pointing upward")
                    finger_count = sum([r_pinky is not None, l_pinky is not None, 
                                      r_index is not None, l_index is not None])
                    if finger_count >= 2:
                        details.append(f"{finger_count} finger tips clearly visible")
                return f"You're raising both hands above your shoulders with your arms elevated, {', '.join(details)}"
        
        # Check for Hands on Hips (wrists near hips)
        if (r_hip is not None and l_hip is not None and
            r_wrist is not None and l_wrist is not None):
            
            r_dist = self.calculate_distance(r_wrist, r_hip)
            l_dist = self.calculate_distance(l_wrist, l_hip)
            
            if r_dist < 80 and l_dist < 80:
                details = ["standing in a confident and assertive posture", "hands positioned firmly on your hips", "body language showing readiness"]
                if num_keypoints == 33:
                    if (r_thumb is not None and r_thumb[0] > r_wrist[0]) or (l_thumb is not None and l_thumb[0] < l_wrist[0]):
                        details.append("thumbs pointing forward")
                    if r_pinky is not None or l_pinky is not None:
                        details.append("fingers spread across your hip area")
                return f"You have your hands on your hips, {', '.join(details)}"
        
        # Check for One Arm Raised
        if (r_shoulder is not None and l_shoulder is not None):
            shoulder_y_avg = (r_shoulder[1] + l_shoulder[1]) / 2
            
            if r_wrist is not None and r_wrist[1] < shoulder_y_avg - 30:
                # Check if it's a wave or just raised
                if r_elbow is not None:
                    elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                    if 90 < elbow_angle < 150:
                        details = ["making a friendly waving motion with your right hand", "elbow bent in a natural greeting gesture", "showing a welcoming interaction"]
                        if num_keypoints == 33:
                            if r_index is not None and r_pinky is not None:
                                details.append("fingers spread in an open hand position")
                            if r_thumb is not None:
                                details.append("thumb clearly visible")
                        return f"You're waving with your right hand, {', '.join(details)}"
                    elif r_wrist[1] < r_elbow[1]:
                        details = ["your right arm fully extended upward", "reaching high above your head", "hand positioned well above your shoulder"]
                        if num_keypoints == 33:
                            if r_index is not None and r_index[1] < r_wrist[1]:
                                details.append("index finger pointing straight up")
                            if r_pinky is not None:
                                details.append("entire hand clearly visible")
                        return f"You're raising your right hand straight up above your head, {', '.join(details)}"
                details = ["your right arm elevated above your shoulder", "actively signaling or reaching upward", "right hand positioned higher than your shoulder level"]
                if num_keypoints == 33:
                    if r_index is not None:
                        details.append("index finger clearly detected")
                    if r_thumb is not None:
                        details.append("thumb visible and extended")
                return f"You're raising your right hand up above your shoulder, {', '.join(details)}"
            if l_wrist is not None and l_wrist[1] < shoulder_y_avg - 30:
                # Check if it's a wave or just raised
                if l_elbow is not None:
                    elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
                    if 90 < elbow_angle < 150:
                        details = ["making a friendly waving motion with your left hand", "elbow bent in a natural greeting gesture", "showing a welcoming interaction"]
                        if num_keypoints == 33:
                            if l_index is not None and l_pinky is not None:
                                details.append("fingers spread in an open hand position")
                            if l_thumb is not None:
                                details.append("thumb clearly visible")
                        return f"You're waving with your left hand, {', '.join(details)}"
                    elif l_wrist[1] < l_elbow[1]:
                        details = ["your left arm fully extended upward", "reaching high above your head", "hand positioned well above your shoulder"]
                        if num_keypoints == 33:
                            if l_index is not None and l_index[1] < l_wrist[1]:
                                details.append("index finger pointing straight up")
                            if l_pinky is not None:
                                details.append("entire hand clearly visible")
                        return f"You're raising your left hand straight up above your head, {', '.join(details)}"
                details = ["your left arm elevated above your shoulder", "actively signaling or reaching upward", "left hand positioned higher than your shoulder level"]
                if num_keypoints == 33:
                    if l_index is not None:
                        details.append("index finger clearly detected")
                    if l_thumb is not None:
                        details.append("thumb visible and extended")
                return f"You're raising your left hand up above your shoulder, {', '.join(details)}"
        
        # Check for Sitting (knees bent, hips lower relative to knees)
        if (r_hip is not None and l_hip is not None and
            r_knee is not None and l_knee is not None):
            
            hip_y_avg = (r_hip[1] + l_hip[1]) / 2
            knee_y_avg = (r_knee[1] + l_knee[1]) / 2
            
            # If hips are close to knees (bent position)
            if abs(hip_y_avg - knee_y_avg) < 100:
                details = ["sitting down with your knees bent", "in a seated position", "hips positioned close to your knees"]
                if num_keypoints == 33:
                    r_ankle_pt = get_point(28)
                    l_ankle_pt = get_point(27)
                    if r_ankle_pt is not None and l_ankle_pt is not None:
                        details.append("feet positioned on the ground")
                return f"You're {', '.join(details)}"
        
        # Check for Standing Straight (hips and shoulders aligned vertically)
        if (thorax is not None and r_hip is not None and l_hip is not None):
            hip_y_avg = (r_hip[1] + l_hip[1]) / 2
            if abs(thorax[0] - (r_hip[0] + l_hip[0]) / 2) < 30:
                details = ["standing straight up with good posture", "body aligned vertically", "shoulders and hips in proper alignment"]
                return f"You're {', '.join(details)}"
        
        # Check for waving with one arm
        if r_wrist is not None and r_elbow is not None and r_shoulder is not None:
            r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            if 80 < r_angle < 140 and r_wrist[1] < r_shoulder[1]:
                details = ["making a greeting gesture", "moving your right hand in a friendly waving motion"]
                if num_keypoints == 33:
                    if r_index is not None or r_pinky is not None:
                        details.append("fingers visible in the waving motion")
                return f"You're waving with your right hand, {', '.join(details)}"
        
        if l_wrist is not None and l_elbow is not None and l_shoulder is not None:
            l_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            if 80 < l_angle < 140 and l_wrist[1] < l_shoulder[1]:
                details = ["making a greeting gesture", "moving your left hand in a friendly waving motion"]
                if num_keypoints == 33:
                    if l_index is not None or l_pinky is not None:
                        details.append("fingers visible in the waving motion")
                return f"You're waving with your left hand, {', '.join(details)}"
        
        # Check for pointing gesture
        if r_wrist is not None and r_elbow is not None and r_shoulder is not None:
            r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            if r_angle > 160 and r_wrist[0] > r_shoulder[0]:
                details = ["extending your right arm straight forward", "directing attention ahead of you", "arm fully extended in front"]
                if num_keypoints == 33:
                    if r_index is not None and r_index[0] > r_wrist[0]:
                        details.append("index finger extended and pointing forward")
                    if r_thumb is not None:
                        details.append("thumb clearly visible")
                return f"You're pointing forward with your right hand, {', '.join(details)}"
        
        if l_wrist is not None and l_elbow is not None and l_shoulder is not None:
            l_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            if l_angle > 160 and l_wrist[0] < l_shoulder[0]:
                details = ["extending your left arm straight forward", "directing attention ahead of you", "arm fully extended in front"]
                if num_keypoints == 33:
                    if l_index is not None and l_index[0] < l_wrist[0]:
                        details.append("index finger extended and pointing forward")
                    if l_thumb is not None:
                        details.append("thumb clearly visible")
                return f"You're pointing forward with your left hand, {', '.join(details)}"
        
        # Try to classify with minimal keypoints (at least shoulders and hips)
        if (r_shoulder is not None or l_shoulder is not None) and (r_hip is not None or l_hip is not None):
            # Check if arms are down
            if r_wrist is not None and l_wrist is not None:
                shoulder_y_avg = ((r_shoulder[1] if r_shoulder is not None else 0) + 
                                (l_shoulder[1] if l_shoulder is not None else 0)) / 2
                if r_wrist[1] > shoulder_y_avg + 50 and l_wrist[1] > shoulder_y_avg + 50:
                    details = ["standing upright with your arms hanging straight down at your sides", "in a relaxed and natural posture", "body in a balanced stance"]
                    if num_keypoints == 33:
                        finger_count = sum([r_pinky is not None, l_pinky is not None, 
                                          r_index is not None, l_index is not None,
                                          r_thumb is not None, l_thumb is not None])
                        if finger_count > 0:
                            details.append(f"{finger_count} finger keypoints clearly visible")
                        if r_thumb is not None or l_thumb is not None:
                            details.append("thumbs detected and visible")
                    return f"You're {', '.join(details)}"
            # Add more context about body position
            details = ["standing in a neutral pose", "body in a balanced and centered position", "maintaining an upright stance"]
            if num_keypoints == 33:
                if head is not None and thorax is not None:
                    head_angle = np.arctan2(head[1] - thorax[1], head[0] - thorax[0]) * 180 / np.pi
                    if abs(head_angle) < 20:
                        details.append("head aligned with your body")
                finger_count = sum([r_pinky is not None, l_pinky is not None, 
                                  r_index is not None, l_index is not None])
                if finger_count > 0:
                    details.append(f"{finger_count} finger tips clearly visible")
            return f"You're {', '.join(details)}"
        
        # If we have any keypoints at all, show detecting
        if visible_count > 0:
            return f"Detecting pose... ({visible_count}/{num_keypoints} keypoints visible)"
        
        # Check for Crossed Arms
        if (r_shoulder is not None and l_shoulder is not None and
            r_wrist is not None and l_wrist is not None):
            
            # Check if wrists are on opposite sides of body center
            body_center_x = (r_shoulder[0] + l_shoulder[0]) / 2
            r_wrist_crossed = r_wrist[0] < body_center_x  # Right wrist on left side
            l_wrist_crossed = l_wrist[0] > body_center_x  # Left wrist on right side
            
            if r_wrist_crossed and l_wrist_crossed:
                # Check if arms are bent (elbows visible and positioned correctly)
                if r_elbow is not None and l_elbow is not None:
                    details = ["crossing your arms in front of your body", "in a defensive or thoughtful posture", "arms positioned across your chest"]
                    if num_keypoints == 33:
                        if r_thumb is not None or l_thumb is not None:
                            details.append("hands and fingers clearly visible")
                    return f"You're {', '.join(details)}"
        
        # Check for Hands Behind Head
        if (head is not None and r_wrist is not None and l_wrist is not None):
            # Check if wrists are behind and above head
            if (r_wrist[1] < head[1] and l_wrist[1] < head[1] and
                abs(r_wrist[0] - head[0]) < 100 and abs(l_wrist[0] - head[0]) < 100):
                details = ["placing your hands behind your head", "in a relaxed or stretching posture", "arms elevated behind your head"]
                if num_keypoints == 33:
                    if r_index is not None or l_index is not None:
                        details.append("fingers extended and visible")
                return f"You're {', '.join(details)}"
        
        # Check for Leaning (body not vertical)
        if (thorax is not None and r_hip is not None and l_hip is not None):
            hip_center = np.array([(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2])
            body_angle = np.arctan2(thorax[0] - hip_center[0], thorax[1] - hip_center[1]) * 180 / np.pi
            
            if abs(body_angle) > 15:  # Leaning more than 15 degrees
                direction = "left" if body_angle < 0 else "right"
                details = [f"leaning to your {direction}", "body shifted from vertical", "in a dynamic posture"]
                return f"You're {', '.join(details)}"
        
        # Check for Squatting (knees bent, hips low)
        if num_keypoints == 33:
            r_ankle_pt = get_point(28)  # right_ankle
            l_ankle_pt = get_point(27)  # left_ankle
            r_heel_pt = get_point(30)   # right_heel
            l_heel_pt = get_point(29)   # left_heel
        else:
            r_ankle_pt = get_point(0)   # r_ankle (MPII)
            l_ankle_pt = get_point(5)   # l_ankle (MPII)
            r_heel_pt = None
            l_heel_pt = None
        
        if (r_hip is not None and l_hip is not None and
            r_knee is not None and l_knee is not None and
            r_ankle_pt is not None and l_ankle_pt is not None):
            
            hip_y_avg = (r_hip[1] + l_hip[1]) / 2
            knee_y_avg = (r_knee[1] + l_knee[1]) / 2
            ankle_y_avg = (r_ankle_pt[1] + l_ankle_pt[1]) / 2
            
            # If hips are close to knees and knees are bent
            if (abs(hip_y_avg - knee_y_avg) < 80 and 
                knee_y_avg < ankle_y_avg - 50):  # Knees above ankles
                details = ["squatting down", "knees bent in a low position", "hips positioned close to your knees"]
                if num_keypoints == 33:
                    if r_heel_pt is not None or l_heel_pt is not None:
                        details.append("feet flat on the ground")
                return f"You're {', '.join(details)}"
        
        # Check for Lunge Position
        if (r_hip is not None and l_hip is not None and
            r_knee is not None and l_knee is not None and
            r_ankle_pt is not None and l_ankle_pt is not None):
            
            # Check if one leg is significantly forward
            r_leg_forward = r_ankle_pt[0] > r_hip[0] + 50
            l_leg_forward = l_ankle_pt[0] < l_hip[0] - 50
            
            if r_leg_forward or l_leg_forward:
                leg = "right" if r_leg_forward else "left"
                details = [f"in a lunge position with your {leg} leg forward", "one leg extended forward", "in a dynamic athletic stance"]
                return f"You're {', '.join(details)}"
        
        # Default: Standing
        return "You're standing in a neutral position"


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
                 confidence_threshold: float = 0.2):
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
            # Lower thresholds for better detection sensitivity
            self.mediapipe_model = MediaPipePoseEstimator(
                static_image_mode=False,
                model_complexity=2,  # Use highest complexity for best accuracy
                min_detection_confidence=0.3,  # Lowered from 0.5 for better detection
                min_tracking_confidence=0.3  # Lowered from 0.5 for better tracking
            )
            print("✓ MediaPipe loaded")
        
        # Performance metrics
        self.frame_count = 0
        self.total_time = 0.0
        self.inference_times = []
        
        # Use 33 MediaPipe keypoints
        self.use_33_keypoints = True
        self.num_keypoints = 33
        
        # MediaPipe 33 keypoint names
        self.joint_names = [
            'nose',  # 0
            'left_eye_inner', 'left_eye', 'left_eye_outer',  # 1-3
            'right_eye_inner', 'right_eye', 'right_eye_outer',  # 4-6
            'left_ear', 'right_ear',  # 7-8
            'mouth_left', 'mouth_right',  # 9-10
            'left_shoulder', 'right_shoulder',  # 11-12
            'left_elbow', 'right_elbow',  # 13-14
            'left_wrist', 'right_wrist',  # 15-16
            'left_pinky', 'right_pinky',  # 17-18
            'left_index', 'right_index',  # 19-20
            'left_thumb', 'right_thumb',  # 21-22
            'left_hip', 'right_hip',  # 23-24
            'left_knee', 'right_knee',  # 25-26
            'left_ankle', 'right_ankle',  # 27-28
            'left_heel', 'right_heel',  # 29-30
            'left_foot_index', 'right_foot_index'  # 31-32
        ]
        
        # MediaPipe skeleton connections (33 keypoints)
        # Using MediaPipe's exact POSE_CONNECTIONS structure
        # This matches the official MediaPipe BlazePose GHUM 3D model
        self.skeleton = [
            [15, 21],  # left wrist to left thumb
            [16, 20],  # right wrist to right index
            [18, 20],  # right pinky to right index
            [3, 7],    # left eye outer to left ear
            [14, 16],  # right elbow to right wrist
            [23, 25],  # left hip to left knee
            [28, 30],  # right ankle to right heel
            [11, 23],  # left shoulder to left hip
            [27, 31],  # left ankle to left foot index
            [6, 8],    # right eye outer to right ear
            [15, 17],  # left wrist to left pinky
            [24, 26],  # right hip to right knee
            [16, 22],  # right wrist to right thumb
            [4, 5],    # right eye inner to right eye
            [5, 6],    # right eye to right eye outer
            [29, 31],  # left heel to left foot index
            [12, 24],  # right shoulder to right hip
            [23, 24],  # left hip to right hip
            [0, 1],    # nose to left eye inner
            [9, 10],   # mouth left to mouth right
            [1, 2],    # left eye inner to left eye
            [0, 4],    # nose to right eye inner
            [11, 13],  # left shoulder to left elbow
            [30, 32],  # right heel to right foot index
            [28, 32],  # right ankle to right foot index
            [15, 19],  # left wrist to left index
            [16, 18],  # right wrist to right pinky
            [25, 27],  # left knee to left ankle
            [26, 28],  # right knee to right ankle
            [12, 14],  # right shoulder to right elbow
            [17, 19],  # left pinky to left index
            [2, 3],    # left eye to left eye outer
            [11, 12],  # left shoulder to right shoulder
            [27, 29],  # left ankle to left heel
            [13, 15]   # left elbow to left wrist
        ]
        
        # Initialize pose classifier
        self.pose_classifier = PoseClassifier()
        
        # Initialize pose history trackers for temporal smoothing
        self.pose_history_trackers = {}  # person_id -> PoseHistoryTracker
        self.person_id_counter = 0
        self.person_tracking = {}  # Track people across frames
    
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
            Keypoints array [16, 3] (x, y, confidence) in image coordinates
        """
        if self.mediapipe_model is None:
            return None
        
        try:
            _, results = self.mediapipe_model.process_image(frame)
            
            # Check if pose was detected
            if results.pose_landmarks is None:
                return None
            
            # Extract keypoints (already in image coordinates)
            # Use 33 keypoints instead of 16
            keypoints = self.mediapipe_model.extract_keypoints(
                results, 
                (frame.shape[0], frame.shape[1]),
                use_33_keypoints=True
            )
            
            # Ensure we have valid keypoints (at least some with visibility > 0.1)
            if keypoints is not None:
                # Check if we have any visible keypoints
                visible_count = np.sum(keypoints[:, 2] > 0.1)
                if visible_count > 0:
                    return keypoints
                else:
                    # Debug: print what we got
                    print(f"Debug: MediaPipe detected pose but all keypoints have low visibility")
                    print(f"Max visibility: {np.max(keypoints[:, 2]):.3f}")
            
            return None
        except Exception as e:
            print(f"Error in MediaPipe inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_bbox_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            bbox1: [x, y, w, h]
            bbox2: [x, y, w, h]
        
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _assign_person_id(self, bbox: List[int], center: Tuple[float, float]) -> int:
        """
        Assign or match person ID based on position tracking.
        
        Args:
            bbox: Bounding box [x, y, w, h]
            center: Center point (x, y)
            
        Returns:
            Person ID (existing or new)
        """
        # Check if this detection matches an existing person
        for person_id, (prev_bbox, prev_center, frames_since_seen) in self.person_tracking.items():
            if frames_since_seen > 10:  # Person lost for too long, remove
                continue
            
            # Calculate distance to previous position
            center_dist = np.sqrt((center[0] - prev_center[0])**2 + 
                                 (center[1] - prev_center[1])**2)
            
            # Calculate IoU with previous bbox
            iou = self._calculate_bbox_iou(bbox, prev_bbox)
            
            # If close enough, it's the same person
            avg_size = (bbox[2] + bbox[3] + prev_bbox[2] + prev_bbox[3]) / 4
            threshold = max(100, avg_size * 0.4)
            
            if center_dist < threshold or iou > 0.3:
                # Update tracking info
                self.person_tracking[person_id] = (bbox, center, 0)
                return person_id
        
        # New person detected
        new_id = self.person_id_counter
        self.person_id_counter += 1
        self.person_tracking[new_id] = (bbox, center, 0)
        
        # Initialize history tracker for new person
        if new_id not in self.pose_history_trackers:
            self.pose_history_trackers[new_id] = PoseHistoryTracker(history_size=10, smoothing_alpha=0.7)
        
        return new_id
    
    def _update_person_tracking(self):
        """Update frames since seen for all tracked people."""
        to_remove = []
        for person_id, (bbox, center, frames_since_seen) in self.person_tracking.items():
            self.person_tracking[person_id] = (bbox, center, frames_since_seen + 1)
            if frames_since_seen > 20:  # Remove if not seen for 20 frames
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_tracking[person_id]
            if person_id in self.pose_history_trackers:
                del self.pose_history_trackers[person_id]
    
    def _is_duplicate_detection(self, new_bbox: List[int], existing_bboxes: List[List[int]], 
                                new_center: Tuple[float, float], 
                                existing_centers: List[Tuple[float, float]]) -> bool:
        """
        Check if a new detection is a duplicate of existing detections.
        
        Args:
            new_bbox: New bounding box [x, y, w, h]
            existing_bboxes: List of existing bounding boxes
            new_center: Center point of new detection (x, y)
            existing_centers: List of existing center points
        
        Returns:
            True if duplicate, False if new person
        """
        for existing_bbox, existing_center in zip(existing_bboxes, existing_centers):
            # Check IoU overlap
            iou = self._calculate_bbox_iou(new_bbox, existing_bbox)
            if iou > 0.3:  # More than 30% overlap = likely same person
                return True
            
            # Check center distance (more strict)
            center_dist = np.sqrt((new_center[0] - existing_center[0])**2 + 
                                 (new_center[1] - existing_center[1])**2)
            # Use dynamic threshold based on bbox size
            avg_bbox_size = (new_bbox[2] + new_bbox[3] + existing_bbox[2] + existing_bbox[3]) / 4
            threshold = max(150, avg_bbox_size * 0.5)  # At least 150px or 50% of average size
            
            if center_dist < threshold:
                return True
        
        return False
    
    def predict_multiple_people(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect multiple people in frame, ensuring only unique people are counted.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            List of (keypoints, bbox) tuples for each detected person
            bbox is [x, y, w, h] bounding box
        """
        if self.mediapipe_model is None:
            return []
        
        detected_people = []
        detected_bboxes = []
        detected_centers = []
        
        try:
            # Try full frame first
            _, results = self.mediapipe_model.process_image(frame)
            if results.pose_landmarks is not None:
                keypoints = self.mediapipe_model.extract_keypoints(
                    results, 
                    (frame.shape[0], frame.shape[1]),
                    use_33_keypoints=True
                )
                if keypoints is not None and np.sum(keypoints[:, 2] > 0.1) > 5:
                    # Calculate bounding box from keypoints
                    visible_kpts = keypoints[keypoints[:, 2] > 0.1]
                    if len(visible_kpts) > 0:
                        x_min = max(0, int(np.min(visible_kpts[:, 0])))
                        y_min = max(0, int(np.min(visible_kpts[:, 1])))
                        x_max = min(frame.shape[1], int(np.max(visible_kpts[:, 0])))
                        y_max = min(frame.shape[0], int(np.max(visible_kpts[:, 1])))
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        center = (np.mean(visible_kpts[:, 0]), np.mean(visible_kpts[:, 1]))
                        
                        detected_people.append((keypoints, bbox))
                        detected_bboxes.append(bbox)
                        detected_centers.append(center)
            
            # Only try quadrants if we haven't found multiple people yet
            # This prevents duplicate detections of the same person
            if len(detected_people) < 2:
                h, w = frame.shape[:2]
                quadrants = [
                    frame[0:h//2, 0:w//2],  # Top-left
                    frame[0:h//2, w//2:w],  # Top-right
                    frame[h//2:h, 0:w//2],  # Bottom-left
                    frame[h//2:h, w//2:w]   # Bottom-right
                ]
                offsets = [
                    (0, 0),           # Top-left offset
                    (w//2, 0),         # Top-right offset
                    (0, h//2),         # Bottom-left offset
                    (w//2, h//2)       # Bottom-right offset
                ]
                
                for quadrant, (offset_x, offset_y) in zip(quadrants, offsets):
                    _, results = self.mediapipe_model.process_image(quadrant)
                    if results.pose_landmarks is not None:
                        keypoints = self.mediapipe_model.extract_keypoints(
                            results,
                            (quadrant.shape[0], quadrant.shape[1]),
                            use_33_keypoints=True
                        )
                        if keypoints is not None and np.sum(keypoints[:, 2] > 0.1) > 5:
                            # Adjust keypoints to full frame coordinates
                            keypoints[:, 0] += offset_x
                            keypoints[:, 1] += offset_y
                            
                            # Calculate bounding box and center
                            visible_kpts = keypoints[keypoints[:, 2] > 0.1]
                            if len(visible_kpts) > 0:
                                x_min = max(0, int(np.min(visible_kpts[:, 0])))
                                y_min = max(0, int(np.min(visible_kpts[:, 1])))
                                x_max = min(frame.shape[1], int(np.max(visible_kpts[:, 0])))
                                y_max = min(frame.shape[0], int(np.max(visible_kpts[:, 1])))
                                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                                center = (np.mean(visible_kpts[:, 0]), np.mean(visible_kpts[:, 1]))
                                
                                # Check if this is a new person (not duplicate)
                                if not self._is_duplicate_detection(bbox, detected_bboxes, center, detected_centers):
                                    detected_people.append((keypoints, bbox))
                                    detected_bboxes.append(bbox)
                                    detected_centers.append(center)
                                    
                                    # Stop if we found enough people
                                    if len(detected_people) >= 5:
                                        break
            
            return detected_people
            
        except Exception as e:
            print(f"Error in multi-person detection: {e}")
            return detected_people
    
    def visualize_pose(self,
                      frame: np.ndarray,
                      keypoints: np.ndarray,
                      label: str = "Pose",
                      color: Tuple[int, int, int] = None,
                      skeleton_color: Tuple[int, int, int] = None,
                      keypoint_color: Tuple[int, int, int] = None,
                      show_pose_name: bool = True,
                      keypoints_already_scaled: bool = False) -> np.ndarray:
        """
        Draw skeleton on frame with pose classification.
        
        Args:
            frame: Input frame
            keypoints: Keypoints array [num_joints, 3]
            label: Label to display (e.g., "Trained" or "MediaPipe")
            color: Color for both skeleton and keypoints (BGR) - deprecated, use skeleton_color/keypoint_color
            skeleton_color: Color for skeleton lines (BGR) - default: teal/cyan (0, 255, 255)
            keypoint_color: Color for keypoint circles (BGR) - default: pink (203, 192, 255)
            show_pose_name: Whether to display classified pose name
            keypoints_already_scaled: If True, keypoints are already in image coordinates
        
        Returns:
            Frame with skeleton drawn and pose labeled
        """
        vis_frame = frame.copy()
        
        # Set default colors to match reference image (pink keypoints, teal skeleton)
        if skeleton_color is None:
            skeleton_color = color if color is not None else (0, 255, 255)  # Teal/Cyan
        if keypoint_color is None:
            keypoint_color = color if color is not None else (203, 192, 255)  # Pink
        
        # Scale keypoints to image size (if needed)
        if keypoints_already_scaled:
            scaled_keypoints = keypoints.copy()
        else:
            scale_x = frame.shape[1] / self.input_size[1]
            scale_y = frame.shape[0] / self.input_size[0]
            
            scaled_keypoints = keypoints.copy()
            scaled_keypoints[:, 0] *= scale_x
            scaled_keypoints[:, 1] *= scale_y
        
        # Classify the pose
        pose_name = "Unknown"
        if show_pose_name:
            pose_name = self.pose_classifier.classify_pose(scaled_keypoints)
        
        # Draw skeleton connections (thicker lines for better visibility)
        # Use lower threshold for drawing to show more connections
        draw_threshold = max(0.15, self.confidence_threshold * 0.7)
        for connection in self.skeleton:
            start_joint = connection[0]
            end_joint = connection[1]
            
            if (scaled_keypoints[start_joint, 2] > draw_threshold and
                scaled_keypoints[end_joint, 2] > draw_threshold):
                
                start_point = (int(scaled_keypoints[start_joint, 0]),
                             int(scaled_keypoints[start_joint, 1]))
                end_point = (int(scaled_keypoints[end_joint, 0]),
                           int(scaled_keypoints[end_joint, 1]))
                
                # Draw thicker lines for better visibility (teal/cyan skeleton)
                cv2.line(vis_frame, start_point, end_point, skeleton_color, 3)
        
        # Draw joints (larger circles for better visibility)
        # Use lower threshold for drawing to show more keypoints
        draw_threshold = max(0.15, self.confidence_threshold * 0.7)
        num_kpts = len(scaled_keypoints)
        
        for i, (x, y, conf) in enumerate(scaled_keypoints):
            if conf > draw_threshold:
                # Check if this is a finger keypoint (MediaPipe 33 keypoints)
                is_finger = False
                if num_kpts == 33:
                    # MediaPipe finger indices: 17-22 (pinky, index, thumb)
                    finger_indices = [17, 18, 19, 20, 21, 22]
                    is_finger = i in finger_indices
                
                # Make finger keypoints slightly larger and more visible
                if is_finger:
                    circle_size = int(5 + conf * 3)  # Larger for fingers
                    # Use slightly different color for fingers (brighter pink)
                    finger_color = (min(255, keypoint_color[0] + 30), 
                                  min(255, keypoint_color[1] + 20), 
                                  min(255, keypoint_color[2] + 10))
                    cv2.circle(vis_frame, (int(x), int(y)), circle_size, finger_color, -1)
                    cv2.circle(vis_frame, (int(x), int(y)), circle_size, (255, 255, 0), 2)  # Yellow outline for fingers
                else:
                    # Regular keypoints
                    circle_size = int(4 + conf * 2)
                    cv2.circle(vis_frame, (int(x), int(y)), circle_size, keypoint_color, -1)
                    cv2.circle(vis_frame, (int(x), int(y)), circle_size, (255, 255, 255), 1)  # White outline
        
        # Display pose name prominently at the top
        if show_pose_name and pose_name:
            # Use white text color for better visibility
            text_color = (255, 255, 255)  # White text
            border_color = (0, 255, 255)  # Teal border to match skeleton
            
            # Get text size for proper positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(pose_name, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 50
            
            # Draw background rectangle with padding
            padding = 15
            cv2.rectangle(vis_frame, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 0), -1)  # Black background
            cv2.rectangle(vis_frame,
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         border_color, 2)  # Teal border
            
            # Draw pose name text with white color
            cv2.putText(vis_frame, pose_name, (text_x, text_y),
                       font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return vis_frame
    
    def draw_side_panel(self, frame: np.ndarray, people_data: List[Tuple]) -> np.ndarray:
        """
        Draw chatbot-style side panel showing pose descriptions.
        
        Args:
            frame: Input frame
            people_data: List of (person_id, pose_name, keypoints) tuples
        
        Returns:
            Frame with chatbot-style side panel drawn
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Chatbot-style panel dimensions - wider and taller
        panel_width = 400
        panel_height = h - 20  # Full height
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Draw panel background (chatbot style - darker)
        cv2.rectangle(vis_frame, 
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (25, 25, 35), -1)  # Dark blue-gray background
        cv2.rectangle(vis_frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 255, 255), 2)  # Teal border
        
        # Panel header (chatbot style)
        header_height = 40
        cv2.rectangle(vis_frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + header_height),
                     (40, 40, 50), -1)  # Slightly lighter header
        cv2.line(vis_frame,
                (panel_x, panel_y + header_height),
                (panel_x + panel_width, panel_y + header_height),
                (0, 255, 255), 2)
        
        # Panel title
        title = "Pose Analysis"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        title_x = panel_x + (panel_width - title_size[0]) // 2
        cv2.putText(vis_frame, title, (title_x, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Chatbot message area (scrollable-like appearance)
        chat_start_y = panel_y + header_height + 10
        chat_area_height = panel_height - header_height - 20
        
        # Draw person information in chatbot message bubbles
        y_offset = chat_start_y
        line_spacing = 18  # Smaller spacing for more text
        
        for person_data in people_data:
            # Handle both old format (3 items) and new format (6 items)
            if len(person_data) >= 6:
                person_id, pose_name, keypoints, quality_feedback, body_details, movement_info = person_data[:6]
            else:
                person_id, pose_name, keypoints = person_data[:3]
                quality_feedback = {}
                body_details = []
                movement_info = {}
            # Person label (smaller, chatbot style)
            person_label = f"Person {person_id + 1}:"
            cv2.putText(vis_frame, person_label, (panel_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_offset += line_spacing
            
            # Split pose description into words for wrapping
            words = pose_name.split()
            max_width = panel_width - 40  # Leave margins
            font_scale = 0.4  # Smaller font for chatbot style
            font_thickness = 1
            
            # Wrap text into multiple lines
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                if test_size[0] > max_width and current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            # Draw each line in a message bubble style
            for line in lines:
                # Calculate text size
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # Draw message bubble background (chatbot style)
                bubble_padding = 8
                bubble_x = panel_x + 15
                bubble_y = y_offset - text_size[1] - 5
                bubble_w = text_size[0] + bubble_padding * 2
                bubble_h = text_size[1] + bubble_padding * 2
                
                # Rounded rectangle effect (using filled rectangle)
                cv2.rectangle(vis_frame,
                             (bubble_x, bubble_y),
                             (bubble_x + bubble_w, bubble_y + bubble_h),
                             (50, 50, 60), -1)  # Dark gray bubble
                cv2.rectangle(vis_frame,
                             (bubble_x, bubble_y),
                             (bubble_x + bubble_w, bubble_y + bubble_h),
                             (100, 100, 120), 1)  # Light border
                
                # Draw text
                cv2.putText(vis_frame, line, 
                           (bubble_x + bubble_padding, bubble_y + text_size[1] + bubble_padding - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 240), font_thickness)
                
                y_offset += bubble_h + 5
            
            # Keypoint info (smaller, compact)
            if keypoints is not None:
                visible_count = np.sum(keypoints[:, 2] > 0.1)
                avg_confidence = np.mean(keypoints[keypoints[:, 2] > 0.1, 2]) if visible_count > 0 else 0
                kpt_text = f"Keypoints: {visible_count}/33 | Confidence: {avg_confidence:.0%}"
                
                # Draw in smaller bubble
                kpt_size = cv2.getTextSize(kpt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                cv2.rectangle(vis_frame,
                             (panel_x + 15, y_offset - 2),
                             (panel_x + 15 + kpt_size[0] + 10, y_offset + kpt_size[1] + 4),
                             (30, 30, 40), -1)
                cv2.putText(vis_frame, kpt_text, (panel_x + 20, y_offset + kpt_size[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 200), 1)
                y_offset += kpt_size[1] + 10
            
            # Quality feedback section
            if quality_feedback:
                y_offset += 5
                quality_title = "Posture Quality:"
                title_size = cv2.getTextSize(quality_title, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(vis_frame, quality_title, (panel_x + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)
                y_offset += 15
                
                for key, feedback_text in quality_feedback.items():
                    # Color code: green for good, yellow for warnings
                    color = (100, 255, 100) if "✓" in feedback_text else (100, 200, 255)
                    
                    # Wrap feedback text
                    words = feedback_text.split()
                    current_line = ""
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                        if test_size[0] > max_width and current_line:
                            # Draw current line
                            text_size = cv2.getTextSize(current_line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                            cv2.rectangle(vis_frame,
                                         (panel_x + 20, y_offset - text_size[1] - 2),
                                         (panel_x + 20 + text_size[0] + 8, y_offset + 2),
                                         (25, 25, 35), -1)
                            cv2.putText(vis_frame, current_line, (panel_x + 24, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                            y_offset += text_size[1] + 5
                            current_line = word
                        else:
                            current_line = test_line
                    
                    # Draw remaining line
                    if current_line:
                        text_size = cv2.getTextSize(current_line, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                        cv2.rectangle(vis_frame,
                                     (panel_x + 20, y_offset - text_size[1] - 2),
                                     (panel_x + 20 + text_size[0] + 8, y_offset + 2),
                                     (25, 25, 35), -1)
                        cv2.putText(vis_frame, current_line, (panel_x + 24, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                        y_offset += text_size[1] + 8
            
            # Body part details section
            if body_details:
                y_offset += 5
                details_title = "Body Position Details:"
                title_size = cv2.getTextSize(details_title, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(vis_frame, details_title, (panel_x + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
                y_offset += 15
                
                for detail in body_details[:4]:  # Limit to 4 details to save space
                    text_size = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                    cv2.rectangle(vis_frame,
                                 (panel_x + 20, y_offset - text_size[1] - 2),
                                 (panel_x + 20 + text_size[0] + 8, y_offset + 2),
                                 (35, 35, 45), -1)
                    cv2.putText(vis_frame, detail, (panel_x + 24, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 240), 1)
                    y_offset += text_size[1] + 6
            
            if not quality_feedback and not body_details:
                y_offset += 10
            
            # Add separator between people (if multiple)
            if person_id < len(people_data) - 1:
                cv2.line(vis_frame, 
                        (panel_x + 20, y_offset + 5),
                        (panel_x + panel_width - 20, y_offset + 5),
                        (60, 60, 70), 1)
                y_offset += 15
        
        return vis_frame
    
    def process_webcam(self, camera_index: int = 0, multi_person: bool = True) -> None:
        """
        Process webcam feed with pose estimation.
        
        Args:
            camera_index: Webcam index (default: 0)
            multi_person: If True, detect multiple people; if False, single person mode
        
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
        
        print(f"\nMulti-person mode: {'ON' if multi_person else 'OFF'}")
        print("\nControls:")
        print("  q: Quit")
        print("  s: Save frame")
        if self.hybrid_mode:
            print("  m: Toggle method")
        print("\nTips for better detection:")
        print("  - Stand 3-6 feet from camera")
        print("  - Ensure good lighting")
        print("  - Face the camera directly")
        print("  - Keep your full body in frame")
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
            
            # Multi-person detection
            if multi_person and self.mediapipe_model:
                detected_people = self.predict_multiple_people(frame)
                vis_frame = frame.copy()
                people_data = []
                
                # Different colors for different people
                person_colors = [
                    ((0, 255, 255), (203, 192, 255)),  # Person 1: Teal/Pink
                    ((0, 255, 0), (255, 192, 203)),    # Person 2: Green/Pink
                    ((255, 0, 255), (192, 203, 255)),  # Person 3: Magenta/Blue
                    ((255, 255, 0), (203, 255, 192)),  # Person 4: Yellow/Green
                    ((0, 165, 255), (255, 203, 192)),  # Person 5: Orange/Peach
                ]
                
                # Process each detected person with tracking and smoothing
                for idx, (keypoints, bbox) in enumerate(detected_people):
                    # Calculate center for tracking
                    visible_kpts = keypoints[keypoints[:, 2] > 0.1] if keypoints is not None else None
                    if visible_kpts is None or len(visible_kpts) == 0:
                        continue
                    
                    center = (np.mean(visible_kpts[:, 0]), np.mean(visible_kpts[:, 1]))
                    
                    # Assign or match person ID
                    person_id = self._assign_person_id(bbox, center)
                    
                    if person_id >= len(person_colors):
                        continue
                    
                    # Get or create history tracker for this person
                    if person_id not in self.pose_history_trackers:
                        self.pose_history_trackers[person_id] = PoseHistoryTracker(history_size=10, smoothing_alpha=0.7)
                    
                    tracker = self.pose_history_trackers[person_id]
                    
                    # Smooth keypoints using temporal smoothing
                    smoothed_keypoints = tracker.smooth_keypoints(keypoints)
                    if smoothed_keypoints is not None:
                        keypoints = smoothed_keypoints
                    
                    # Calculate pose confidence
                    pose_confidence = self.pose_classifier.calculate_pose_confidence(keypoints)
                    
                    # Classify pose for this person
                    pose_name = self.pose_classifier.classify_pose(keypoints)
                    
                    # Add to history and get smoothed pose
                    tracker.add_frame(pose_name, keypoints, pose_confidence)
                    smoothed_pose = tracker.get_smoothed_pose()
                    
                    # Detect activity
                    activity = tracker.detect_activity()
                    
                    # Get movement info for key joints
                    num_kpts = len(keypoints)
                    movement_info = {}
                    if num_kpts == 33:
                        # Track right wrist movement
                        r_wrist_movement = tracker.get_movement_info(16)
                        l_wrist_movement = tracker.get_movement_info(15)
                        movement_info['right_hand'] = r_wrist_movement
                        movement_info['left_hand'] = l_wrist_movement
                    
                    # Get quality feedback
                    quality_feedback = tracker.get_quality_feedback(keypoints, num_kpts)
                    
                    # Get detailed body descriptions
                    body_details = tracker.get_detailed_description(keypoints, num_kpts)
                    
                    # Build enhanced pose description
                    if activity:
                        pose_name = f"{smoothed_pose} - {activity}"
                    else:
                        pose_name = smoothed_pose
                    
                    # Add movement info to pose name if significant
                    if movement_info.get('right_hand', {}).get('is_moving') or movement_info.get('left_hand', {}).get('is_moving'):
                        moving_hands = []
                        if movement_info.get('right_hand', {}).get('is_moving'):
                            dir_r = movement_info['right_hand']['direction']
                            moving_hands.append(f"right hand moving {dir_r}")
                        if movement_info.get('left_hand', {}).get('is_moving'):
                            dir_l = movement_info['left_hand']['direction']
                            moving_hands.append(f"left hand moving {dir_l}")
                        if moving_hands:
                            pose_name += f" ({', '.join(moving_hands)})"
                    
                    # Get colors for this person
                    skeleton_color, keypoint_color = person_colors[person_id % len(person_colors)]
                    
                    # Visualize this person's pose
                    vis_frame = self.visualize_pose(
                        vis_frame, keypoints, f"Person {person_id + 1}",
                        skeleton_color=skeleton_color,
                        keypoint_color=keypoint_color,
                        show_pose_name=False,  # Don't show pose name on person (will show in panel)
                        keypoints_already_scaled=True
                    )
                    
                    # Draw bounding box with person label and confidence
                    x, y, w, h = bbox
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), skeleton_color, 3)
                    label = f"Person {person_id + 1} ({pose_confidence:.0%})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(vis_frame, (x, y - label_size[1] - 15),
                                 (x + label_size[0] + 15, y), skeleton_color, -1)
                    cv2.putText(vis_frame, label, (x + 8, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Store data for side panel with pose description and additional info
                    people_data.append((person_id, pose_name, keypoints, quality_feedback, body_details, movement_info))
                
                # Update person tracking (increment frames since seen)
                self._update_person_tracking()
                
                # Draw side panel with detailed pose information for all people
                if len(people_data) > 0:
                    vis_frame = self.draw_side_panel(vis_frame, people_data)
                
            else:
                # Single person mode (original behavior)
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
                
                # Visualize with pose classification
                vis_frame = frame.copy()
                
                # Use whichever keypoints are available
                keypoints_to_use = None
                people_data = []
                
                # Initialize tracker for single person mode
                if 0 not in self.pose_history_trackers:
                    self.pose_history_trackers[0] = PoseHistoryTracker(history_size=10, smoothing_alpha=0.7)
                tracker = self.pose_history_trackers[0]
                
                if trained_kpts is not None:
                    keypoints_to_use = trained_kpts
                    
                    # Smooth keypoints
                    smoothed_keypoints = tracker.smooth_keypoints(trained_kpts)
                    if smoothed_keypoints is not None:
                        keypoints_to_use = smoothed_keypoints
                    
                    # Calculate confidence
                    pose_confidence = self.pose_classifier.calculate_pose_confidence(keypoints_to_use)
                    
                    # Classify pose
                    pose_name = self.pose_classifier.classify_pose(keypoints_to_use)
                    
                    # Add to history and get smoothed pose
                    tracker.add_frame(pose_name, keypoints_to_use, pose_confidence)
                    smoothed_pose = tracker.get_smoothed_pose()
                    
                    # Detect activity
                    activity = tracker.detect_activity()
                    
                    # Get movement info for key joints
                    num_kpts = len(keypoints_to_use)
                    movement_info = {}
                    if num_kpts == 33:
                        # Track right wrist movement
                        r_wrist_movement = tracker.get_movement_info(16)
                        l_wrist_movement = tracker.get_movement_info(15)
                        movement_info['right_hand'] = r_wrist_movement
                        movement_info['left_hand'] = l_wrist_movement
                    
                    # Get quality feedback
                    quality_feedback = tracker.get_quality_feedback(keypoints_to_use, num_kpts)
                    
                    # Get detailed body descriptions
                    body_details = tracker.get_detailed_description(keypoints_to_use, num_kpts)
                    
                    # Build enhanced pose description
                    if activity:
                        pose_name = f"{smoothed_pose} - {activity}"
                    else:
                        pose_name = smoothed_pose
                    
                    # Add movement info to pose name if significant
                    if movement_info.get('right_hand', {}).get('is_moving') or movement_info.get('left_hand', {}).get('is_moving'):
                        moving_hands = []
                        if movement_info.get('right_hand', {}).get('is_moving'):
                            dir_r = movement_info['right_hand']['direction']
                            moving_hands.append(f"right hand moving {dir_r}")
                        if movement_info.get('left_hand', {}).get('is_moving'):
                            dir_l = movement_info['left_hand']['direction']
                            moving_hands.append(f"left hand moving {dir_l}")
                        if moving_hands:
                            pose_name += f" ({', '.join(moving_hands)})"
                    
                    vis_frame = self.visualize_pose(vis_frame, keypoints_to_use, "Trained", 
                                                   skeleton_color=(0, 255, 255),
                                                   keypoint_color=(203, 192, 255),
                                                   show_pose_name=False, keypoints_already_scaled=False)
                    people_data.append((0, pose_name, keypoints_to_use, quality_feedback, body_details, movement_info))
                elif mediapipe_kpts is not None:
                    keypoints_to_use = mediapipe_kpts
                    
                    # Smooth keypoints
                    smoothed_keypoints = tracker.smooth_keypoints(mediapipe_kpts)
                    if smoothed_keypoints is not None:
                        keypoints_to_use = smoothed_keypoints
                    
                    # Calculate confidence
                    pose_confidence = self.pose_classifier.calculate_pose_confidence(keypoints_to_use)
                    
                    # Classify pose
                    pose_name = self.pose_classifier.classify_pose(keypoints_to_use)
                    
                    # Add to history and get smoothed pose
                    tracker.add_frame(pose_name, keypoints_to_use, pose_confidence)
                    smoothed_pose = tracker.get_smoothed_pose()
                    
                    # Detect activity
                    activity = tracker.detect_activity()
                    
                    # Get movement info for key joints
                    num_kpts = len(keypoints_to_use)
                    movement_info = {}
                    if num_kpts == 33:
                        # Track right wrist movement
                        r_wrist_movement = tracker.get_movement_info(16)
                        l_wrist_movement = tracker.get_movement_info(15)
                        movement_info['right_hand'] = r_wrist_movement
                        movement_info['left_hand'] = l_wrist_movement
                    
                    # Get quality feedback
                    quality_feedback = tracker.get_quality_feedback(keypoints_to_use, num_kpts)
                    
                    # Get detailed body descriptions
                    body_details = tracker.get_detailed_description(keypoints_to_use, num_kpts)
                    
                    # Build enhanced pose description
                    if activity:
                        pose_name = f"{smoothed_pose} - {activity}"
                    else:
                        pose_name = smoothed_pose
                    
                    # Add movement info to pose name if significant
                    if movement_info.get('right_hand', {}).get('is_moving') or movement_info.get('left_hand', {}).get('is_moving'):
                        moving_hands = []
                        if movement_info.get('right_hand', {}).get('is_moving'):
                            dir_r = movement_info['right_hand']['direction']
                            moving_hands.append(f"right hand moving {dir_r}")
                        if movement_info.get('left_hand', {}).get('is_moving'):
                            dir_l = movement_info['left_hand']['direction']
                            moving_hands.append(f"left hand moving {dir_l}")
                        if moving_hands:
                            pose_name += f" ({', '.join(moving_hands)})"
                    
                    vis_frame = self.visualize_pose(vis_frame, keypoints_to_use, "MediaPipe", 
                                                   skeleton_color=(0, 255, 255),
                                                   keypoint_color=(203, 192, 255),
                                                   show_pose_name=False, keypoints_already_scaled=True)
                    people_data.append((0, pose_name, keypoints_to_use, quality_feedback, body_details, movement_info))
                
                # Draw chatbot-style side panel for single person
                if len(people_data) > 0:
                    vis_frame = self.draw_side_panel(vis_frame, people_data)
            
            inference_time = time.time() - inference_start
            self.inference_times.append(inference_time)
            
            # Add metrics
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
            
            # Display FPS and performance metrics
            cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, f'Inference: {inference_time*1000:.1f}ms', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            if self.hybrid_mode and not multi_person:
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
    parser.add_argument('--use_mediapipe', action='store_true', default=True,
                       help='Use MediaPipe for pose estimation')
    parser.add_argument('--hybrid', action='store_true',
                       help='Compare trained model vs MediaPipe (requires --model_path)')
    parser.add_argument('--confidence', type=float, default=0.2,
                       help='Confidence threshold for visualization (lower = more keypoints shown)')
    parser.add_argument('--multi_person', action='store_true', default=False,
                       help='Enable multi-person detection')
    parser.add_argument('--single_person', action='store_true', default=True,
                       help='Use single person mode (default: True)')
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
    
    # Determine multi-person mode (default to single person)
    multi_person = args.multi_person
    
    # Run webcam processing (always use camera 0)
    system.process_webcam(camera_index=0, multi_person=multi_person)


if __name__ == '__main__':
    main()
