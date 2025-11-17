import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import json
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional
import random

class MPIIPoseDataset(Dataset):
    """
    MPII Human Pose Dataset loader
    """
    
    # MPII joint indices
    JOINT_NAMES = [
        'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
        'pelvis', 'thorax', 'upper_neck', 'head_top',
        'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
    ]
    
    # Skeleton connections for visualization
    SKELETON = [
        [0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],  # legs
        [6, 7], [7, 8], [8, 9],  # torso and head
        [7, 12], [12, 11], [11, 10],  # right arm
        [7, 13], [13, 14], [14, 15]  # left arm
    ]
    
    def __init__(self, 
                 images_dir: str,
                 annotations_file: str,
                 input_size: Tuple[int, int] = (256, 256),
                 output_size: Tuple[int, int] = (64, 64),
                 is_training: bool = True,
                 use_flip: bool = True,
                 rotation_range: float = 30.0,
                 scale_range: Tuple[float, float] = (0.7, 1.3)):
        """
        Args:
            images_dir: Directory containing images
            annotations_file: Path to MPII annotations (.mat file)
            input_size: Input image size (height, width)
            output_size: Output heatmap size (height, width)
            is_training: Whether this is training mode
            use_flip: Whether to use horizontal flipping
            rotation_range: Random rotation range in degrees
            scale_range: Random scale range
        """
        self.images_dir = images_dir
        self.input_size = input_size
        self.output_size = output_size
        self.is_training = is_training
        self.use_flip = use_flip
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        
        # Get list of available images first (needed for dummy annotations)
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(images_dir) else []
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        # Filter annotations to only include images we have
        self.valid_annotations = []
        for ann in self.annotations:
            if ann['image_name'] in self.image_files:
                self.valid_annotations.append(ann)
        
        print(f"Loaded {len(self.valid_annotations)} valid annotations from {len(self.image_files)} images")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        """Load MPII annotations from .mat file"""
        try:
            mat_data = loadmat(annotations_file)
            release = mat_data['RELEASE']
            
            annotations = []
            annolist = release['annolist'][0, 0]
            img_train = release['img_train'][0, 0]
            
            for i in range(len(annolist)):
                # Check if this is a training image
                is_train = img_train[i, 0] == 1
                
                if is_train != self.is_training:
                    continue
                
                image_info = annolist[i, 0]
                image_name = image_info['image'][0, 0]['name'][0]
                
                # Extract person annotations
                if 'annorect' in image_info.dtype.names:
                    annorect = image_info['annorect'][0, 0]
                    
                    for j in range(len(annorect)):
                        person_ann = annorect[j, 0]
                        
                        # Extract keypoints
                        keypoints = np.zeros((16, 3))  # x, y, visibility
                        
                        if 'annopoints' in person_ann.dtype.names and person_ann['annopoints'].size > 0:
                            points = person_ann['annopoints'][0, 0]['point'][0]
                            
                            for point in points:
                                joint_id = point['id'][0, 0]
                                if 0 <= joint_id < 16:
                                    keypoints[joint_id, 0] = point['x'][0, 0]
                                    keypoints[joint_id, 1] = point['y'][0, 0]
                                    keypoints[joint_id, 2] = point['is_visible'][0, 0]
                        
                        # Extract bounding box and scale
                        if 'scale' in person_ann.dtype.names:
                            scale = person_ann['scale'][0, 0]
                        else:
                            scale = 1.0
                        
                        if 'objpos' in person_ann.dtype.names:
                            objpos = person_ann['objpos'][0, 0]
                            center_x = objpos['x'][0, 0]
                            center_y = objpos['y'][0, 0]
                        else:
                            # Calculate center from keypoints
                            valid_points = keypoints[keypoints[:, 2] > 0]
                            if len(valid_points) > 0:
                                center_x = np.mean(valid_points[:, 0])
                                center_y = np.mean(valid_points[:, 1])
                            else:
                                center_x = center_y = 0
                        
                        annotations.append({
                            'image_name': image_name,
                            'keypoints': keypoints,
                            'center': (center_x, center_y),
                            'scale': scale
                        })
            
            return annotations
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            print("Creating dummy annotations for images without MPII data...")
            return self._create_dummy_annotations()
    
    def _create_dummy_annotations(self) -> List[Dict]:
        """Create dummy annotations for images without MPII data"""
        annotations = []
        # Use image_files if available, otherwise read from directory
        if hasattr(self, 'image_files') and len(self.image_files) > 0:
            image_list = self.image_files
        else:
            # Fallback: read images directory directly
            if os.path.exists(self.images_dir):
                image_list = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            else:
                image_list = []
        
        for img_file in image_list[:1000]:  # Limit to first 1000 for demo
            annotations.append({
                'image_name': img_file,
                'keypoints': np.zeros((16, 3)),
                'center': (128, 128),
                'scale': 1.0
            })
        return annotations
    
    def __len__(self):
        return len(self.valid_annotations)
    
    def __getitem__(self, idx):
        ann = self.valid_annotations[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, ann['image_name'])
        image = cv2.imread(img_path)
        if image is None:
            # If image doesn't exist, create a dummy one
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get keypoints
        keypoints = ann['keypoints'].copy()
        center = ann['center']
        scale = ann['scale']
        
        # Data augmentation
        if self.is_training:
            image, keypoints, center, scale = self._augment_data(
                image, keypoints, center, scale
            )
        
        # Crop and resize image
        image, keypoints = self._crop_and_resize(image, keypoints, center, scale)
        
        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints)
        
        # Convert to tensor
        image_tensor = self.transform(Image.fromarray(image))
        
        return {
            'image': image_tensor,
            'heatmaps': torch.FloatTensor(heatmaps),
            'keypoints': torch.FloatTensor(keypoints),
            'image_name': ann['image_name']
        }
    
    def _augment_data(self, image, keypoints, center, scale):
        """Apply data augmentation"""
        height, width = image.shape[:2]
        
        # Random horizontal flip
        if self.use_flip and random.random() < 0.5:
            image = cv2.flip(image, 1)
            keypoints[:, 0] = width - keypoints[:, 0]
            center = (width - center[0], center[1])
            
            # Flip joint pairs
            flip_pairs = [(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)]
            for pair in flip_pairs:
                keypoints[pair[0]], keypoints[pair[1]] = keypoints[pair[1]].copy(), keypoints[pair[0]].copy()
        
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (width, height))
            
            # Rotate keypoints
            ones = np.ones(shape=(len(keypoints), 1))
            points_ones = np.hstack([keypoints[:, :2], ones])
            rotated_points = M.dot(points_ones.T).T
            keypoints[:, :2] = rotated_points
        
        # Random scale
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            scale *= scale_factor
        
        return image, keypoints, center, scale
    
    def _crop_and_resize(self, image, keypoints, center, scale):
        """Crop and resize image around the person"""
        height, width = image.shape[:2]
        
        # Calculate crop size
        crop_size = int(200 * scale)
        crop_size = min(crop_size, min(height, width))
        
        # Calculate crop coordinates
        x1 = max(0, int(center[0] - crop_size // 2))
        y1 = max(0, int(center[1] - crop_size // 2))
        x2 = min(width, x1 + crop_size)
        y2 = min(height, y1 + crop_size)
        
        # Adjust if crop goes out of bounds
        if x2 - x1 < crop_size:
            if x1 == 0:
                x2 = min(width, x1 + crop_size)
            else:
                x1 = max(0, x2 - crop_size)
        
        if y2 - y1 < crop_size:
            if y1 == 0:
                y2 = min(height, y1 + crop_size)
            else:
                y1 = max(0, y2 - crop_size)
        
        # Crop image
        cropped_image = image[y1:y2, x1:x2]
        
        # Adjust keypoints
        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1
        
        # Resize to input size
        cropped_image = cv2.resize(cropped_image, self.input_size)
        
        # Scale keypoints
        scale_x = self.input_size[1] / (x2 - x1)
        scale_y = self.input_size[0] / (y2 - y1)
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        return cropped_image, keypoints
    
    def _generate_heatmaps(self, keypoints):
        """Generate Gaussian heatmaps for keypoints"""
        num_joints = len(self.JOINT_NAMES)
        heatmaps = np.zeros((num_joints, self.output_size[0], self.output_size[1]))
        
        # Scale factor from input to output
        scale_x = self.output_size[1] / self.input_size[1]
        scale_y = self.output_size[0] / self.input_size[0]
        
        for i in range(num_joints):
            if keypoints[i, 2] > 0:  # If joint is visible
                x = int(keypoints[i, 0] * scale_x)
                y = int(keypoints[i, 1] * scale_y)
                
                if 0 <= x < self.output_size[1] and 0 <= y < self.output_size[0]:
                    heatmaps[i] = self._generate_gaussian_heatmap(
                        self.output_size, (x, y), sigma=2.0
                    )
        
        return heatmaps
    
    def _generate_gaussian_heatmap(self, size, center, sigma=2.0):
        """Generate a 2D Gaussian heatmap"""
        height, width = size
        x = np.arange(0, width, 1, dtype=np.float32)
        y = np.arange(0, height, 1, dtype=np.float32)
        y = y[:, np.newaxis]
        
        x0, y0 = center
        heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        return heatmap


def create_data_loaders(images_dir: str, 
                       annotations_file: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Create full dataset
    full_dataset = MPIIPoseDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        is_training=True
    )
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update training flag for validation dataset
    val_dataset.dataset.is_training = False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    images_dir = "images"
    annotations_file = "mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
    
    dataset = MPIIPoseDataset(images_dir, annotations_file)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"Keypoints shape: {sample['keypoints'].shape}")

