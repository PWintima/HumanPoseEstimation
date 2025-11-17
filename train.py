"""
Training Pipeline for Human Pose Estimation Models

This module provides a comprehensive training system with:
- Model training with configurable optimizers and schedulers
- Validation and evaluation metrics (PCK accuracy)
- Checkpointing and model saving
- Training curve visualization
- Early stopping support
- Gradient clipping
- Comprehensive logging

The trainer supports both HRNet and SimpleBaseline architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

from dataset import MPIIPoseDataset, create_data_loaders
from model import create_model


class PoseTrainer:
    """
    Comprehensive trainer for pose estimation models.
    
    Handles the complete training lifecycle including:
    - Training loop with progress tracking
    - Validation and metric computation
    - Model checkpointing (best and latest)
    - Learning rate scheduling
    - Early stopping
    - Training curve visualization
    - Comprehensive logging
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 config: Dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        self.output_dir = config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _create_optimizer(self):
        """Create optimizer based on config"""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'step').lower()
        
        if scheduler_type == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = self.config.get('epochs', 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'plateau':
            patience = self.config.get('patience', 10)
            factor = self.config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience, factor=factor)
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        """
        Train the model for one complete epoch.
        
        Processes all batches in the training set, computes loss, performs
        backpropagation, and updates model parameters. Includes gradient
        clipping and progress tracking.
        
        Returns:
            Average training loss for the epoch
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Create progress bar for visual feedback
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device (GPU or CPU)
            images = batch['image'].to(self.device)
            heatmaps = batch['heatmaps'].to(self.device)  # Ground truth heatmaps
            
            # Forward pass: predict heatmaps from images
            self.optimizer.zero_grad()  # Clear gradients from previous iteration
            predicted_heatmaps = self.model(images)
            
            # Compute loss: MSE between predicted and ground truth heatmaps
            loss = self.criterion(predicted_heatmaps, heatmaps)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Gradient clipping: prevent exploding gradients
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            # Update model parameters
            self.optimizer.step()
            
            # Accumulate loss for epoch average
            total_loss += loss.item()
            
            # Update progress bar with current and average loss
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.6f}'
            })
            
            # Log batch loss at specified intervals
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
        
        # Compute and store average loss for this epoch
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model on the validation set.
        
        Evaluates model performance without gradient computation (faster).
        Computes both loss and PCK (Percentage of Correct Keypoints) accuracy.
        
        Returns:
            Tuple of (average validation loss, average PCK@0.5 accuracy)
        """
        # Set model to evaluation mode (disables dropout, uses batch norm stats)
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_loader)
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                images = batch['image'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)  # Ground truth heatmaps
                keypoints = batch['keypoints'].to(self.device)  # Ground truth keypoint coordinates
                
                # Forward pass: predict heatmaps
                predicted_heatmaps = self.model(images)
                
                # Compute loss: MSE between predicted and ground truth heatmaps
                loss = self.criterion(predicted_heatmaps, heatmaps)
                total_loss += loss.item()
                
                # Compute PCK accuracy: percentage of keypoints within threshold distance
                accuracy = self._compute_pck_accuracy(predicted_heatmaps, keypoints)
                total_accuracy += accuracy
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'PCK@0.5': f'{accuracy:.4f}'
                })
        
        # Compute averages across all batches
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Store metrics for plotting
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def _compute_pck_accuracy(self, predicted_heatmaps: torch.Tensor, keypoints: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Compute PCK (Percentage of Correct Keypoints) accuracy.
        
        PCK measures the percentage of predicted keypoints that are within a
        threshold distance from the ground truth keypoints. This is a standard
        metric for pose estimation evaluation.
        
        Args:
            predicted_heatmaps: Predicted heatmaps [B, num_joints, H, W]
            keypoints: Ground truth keypoints [B, num_joints, 3] (x, y, visibility)
            threshold: Distance threshold in pixels (default: 0.5)
            
        Returns:
            PCK accuracy as a float (0.0 to 1.0)
        """
        batch_size, num_joints, height, width = predicted_heatmaps.shape
        
        # Extract predicted keypoint locations from heatmaps
        # Find the location with maximum heatmap value for each joint
        predicted_heatmaps_flat = predicted_heatmaps.view(batch_size, num_joints, -1)
        _, max_indices = torch.max(predicted_heatmaps_flat, dim=2)  # Find max value index
        pred_y = max_indices // width  # Convert flat index to (y, x) coordinates
        pred_x = max_indices % width
        
        # Scale ground truth keypoints from image coordinates to heatmap coordinates
        # Assuming input images are resized to 256x256 before being fed to the model
        scale_x = width / 256.0
        scale_y = height / 256.0
        
        gt_x = keypoints[:, :, 0] * scale_x  # Ground truth x coordinates
        gt_y = keypoints[:, :, 1] * scale_y  # Ground truth y coordinates
        
        # Compute Euclidean distance between predicted and ground truth keypoints
        distances = torch.sqrt((pred_x.float() - gt_x) ** 2 + (pred_y.float() - gt_y) ** 2)
        
        # Only consider visible keypoints (visibility flag > 0)
        visible = keypoints[:, :, 2] > 0
        
        # Count keypoints that are within threshold distance and visible
        correct = (distances < threshold) & visible
        
        # Compute accuracy: correct keypoints / total visible keypoints
        accuracy = correct.sum().float() / visible.sum().float()
        
        return accuracy.item()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'Saved best model with validation loss: {self.best_val_loss:.6f}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        
        self.logger.info(f'Loaded checkpoint from epoch {self.current_epoch}')
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.val_accuracies, 'g-', label='PCK@0.5')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()
    
    def train(self):
        """Main training loop"""
        epochs = self.config.get('epochs', 100)
        save_interval = self.config.get('save_interval', 10)
        early_stopping_patience = self.config.get('early_stopping_patience', 20)
        
        self.logger.info(f'Starting training for {epochs} epochs')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val PCK@0.5: {val_accuracy:.4f}, LR: {current_lr:.6f}')
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Plot training curves
            if epoch % save_interval == 0:
                self.plot_training_curves()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}')
                break
        
        self.logger.info(f'Training completed. Best validation loss: {self.best_val_loss:.6f} at epoch {best_epoch}')
        
        # Save final model
        self.save_checkpoint(is_best=True)
        self.plot_training_curves()


def create_config() -> Dict:
    """
    Create default training configuration dictionary.
    
    Returns a comprehensive configuration dictionary with all training
    hyperparameters, model settings, data settings, and logging options.
    This can be customized before training to adjust model behavior.
    
    Returns:
        Dictionary containing all training configuration parameters
    """
    return {
        # Model settings
        'model_type': 'simplebaseline',  # 'hrnet' or 'simplebaseline'
        'num_joints': 16,
        'backbone': 'resnet50',  # For SimpleBaseline
        
        # Training settings
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adam',  # 'adam', 'adamw', 'sgd'
        'momentum': 0.9,  # For SGD
        'grad_clip': 5.0,
        
        # Learning rate scheduler
        'scheduler': 'step',  # 'step', 'cosine', 'plateau'
        'step_size': 30,
        'gamma': 0.1,
        'patience': 10,  # For ReduceLROnPlateau
        'factor': 0.5,  # For ReduceLROnPlateau
        
        # Data settings
        'train_split': 0.8,
        'num_workers': 4,
        'input_size': (256, 256),
        'output_size': (64, 64),
        
        # Logging and saving
        'output_dir': 'outputs',
        'log_level': 'INFO',
        'log_interval': 100,
        'save_interval': 10,
        'early_stopping_patience': 20,
        
        # Data augmentation
        'use_flip': True,
        'rotation_range': 30.0,
        'scale_range': (0.7, 1.3)
    }


def main():
    """
    Main training function.
    
    Orchestrates the complete training pipeline:
    1. Setup device (GPU/CPU)
    2. Load configuration
    3. Create data loaders
    4. Initialize model
    5. Create trainer
    6. Start training loop
    
    All outputs (checkpoints, logs, plots) are saved to a timestamped
    directory under outputs/training_YYYYMMDD_HHMMSS/
    """
    # Detect and set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load training configuration
    config = create_config()
    
    # Create timestamped output directory for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = f"outputs/training_{timestamp}"
    
    # Save configuration to JSON file for reproducibility
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders for training and validation sets
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        images_dir='images',
        annotations_file='mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_split=config['train_split']
    )
    
    # Check if dataset is empty (no images found)
    if train_loader is None or val_loader is None:
        print("\n" + "="*60)
        print("ERROR: No training data found!")
        print("="*60)
        print("Please ensure:")
        print("1. Images are placed in the 'images/' directory")
        print("2. Images are in .jpg, .jpeg, or .png format")
        print("3. The 'images/' directory exists and is accessible")
        print("\nAlternatively, you can:")
        print("- Run stage3_main.py to prepare your dataset")
        print("- Or place your images in the 'images/' directory")
        print("="*60)
        return
    
    # Display dataset statistics
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create model instance based on configuration
    print("Creating model...")
    model = create_model(
        model_type=config['model_type'],
        num_joints=config['num_joints'],
        backbone=config.get('backbone', 'resnet50')
    )
    
    # Initialize trainer with model, data loaders, and configuration
    trainer = PoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Start the training loop
    trainer.train()


if __name__ == "__main__":
    main()

