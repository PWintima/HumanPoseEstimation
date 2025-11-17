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
    """Comprehensive trainer for pose estimation models"""
    
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            heatmaps = batch['heatmaps'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_heatmaps = self.model(images)
            
            # Compute loss
            loss = self.criterion(predicted_heatmaps, heatmaps)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.6f}'
            })
            
            # Log batch loss
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device)
                heatmaps = batch['heatmaps'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                
                # Forward pass
                predicted_heatmaps = self.model(images)
                
                # Compute loss
                loss = self.criterion(predicted_heatmaps, heatmaps)
                total_loss += loss.item()
                
                # Compute accuracy (PCK@0.5)
                accuracy = self._compute_pck_accuracy(predicted_heatmaps, keypoints)
                total_accuracy += accuracy
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'PCK@0.5': f'{accuracy:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def _compute_pck_accuracy(self, predicted_heatmaps: torch.Tensor, keypoints: torch.Tensor, threshold: float = 0.5) -> float:
        """Compute PCK (Percentage of Correct Keypoints) accuracy"""
        batch_size, num_joints, height, width = predicted_heatmaps.shape
        
        # Get predicted keypoint locations
        predicted_heatmaps_flat = predicted_heatmaps.view(batch_size, num_joints, -1)
        _, max_indices = torch.max(predicted_heatmaps_flat, dim=2)
        pred_y = max_indices // width
        pred_x = max_indices % width
        
        # Scale keypoints to heatmap size
        scale_x = width / 256.0  # Assuming input image size is 256x256
        scale_y = height / 256.0
        
        gt_x = keypoints[:, :, 0] * scale_x
        gt_y = keypoints[:, :, 1] * scale_y
        
        # Compute distances
        distances = torch.sqrt((pred_x.float() - gt_x) ** 2 + (pred_y.float() - gt_y) ** 2)
        
        # Check visibility (keypoints[:, :, 2] > 0)
        visible = keypoints[:, :, 2] > 0
        
        # Compute PCK
        correct = (distances < threshold) & visible
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
    """Create default training configuration"""
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
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create configuration
    config = create_config()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = f"outputs/training_{timestamp}"
    
    # Save configuration
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        images_dir='images',
        annotations_file='mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_split=config['train_split']
    )
    
    # Check if dataset is empty
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
    
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=config['model_type'],
        num_joints=config['num_joints'],
        backbone=config.get('backbone', 'resnet50')
    )
    
    # Create trainer
    trainer = PoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

