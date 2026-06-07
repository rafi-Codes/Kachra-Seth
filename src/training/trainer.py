"""Trainer class for stack count prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time
from tqdm import tqdm

from ..models import StackCountModel
from .losses import CombinedLoss
from .metrics import MetricsTracker, PerformanceThresholds
from .schedulers import WarmupCosineScheduler


class StackCountTrainer:
    """Trainer for stack count prediction model."""
    
    def __init__(
        self,
        model: StackCountModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """Initialize trainer.
        
        Args:
            model: StackCountModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training configuration
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 100)
        self.batch_size = training_config.get('batch_size', 32)
        
        # Optimizer
        optimizer_config = training_config.get('optimizer', {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_config.get('lr', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        scheduler_config = training_config.get('scheduler', {})
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=self.epochs
        )
        
        # Loss function
        self.loss_fn = CombinedLoss(
            primary_weight=0.8,
            confidence_weight=0.2
        )
        
        # Metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Paths
        self.checkpoint_dir = Path(config.get('paths', {}).get('checkpoints', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        early_stopping_config = training_config.get('early_stopping', {})
        self.early_stopping_patience = early_stopping_config.get('patience', 15)
        self.early_stopping_enabled = self.early_stopping_patience > 0
    
    def train_epoch(self, epoch: int, phase: int = 1) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            phase: Curriculum learning phase
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} (Train)")
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            counts = batch['count'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            count_pred = outputs['count']
            confidence_pred = outputs['confidence']
            
            # Calculate loss
            loss, loss_components = self.loss_fn(count_pred, confidence_pred, counts)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(
                count_pred, counts, confidence_pred,
                loss_dict=loss_components
            )
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'mae': self.train_metrics.compute().get('mae', 0)
            })
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self, epoch: int, phase: int = 1) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            phase: Curriculum learning phase
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} (Val)")
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                images = batch['image'].to(self.device)
                counts = batch['count'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                count_pred = outputs['count']
                confidence_pred = outputs['confidence']
                
                # Calculate loss
                loss, loss_components = self.loss_fn(count_pred, confidence_pred, counts)
                
                # Update metrics
                self.val_metrics.update(
                    count_pred, counts, confidence_pred,
                    loss_dict=loss_components
                )
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mae': self.val_metrics.compute().get('mae', 0)
                })
        
        # Compute epoch metrics
        metrics = self.val_metrics.compute()
        metrics['loss'] = total_loss / len(self.val_loader)
        
        # Get stratified metrics
        count_ranges = [(5, 50), (51, 150), (151, 500)]
        strata_metrics = self.val_metrics.get_strata_metrics(count_ranges)
        metrics['strata'] = strata_metrics
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Complete training loop with curriculum learning.
        
        Returns:
            Dictionary with training history
        """
        history = {
            'train': [],
            'val': []
        }
        
        # Get curriculum configuration
        curriculum_config = self.config.get('training', {}).get('curriculum', {})
        
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            
            # Determine curriculum phase
            phase = self._get_curriculum_phase(epoch, curriculum_config)
            
            # Update data loader phase if needed
            if hasattr(self.train_loader.dataset, 'update_phase'):
                self.train_loader.dataset.update_phase(phase)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.epochs} - Curriculum Phase {phase}")
            print(f"{'='*60}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch, phase)
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate(epoch, phase)
            history['val'].append(val_metrics)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Check for improvement
            val_mae = val_metrics.get('mae', float('inf'))
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best checkpoint
                self._save_checkpoint('best')
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if epoch % 5 == 0:
                self._save_checkpoint(f'epoch_{epoch}')
            
            # Early stopping
            if self.early_stopping_enabled and self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation MAE: {self.best_val_mae:.4f} at epoch {self.best_epoch}")
                break
        
        # Load best model
        self._load_checkpoint('best')
        
        # Print training summary
        self._print_training_summary(history)
        
        return history
    
    def _get_curriculum_phase(self, epoch: int, config: Dict) -> int:
        """Determine curriculum learning phase.
        
        Args:
            epoch: Current epoch
            config: Curriculum configuration
            
        Returns:
            Phase number (1, 2, or 3)
        """
        phase1 = config.get('phase1', {}).get('epochs', [1, 30])
        phase2 = config.get('phase2', {}).get('epochs', [31, 70])
        phase3 = config.get('phase3', {}).get('epochs', [71, 100])
        
        if phase1[0] <= epoch <= phase1[1]:
            return 1
        elif phase2[0] <= epoch <= phase2[1]:
            return 2
        else:
            return 3
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict
    ):
        """Print epoch summary.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics.get('mae', 0):.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics.get('mae', 0):.4f}")
        print(f"  Val MAPE: {val_metrics.get('mape', 0):.4f}%")
        print(f"  Val Within 5%: {val_metrics.get('within_5pct', 0):.4f}")
        print(f"  Val ECE: {val_metrics.get('confidence_ece', 0):.4f}")
        
        if 'strata' in val_metrics:
            print(f"\n  Stratified Metrics:")
            for range_name, metrics in val_metrics['strata'].items():
                print(f"    {range_name}: MAE={metrics['mae']:.4f}, Acc={metrics['within_5pct']:.4f}")
        
        print(f"  Best Val MAE: {self.best_val_mae:.4f} (Epoch {self.best_epoch})")
    
    def _print_training_summary(self, history: Dict):
        """Print overall training summary.
        
        Args:
            history: Training history
        """
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"Total epochs trained: {len(history['train'])}")
        print(f"Best validation MAE: {self.best_val_mae:.4f} (Epoch {self.best_epoch})")
        
        # Final validation metrics
        final_val = history['val'][-1]
        print(f"\nFinal Validation Metrics:")
        print(f"  MAE: {final_val.get('mae', 0):.4f}")
        print(f"  MAPE: {final_val.get('mape', 0):.4f}%")
        print(f"  Within 5%: {final_val.get('within_5pct', 0):.4f}")
        print(f"  ECE: {final_val.get('confidence_ece', 0):.4f}")
        
        # Check performance thresholds
        check_results = PerformanceThresholds.check_metrics(final_val)
        PerformanceThresholds.print_results(check_results)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, name: str):
        """Load model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = self.checkpoint_dir / f'{name}.pth'
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_mae = checkpoint['best_val_mae']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
