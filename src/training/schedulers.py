"""Learning rate schedulers with warmup support."""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Optional


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7
    ):
        """Initialize warmup cosine scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            min_lr: Minimum learning rate
            warmup_start_lr: Learning rate at start of warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate."""
        self.current_epoch += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """Calculate learning rate for current epoch."""
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return float(lr)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """Get scheduler state."""
        return {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'min_lr': self.min_lr,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_epochs = state_dict['max_epochs']
        self.min_lr = state_dict['min_lr']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.base_lr = state_dict['base_lr']


class WarmupScheduler:
    """Simple linear warmup scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-7
    ):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Learning rate at start of warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
        else:
            lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """Get scheduler state."""
        return {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.base_lr = state_dict['base_lr']


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: dict
):
    """Create scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        
    Returns:
        Scheduler instance
    """
    scheduler_config = config.get('training', {}).get('scheduler', {})
    scheduler_name = scheduler_config.get('name', 'cosine_annealing')
    
    if scheduler_name == 'cosine_annealing':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=scheduler_config.get('warmup_epochs', 5),
            max_epochs=config.get('training', {}).get('epochs', 100),
            min_lr=1e-6
        )
    elif scheduler_name == 'warmup':
        return WarmupScheduler(
            optimizer,
            warmup_epochs=scheduler_config.get('warmup_epochs', 5)
        )
    else:
        # Default PyTorch schedulers
        if scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        else:
            return WarmupCosineScheduler(optimizer)


# Test schedulers
if __name__ == "__main__":
    print("Testing Schedulers...")
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Test WarmupCosineScheduler
    print("\n1. Testing WarmupCosineScheduler...")
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=100
    )
    
    lrs = []
    for epoch in range(1, 21):
        scheduler.step()
        lrs.append(scheduler.get_lr())
    
    print(f"First 5 epochs (warmup): {[f'{lr:.2e}' for lr in lrs[:5]]}")
    print(f"Epochs 6-10: {[f'{lr:.2e}' for lr in lrs[5:10]]}")
    print(f"Epochs 15-20: {[f'{lr:.2e}' for lr in lrs[14:]]}")
    
    # Test WarmupScheduler
    print("\n2. Testing WarmupScheduler...")
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler2 = WarmupScheduler(optimizer2, warmup_epochs=5)
    
    lrs2 = []
    for epoch in range(1, 11):
        scheduler2.step()
        lrs2.append(scheduler2.get_lr())
    
    print(f"Learning rates: {[f'{lr:.2e}' for lr in lrs2]}")
    
    print("\nAll scheduler tests passed!")
