"""
Learning rate schedulers.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of epochs
        warmup_start_lr: Starting learning rate for warmup
        min_lr: Minimum learning rate
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            lrs = [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
        
        return lrs


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        max_epochs: Total number of epochs
        power: Polynomial power
        min_lr: Minimum learning rate
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer,
        max_epochs: int,
        power: float = 0.9,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate."""
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        lrs = [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]
        return lrs