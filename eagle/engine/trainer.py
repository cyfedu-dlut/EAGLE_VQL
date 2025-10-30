"""
Training engine for EAGLE models.
"""

import os
import time
import logging
from typing import Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from ..modeling.meta_arch import build_model, save_checkpoint
from ..utils.logger import setup_logger
from ..utils.metrics import MetricTracker
from ..utils.visualization import visualize_predictions

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training engine for EAGLE models.
    
    Args:
        cfg: Configuration object
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
    """
    
    def __init__(
        self,
        cfg,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Mixed precision training
        self.use_amp = cfg.MIXED_PRECISION
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.current_epoch = 0
        self.current_iter = 0
        self.best_metric = 0.0
        
        self.metric_tracker = MetricTracker()
        
        # Output directories
        self.output_dir = cfg.OUTPUT.DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Logging
        if cfg.OUTPUT.USE_TENSORBOARD:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        else:
            self.tb_writer = None
        
        if cfg.OUTPUT.USE_WANDB:
            import wandb
            wandb.init(
                project=cfg.OUTPUT.WANDB_PROJECT,
                entity=cfg.OUTPUT.WANDB_ENTITY,
                config=cfg,
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.cfg.SOLVER.MAX_EPOCHS}")
        logger.info(f"Total iterations per epoch: {len(self.train_loader)}")
        
        for epoch in range(self.current_epoch, self.cfg.SOLVER.MAX_EPOCHS):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            # Log metrics
            self._log_metrics(train_metrics, 'train', epoch)
            
            # Validation
            if self.val_loader is not None and (epoch + 1) % self.cfg.SOLVER.EVAL_PERIOD == 0:
                val_metrics = self.validate()
                self._log_metrics(val_metrics, 'val', epoch)
                
                # Save best model
                current_metric = val_metrics.get('iou', 0.0)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self._save_checkpoint('best_model.pth', is_best=True)
                    logger.info(f"New best model! IoU: {self.best_metric:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed!")
        
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        self.metric_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self._to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    query_images=batch['query_image'],
                    query_masks=batch['query_mask'],
                    search_images=batch['search_image'],
                    search_masks=batch['search_mask'],
                    frame_indices=batch.get('frame_indices'),
                )
                
                losses = outputs['losses']
                loss = losses['loss_total']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.cfg.SOLVER.CLIP_GRAD_NORM > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.SOLVER.CLIP_GRAD_NORM
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.cfg.SOLVER.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.SOLVER.CLIP_GRAD_NORM
                    )
                
                self.optimizer.step()
            
            # Update metrics
            self.metric_tracker.update({
                'loss': loss.item(),
                'loss_amm': losses['loss_amm'].item(),
                'loss_glm': losses['loss_glm'].item(),
                'loss_fused': losses['loss_fused'].item(),
            })
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })
            
            self.current_iter += 1
        
        return self.metric_tracker.get_average()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        logger.info("Running validation...")
        self.model.eval()
        self.metric_tracker.reset()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self._to_device(batch)
            
            # Forward pass
            outputs = self.model(
                query_images=batch['query_image'],
                query_masks=batch['query_mask'],
                search_images=batch['search_image'],
                search_masks=batch['search_mask'],
                frame_indices=batch.get('frame_indices'),
            )
            
            predictions = outputs['predictions']
            targets = batch['search_mask']
            
            # Compute metrics
            metrics = self._compute_metrics(predictions, targets)
            self.metric_tracker.update(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'iou': f"{metrics.get('iou', 0.0):.4f}",
            })
        
        return self.metric_tracker.get_average()
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Threshold predictions
        pred_binary = (predictions > 0.5).float()
        
        # Compute IoU
        intersection = (pred_binary * targets).sum(dim=[1, 2, 3, 4])
        union = ((pred_binary + targets) > 0).float().sum(dim=[1, 2, 3, 4])
        iou = (intersection / (union + 1e-8)).mean().item()
        
        # Compute pixel accuracy
        correct = (pred_binary == targets).float().sum(dim=[1, 2, 3, 4])
        total = torch.tensor(targets.numel() / targets.size(0), device=targets.device)
        pixel_acc = (correct / total).mean().item()
        
        return {
            'iou': iou,
            'pixel_accuracy': pixel_acc,
        }
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str, epoch: int):
        """Log metrics to tensorboard/wandb."""
        # Console logging
        logger.info(f"{prefix.capitalize()} metrics - Epoch {epoch+1}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Tensorboard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{key}", value, epoch)
        
        # Wandb
        if self.use_wandb:
            import wandb
            wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=epoch)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            save_path=checkpoint_path,
            best_metric=self.best_metric,
            scheduler=self.scheduler,
        )
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")