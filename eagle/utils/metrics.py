"""
Evaluation metrics.
"""

import torch
import numpy as np
from typing import Dict
from scipy import ndimage


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU).
    
    Args:
        predictions: Binary predictions (B, T, 1, H, W) or (B, 1, H, W)
        targets: Binary targets (B, T, 1, H, W) or (B, 1, H, W)
    
    Returns:
        IoU per sample (B,) or (B, T)
    """
    # Flatten spatial dimensions
    predictions_flat = predictions.flatten(start_dim=-2)  # (B, T, 1, H*W)
    targets_flat = targets.flatten(start_dim=-2)
    
    # Compute intersection and union
    intersection = (predictions_flat * targets_flat).sum(dim=-1)  # (B, T, 1)
    union = ((predictions_flat + targets_flat) > 0).float().sum(dim=-1)
    
    # Compute IoU
    iou = intersection / (union + 1e-8)
    
    # Remove channel dimension
    iou = iou.squeeze(-1)
    
    return iou


def compute_pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-wise accuracy.
    
    Args:
        predictions: Binary predictions (B, T, 1, H, W) or (B, 1, H, W)
        targets: Binary targets (B, T, 1, H, W) or (B, 1, H, W)
    
    Returns:
        Accuracy per sample (B,) or (B, T)
    """
    # Flatten spatial dimensions
    predictions_flat = predictions.flatten(start_dim=-2)
    targets_flat = targets.flatten(start_dim=-2)
    
    # Compute correct predictions
    correct = (predictions_flat == targets_flat).float().sum(dim=-1)
    total = predictions_flat.shape[-1]
    
    # Compute accuracy
    accuracy = correct / total
    
    # Remove channel dimension
    accuracy = accuracy.squeeze(-1)
    
    return accuracy


def compute_boundary_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dilation_ratio: float = 0.02,
) -> torch.Tensor:
    """
    Compute boundary IoU.
    Focuses on the boundary region of the mask.
    
    Args:
        predictions: Binary predictions (B, T, 1, H, W) or (B, 1, H, W)
        targets: Binary targets (B, T, 1, H, W) or (B, 1, H, W)
        dilation_ratio: Ratio of mask size for boundary dilation
    
    Returns:
        Boundary IoU per sample (B,) or (B, T)
    """
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    shape = predictions_np.shape
    H, W = shape[-2:]
    
    # Determine dilation size
    dilation_size = int(dilation_ratio * np.sqrt(H * W))
    
    # Flatten batch dimensions
    original_shape = predictions_np.shape
    predictions_np = predictions_np.reshape(-1, H, W)
    targets_np = targets_np.reshape(-1, H, W)
    
    boundary_ious = []
    
    for pred, target in zip(predictions_np, targets_np):
        # Extract boundaries
        pred_boundary = _extract_boundary(pred, dilation_size)
        target_boundary = _extract_boundary(target, dilation_size)
        
        # Compute IoU on boundaries
        intersection = (pred_boundary & target_boundary).sum()
        union = (pred_boundary | target_boundary).sum()
        
        if union == 0:
            boundary_iou = 0.0
        else:
            boundary_iou = intersection / union
        
        boundary_ious.append(boundary_iou)
    
    boundary_ious = np.array(boundary_ious)
    boundary_ious = boundary_ious.reshape(original_shape[:-2])
    
    return torch.from_numpy(boundary_ious).float()


def _extract_boundary(mask: np.ndarray, dilation_size: int) -> np.ndarray:
    """Extract boundary of a binary mask."""
    # Dilate
    dilated = ndimage.binary_dilation(mask, iterations=dilation_size)
    
    # Erode
    eroded = ndimage.binary_erosion(mask, iterations=dilation_size)
    
    # Boundary is dilated - eroded
    boundary = dilated & (~eroded)
    
    return boundary


def compute_precision_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple:
    """
    Compute precision and recall.
    
    Args:
        predictions: Binary predictions (B, T, 1, H, W)
        targets: Binary targets (B, T, 1, H, W)
    
    Returns:
        Tuple of (precision, recall)
    """
    # Flatten
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # True positives, false positives, false negatives
    tp = ((predictions_flat == 1) & (targets_flat == 1)).float().sum()
    fp = ((predictions_flat == 1) & (targets_flat == 0)).float().sum()
    fn = ((predictions_flat == 0) & (targets_flat == 1)).float().sum()
    
    # Compute precision and recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return precision.item(), recall.item()


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Binary predictions
        targets: Binary targets
    
    Returns:
        F1 score
    """
    precision, recall = compute_precision_recall(predictions, targets)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1


class MetricTracker:
    """
    Track and compute metrics over multiple batches.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * count
            self.counts[key] += count
    
    def get_average(self) -> Dict[str, float]:
        """Get average metrics."""
        avg_metrics = {}
        for key in self.metrics:
            avg_metrics[key] = self.metrics[key] / max(self.counts[key], 1)
        return avg_metrics
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def get_current(self) -> Dict[str, float]:
        """Get current accumulated metrics (without averaging)."""
        return self.metrics.copy()