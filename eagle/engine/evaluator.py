"""
Evaluation engine for EAGLE models.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.metrics import compute_iou, compute_pixel_accuracy, compute_boundary_iou
from ..utils.visualization import visualize_predictions

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluation engine for EAGLE models.
    
    Args:
        cfg: Configuration object
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        save_predictions: Whether to save predictions
        visualize: Whether to generate visualizations
    """
    
    def __init__(
        self,
        cfg,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        save_predictions: bool = False,
        visualize: bool = False,
    ):
        self.cfg = cfg
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_predictions = save_predictions
        self.visualize = visualize
        
        # Output directories
        if save_predictions:
            self.pred_dir = cfg.TEST.PRED_DIR
            os.makedirs(self.pred_dir, exist_ok=True)
        
        if visualize:
            self.vis_dir = cfg.TEST.VIS_DIR
            os.makedirs(self.vis_dir, exist_ok=True)
        
        # Results storage
        self.results = []
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting evaluation...")
        self.model.eval()
        
        all_metrics = {
            'iou': [],
            'pixel_accuracy': [],
            'boundary_iou': [],
        }
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self._to_device(batch)
            
            # Forward pass
            outputs = self.model(
                query_images=batch['query_image'],
                query_masks=batch['query_mask'],
                search_images=batch['search_image'],
                frame_indices=batch.get('frame_indices'),
            )
            
            predictions = outputs['predictions']
            targets = batch.get('search_mask')
            
            # Compute metrics if ground truth is available
            if targets is not None:
                metrics = self._compute_metrics(predictions, targets)
                
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
                
                pbar.set_postfix({
                    'iou': f"{metrics.get('iou', 0.0):.4f}",
                })
            
            # Save predictions
            if self.save_predictions:
                self._save_predictions(
                    predictions,
                    batch.get('video_id'),
                    batch.get('frame_indices'),
                    batch_idx,
                )
            
            # Visualize
            if self.visualize:
                self._visualize(
                    batch['query_image'],
                    batch['query_mask'],
                    batch['search_image'],
                    predictions,
                    targets,
                    batch_idx,
                )
        
        # Aggregate metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            if len(values) > 0:
                final_metrics[key] = np.mean(values)
                final_metrics[f'{key}_std'] = np.std(values)
        
        # Log results
        logger.info("Evaluation Results:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save results
        self._save_results(final_metrics)
        
        return final_metrics
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Handle single frame vs multi-frame
        if predictions.dim() == 4:
            predictions = predictions.unsqueeze(1)
            targets = targets.unsqueeze(1)
        
        # Threshold predictions
        pred_binary = (predictions > 0.5).float()
        
        # Compute IoU
        iou = compute_iou(pred_binary, targets).mean().item()
        
        # Compute pixel accuracy
        pixel_acc = compute_pixel_accuracy(pred_binary, targets).mean().item()
        
        # Compute boundary IoU
        boundary_iou = compute_boundary_iou(pred_binary, targets).mean().item()
        
        return {
            'iou': iou,
            'pixel_accuracy': pixel_acc,
            'boundary_iou': boundary_iou,
        }
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def _save_predictions(
        self,
        predictions: torch.Tensor,
        video_ids: Optional[List[str]],
        frame_indices: Optional[List[int]],
        batch_idx: int,
    ):
        """Save predictions to disk."""
        predictions_np = predictions.cpu().numpy()
        
        B = predictions_np.shape[0]
        
        for b in range(B):
            if video_ids is not None:
                video_id = video_ids[b]
            else:
                video_id = f"video_{batch_idx}_{b}"
            
            # Create directory for this video
            video_dir = os.path.join(self.pred_dir, video_id)
            os.makedirs(video_dir, exist_ok=True)
            
            # Save predictions
            if predictions_np.ndim == 5:
                # Multi-frame
                T = predictions_np.shape[1]
                for t in range(T):
                    frame_idx = frame_indices[t] if frame_indices is not None else t
                    pred_t = predictions_np[b, t, 0]
                    
                    save_path = os.path.join(video_dir, f"frame_{frame_idx:06d}.npy")
                    np.save(save_path, pred_t)
            else:
                # Single frame
                pred = predictions_np[b, 0]
                save_path = os.path.join(video_dir, "prediction.npy")
                np.save(save_path, pred)
    
    def _visualize(
        self,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        search_images: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor],
        batch_idx: int,
    ):
        """Generate visualizations."""
        B = query_images.shape[0]
        
        for b in range(B):
            vis_path = os.path.join(self.vis_dir, f"batch_{batch_idx}_sample_{b}.png")
            
            visualize_predictions(
                query_image=query_images[b],
                query_mask=query_masks[b],
                search_image=search_images[b],
                prediction=predictions[b],
                target=targets[b] if targets is not None else None,
                save_path=vis_path,
            )
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save evaluation results to JSON."""
        results_path = os.path.join(self.cfg.OUTPUT.DIR, 'evaluation_results.json')
        
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Results saved to {results_path}")


class VQ2DEvaluator(Evaluator):
    """
    Evaluator specifically for VQ2D task.
    Includes VQ2D-specific metrics and visualization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute VQ2D-specific metrics."""
        # Base metrics
        metrics = super()._compute_metrics(predictions, targets)
        
        # Additional VQ2D metrics
        # Center distance
        center_dist = self._compute_center_distance(predictions, targets)
        metrics['center_distance'] = center_dist
        
        # Detection success rate (IoU > 0.5)
        pred_binary = (predictions > 0.5).float()
        iou_per_sample = compute_iou(pred_binary, targets)
        detection_rate = (iou_per_sample > 0.5).float().mean().item()
        metrics['detection_rate'] = detection_rate
        
        return metrics
    
    def _compute_center_distance(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute average center distance between predictions and targets."""
        pred_binary = (predictions > 0.5).float()
        
        B, T = pred_binary.shape[:2]
        distances = []
        
        for b in range(B):
            for t in range(T):
                pred_mask = pred_binary[b, t, 0]
                target_mask = targets[b, t, 0]
                
                # Compute centers
                pred_center = self._compute_mask_center(pred_mask)
                target_center = self._compute_mask_center(target_mask)
                
                if pred_center is not None and target_center is not None:
                    dist = torch.norm(pred_center - target_center).item()
                    distances.append(dist)
        
        return np.mean(distances) if len(distances) > 0 else 0.0
    
    def _compute_mask_center(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute center of mass of a mask."""
        if mask.sum() == 0:
            return None
        
        H, W = mask.shape
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=mask.device, dtype=torch.float32),
            torch.arange(W, device=mask.device, dtype=torch.float32),
            indexing='ij'
        )
        
        center_y = (mask * y_coords).sum() / mask.sum()
        center_x = (mask * x_coords).sum() / mask.sum()
        
        return torch.stack([center_x, center_y])


class VQ3DEvaluator(Evaluator):
    """
    Evaluator specifically for VQ3D task.
    Includes temporal consistency and 3D metrics.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute VQ3D-specific metrics."""
        # Base metrics
        metrics = super()._compute_metrics(predictions, targets)
        
        # Temporal consistency
        temporal_consistency = self._compute_temporal_consistency(predictions)
        metrics['temporal_consistency'] = temporal_consistency
        
        # Temporal IoU (IoU over time)
        temporal_iou = self._compute_temporal_iou(predictions, targets)
        metrics['temporal_iou'] = temporal_iou
        
        # Tracking accuracy (for video sequences)
        tracking_acc = self._compute_tracking_accuracy(predictions, targets)
        metrics['tracking_accuracy'] = tracking_acc
        
        return metrics
    
    def _compute_temporal_consistency(self, predictions: torch.Tensor) -> float:
        """
        Compute temporal consistency of predictions.
        Measures smoothness of predictions over time.
        """
        if predictions.shape[1] < 2:
            return 1.0
        
        # Compute differences between consecutive frames
        diffs = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 norm of differences
        consistency = 1.0 - torch.mean(torch.abs(diffs)).item()
        
        return max(0.0, consistency)
    
    def _compute_temporal_iou(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute IoU averaged over all frames."""
        pred_binary = (predictions > 0.5).float()
        
        # Compute IoU per frame
        ious = compute_iou(pred_binary, targets)
        
        return ious.mean().item()
    
    def _compute_tracking_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute tracking accuracy.
        Percentage of frames where object is successfully tracked (IoU > 0.5).
        """
        pred_binary = (predictions > 0.5).float()
        
        # Compute IoU per frame
        ious = compute_iou(pred_binary, targets)
        
        # Count successful tracks
        successful = (ious > 0.5).float().mean().item()
        
        return successful