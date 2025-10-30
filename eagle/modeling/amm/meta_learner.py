"""
Meta-Learner G_θ with Steepest Descent.
Implements iterative refinement of segmentation predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MetaLearner(nn.Module):
    """
    Meta-Learner G_θ.
    
    Performs iterative refinement using steepest descent optimization.
    Each iteration updates the prediction based on gradient of loss.
    
    Algorithm:
    1. Initialize prediction from correlation
    2. For k iterations:
        a. Compute gradient of loss w.r.t. prediction
        b. Update: prediction = prediction - lr * gradient
        c. Refine with learned network
    
    Args:
        feature_dim: Input feature dimension
        num_iterations: Number of meta-learning iterations
        learning_rate: Step size for gradient descent
        regularizer: Regularization weight
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_iterations: int = 3,
        learning_rate: float = 0.1,
        regularizer: float = 0.01,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        
        # Refinement network for each iteration
        self.refinement_networks = nn.ModuleList([
            self._build_refinement_network()
            for _ in range(num_iterations)
        ])
        
        # Learnable step sizes (one per iteration)
        self.step_sizes = nn.Parameter(
            torch.ones(num_iterations) * learning_rate
        )
    
    def _build_refinement_network(self) -> nn.Module:
        """
        Build refinement network for one iteration.
        Uses residual architecture for stable training.
        """
        return nn.Sequential(
            nn.Conv2d(self.feature_dim + 1, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, 1),
        )
    
    def forward(
        self,
        search_features: torch.Tensor,
        initial_prediction: torch.Tensor,
        pseudo_labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Meta-learning refinement.
        
        Args:
            search_features: Search frame features (B, D, H, W)
            initial_prediction: Initial prediction (B, 1, H, W)
            pseudo_labels: Pseudo labels (B, C, H, W)
            weights: Spatial weights (B, 1, H, W), optional
        
        Returns:
            - Final prediction (B, 1, H, W)
            - List of intermediate predictions
        """
        B, D, H, W = search_features.shape
        
        # Initialize
        prediction = initial_prediction
        predictions_history = [prediction]
        
        if weights is None:
            weights = torch.ones_like(prediction)
        
        # Iterative refinement
        for iter_idx in range(self.num_iterations):
            # Compute gradient of loss w.r.t. prediction
            gradient = self._compute_gradient(
                prediction, 
                pseudo_labels, 
                weights
            )
            
            # Steepest descent update
            step_size = self.step_sizes[iter_idx]
            prediction_updated = prediction - step_size * gradient
            
            # Refine with learned network
            refinement_input = torch.cat([search_features, prediction_updated], dim=1)
            residual = self.refinement_networks[iter_idx](refinement_input)
            
            # Update prediction (with residual connection)
            prediction = prediction_updated + residual
            
            # Clamp to valid range
            prediction = torch.clamp(prediction, 0.0, 1.0)
            
            predictions_history.append(prediction)
        
        return prediction, predictions_history
    
    def _compute_gradient(
        self,
        prediction: torch.Tensor,
        pseudo_labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of weighted loss w.r.t. prediction.
        
        Uses Lovász hinge loss gradient as in paper.
        
        Args:
            prediction: Current prediction (B, 1, H, W)
            pseudo_labels: Pseudo labels (B, C, H, W)
            weights: Spatial weights (B, 1, H, W)
        
        Returns:
            Gradient (B, 1, H, W)
        """
        # Compute loss gradient analytically
        # For Lovász hinge loss: gradient ≈ sign(prediction - target)
        
        # Aggregate pseudo labels (average over channels)
        target = pseudo_labels.mean(dim=1, keepdim=True)
        
        # Compute error
        error = prediction - target
        
        # Apply weights
        weighted_error = error * weights
        
        # Add regularization
        gradient = weighted_error + self.regularizer * prediction
        
        return gradient
    
    def lovasz_hinge_gradient(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of Lovász hinge loss.
        
        Lovász hinge loss is a convex surrogate for IoU loss.
        
        Args:
            prediction: Predictions (B, 1, H, W)
            target: Targets (B, 1, H, W)
        
        Returns:
            Gradient (B, 1, H, W)
        """
        B, _, H, W = prediction.shape
        
        # Flatten spatial dimensions
        pred_flat = prediction.view(B, -1)
        target_flat = target.view(B, -1)
        
        # Compute errors
        errors = torch.abs(pred_flat - target_flat)
        
        # Sort errors
        errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
        target_sorted = target_flat.gather(1, perm)
        
        # Compute Lovász extension gradient
        grad_flat = torch.zeros_like(pred_flat)
        
        for b in range(B):
            inter = target_sorted[b].sum() - target_sorted[b].float().cumsum(0)
            union = target_sorted[b].sum() + (1 - target_sorted[b]).float().cumsum(0)
            jaccard = 1.0 - inter / union
            
            if len(jaccard) > 1:
                jaccard[1:] = jaccard[1:] - jaccard[:-1]
            
            # Scatter gradient back
            grad_flat[b] = jaccard.gather(0, perm[b].argsort())
        
        # Reshape gradient
        gradient = grad_flat.view(B, 1, H, W)
        
        # Sign based on error direction
        gradient = gradient * torch.sign(prediction - target)
        
        return gradient


class SteepestDescentOptimizer(nn.Module):
    """
    Standalone steepest descent optimizer module.
    Can be used independently for ablation studies.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def step(
        self,
        prediction: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of steepest descent with momentum.
        
        Args:
            prediction: Current prediction (B, 1, H, W)
            gradient: Gradient (B, 1, H, W)
        
        Returns:
            Updated prediction (B, 1, H, W)
        """
        if self.velocity is None:
            self.velocity = torch.zeros_like(gradient)
        
        # Update velocity
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        
        # Update prediction
        prediction_new = prediction + self.velocity
        
        return prediction_new
    
    def reset(self):
        """Reset velocity for new sequence."""
        self.velocity = None