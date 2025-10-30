"""
Discriminative Correlation Filter (DCF) for geometric localization.
Based on correlation filter tracking literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DiscriminativeCorrelationFilter(nn.Module):
    """
    Discriminative Correlation Filter (DCF).
    
    Learns a correlation filter that maximizes response at target location
    while minimizing response at background.
    
    Uses hinge loss instead of traditional MSE for better discrimination.
    
    Args:
        feature_dim: Input feature dimension
        lambda_reg: Regularization parameter
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for filter optimization
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        lambda_reg: float = 0.01,
        num_iterations: int = 5,
        learning_rate: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        
        # Learnable filter initialization
        self.filter_init = nn.Parameter(
            torch.randn(1, feature_dim, 1, 1) * 0.01
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
        )
    
    def forward(
        self,
        search_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply DCF to compute response map.
        
        Args:
            search_features: Search region features (B, D, H, W)
            target_features: Target template features (B, D, H', W')
            target_mask: Target mask (B, 1, H', W')
        
        Returns:
            Response map (B, 1, H, W)
        """
        B, D, H, W = search_features.shape
        
        # Project features
        search_proj = self.feature_proj(search_features)
        target_proj = self.feature_proj(target_features)
        
        # Train correlation filter on target
        correlation_filter = self._train_filter(target_proj, target_mask)
        
        # Apply filter to search region
        response = self._apply_filter(search_proj, correlation_filter)
        
        return response
    
    def _train_filter(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Train correlation filter on target features.
        
        Uses iterative optimization with hinge loss.
        
        Args:
            features: Target features (B, D, H, W)
            mask: Target mask (B, 1, H, W)
        
        Returns:
            Learned filter (B, D, 1, 1)
        """
        B, D, H, W = features.shape
        
        # Initialize filter
        filter_w = self.filter_init.expand(B, -1, -1, -1).clone()
        
        # Generate Gaussian label
        label = self._generate_gaussian_label(mask)
        
        # Iterative optimization
        for _ in range(self.num_iterations):
            # Compute response
            response = self._apply_filter(features, filter_w)
            
            # Compute gradient of hinge loss
            gradient = self._compute_hinge_gradient(response, label, mask)
            
            # Update filter
            filter_w = filter_w - self.learning_rate * (
                gradient + self.lambda_reg * filter_w
            )
        
        return filter_w
    
    def _apply_filter(
        self,
        features: torch.Tensor,
        correlation_filter: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply correlation filter to features.
        
        Args:
            features: Input features (B, D, H, W)
            correlation_filter: Filter weights (B, D, 1, 1)
        
        Returns:
            Response map (B, 1, H, W)
        """
        # Compute correlation via element-wise product and sum
        response = (features * correlation_filter).sum(dim=1, keepdim=True)
        
        return response
    
    def _generate_gaussian_label(
        self,
        mask: torch.Tensor,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        """
        Generate Gaussian label centered at mask centroid.
        
        Args:
            mask: Binary mask (B, 1, H, W)
            sigma: Gaussian standard deviation
        
        Returns:
            Gaussian label (B, 1, H, W)
        """
        B, _, H, W = mask.shape
        
        # Find mask centroid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=mask.device),
            torch.arange(W, device=mask.device),
            indexing='ij'
        )
        
        y_coords = y_coords.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        x_coords = x_coords.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # Compute centroid
        mask_sum = mask.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
        cy = (mask * y_coords).sum(dim=[2, 3], keepdim=True) / mask_sum
        cx = (mask * x_coords).sum(dim=[2, 3], keepdim=True) / mask_sum
        
        # Generate Gaussian
        gaussian = torch.exp(
            -((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma ** 2)
        )
        
        return gaussian
    
    def _compute_hinge_gradient(
        self,
        response: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of hinge loss.
        
        Hinge loss: max(0, margin - response * label)
        
        Args:
            response: Filter response (B, 1, H, W)
            label: Target label (B, 1, H, W)
            mask: Valid region mask (B, 1, H, W)
        
        Returns:
            Gradient w.r.t. filter (B, D, 1, 1)
        """
        # Compute hinge loss margin
        margin = 1.0
        loss_per_pixel = torch.clamp(margin - response * label, min=0.0)
        
        # Gradient: -label where loss > 0, 0 elsewhere
        gradient = torch.where(
            loss_per_pixel > 0,
            -label * mask,
            torch.zeros_like(label)
        )
        
        # Average gradient
        gradient = gradient.mean(dim=[2, 3], keepdim=True)
        
        return gradient


class AdaptiveDCF(nn.Module):
    """
    Adaptive DCF that updates filter online.
    Maintains running average of filters for temporal consistency.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        lambda_reg: float = 0.01,
        num_iterations: int = 5,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
    ):
        super().__init__()
        
        self.dcf = DiscriminativeCorrelationFilter(
            feature_dim=feature_dim,
            lambda_reg=lambda_reg,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
        )
        
        self.momentum = momentum
        
        # Running filter
        self.register_buffer(
            'running_filter',
            torch.zeros(1, feature_dim, 1, 1)
        )
        self.register_buffer('initialized', torch.tensor(False))
    
    def forward(
        self,
        search_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Forward with online adaptation.
        
        Args:
            search_features: Search features (B, D, H, W)
            target_features: Target features (B, D, H', W')
            target_mask: Target mask (B, 1, H', W')
            update: Whether to update running filter
        
        Returns:
            Response map (B, 1, H, W)
        """
        # Compute current filter
        current_response = self.dcf(search_features, target_features, target_mask)
        
        if update:
            # Train new filter
            new_filter = self.dcf._train_filter(target_features, target_mask)
            
            # Update running filter with momentum
            if not self.initialized:
                self.running_filter = new_filter.mean(dim=0, keepdim=True)
                self.initialized = torch.tensor(True)
            else:
                self.running_filter = (
                    self.momentum * self.running_filter +
                    (1 - self.momentum) * new_filter.mean(dim=0, keepdim=True)
                )
        
        # Apply running filter
        if self.initialized:
            response = self.dcf._apply_filter(
                search_features,
                self.running_filter.expand(search_features.shape[0], -1, -1, -1)
            )
        else:
            response = current_response
        
        return response
    
    def reset(self):
        """Reset running filter."""
        self.running_filter.zero_()
        self.initialized = torch.tensor(False)