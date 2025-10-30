"""
Prediction Head for final output generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PredictionHead(nn.Module):
    """
    Prediction Head for generating final segmentation masks.
    
    Converts fused features to binary segmentation masks.
    
    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes (1 for binary)
        use_crf: Use CRF for post-processing
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 1,
        use_crf: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_crf = use_crf
        
        # Main prediction network
        self.pred_net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim // 4, num_classes, 1),
        )
        
        # Optional CRF
        if use_crf:
            self.crf = DenseCRF(
                num_classes=num_classes,
            )
    
    def forward(
        self,
        features: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate predictions.
        
        Args:
            features: Input features (B, D, H, W) or (B, T, D, H, W)
            image: Optional RGB image for CRF (B, 3, H, W) or (B, T, 3, H, W)
        
        Returns:
            Predictions (B, C, H, W) or (B, T, C, H, W)
        """
        # Handle temporal dimension
        squeeze_output = False
        if features.dim() == 5:
            B, T, D, H, W = features.shape
            features = features.reshape(B * T, D, H, W)
            if image is not None:
                image = image.reshape(B * T, 3, H, W)
        else:
            squeeze_output = True
            T = 1
        
        # Generate predictions
        logits = self.pred_net(features)
        predictions = torch.sigmoid(logits)
        
        # Apply CRF if enabled
        if self.use_crf and image is not None:
            predictions = self.crf(predictions, image)
        
        # Reshape back
        if not squeeze_output:
            predictions = predictions.reshape(B, T, self.num_classes, H, W)
        
        return predictions


class DenseCRF(nn.Module):
    """
    Dense CRF for post-processing segmentation.
    Refines predictions using image appearance.
    
    Note: This is a simplified differentiable approximation.
    For full CRF, use pydensecrf library.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        num_iterations: int = 5,
        spatial_weight: float = 3.0,
        bilateral_weight: float = 5.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.spatial_weight = spatial_weight
        self.bilateral_weight = bilateral_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply CRF refinement.
        
        Args:
            predictions: Initial predictions (B, C, H, W)
            image: RGB image (B, 3, H, W)
        
        Returns:
            Refined predictions (B, C, H, W)
        """
        B, C, H, W = predictions.shape
        
        # Compute pairwise potentials
        spatial_kernel = self._spatial_kernel(H, W, predictions.device)
        bilateral_kernel = self._bilateral_kernel(image)
        
        # Iterative mean-field inference
        Q = predictions.clone()
        
        for _ in range(self.num_iterations):
            # Message passing
            spatial_message = self._apply_kernel(Q, spatial_kernel, self.spatial_weight)
            bilateral_message = self._apply_kernel(Q, bilateral_kernel, self.bilateral_weight)
            
            # Update
            Q = predictions - spatial_message - bilateral_message
            Q = torch.sigmoid(Q)
        
        return Q
    
    def _spatial_kernel(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute spatial Gaussian kernel."""
        # Simple spatial kernel (can be optimized)
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten
        positions = torch.stack([y.flatten(), x.flatten()], dim=0)  # (2, H*W)
        
        # Pairwise distances
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (2, H*W, H*W)
        distances = (diff ** 2).sum(dim=0)  # (H*W, H*W)
        
        # Gaussian kernel
        kernel = torch.exp(-distances / (2 * self.spatial_weight ** 2))
        
        return kernel
    
    def _bilateral_kernel(self, image: torch.Tensor) -> torch.Tensor:
        """Compute bilateral kernel based on image appearance."""
        B, C, H, W = image.shape
        
        # Flatten image
        image_flat = image.view(B, C, -1)  # (B, 3, H*W)
        
        # Pairwise color differences
        color_diff = image_flat.unsqueeze(3) - image_flat.unsqueeze(2)  # (B, 3, H*W, H*W)
        color_distances = (color_diff ** 2).sum(dim=1)  # (B, H*W, H*W)
        
        # Bilateral kernel
        kernel = torch.exp(-color_distances / (2 * self.bilateral_weight ** 2))
        
        return kernel
    
    def _apply_kernel(
        self,
        Q: torch.Tensor,
        kernel: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        """Apply kernel to predictions."""
        B, C, H, W = Q.shape
        
        # Flatten predictions
        Q_flat = Q.view(B, C, -1)  # (B, C, H*W)
        
        # Apply kernel
        if kernel.dim() == 2:
            # Spatial kernel (same for all batches)
            kernel = kernel.unsqueeze(0).expand(B, -1, -1)
        
        # Message passing
        message = torch.bmm(Q_flat, kernel)  # (B, C, H*W)
        
        # Reshape back
        message = message.view(B, C, H, W)
        
        return weight * message