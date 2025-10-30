"""
Pseudo-Label Modulator P_θ.
Generates pseudo labels from query information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PseudoLabelModulator(nn.Module):
    """
    Pseudo-Label Modulator P_θ.
    
    Converts query mask M^q into multi-channel pseudo label P(M^q).
    Uses learnable convolutions to generate discriminative pseudo labels.
    
    Args:
        input_dim: Input feature dimension
        output_channels: Number of pseudo label channels (default: 32)
        kernel_size: Convolution kernel size
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        
        # Feature extraction from query
        self.query_encoder = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Mask processing branch
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Fusion and pseudo label generation
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, 1),
        )
        
        # Optional: Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(output_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate pseudo labels.
        
        Args:
            query_features: Query features (B, D, H, W)
            query_mask: Query binary mask (B, 1, H, W)
        
        Returns:
            Pseudo labels P(M^q) (B, C, H, W)
        """
        # Encode query features
        query_feat = self.query_encoder(query_features)
        
        # Encode mask
        mask_feat = self.mask_encoder(query_mask)
        
        # Fuse features
        fused = torch.cat([query_feat, mask_feat], dim=1)
        
        # Generate pseudo labels
        pseudo_labels = self.fusion(fused)
        
        # Apply spatial attention (optional)
        attention = self.spatial_attention(pseudo_labels)
        pseudo_labels = pseudo_labels * attention
        
        return pseudo_labels
    
    def extract_target_features(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract compact target representation.
        Averages features within mask region.
        
        Args:
            query_features: Query features (B, D, H, W)
            query_mask: Query mask (B, 1, H, W)
        
        Returns:
            Target features (B, D)
        """
        # Mask features
        masked_features = query_features * query_mask
        
        # Global average pooling within mask
        mask_sum = query_mask.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
        target_features = masked_features.sum(dim=[2, 3]) / mask_sum.squeeze()
        
        return target_features