"""
Target Reweighting Network W_θ.
Generates instance-specific loss weights based on meta-learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetReweightingNetwork(nn.Module):
    """
    Target Reweighting Network W_θ.
    
    Generates spatial weights for loss computation based on:
    1. Query-target similarity
    2. Prediction confidence
    3. Spatial context
    
    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Similarity computation branch
        self.similarity_branch = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        
        # Confidence branch
        self.confidence_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        
        # Weight fusion
        self.weight_fusion = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),  # Ensure weights in [0, 1]
        )
    
    def forward(
        self,
        target_features: torch.Tensor,
        search_features: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reweighting map.
        
        Args:
            target_features: Target features from query (B, D, H, W)
            search_features: Search frame features (B, D, H, W)
            prediction: Current prediction (B, 1, H, W)
        
        Returns:
            Weight map (B, 1, H, W)
        """
        # Project features
        target_proj = self.feature_proj(target_features)
        search_proj = self.feature_proj(search_features)
        
        # Compute similarity
        concat_features = torch.cat([target_proj, search_proj], dim=1)
        similarity_map = self.similarity_branch(concat_features)
        
        # Compute confidence weight
        confidence_map = self.confidence_branch(prediction)
        
        # Fuse to get final weights
        weight_input = torch.cat([similarity_map, confidence_map], dim=1)
        weights = self.weight_fusion(weight_input)
        
        return weights
    
    def compute_similarity(
        self,
        query_feat: torch.Tensor,
        search_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and search features.
        
        Args:
            query_feat: Query features (B, D)
            search_feat: Search features (B, D, H, W)
        
        Returns:
            Similarity map (B, 1, H, W)
        """
        # Normalize features
        query_norm = F.normalize(query_feat, p=2, dim=1)  # (B, D)
        search_norm = F.normalize(search_feat, p=2, dim=1)  # (B, D, H, W)
        
        # Compute cosine similarity
        query_norm = query_norm.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        similarity = (query_norm * search_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        return similarity