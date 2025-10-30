"""
AMM Head: Complete Appearance-Aware Meta-Learning Memory module.
Integrates all AMM components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .pseudo_label_modulator import PseudoLabelModulator
from .target_reweighting import TargetReweightingNetwork
from .meta_learner import MetaLearner


class AMMHead(nn.Module):
    """
    Complete AMM (Appearance-Aware Meta-Learning Memory) Head.
    
    Pipeline:
    1. Extract query target features
    2. Generate pseudo labels P_θ(M^q)
    3. Compute initial prediction via correlation
    4. Iteratively refine with meta-learner G_θ
    5. Apply target reweighting W_θ
    
    Args:
        feature_dim: Feature dimension from backbone
        memory_size: Size of memory bank
        pseudo_label_channels: Number of pseudo label channels
        meta_learner_iters: Number of meta-learning iterations
        learning_rate: Learning rate for meta-learner
        regularizer: Regularization weight
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        memory_size: int = 50,
        pseudo_label_channels: int = 32,
        meta_learner_iters: int = 3,
        learning_rate: float = 0.1,
        regularizer: float = 0.01,
        confidence_threshold: float = 0.6,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.confidence_threshold = confidence_threshold
        
        # Components
        self.pseudo_label_modulator = PseudoLabelModulator(
            input_dim=feature_dim,
            output_channels=pseudo_label_channels,
        )
        
        self.target_reweighting = TargetReweightingNetwork(
            feature_dim=feature_dim,
        )
        
        self.meta_learner = MetaLearner(
            feature_dim=feature_dim,
            num_iterations=meta_learner_iters,
            learning_rate=learning_rate,
            regularizer=regularizer,
        )
        
        # Memory bank for target features
        self.register_buffer(
            'memory_features',
            torch.zeros(1, memory_size, feature_dim)
        )
        self.register_buffer(
            'memory_masks',
            torch.zeros(1, memory_size, 1, 1, 1)
        )
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Feature projection for correlation
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
        search_features: torch.Tensor,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query_features: Query frame features (B, D, H, W)
            query_mask: Query mask (B, 1, H, W)
            search_features: Search frame features (B, T, D, H, W) or (B, D, H, W)
            update_memory: Whether to update memory bank
        
        Returns:
            Dictionary with:
            - predictions: Final predictions (B, T, 1, H, W)
            - intermediate_predictions: List of intermediate predictions
            - pseudo_labels: Generated pseudo labels
            - weights: Reweighting maps
        """
        B = query_features.shape[0]
        
        # Handle single frame or multi-frame input
        if search_features.dim() == 4:
            search_features = search_features.unsqueeze(1)  # Add temporal dim
        
        B, T, D, H, W = search_features.shape
        
        # Step 1: Generate pseudo labels
        pseudo_labels = self.pseudo_label_modulator(query_features, query_mask)
        
        # Step 2: Extract target features
        target_features = self.pseudo_label_modulator.extract_target_features(
            query_features, query_mask
        )
        
        # Step 3: Update memory (optional)
        if update_memory:
            self._update_memory(target_features, query_mask)
        
        # Step 4: Process each search frame
        all_predictions = []
        all_weights = []
        all_intermediate = []
        
        for t in range(T):
            search_feat_t = search_features[:, t]  # (B, D, H, W)
            
            # Compute initial prediction via correlation
            initial_pred = self._compute_correlation_prediction(
                target_features, search_feat_t
            )
            
            # Compute reweighting
            weights = self.target_reweighting(
                query_features, search_feat_t, initial_pred
            )
            
            # Meta-learning refinement
            final_pred, intermediate_preds = self.meta_learner(
                search_feat_t,
                initial_pred,
                pseudo_labels,
                weights,
            )
            
            all_predictions.append(final_pred)
            all_weights.append(weights)
            all_intermediate.append(intermediate_preds)
        
        # Stack results
        predictions = torch.stack(all_predictions, dim=1)  # (B, T, 1, H, W)
        weights = torch.stack(all_weights, dim=1)
        
        return {
            'predictions': predictions,
            'intermediate_predictions': all_intermediate,
            'pseudo_labels': pseudo_labels,
            'weights': weights,
            'target_features': target_features,
        }
    
    def _compute_correlation_prediction(
        self,
        target_features: torch.Tensor,
        search_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute initial prediction via correlation.
        
        Args:
            target_features: Target features (B, D)
            search_features: Search features (B, D, H, W)
        
        Returns:
            Initial prediction (B, 1, H, W)
        """
        # Project features
        search_proj = self.feature_proj(search_features)  # (B, 256, H, W)
        
        # Normalize
        target_norm = F.normalize(target_features, p=2, dim=1)
        search_norm = F.normalize(search_proj, p=2, dim=1)
        
        # Compute correlation
        target_norm = target_norm.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        correlation = (target_norm * search_norm).sum(dim=1, keepdim=True)
        
        # Apply sigmoid to get initial prediction
        prediction = torch.sigmoid(correlation * 10.0)  # Scale factor for sharpness
        
        return prediction
    
    def _update_memory(
        self,
        target_features: torch.Tensor,
        query_mask: torch.Tensor,
    ):
        """
        Update memory bank with new target features.
        
        Args:
            target_features: Target features (B, D)
            query_mask: Query mask (B, 1, H, W)
        """
        B, D = target_features.shape
        
        # Expand memory if needed
        if self.memory_features.shape[0] != B:
            self.memory_features = self.memory_features.expand(B, -1, -1).clone()
            self.memory_masks = self.memory_masks.expand(B, -1, -1, -1, -1).clone()
            self.memory_ptr = self.memory_ptr.expand(B).clone()
        
        # Update each batch item
        for b in range(B):
            ptr = self.memory_ptr[b].item()
            
            # Store features and mask
            self.memory_features[b, ptr] = target_features[b]
            self.memory_masks[b, ptr] = query_mask[b].unsqueeze(0)
            
            # Update pointer (circular buffer)
            self.memory_ptr[b] = (ptr + 1) % self.memory_size
    
    def get_memory_features(self) -> torch.Tensor:
        """Get all features in memory bank."""
        return self.memory_features
    
    def reset_memory(self):
        """Reset memory bank."""
        self.memory_features.zero_()
        self.memory_masks.zero_()
        self.memory_ptr.zero_()


class AMMMemoryBank(nn.Module):
    """
    Standalone memory bank for AMM.
    Stores historical target features for temporal consistency.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        memory_size: int = 50,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Memory buffers
        self.register_buffer(
            'features',
            torch.zeros(memory_size, feature_dim)
        )
        self.register_buffer(
            'timestamps',
            torch.zeros(memory_size, dtype=torch.long)
        )
        self.register_buffer('ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
    
    def update(
        self,
        features: torch.Tensor,
        timestamp: int,
    ):
        """
        Add features to memory.
        
        Args:
            features: Features to add (D,) or (B, D)
            timestamp: Frame timestamp
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        for feat in features:
            ptr = self.ptr.item()
            
            self.features[ptr] = feat
            self.timestamps[ptr] = timestamp
            
            self.ptr = (self.ptr + 1) % self.memory_size
            self.count = min(self.count + 1, self.memory_size)
    
    def get_recent(self, k: int) -> torch.Tensor:
        """
        Get k most recent features.
        
        Args:
            k: Number of features to retrieve
        
        Returns:
            Recent features (k, D)
        """
        k = min(k, self.count.item())
        
        if k == 0:
            return torch.zeros(0, self.feature_dim, device=self.features.device)
        
        indices = torch.arange(self.ptr - k, self.ptr) % self.memory_size
        return self.features[indices]
    
    def get_all(self) -> torch.Tensor:
        """Get all valid features in memory."""
        count = self.count.item()
        if count == 0:
            return torch.zeros(0, self.feature_dim, device=self.features.device)
        
        return self.features[:count]
    
    def reset(self):
        """Reset memory."""
        self.features.zero_()
        self.timestamps.zero_()
        self.ptr.zero_()
        self.count.zero_()