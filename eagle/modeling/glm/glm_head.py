"""
GLM Head: Complete Geometry-Aware Localization Memory module.
Integrates DCF and memory bank for robust tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .dcf import DiscriminativeCorrelationFilter, AdaptiveDCF
from .memory_bank import GLMMemoryBank, SpatialMemoryAggregator


class GLMHead(nn.Module):
    """
    Complete GLM (Geometry-Aware Localization Memory) Head.
    
    Pipeline:
    1. Extract query spatial features
    2. Initialize DCF filter from query
    3. For each search frame:
        a. Predict motion from memory
        b. Apply DCF to get response map
        c. Refine with memory-guided features
        d. Update memory bank
    
    Args:
        feature_dim: Feature dimension from backbone
        memory_size: Size of memory bank
        dcf_iters: Number of DCF iterations
        lambda_reg: DCF regularization
        use_static: Use static query memory
        scale_factor: Scale variation tolerance
        update_first_n: Update memory only for first N frames
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        memory_size: int = 50,
        dcf_iters: int = 5,
        lambda_reg: float = 0.01,
        use_static: bool = False,
        scale_factor: float = 1.5,
        update_first_n: int = 100,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.scale_factor = scale_factor
        self.update_first_n = update_first_n
        
        # Components
        self.dcf = AdaptiveDCF(
            feature_dim=feature_dim,
            lambda_reg=lambda_reg,
            num_iterations=dcf_iters,
        )
        
        self.memory_bank = GLMMemoryBank(
            feature_dim=feature_dim,
            memory_size=memory_size,
            use_static=use_static,
        )
        
        self.memory_aggregator = SpatialMemoryAggregator(
            feature_dim=feature_dim,
        )
        
        # Feature refinement network
        self.refinement_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2 + 1, 256, 3, padding=1),
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
        
        # Gaussian label generator
        self.gaussian_sigma = 2.0
    
    def forward(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
        search_features: torch.Tensor,
        update_memory: bool = True,
        frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query_features: Query frame features (B, D, H, W)
            query_mask: Query mask (B, 1, H, W)
            search_features: Search frame features (B, T, D, H, W) or (B, D, H, W)
            update_memory: Whether to update memory
            frame_indices: Frame indices for temporal tracking
        
        Returns:
            Dictionary with predictions and tracking info
        """
        B = query_features.shape[0]
        
        # Handle single frame
        if search_features.dim() == 4:
            search_features = search_features.unsqueeze(1)
            if frame_indices is None:
                frame_indices = [0]
        
        B, T, D, H, W = search_features.shape
        
        if frame_indices is None:
            frame_indices = list(range(T))
        
        # Initialize from query
        self._initialize_from_query(query_features, query_mask)
        
        # Process each frame
        all_predictions = []
        all_response_maps = []
        all_locations = []
        all_scales = []
        
        for t in range(T):
            search_feat_t = search_features[:, t]  # (B, D, H, W)
            frame_idx = frame_indices[t]
            
            # Predict motion from memory
            predicted_location, predicted_scale = self.memory_bank.predict_motion(
                frame_idx, window=5
            )
            
            # Get DCF response
            response_map = self.dcf(
                search_feat_t,
                query_features,
                query_mask,
                update=(update_memory and frame_idx < self.update_first_n)
            )
            
            # Get memory-guided features
            memory_data = self.memory_bank.get_recent(k=10)
            if len(memory_data['features']) > 0:
                memory_feature = self.memory_aggregator(
                    memory_data,
                    query_feature=self._extract_query_feature(query_features, query_mask)
                )
                
                # Broadcast memory feature to spatial dimensions
                memory_feature_spatial = memory_feature.view(1, -1, 1, 1).expand(
                    B, -1, H, W
                )
            else:
                memory_feature_spatial = torch.zeros(B, D, H, W, device=search_feat_t.device)
            
            # Refine prediction
            refinement_input = torch.cat([
                search_feat_t,
                memory_feature_spatial,
                response_map,
            ], dim=1)
            
            refinement = self.refinement_net(refinement_input)
            final_prediction = torch.sigmoid(response_map + refinement)
            
            # Extract location and scale from prediction
            location, scale, confidence = self._extract_bbox_from_prediction(
                final_prediction
            )
            
            # Update memory
            if update_memory and frame_idx < self.update_first_n:
                query_feature = self._extract_query_feature(search_feat_t, final_prediction)
                self.memory_bank.update(
                    features=query_feature,
                    location=location[0],  # Batch dim
                    scale=scale[0],
                    timestamp=frame_idx,
                    confidence=confidence[0].item(),
                )
            
            all_predictions.append(final_prediction)
            all_response_maps.append(response_map)
            all_locations.append(location)
            all_scales.append(scale)
        
        # Stack results
        predictions = torch.stack(all_predictions, dim=1)  # (B, T, 1, H, W)
        response_maps = torch.stack(all_response_maps, dim=1)
        locations = torch.stack(all_locations, dim=1)  # (B, T, 2)
        scales = torch.stack(all_scales, dim=1)  # (B, T, 2)
        
        return {
            'predictions': predictions,
            'response_maps': response_maps,
            'locations': locations,
            'scales': scales,
        }
    
    def _initialize_from_query(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
    ):
        """
        Initialize tracking from query.
        
        Args:
            query_features: Query features (B, D, H, W)
            query_mask: Query mask (B, 1, H, W)
        """
        # Extract query feature vector
        query_feature = self._extract_query_feature(query_features, query_mask)
        
        # Extract query location
        query_location, query_scale, _ = self._extract_bbox_from_prediction(query_mask)
        
        # Initialize static memory if used
        if self.memory_bank.use_static:
            self.memory_bank.init_static_query(
                query_feature[0],  # Batch dim
                query_location[0],
            )
        
        # Initialize memory with query
        self.memory_bank.update(
            features=query_feature[0],
            location=query_location[0],
            scale=query_scale[0],
            timestamp=0,
            confidence=1.0,
        )
    
    def _extract_query_feature(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract feature vector from masked region.
        
        Args:
            features: Features (B, D, H, W)
            mask: Mask (B, 1, H, W)
        
        Returns:
            Feature vector (B, D)
        """
        # Average pooling within mask
        masked_features = features * mask
        mask_sum = mask.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
        feature_vector = masked_features.sum(dim=[2, 3]) / mask_sum.squeeze()
        
        return feature_vector
    
    def _extract_bbox_from_prediction(
        self,
        prediction: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract bounding box from prediction map.
        
        Args:
            prediction: Prediction map (B, 1, H, W)
            threshold: Threshold for binarization
        
        Returns:
            - Center location (B, 2) [x, y]
            - Scale (B, 2) [width, height]
            - Confidence (B,)
        """
        B, _, H, W = prediction.shape
        
        # Threshold prediction
        binary_mask = (prediction > threshold).float()
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=prediction.device, dtype=torch.float32),
            torch.arange(W, device=prediction.device, dtype=torch.float32),
            indexing='ij'
        )
        
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # Compute center
        mask_sum = binary_mask.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
        center_y = (binary_mask * y_coords).sum(dim=[2, 3]) / mask_sum.squeeze()
        center_x = (binary_mask * x_coords).sum(dim=[2, 3]) / mask_sum.squeeze()
        
        location = torch.stack([center_x, center_y], dim=1)  # (B, 2)
        
        # Compute scale (bounding box size)
        y_min = (binary_mask * y_coords + (1 - binary_mask) * H).min(dim=2)[0].min(dim=1)[0]
        y_max = (binary_mask * y_coords).max(dim=2)[0].max(dim=1)[0]
        x_min = (binary_mask * x_coords + (1 - binary_mask) * W).min(dim=2)[0].min(dim=1)[0]
        x_max = (binary_mask * x_coords).max(dim=2)[0].max(dim=1)[0]
        
        width = (x_max - x_min).clamp(min=1.0)
        height = (y_max - y_min).clamp(min=1.0)
        
        scale = torch.stack([width, height], dim=1)  # (B, 2)
        
        # Compute confidence (average prediction in mask region)
        confidence = (prediction * binary_mask).sum(dim=[2, 3]) / mask_sum.squeeze()
        confidence = confidence.squeeze(1)  # (B,)
        
        return location, scale, confidence
    
    def reset(self):
        """Reset memory and filter."""
        self.memory_bank.reset()
        self.dcf.reset()