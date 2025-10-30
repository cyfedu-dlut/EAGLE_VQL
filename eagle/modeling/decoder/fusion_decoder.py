"""
Fusion Decoder for combining AMM and GLM predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FusionDecoder(nn.Module):
    """
    Fusion Decoder.
    
    Combines predictions from AMM and GLM branches using:
    1. Learned attention weights
    2. Feature-level fusion
    3. Multi-scale refinement
    
    Args:
        feature_dim: Feature dimension
        fusion_type: Type of fusion ('learned', 'add', 'max')
        num_refinement_stages: Number of refinement stages
        use_multi_scale: Use multi-scale features
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        fusion_type: str = 'learned',
        num_refinement_stages: int = 2,
        use_multi_scale: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.num_refinement_stages = num_refinement_stages
        self.use_multi_scale = use_multi_scale
        
        # Learned fusion weights
        if fusion_type == 'learned':
            self.fusion_weight_net = nn.Sequential(
                nn.Conv2d(2, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1),
                nn.Softmax(dim=1),  # Normalize weights
            )
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Refinement stages
        self.refinement_stages = nn.ModuleList([
            self._build_refinement_stage(feature_dim)
            for _ in range(num_refinement_stages)
        ])
        
        # Multi-scale processing
        if use_multi_scale:
            self.multiscale_fusion = MultiScaleFusion(feature_dim)
    
    def _build_refinement_stage(self, feature_dim: int) -> nn.Module:
        """Build one refinement stage."""
        return nn.Sequential(
            nn.Conv2d(feature_dim + 1, 256, 3, padding=1),
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
        amm_prediction: torch.Tensor,
        glm_prediction: torch.Tensor,
        amm_features: Optional[torch.Tensor] = None,
        glm_features: Optional[torch.Tensor] = None,
        search_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse predictions from AMM and GLM.
        
        Args:
            amm_prediction: AMM prediction (B, T, 1, H, W) or (B, 1, H, W)
            glm_prediction: GLM prediction (B, T, 1, H, W) or (B, 1, H, W)
            amm_features: AMM features (B, T, D, H, W), optional
            glm_features: GLM features (B, T, D, H, W), optional
            search_features: Original search features (B, T, D, H, W), optional
        
        Returns:
            Dictionary with fused predictions
        """
        # Handle single frame case
        squeeze_output = False
        if amm_prediction.dim() == 4:
            amm_prediction = amm_prediction.unsqueeze(1)
            glm_prediction = glm_prediction.unsqueeze(1)
            squeeze_output = True
            
            if amm_features is not None:
                amm_features = amm_features.unsqueeze(1)
            if glm_features is not None:
                glm_features = glm_features.unsqueeze(1)
            if search_features is not None:
                search_features = search_features.unsqueeze(1)
        
        B, T, _, H, W = amm_prediction.shape
        
        # Fuse predictions frame by frame
        fused_predictions = []
        fusion_weights_list = []
        
        for t in range(T):
            amm_pred_t = amm_prediction[:, t]  # (B, 1, H, W)
            glm_pred_t = glm_prediction[:, t]
            
            # Compute fusion weights
            if self.fusion_type == 'learned':
                pred_concat = torch.cat([amm_pred_t, glm_pred_t], dim=1)  # (B, 2, H, W)
                fusion_weights = self.fusion_weight_net(pred_concat)  # (B, 2, H, W)
                
                # Weighted combination
                fused = (
                    fusion_weights[:, 0:1] * amm_pred_t +
                    fusion_weights[:, 1:2] * glm_pred_t
                )
                
                fusion_weights_list.append(fusion_weights)
            
            elif self.fusion_type == 'add':
                fused = (amm_pred_t + glm_pred_t) / 2.0
            
            elif self.fusion_type == 'max':
                fused = torch.max(amm_pred_t, glm_pred_t)
            
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
            # Feature-level fusion if available
            if amm_features is not None and glm_features is not None:
                amm_feat_t = amm_features[:, t]  # (B, D, H, W)
                glm_feat_t = glm_features[:, t]
                
                # Fuse features
                feat_concat = torch.cat([amm_feat_t, glm_feat_t], dim=1)
                fused_features = self.feature_fusion(feat_concat)
            elif search_features is not None:
                fused_features = search_features[:, t]
            else:
                fused_features = None
            
            # Refinement stages
            refined = fused
            for stage in self.refinement_stages:
                if fused_features is not None:
                    stage_input = torch.cat([fused_features, refined], dim=1)
                else:
                    # Repeat prediction to match feature dim
                    repeated_pred = refined.expand(-1, self.feature_dim, -1, -1)
                    stage_input = torch.cat([repeated_pred, refined], dim=1)
                
                residual = stage(stage_input)
                refined = refined + residual
                refined = torch.clamp(refined, 0.0, 1.0)
            
            fused_predictions.append(refined)
        
        # Stack results
        fused_predictions = torch.stack(fused_predictions, dim=1)  # (B, T, 1, H, W)
        
        if squeeze_output:
            fused_predictions = fused_predictions.squeeze(1)
        
        result = {
            'predictions': fused_predictions,
        }
        
        if len(fusion_weights_list) > 0:
            fusion_weights = torch.stack(fusion_weights_list, dim=1)
            result['fusion_weights'] = fusion_weights
        
        return result


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    Combines features at different scales for better segmentation.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        scales: list = [1, 2, 4],
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.scales = scales
        
        # Per-scale processing
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // len(scales), 3, padding=1),
                nn.BatchNorm2d(feature_dim // len(scales)),
                nn.ReLU(inplace=True),
            )
            for _ in scales
        ])
        
        # Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale processing.
        
        Args:
            features: Input features (B, D, H, W)
        
        Returns:
            Fused features (B, D, H, W)
        """
        B, D, H, W = features.shape
        
        multiscale_features = []
        
        for scale, conv in zip(self.scales, self.scale_convs):
            if scale == 1:
                scaled_feat = features
            else:
                # Downsample
                scaled_feat = F.avg_pool2d(features, kernel_size=scale, stride=scale)
            
            # Process
            processed = conv(scaled_feat)
            
            # Upsample back to original size
            if scale != 1:
                processed = F.interpolate(
                    processed,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            
            multiscale_features.append(processed)
        
        # Concatenate and fuse
        concat_features = torch.cat(multiscale_features, dim=1)
        fused = self.fusion_conv(concat_features)
        
        return fused


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to weight AMM vs GLM based on input.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Global context for fusion decision
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fusion weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )
    
    def forward(
        self,
        amm_prediction: torch.Tensor,
        glm_prediction: torch.Tensor,
        amm_features: torch.Tensor,
        glm_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive fusion weights.
        
        Args:
            amm_prediction: AMM prediction (B, 1, H, W)
            glm_prediction: GLM prediction (B, 1, H, W)
            amm_features: AMM features (B, D, H, W)
            glm_features: GLM features (B, D, H, W)
        
        Returns:
            - Fused prediction (B, 1, H, W)
            - Fusion weights (B, 2)
        """
        B = amm_prediction.shape[0]
        
        # Extract global features
        amm_global = self.global_pool(amm_features).view(B, -1)
        glm_global = self.global_pool(glm_features).view(B, -1)
        
        # Get prediction statistics
        amm_conf = amm_prediction.mean(dim=[2, 3])  # (B, 1)
        glm_conf = glm_prediction.mean(dim=[2, 3])  # (B, 1)
        
        # Concatenate all information
        fusion_input = torch.cat([
            amm_global,
            glm_global,
            amm_conf,
            glm_conf,
        ], dim=1)
        
        # Predict fusion weights
        fusion_weights = self.weight_predictor(fusion_input)  # (B, 2)
        
        # Apply weights
        fused_prediction = (
            fusion_weights[:, 0:1, None, None] * amm_prediction +
            fusion_weights[:, 1:2, None, None] * glm_prediction
        )
        
        return fused_prediction, fusion_weights