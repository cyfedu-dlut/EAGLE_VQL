"""
DINOv2 backbone for feature extraction.
Reference: https://github.com/facebookresearch/dinov2
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DINOv2Backbone(nn.Module):
    """
    DINOv2 Vision Transformer backbone.
    
    Features:
    - Self-supervised pretrained on diverse data
    - Strong semantic features without fine-tuning
    - Multiple output scales via intermediate layers
    
    Args:
        model_name: Name of DINOv2 model ('dinov2_vitb14', 'dinov2_vitl14', etc.)
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone weights
        feature_dim: Output feature dimension
        use_checkpoint: Use gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitb14',
        pretrained: bool = True,
        freeze: bool = True,
        feature_dim: int = 768,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.use_checkpoint = use_checkpoint
        
        # Load DINOv2 model
        self.backbone = self._load_dinov2_model(model_name, pretrained)
        
        # Freeze if specified
        if freeze:
            self._freeze_backbone()
        
        # Get patch size and embedding dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.embed_dim = self.backbone.embed_dim
        
        logger.info(f"Loaded DINOv2 backbone: {model_name}")
        logger.info(f"Patch size: {self.patch_size}, Embed dim: {self.embed_dim}")
        logger.info(f"Frozen: {freeze}")
    
    def _load_dinov2_model(self, model_name: str, pretrained: bool):
        """Load DINOv2 model from torch.hub."""
        try:
            if pretrained:
                model = torch.hub.load('facebookresearch/dinov2', model_name)
            else:
                # Load architecture without weights
                model = torch.hub.load(
                    'facebookresearch/dinov2', 
                    model_name,
                    pretrained=False
                )
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise
        
        return model
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.backbone.eval()
    
    def forward(
        self, 
        x: torch.Tensor,
        return_multilevel: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_multilevel: Whether to return intermediate features
        
        Returns:
            Dictionary with features:
            - 'features': Final features (B, D, H', W')
            - 'patch_tokens': Patch tokens (B, N, D)
            - 'cls_token': CLS token (B, D)
            - 'intermediate_X': Intermediate features if requested
        """
        B, C, H, W = x.shape
        
        # Forward through backbone
        if return_multilevel:
            output = self.backbone.get_intermediate_layers(
                x, 
                n=[3, 6, 9, 12],  # Get features from layers 3, 6, 9, 12
                return_class_token=True
            )
            
            features_list = []
            cls_tokens = []
            
            for feat, cls_token in output:
                features_list.append(feat)
                cls_tokens.append(cls_token)
            
            # Reshape final features to spatial grid
            final_features = features_list[-1]
            N = final_features.shape[1]
            H_feat = W_feat = int(N ** 0.5)
            
            final_features = final_features.reshape(B, H_feat, W_feat, -1)
            final_features = final_features.permute(0, 3, 1, 2)  # (B, D, H', W')
            
            result = {
                'features': final_features,
                'patch_tokens': features_list[-1],
                'cls_token': cls_tokens[-1],
            }
            
            # Add intermediate features
            for i, (feat, cls_token) in enumerate(zip(features_list[:-1], cls_tokens[:-1])):
                H_i = W_i = int(feat.shape[1] ** 0.5)
                feat = feat.reshape(B, H_i, W_i, -1).permute(0, 3, 1, 2)
                result[f'intermediate_{i}'] = feat
        
        else:
            # Standard forward (CLS token + patch tokens)
            output = self.backbone.forward_features(x)
            
            # Extract patch tokens (without CLS token)
            patch_tokens = output['x_norm_patchtokens']  # (B, N, D)
            cls_token = output['x_norm_clstoken']  # (B, D)
            
            # Reshape patch tokens to spatial grid
            N = patch_tokens.shape[1]
            H_feat = W_feat = int(N ** 0.5)
            
            features = patch_tokens.reshape(B, H_feat, W_feat, -1)
            features = features.permute(0, 3, 1, 2)  # (B, D, H', W')
            
            result = {
                'features': features,
                'patch_tokens': patch_tokens,
                'cls_token': cls_token,
            }
        
        return result
    
    def train(self, mode=True):
        """Override train mode to keep backbone frozen if specified."""
        super().train(mode)
        
        # Keep backbone in eval mode if frozen
        if hasattr(self, 'backbone'):
            frozen = not any(p.requires_grad for p in self.backbone.parameters())
            if frozen:
                self.backbone.eval()
        
        return self
    
    @property
    def output_shape(self):
        """Get output shape info."""
        return {
            'feature_dim': self.embed_dim,
            'patch_size': self.patch_size,
        }


class DINOv2FeatureExtractor(nn.Module):
    """
    Wrapper for extracting DINOv2 features with additional processing.
    Used for both query and search frame feature extraction.
    """
    
    def __init__(
        self,
        backbone: DINOv2Backbone,
        feature_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.feature_dim = backbone.embed_dim
        
        # Optional feature projection
        if feature_projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Conv2d(self.feature_dim, feature_projection_dim, 1),
                nn.BatchNorm2d(feature_projection_dim),
                nn.ReLU(inplace=True),
            )
            self.output_dim = feature_projection_dim
        else:
            self.projection = None
            self.output_dim = self.feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features.
        
        Args:
            x: Input images (B, C, H, W)
        
        Returns:
            Features (B, D, H', W')
        """
        # Extract features from backbone
        backbone_output = self.backbone(x)
        features = backbone_output['features']
        
        # Apply projection if exists
        if self.projection is not None:
            features = self.projection(features)
        
        return features