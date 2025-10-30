"""
CLIP backbone for feature extraction.
Reference: https://github.com/openai/CLIP
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CLIPBackbone(nn.Module):
    """
    CLIP Vision Transformer backbone.
    
    Alternative to DINOv2, provides vision-language aligned features.
    
    Args:
        model_name: CLIP model name ('ViT-B/16', 'ViT-L/14', etc.)
        pretrained: Whether to load pretrained weights
        freeze: Whether to freeze backbone
        feature_dim: Output feature dimension
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-B/16',
        pretrained: bool = True,
        freeze: bool = True,
        feature_dim: int = 512,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        
        # Load CLIP model
        self.model, self.preprocess = self._load_clip_model(model_name, pretrained)
        
        # Extract vision encoder
        self.backbone = self.model.visual
        
        if freeze:
            self._freeze_backbone()
        
        logger.info(f"Loaded CLIP backbone: {model_name}")
        logger.info(f"Feature dim: {self.feature_dim}, Frozen: {freeze}")
    
    def _load_clip_model(self, model_name: str, pretrained: bool):
        """Load CLIP model."""
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        
        return model, preprocess
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Dictionary with features
        """
        # CLIP visual encoder
        features = self.backbone(x)
        
        # CLIP outputs are typically (B, D) pooled features
        # Need to extract spatial features for dense prediction
        
        # Get patch embeddings before pooling
        if hasattr(self.backbone, 'conv1'):
            # ResNet-style CLIP
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.avgpool(x)
            
            # Get spatial features
            for layer in [self.backbone.layer1, self.backbone.layer2, 
                         self.backbone.layer3, self.backbone.layer4]:
                x = layer(x)
            
            features_spatial = x
        else:
            # ViT-style CLIP
            x = self.backbone.conv1(x)  # Patch embedding
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            
            # Add positional embedding and cls token
            x = torch.cat([
                self.backbone.class_embedding.expand(x.shape[0], 1, -1), 
                x
            ], dim=1)
            x = x + self.backbone.positional_embedding
            
            # Transformer
            x = self.backbone.transformer(x)
            
            # Extract spatial tokens (exclude CLS)
            patch_tokens = x[:, 1:, :]
            cls_token = x[:, 0, :]
            
            # Reshape to spatial
            B, N, D = patch_tokens.shape
            H = W = int(N ** 0.5)
            features_spatial = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        result = {
            'features': features_spatial,
            'global_features': features,
        }
        
        return result
    
    def train(self, mode=True):
        """Override train to keep frozen."""
        super().train(mode)
        if hasattr(self, 'backbone'):
            frozen = not any(p.requires_grad for p in self.backbone.parameters())
            if frozen:
                self.backbone.eval()
        return self