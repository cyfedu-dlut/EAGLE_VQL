"""
Build backbone networks.
"""

import logging
from .dinov2 import DINOv2Backbone, DINOv2FeatureExtractor
from .clip import CLIPBackbone

logger = logging.getLogger(__name__)


def build_backbone(cfg):
    """
    Build backbone network based on config.
    
    Args:
        cfg: Config object
    
    Returns:
        Backbone network
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    logger.info(f"Building backbone: {backbone_name}")
    
    if 'dinov2' in backbone_name.lower():
        backbone = DINOv2Backbone(
            model_name=backbone_name,
            pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
            freeze=cfg.MODEL.BACKBONE.FREEZE,
            feature_dim=cfg.MODEL.BACKBONE.FEATURE_DIM,
            use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        )
    
    elif 'clip' in backbone_name.lower():
        backbone = CLIPBackbone(
            model_name=cfg.MODEL.BACKBONE.CLIP_MODEL,
            pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
            freeze=cfg.MODEL.BACKBONE.FREEZE,
            feature_dim=cfg.MODEL.BACKBONE.FEATURE_DIM,
        )
    
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    return backbone