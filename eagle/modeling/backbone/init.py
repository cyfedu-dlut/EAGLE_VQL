"""
Backbone networks for feature extraction.
"""

from .dinov2 import DINOv2Backbone
from .clip import CLIPBackbone
from .build import build_backbone

__all__ = [
    'DINOv2Backbone',
    'CLIPBackbone',
    'build_backbone',
]