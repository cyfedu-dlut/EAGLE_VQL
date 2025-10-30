"""
Model architectures for EAGLE.
"""

from .backbone import build_backbone
from .meta_arch import (
    EAGLE_VQ2D,
    EAGLE_VQ3D,
    build_model,
)

__all__ = [
    'build_backbone',
    'EAGLE_VQ2D',
    'EAGLE_VQ3D',
    'build_model',
]