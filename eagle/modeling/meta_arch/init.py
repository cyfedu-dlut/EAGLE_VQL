"""
Meta-architectures for EAGLE models.
"""

from .eagle_vq2d import EAGLE_VQ2D
from .eagle_vq3d import EAGLE_VQ3D
from .build import build_model

__all__ = [
    'EAGLE_VQ2D',
    'EAGLE_VQ3D',
    'build_model',
]