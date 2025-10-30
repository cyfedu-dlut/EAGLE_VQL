"""
EAGLE: Efficient Adaptive Geometry-Aware Learning for Visual Query Localization

This package provides implementations for VQ2D and VQ3D tasks on Ego4D dataset.
"""

from .version import __version__, get_version, get_version_info

# Import main components
from .config import get_cfg
from .modeling import build_model
from .data import build_vq_dataloader
from .evaluation import VQ2DEvaluator, VQ3DEvaluator

__all__ = [
    '__version__',
    'get_version',
    'get_version_info',
    'get_cfg',
    'build_model',
    'build_vq_dataloader',
    'VQ2DEvaluator',
    'VQ3DEvaluator',
]

# Package metadata
__author__ = ''
__email__ = ''
__url__ = ''
__description__ = 'EAGLE'
__license__ = 'MIT'