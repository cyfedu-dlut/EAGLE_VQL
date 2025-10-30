"""Configuration system for EAGLE."""

from .config import CfgNode, get_cfg, global_cfg, set_global_cfg
from .defaults import _C

__all__ = [
    'CfgNode',
    'get_cfg',
    'global_cfg',
    'set_global_cfg',
]