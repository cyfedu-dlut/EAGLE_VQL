"""
Configuration system based on YACS.
Similar to detectron2 config system.
"""

import os
import yaml
from typing import Any, Optional
from yacs.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):
    """
    Extended CfgNode with additional utilities.
    """
    
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)
    
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True):
        """Load config from a YAML file."""
        assert os.path.exists(cfg_filename), f"Config file {cfg_filename} not found"
        with open(cfg_filename, 'r') as f:
            cfg = yaml.safe_load(f)
        self.merge_from_other_cfg(CfgNode(cfg))
    
    def dump(self, *args, **kwargs):
        """Dump config to string."""
        return super().dump(*args, **kwargs)
    
    def clone(self):
        """Deep copy the config."""
        return CfgNode(self.__dict__)


# Global config object
_global_cfg = CfgNode()


def get_cfg() -> CfgNode:
    """
    Get a copy of the global config object.
    
    Returns:
        CfgNode: A copy of the global config
    """
    from .defaults import _C
    return _C.clone()


def global_cfg() -> CfgNode:
    """
    Get the global config object (not a copy).
    
    Returns:
        CfgNode: The global config
    """
    return _global_cfg


def set_global_cfg(cfg: CfgNode):
    """
    Set the global config object.
    
    Args:
        cfg: Config to set as global
    """
    global _global_cfg
    _global_cfg.clear()
    _global_cfg.update(cfg)


def configurable(init_func=None, *, from_config=None):
    """
    Decorator for functions/classes that take config as input.
    Similar to detectron2's configurable.
    """
    if init_func is not None:
        # Used as @configurable without arguments
        return _make_configurable(init_func)
    
    def wrapper(func):
        return _make_configurable(func, from_config)
    
    return wrapper


def _make_configurable(func, from_config=None):
    """Helper for configurable decorator."""
    if from_config is None:
        from_config = func
    
    def wrapped(cfg, *args, **kwargs):
        explicit_args = from_config(cfg)
        explicit_args.update(kwargs)
        return func(*args, **explicit_args)
    
    wrapped.from_config = from_config
    return wrapped