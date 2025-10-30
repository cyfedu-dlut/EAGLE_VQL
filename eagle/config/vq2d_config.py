"""
Configuration for VQ2D task.
"""

from .defaults import get_cfg_defaults


def get_vq2d_cfg():
    """
    Get VQ2D-specific configuration.
    
    Returns:
        CfgNode: VQ2D configuration
    """
    cfg = get_cfg_defaults()
    
    # Task-specific settings
    cfg.MODEL.TASK = 'vq2d'
    
    # VQ2D typically uses single frame or short clips
    cfg.DATALOADER.NUM_FRAMES = 1
    cfg.DATALOADER.SAMPLING_STRATEGY = 'random'
    
    # VQ2D datasets
    cfg.DATASETS.TRAIN = ('ego4d_vq2d_train',)
    cfg.DATASETS.TEST = ('ego4d_vq2d_val',)
    
    # Adjust batch size for single-frame processing
    cfg.DATALOADER.BATCH_SIZE = 16
    
    # VQ2D-specific augmentation
    cfg.AUGMENTATION.ENABLED = True
    cfg.AUGMENTATION.HORIZONTAL_FLIP = 0.5
    cfg.AUGMENTATION.ROTATION_DEGREES = 10
    
    # Training settings
    cfg.SOLVER.MAX_EPOCHS = 50
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.LR_SCHEDULER = 'warmup_cosine'
    cfg.SOLVER.WARMUP_EPOCHS = 3
    
    return cfg