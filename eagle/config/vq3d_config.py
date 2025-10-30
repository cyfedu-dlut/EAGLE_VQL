"""
Configuration for VQ3D task.
"""

from .defaults import get_cfg_defaults


def get_vq3d_cfg():
    """
    Get VQ3D-specific configuration.
    
    Returns:
        CfgNode: VQ3D configuration
    """
    cfg = get_cfg_defaults()
    
    # Task-specific settings
    cfg.MODEL.TASK = 'vq3d'
    
    # VQ3D uses longer video clips
    cfg.DATALOADER.NUM_FRAMES = 8
    cfg.DATALOADER.SAMPLING_STRATEGY = 'uniform'
    
    # Enable VQ3D-specific modules
    cfg.MODEL.VQ3D.USE_3D_REASONING = True
    cfg.MODEL.VQ3D.USE_LONG_TERM_MEMORY = True
    cfg.MODEL.VQ3D.LONG_TERM_MEMORY_SIZE = 50
    
    # Temporal modeling
    cfg.MODEL.TEMPORAL.AGGREGATION_TYPE = 'transformer'
    cfg.MODEL.TEMPORAL.PREDICTION_HORIZON = 5
    
    # VQ3D datasets
    cfg.DATASETS.TRAIN = ('ego4d_vq3d_train',)
    cfg.DATASETS.TEST = ('ego4d_vq3d_val',)
    
    # Adjust batch size for multi-frame processing
    cfg.DATALOADER.BATCH_SIZE = 4
    
    # VQ3D-specific augmentation
    cfg.AUGMENTATION.ENABLED = True
    cfg.AUGMENTATION.HORIZONTAL_FLIP = 0.3
    cfg.AUGMENTATION.ROTATION_DEGREES = 5
    cfg.AUGMENTATION.RANDOM_SCALE.ENABLED = True
    
    # Training settings
    cfg.SOLVER.MAX_EPOCHS = 100
    cfg.SOLVER.BASE_LR = 0.00005
    cfg.SOLVER.LR_SCHEDULER = 'warmup_cosine'
    cfg.SOLVER.WARMUP_EPOCHS = 5
    
    # Add temporal consistency loss weight
    cfg.MODEL.LOSS_WEIGHTS.TEMPORAL = 0.1
    cfg.MODEL.LOSS_WEIGHTS.MOTION = 0.05
    
    return cfg