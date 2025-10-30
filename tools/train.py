"""
Training script for EAGLE models.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eagle.config import get_cfg_defaults
from eagle.modeling.meta_arch import build_model
from eagle.data import build_train_loader, build_val_loader
from eagle.engine import Trainer
from eagle.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train EAGLE model')
    
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        help='Path to config file',
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='vq2d',
        choices=['vq2d', 'vq3d'],
        help='Task to train on',
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='Path to checkpoint to resume from',
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation',
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use',
    )
    
    parser.add_argument(
        '--dist-url',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL for distributed training',
    )
    
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using command line',
    )
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration."""
    cfg = get_cfg_defaults()
    
    # Load task-specific config
    if args.task == 'vq2d':
        from eagle.config.vq2d_config import get_vq2d_cfg
        cfg = get_vq2d_cfg()
    elif args.task == 'vq3d':
        from eagle.config.vq3d_config import get_vq3d_cfg
        cfg = get_vq3d_cfg()
    
    # Merge from config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # Merge from command line
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Set number of GPUs
    cfg.NUM_GPUS = args.num_gpus
    cfg.DISTRIBUTED = args.num_gpus > 1
    
    # Freeze config
    cfg.freeze()
    
    return cfg


def setup_environment(cfg):
    """Setup random seeds and CUDA settings."""
    # Random seeds
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    
    # CUDA settings
    if cfg.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def setup_distributed(args):
    """Setup distributed training."""
    if args.num_gpus > 1:
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.num_gpus,
            rank=int(os.environ.get('LOCAL_RANK', 0)),
        )
        
        # Set device
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        
        return local_rank
    
    return 0


def build_optimizer(cfg, model):
    """Build optimizer."""
    # Separate backbone and other parameters
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    # Use different learning rates for backbone and other parts
    param_groups = [
        {'params': backbone_params, 'lr': cfg.SOLVER.BASE_LR * 0.1},
        {'params': other_params, 'lr': cfg.SOLVER.BASE_LR},
    ]
    
    # Build optimizer
    if cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=cfg.SOLVER.EPS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            eps=cfg.SOLVER.EPS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.SOLVER.OPTIMIZER}")
    
    return optimizer


def build_scheduler(cfg, optimizer):
    """Build learning rate scheduler."""
    if cfg.SOLVER.LR_SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.SOLVER.STEP_SIZE,
            gamma=cfg.SOLVER.GAMMA,
        )
    
    elif cfg.SOLVER.LR_SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.T_MAX,
        )
    
    elif cfg.SOLVER.LR_SCHEDULER == 'poly':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / cfg.SOLVER.MAX_EPOCHS) ** cfg.SOLVER.POLY_POWER,
        )
    
    elif cfg.SOLVER.LR_SCHEDULER == 'warmup_cosine':
        from eagle.utils.scheduler import WarmupCosineScheduler
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
            max_epochs=cfg.SOLVER.MAX_EPOCHS,
            warmup_start_lr=cfg.SOLVER.WARMUP_START_LR,
            min_lr=cfg.SOLVER.WARMUP_START_LR,
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {cfg.SOLVER.LR_SCHEDULER}")
    
    return scheduler


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    cfg = setup_config(args)
    
    # Setup distributed training
    local_rank = setup_distributed(args)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    logger = setup_logger(
        name='eagle',
        output_dir=cfg.OUTPUT.DIR,
        rank=local_rank,
    )
    
    logger.info(f"Command line arguments: {args}")
    logger.info(f"Running with config:\n{cfg}")
    
    # Setup environment
    setup_environment(cfg)
    
    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if cfg.DISTRIBUTED:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    # Build data loaders
    logger.info("Building data loaders...")
    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg) if cfg.DATASETS.TEST else None
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if cfg.DISTRIBUTED:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Build trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    
    trainer.current_epoch = start_epoch
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("Running evaluation only...")
        if val_loader is None:
            logger.error("No validation dataset specified for evaluation!")
            return
        
        val_metrics = trainer.validate()
        logger.info("Evaluation results:")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        return
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()