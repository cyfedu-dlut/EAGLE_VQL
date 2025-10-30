"""
Testing/Evaluation script for EAGLE models.
"""

import os
import sys
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eagle.config import get_cfg_defaults
from eagle.modeling.meta_arch import build_model, load_checkpoint
from eagle.data import build_test_loader
from eagle.engine import Evaluator, VQ2DEvaluator, VQ3DEvaluator
from eagle.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate EAGLE model')
    
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to config file',
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights',
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='vq2d',
        choices=['vq2d', 'vq3d'],
        help='Task to evaluate on',
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to disk',
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_output',
        help='Directory to save outputs',
    )
    
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using command line',
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    cfg = get_cfg_defaults()
    
    # Load task-specific config
    if args.task == 'vq2d':
        from eagle.config.vq2d_config import get_vq2d_cfg
        cfg = get_vq2d_cfg()
    elif args.task == 'vq3d':
        from eagle.config.vq3d_config import get_vq3d_cfg
        cfg = get_vq3d_cfg()
    
    # Merge from config file
    cfg.merge_from_file(args.config_file)
    
    # Merge from command line
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Override output directory
    cfg.defrost()
    cfg.OUTPUT.DIR = args.output_dir
    cfg.TEST.SAVE_PREDICTIONS = args.save_predictions
    cfg.TEST.VISUALIZE = args.visualize
    cfg.freeze()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    logger = setup_logger(
        name='eagle',
        output_dir=cfg.OUTPUT.DIR,
    )
    
    logger.info(f"Running evaluation with config:\n{cfg}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    
    # Load weights
    logger.info(f"Loading weights from {args.weights}")
    load_checkpoint(
        checkpoint_path=args.weights,
        model=model,
        device=device,
    )
    
    model.eval()
    
    # Build test loader
    logger.info("Building test data loader...")
    test_loader = build_test_loader(cfg)
    
    # Build evaluator
    logger.info("Building evaluator...")
    if args.task == 'vq2d':
        evaluator = VQ2DEvaluator(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            device=device,
            save_predictions=args.save_predictions,
            visualize=args.visualize,
        )
    elif args.task == 'vq3d':
        evaluator = VQ3DEvaluator(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            device=device,
            save_predictions=args.save_predictions,
            visualize=args.visualize,
        )
    else:
        evaluator = Evaluator(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            device=device,
            save_predictions=args.save_predictions,
            visualize=args.visualize,
        )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate()
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    for key, value in results.items():
        logger.info(f"{key}: {value:.4f}")
    logger.info("=" * 50)
    
    logger.info(f"Evaluation completed! Results saved to {cfg.OUTPUT.DIR}")


if __name__ == '__main__':
    main()