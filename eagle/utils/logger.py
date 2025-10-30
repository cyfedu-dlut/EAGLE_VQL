"""
Logging utilities.
"""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = 'eagle',
    output_dir: Optional[str] = None,
    rank: int = 0,
    filename: str = 'log.txt',
) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        output_dir: Directory to save log file
        rank: Process rank (for distributed training)
        filename: Log filename
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    logger.addHandler(console_handler)
    
    # File handler
    if output_dir and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        )
        logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """Wrapper for Tensorboard logging."""
    
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image."""
        self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close writer."""
        self.writer.close()


class WandbLogger:
    """Wrapper for Weights & Biases logging."""
    
    def __init__(self, project: str, entity: str = None, config=None):
        import wandb
        wandb.init(project=project, entity=entity, config=config)
        self.wandb = wandb
    
    def log(self, metrics: dict, step: int = None):
        """Log metrics."""
        self.wandb.log(metrics, step=step)
    
    def log_image(self, key: str, image):
        """Log image."""
        self.wandb.log({key: self.wandb.Image(image)})
    
    def finish(self):
        """Finish logging."""
        self.wandb.finish()