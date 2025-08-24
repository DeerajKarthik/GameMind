"""
Logger utility for GameMind with TensorBoard and W&B support.
"""

import os
import logging
from pathlib import Path


class Logger:
    """Logger class with TensorBoard and W&B support."""
    
    def __init__(self, config: dict):
        """Initialize logger with configuration."""
        self.config = config
        self.log_dir = config.get('logging', {}).get('log_dir', 'logs/')
        self.use_tensorboard = config.get('logging', {}).get('tensorboard', True)
        self.use_wandb = config.get('logging', {}).get('wandb', False)
        self.log_level = config.get('logging', {}).get('log_level', 'INFO')
        
        # Setup basic logging
        self.setup_basic_logging()
        
        # Setup TensorBoard if available
        if self.use_tensorboard:
            self.setup_tensorboard()
        
        # Setup W&B if available
        if self.use_wandb:
            self.setup_wandb()
    
    def setup_basic_logging(self):
        """Setup basic Python logging."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'gamemind.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('GameMind')
        self.logger.info("GameMind Logger initialized")
    
    def setup_tensorboard(self):
        """Setup TensorBoard logging if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
            self.logger.info("TensorBoard logging enabled")
        except ImportError:
            self.logger.warning("TensorBoard not available, falling back to basic logging")
            self.use_tensorboard = False
    
    def setup_wandb(self):
        """Setup W&B logging if available."""
        try:
            import wandb
            wandb.init(
                project="gamemind",
                config=self.config,
                dir=self.log_dir
            )
            self.logger.info("W&B logging enabled")
        except ImportError:
            self.logger.warning("W&B not available, falling back to basic logging")
            self.use_wandb = False
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value."""
        if self.use_tensorboard and hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_scalar(tag, value, step)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({tag: value}, step=step)
            except:
                pass
    
    def log_histogram(self, tag, values, step):
        """Log a histogram."""
        if self.use_tensorboard and hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, image, step):
        """Log an image."""
        if self.use_tensorboard and hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_image(tag, image, step)
    
    def close(self):
        """Close all logging connections."""
        if self.use_tensorboard and hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
        
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass 