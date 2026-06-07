"""Main training script for stack count prediction."""

import sys
import argparse
import torch
from pathlib import Path
from typing import Optional
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import create_model_from_config
from src.data import create_data_loaders_from_config
from src.training.trainer import StackCountTrainer
from src.training.logger import TrainingLogger, create_logger
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train stack count prediction model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Path to data directory'
    )
    
    parser.add_argument(
        '--annotations_dir',
        type=str,
        default='data/annotations',
        help='Path to annotations directory'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name for this experiment'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (small dataset, few epochs)'
    )
    
    return parser.parse_args()


def setup_device(device: str) -> str:
    """Setup training device.
    
    Args:
        device: Desired device ('cuda' or 'cpu')
        
    Returns:
        Actual device being used
    """
    if device == 'cuda':
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA: {device_name}")
            return 'cuda'
        else:
            print("CUDA not available, falling back to CPU")
            return 'cpu'
    return device


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Apply debug mode if specified
    if args.debug:
        print("Running in DEBUG mode")
        config['training']['epochs'] = 3
        config['training']['batch_size'] = 4
        config['training']['early_stopping']['patience'] = 1
    
    # Override paths from command line
    config['paths'] = {
        'data_root': args.data_dir,
        'raw_data': args.data_dir,
        'processed_data': args.data_dir,
        'annotations': args.annotations_dir,
        'checkpoints': 'models/checkpoints',
        'logs': 'logs'
    }
    
    # Setup device
    device = setup_device(args.device)
    config['hardware']['device'] = device
    
    # Setup experiment name
    if args.experiment_name is None:
        from datetime import datetime
        args.experiment_name = f"{config['model']['backbone']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup logging
    print("Setting up logging...")
    logger = create_logger(config)
    logger.experiment_name = args.experiment_name
    logger.log_config(config)
    
    print(f"\n{'='*60}")
    print(f"Training Experiment: {args.experiment_name}")
    print(f"{'='*60}")
    
    # Create model
    print("\nCreating model...")
    model = create_model_from_config(config)
    
    params = model.get_num_parameters()
    print(f"Model parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    data_loader = create_data_loaders_from_config(
        config=config,
        data_dir=args.data_dir,
        annotations_dir=args.annotations_dir,
        current_phase=1
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = StackCountTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Log model architecture
    logger.logger.info(f"Model architecture: {config['model']['backbone']['name']}-{config['model']['backbone']['variant']}")
    logger.logger.info(f"Total parameters: {params['total']:,}")
    
    # Start training
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        history = trainer.train()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Log final metrics
        final_val_metrics = history['val'][-1]
        logger.log_epoch(
            trainer.current_epoch,
            history['train'][-1],
            final_val_metrics,
            trainer.scheduler.get_lr()
        )
        
        logger.logger.info(f"Training time: {training_time:.2f}s")
        logger.logger.info(f"Best validation MAE: {trainer.best_val_mae:.4f} at epoch {trainer.best_epoch}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        training_time = time.time() - start_time
        logger.logger.info(f"Training interrupted after {training_time:.2f}s")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    # Close logger
    logger.close()
    
    print(f"\n{'='*60}")
    print("Training finished!")
    print(f"{'='*60}")
    print(f"Best checkpoint: models/checkpoints/best.pth")
    print(f"Logs: logs/{args.experiment_name}*")
    print(f"\nTo monitor training with TensorBoard:")
    print(f"  tensorboard --logdir logs/tensorboard/{args.experiment_name}")


if __name__ == "__main__":
    main()
