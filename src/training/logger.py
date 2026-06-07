"""Logging and experiment tracking utilities."""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import sys

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class TrainingLogger:
    """Logger for training with console, file, and TensorBoard support."""
    
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_file_logging: bool = True
    ):
        """Initialize training logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name for this experiment
            use_tensorboard: Whether to use TensorBoard
            use_file_logging: Whether to log to file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_file_logging = use_file_logging
        
        # Setup console logging
        self._setup_console_logging()
        
        # Setup file logging
        if use_file_logging:
            self._setup_file_logging()
        
        # Setup TensorBoard
        if self.use_tensorboard:
            tensorboard_dir = self.log_dir / 'tensorboard' / self.experiment_name
            self.writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard logs: {tensorboard_dir}")
        else:
            self.writer = None
        
        # Metrics history
        self.history = {
            'train': [],
            'val': []
        }
        
        logging.info(f"Initialized logger for experiment: {self.experiment_name}")
    
    def _setup_console_logging(self):
        """Setup console logging."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger = logging.getLogger(f"TrainingLogger_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _setup_file_logging(self):
        """Setup file logging."""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
        print(f"File logging: {log_file}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_file}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float
    ):
        """Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            learning_rate: Current learning rate
        """
        # Log to console
        self.logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
            f"Train MAE: {train_metrics.get('mae', 0):.4f}, "
            f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
            f"Val MAE: {val_metrics.get('mae', 0):.4f}, "
            f"LR: {learning_rate:.2e}"
        )
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                if key != 'strata':  # Skip nested dict
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            
            self.writer.add_scalar('learning_rate', learning_rate, epoch)
        
        # Save to history
        self.history['train'].append(train_metrics)
        self.history['val'].append(val_metrics)
    
    def log_strata_metrics(self, epoch: int, strata_metrics: Dict[str, Dict[str, float]]):
        """Log stratified metrics.
        
        Args:
            epoch: Current epoch number
            strata_metrics: Stratified metrics dictionary
        """
        for range_name, metrics in strata_metrics.items():
            self.logger.info(
                f"  {range_name}: MAE={metrics['mae']:.4f}, "
                f"Within 5%={metrics['within_5pct']:.4f}, "
                f"Count={metrics['count']}"
            )
            
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(f'strata/{range_name}/{key}', value, epoch)
    
    def log_best_metric(self, metric_name: str, value: float, epoch: int):
        """Log a new best metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            epoch: Epoch number
        """
        self.logger.info(f"New best {metric_name}: {value:.4f} at epoch {epoch}")
        
        if self.writer is not None:
            self.writer.add_scalar(f'best/{metric_name}', value, epoch)
    
    def save_history(self):
        """Save training history to file."""
        history_file = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"History saved to {history_file}")
    
    def close(self):
        """Close logger and cleanup."""
        if self.writer is not None:
            self.writer.close()
        
        self.save_history()
        self.logger.info("Logger closed")


class ExperimentTracker:
    """Track experiment results for comparison."""
    
    def __init__(self, results_file: str = 'logs/experiment_results.json'):
        """Initialize experiment tracker.
        
        Args:
            results_file: Path to results file
        """
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing results
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []
    
    def add_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        training_time: float,
        notes: str = ""
    ):
        """Add experiment results.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration used
            metrics: Final metrics
            training_time: Training time in seconds
            notes: Additional notes
        """
        experiment = {
            'name': experiment_name,
            'config': config,
            'metrics': metrics,
            'training_time': training_time,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(experiment)
        self._save_results()
    
    def _save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_best_experiment(self, metric: str = 'val_mae', mode: str = 'min') -> Optional[Dict]:
        """Get the best experiment according to a metric.
        
        Args:
            metric: Metric to compare
            mode: 'min' for lower is better, 'max' for higher is better
            
        Returns:
            Best experiment dictionary
        """
        if not self.results:
            return None
        
        if mode == 'min':
            best = min(self.results, key=lambda x: x['metrics'].get(metric, float('inf')))
        else:
            best = max(self.results, key=lambda x: x['metrics'].get(metric, -float('inf')))
        
        return best
    
    def compare_experiments(self, metric: str = 'val_mae') -> str:
        """Generate comparison report.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Comparison report string
        """
        if not self.results:
            return "No experiments to compare."
        
        report = "Experiment Comparison:\n"
        report += "=" * 60 + "\n"
        
        for exp in self.results:
            report += f"{exp['name']}\n"
            report += f"  {metric}: {exp['metrics'].get(metric, 'N/A'):.4f}\n"
            report += f"  Training time: {exp['training_time']:.2f}s\n"
            if exp['notes']:
                report += f"  Notes: {exp['notes']}\n"
            report += "\n"
        
        return report


def create_logger(config: Dict[str, Any]) -> TrainingLogger:
    """Create logger from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrainingLogger instance
    """
    logging_config = config.get('logging', {})
    
    return TrainingLogger(
        log_dir=logging_config.get('tensorboard', {}).get('log_dir', 'logs'),
        experiment_name=logging_config.get('experiment_name'),
        use_tensorboard=logging_config.get('tensorboard', {}).get('enabled', True),
        use_file_logging=True
    )


# Test logger
if __name__ == "__main__":
    print("Testing Logger...")
    
    # Create logger
    logger = TrainingLogger(
        log_dir='logs/test',
        experiment_name='test_experiment',
        use_tensorboard=False,
        use_file_logging=True
    )
    
    # Log config
    test_config = {'epochs': 10, 'batch_size': 32}
    logger.log_config(test_config)
    
    # Log epoch
    train_metrics = {'loss': 0.5, 'mae': 3.2}
    val_metrics = {'loss': 0.6, 'mae': 3.5}
    logger.log_epoch(1, train_metrics, val_metrics, 1e-4)
    
    # Log best metric
    logger.log_best_metric('val_mae', 3.5, 1)
    
    # Close logger
    logger.close()
    
    print("Logger test passed!")
