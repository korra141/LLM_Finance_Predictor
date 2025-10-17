"""
Logging utilities for the financial prediction system.
Handles logging configuration, WandB integration, and experiment tracking.
"""

import logging
import os
import json
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class FinancialLogger:
    """Main logging class for financial prediction experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
        self.wandb_run = None
        self.mlflow_run = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
    
    def _setup_logging(self):
        """Setup basic logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('log_level', 'info').upper()
        
        # Create logger
        self.logger = logging.getLogger('financial_predictor')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = log_config.get('log_file', 'logs/financial_predictor.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Logging setup completed")
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking (WandB, MLflow)."""
        tracking_config = self.config.get('experiment_tracking', {})
        
        # Setup WandB
        if tracking_config.get('use_wandb', False) and WANDB_AVAILABLE:
            self._setup_wandb(tracking_config)
        
        # Setup MLflow
        if tracking_config.get('use_mlflow', False) and MLFLOW_AVAILABLE:
            self._setup_mlflow(tracking_config)
    
    def _setup_wandb(self, config: Dict[str, Any]):
        """Setup WandB experiment tracking."""
        try:
            wandb_config = config.get('wandb', {})
            
            self.wandb_run = wandb.init(
                project=wandb_config.get('project', 'financial-predictor'),
                name=wandb_config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                reinit=True
            )
            
            self.logger.info("WandB experiment tracking initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def _setup_mlflow(self, config: Dict[str, Any]):
        """Setup MLflow experiment tracking."""
        try:
            mlflow_config = config.get('mlflow', {})
            
            # Set tracking URI
            if 'tracking_uri' in mlflow_config:
                mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
            
            # Set experiment
            experiment_name = mlflow_config.get('experiment_name', 'financial-predictor')
            mlflow.set_experiment(experiment_name)
            
            # Start run
            self.mlflow_run = mlflow.start_run(
                run_name=mlflow_config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )
            
            # Log config
            mlflow.log_params(self.config)
            
            self.logger.info("MLflow experiment tracking initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize MLflow: {e}")
            self.mlflow_run = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all configured tracking systems."""
        # Log to console
        self.logger.info(f"Metrics at step {step}: {metrics}")
        
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log to MLflow: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to all configured tracking systems."""
        # Log to console
        self.logger.info(f"Parameters: {params}")
        
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.config.update(params)
            except Exception as e:
                self.logger.warning(f"Failed to log parameters to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_params(params)
            except Exception as e:
                self.logger.warning(f"Failed to log parameters to MLflow: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, str]):
        """Log artifacts to all configured tracking systems."""
        # Log to WandB
        if self.wandb_run:
            try:
                for name, path in artifacts.items():
                    wandb.log_artifact(path, name=name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifacts to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifacts to MLflow: {e}")
    
    def log_model(self, model_path: str, model_name: str = "financial_model"):
        """Log model to all configured tracking systems."""
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.log_model(model_path, name=model_name)
            except Exception as e:
                self.logger.warning(f"Failed to log model to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_model(model_path, model_name)
            except Exception as e:
                self.logger.warning(f"Failed to log model to MLflow: {e}")
    
    def log_dataframe(self, df: pd.DataFrame, name: str, step: Optional[int] = None):
        """Log DataFrame to all configured tracking systems."""
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Table(dataframe=df)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log DataFrame to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                # Save DataFrame to CSV and log as artifact
                csv_path = f"temp_{name}_{step}.csv" if step else f"temp_{name}.csv"
                df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, name)
                os.remove(csv_path)  # Clean up
            except Exception as e:
                self.logger.warning(f"Failed to log DataFrame to MLflow: {e}")
    
    def log_plot(self, plot_path: str, name: str, step: Optional[int] = None):
        """Log plot to all configured tracking systems."""
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Image(plot_path)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log plot to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_artifact(plot_path, name)
            except Exception as e:
                self.logger.warning(f"Failed to log plot to MLflow: {e}")
    
    def log_text(self, text: str, name: str, step: Optional[int] = None):
        """Log text to all configured tracking systems."""
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.log({name: wandb.Html(text)}, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log text to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                # Save text to file and log as artifact
                text_path = f"temp_{name}_{step}.txt" if step else f"temp_{name}.txt"
                with open(text_path, 'w') as f:
                    f.write(text)
                mlflow.log_artifact(text_path, name)
                os.remove(text_path)  # Clean up
            except Exception as e:
                self.logger.warning(f"Failed to log text to MLflow: {e}")
    
    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log experiment summary."""
        # Log to console
        self.logger.info("Experiment Summary:")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")
        
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.summary.update(summary)
            except Exception as e:
                self.logger.warning(f"Failed to log summary to WandB: {e}")
        
        # Log to MLflow
        if self.mlflow_run:
            try:
                mlflow.log_metrics(summary)
            except Exception as e:
                self.logger.warning(f"Failed to log summary to MLflow: {e}")
    
    def finish(self):
        """Finish logging and close all tracking systems."""
        # Finish WandB
        if self.wandb_run:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish WandB: {e}")
        
        # Finish MLflow
        if self.mlflow_run:
            try:
                mlflow.end_run()
            except Exception as e:
                self.logger.warning(f"Failed to finish MLflow: {e}")
        
        self.logger.info("Logging finished")


class ExperimentTracker:
    """Track experiments and compare results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiments = []
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """Start a new experiment."""
        self.current_experiment = {
            'name': experiment_name,
            'config': config,
            'start_time': datetime.now(),
            'metrics': [],
            'artifacts': [],
            'status': 'running'
        }
        
        self.experiments.append(self.current_experiment)
    
    def log_experiment_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log a metric for the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        metric_entry = {
            'name': metric_name,
            'value': value,
            'step': step,
            'timestamp': datetime.now()
        }
        
        self.current_experiment['metrics'].append(metric_entry)
    
    def log_experiment_artifact(self, artifact_name: str, artifact_path: str):
        """Log an artifact for the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        artifact_entry = {
            'name': artifact_name,
            'path': artifact_path,
            'timestamp': datetime.now()
        }
        
        self.current_experiment['artifacts'].append(artifact_entry)
    
    def finish_experiment(self, status: str = 'completed'):
        """Finish the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment to finish.")
        
        self.current_experiment['end_time'] = datetime.now()
        self.current_experiment['status'] = status
        self.current_experiment = None
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments."""
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            # Get final metrics
            final_metrics = {}
            for metric in exp['metrics']:
                if metric['step'] is None or metric['step'] == max([m['step'] for m in exp['metrics'] if m['step'] is not None]):
                    final_metrics[metric['name']] = metric['value']
            
            summary_row = {
                'experiment_name': exp['name'],
                'start_time': exp['start_time'],
                'end_time': exp.get('end_time'),
                'status': exp['status'],
                'duration': (exp.get('end_time', datetime.now()) - exp['start_time']).total_seconds(),
                'num_metrics': len(exp['metrics']),
                'num_artifacts': len(exp['artifacts'])
            }
            summary_row.update(final_metrics)
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def save_experiments(self, filepath: str):
        """Save experiments to file."""
        with open(filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def load_experiments(self, filepath: str):
        """Load experiments from file."""
        with open(filepath, 'r') as f:
            self.experiments = json.load(f)


if __name__ == "__main__":
    # Example usage
    config = {
        'logging': {
            'log_level': 'info',
            'log_file': 'logs/example.log'
        },
        'experiment_tracking': {
            'use_wandb': False,  # Set to True if WandB is available
            'use_mlflow': False,  # Set to True if MLflow is available
            'wandb': {
                'project': 'financial-predictor-test',
                'run_name': 'test_run'
            }
        }
    }
    
    # Initialize logger
    logger = FinancialLogger(config)
    
    # Log some metrics
    logger.log_metrics({
        'accuracy': 0.85,
        'loss': 0.15,
        'f1_score': 0.82
    }, step=100)
    
    # Log parameters
    logger.log_parameters({
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    })
    
    # Finish logging
    logger.finish()
    
    # Example with experiment tracker
    tracker = ExperimentTracker(config)
    
    # Start experiment
    tracker.start_experiment('test_experiment', {'lr': 0.001})
    
    # Log metrics
    tracker.log_experiment_metric('accuracy', 0.85, step=100)
    tracker.log_experiment_metric('loss', 0.15, step=100)
    
    # Finish experiment
    tracker.finish_experiment()
    
    # Get summary
    summary = tracker.get_experiment_summary()
    print("Experiment summary:")
    print(summary)