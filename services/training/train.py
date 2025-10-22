"""
Heimdall RF Source Localization - Training Script with MLflow Integration

Entry point for training the neural network model using:
- PyTorch Lightning for distributed training
- MLflow for experiment tracking and model registry
- HeimdallDataset for loading approved recording sessions

Usage:
    python train.py --config config/model_config.py --epochs 100 --batch-size 32
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
try:
    from pytorch_lightning.loggers import MLflowLogger
except ImportError:
    # pytorch-lightning >= 2.1.0
    from pytorch_lightning.loggers.mlflow import MLflowLogger
import structlog

from src.config import settings
from src.mlflow_setup import initialize_mlflow, MLflowTracker
from src.models.lightning_module import LocalizationLitModule
from src.data.dataset import HeimdallDataset
from src.config.model_config import ModelConfig, BackboneArchitecture

logger = structlog.get_logger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline with MLflow integration.
    
    Responsibilities:
    - Data loading and validation
    - Model initialization
    - PyTorch Lightning trainer setup
    - MLflow experiment tracking
    - Model checkpoint and registry
    """
    
    def __init__(
        self,
        mlflow_tracker: MLflowTracker,
        model_config: ModelConfig,
        training_config: Dict[str, Any],
        output_dir: Path = Path("./outputs"),
    ):
        """
        Initialize training pipeline.
        
        Args:
            mlflow_tracker (MLflowTracker): MLflow tracker instance
            model_config (ModelConfig): Model configuration
            training_config (dict): Training hyperparameters
            output_dir (Path): Output directory for checkpoints/artifacts
        """
        
        self.mlflow_tracker = mlflow_tracker
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "training_pipeline_initialized",
            model_config=str(model_config),
            training_config=training_config,
            output_dir=str(self.output_dir),
        )
    
    def setup_data_loaders(
        self,
        approved_sessions: list,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Setup data loaders for training, validation, and test.
        
        Args:
            approved_sessions (list): List of approved recording session IDs
            val_split (float): Validation set fraction
            test_split (float): Test set fraction
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, val, test loaders
        """
        
        # Split sessions
        num_sessions = len(approved_sessions)
        num_test = max(1, int(num_sessions * test_split))
        num_val = max(1, int((num_sessions - num_test) * val_split))
        num_train = num_sessions - num_val - num_test
        
        train_sessions = approved_sessions[:num_train]
        val_sessions = approved_sessions[num_train:num_train + num_val]
        test_sessions = approved_sessions[num_train + num_val:]
        
        logger.info(
            "data_split",
            total_sessions=num_sessions,
            train=len(train_sessions),
            val=len(val_sessions),
            test=len(test_sessions),
        )
        
        # Create datasets
        train_dataset = HeimdallDataset(
            session_ids=train_sessions,
            split="train",
            augmentation=True,
        )
        
        val_dataset = HeimdallDataset(
            session_ids=val_sessions,
            split="val",
            augmentation=False,
        )
        
        test_dataset = HeimdallDataset(
            session_ids=test_sessions,
            split="test",
            augmentation=False,
        )
        
        # Create loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True,
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True,
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True,
        )
        
        logger.info(
            "data_loaders_created",
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            test_batches=len(test_loader),
        )
        
        return train_loader, val_loader, test_loader
    
    def setup_callbacks(self) -> list:
        """
        Setup PyTorch Lightning callbacks.
        
        Returns:
            list: List of callbacks
        """
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Save best model checkpoint
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch:02d}-{val/loss:.3f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True,
            ),
            
            # Early stopping
            EarlyStopping(
                monitor="val/loss",
                patience=self.training_config.get('early_stopping_patience', 5),
                mode="min",
                verbose=True,
            ),
            
            # Learning rate monitoring
            LearningRateMonitor(logging_interval="epoch"),
        ]
        
        logger.info("callbacks_configured", num_callbacks=len(callbacks))
        
        return callbacks
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        run_name: str = None,
        tags: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Execute training loop with MLflow tracking.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            run_name (str): MLflow run name
            tags (dict): MLflow tags
            
        Returns:
            dict: Training results including metrics and model path
        """
        
        # Generate run name if not provided
        if not run_name:
            run_name = f"{settings.mlflow_run_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Start MLflow run
        run_id = self.mlflow_tracker.start_run(run_name=run_name, tags=tags)
        
        try:
            # Log parameters
            params = {
                'learning_rate': self.training_config['learning_rate'],
                'batch_size': self.training_config['batch_size'],
                'epochs': self.training_config['epochs'],
                'backbone': self.model_config.backbone.value,
                'pretrained': self.model_config.pretrained,
                'freeze_backbone': self.model_config.freeze_backbone,
                'weight_decay': self.training_config.get('weight_decay', 1e-5),
            }
            
            self.mlflow_tracker.log_params(params)
            
            # Create Lightning module
            model = LocalizationLitModule(
                learning_rate=self.training_config['learning_rate'],
                weight_decay=self.training_config.get('weight_decay', 1e-5),
                num_training_steps=len(train_loader) * self.training_config['epochs'],
                pretrained_backbone=self.model_config.pretrained,
                freeze_backbone=self.model_config.freeze_backbone,
                backbone_size=self._get_backbone_size(),
            )
            
            # Setup MLflow logger
            mlflow_logger = MLflowLogger(
                experiment_name=settings.mlflow_experiment_name,
                run_name=run_name,
                tracking_uri=settings.mlflow_tracking_uri,
            )
            mlflow_logger.log_hyperparams(params)
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.training_config['epochs'],
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                logger=mlflow_logger,
                callbacks=self.setup_callbacks(),
                log_every_n_steps=10,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            
            logger.info(
                "trainer_created",
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                max_epochs=self.training_config['epochs'],
            )
            
            # Train
            logger.info("training_started", run_id=run_id)
            trainer.fit(model, train_loader, val_loader)
            
            # Test
            logger.info("testing_started")
            test_results = trainer.test(model, test_loader)
            
            # Get best checkpoint path
            best_model_path = trainer.checkpoint_callback.best_model_path
            
            logger.info(
                "training_completed",
                run_id=run_id,
                best_model_path=best_model_path,
                test_results=test_results,
            )
            
            # Log artifacts
            self.mlflow_tracker.log_artifact(best_model_path, artifact_path="checkpoints")
            
            # Log metrics summary
            if test_results:
                summary_metrics = {
                    'final_test_loss': test_results[0].get('test/loss', None),
                    'final_test_mae': test_results[0].get('test/mae', None),
                }
                self.mlflow_tracker.log_metrics(summary_metrics)
            
            return {
                'run_id': run_id,
                'model_path': best_model_path,
                'metrics': test_results,
                'status': 'SUCCESS',
            }
        
        except Exception as e:
            logger.error(
                "training_failed",
                run_id=run_id,
                error=str(e),
                exc_info=True,
            )
            
            self.mlflow_tracker.end_run(status="FAILED")
            
            raise
        
        finally:
            self.mlflow_tracker.end_run(status="FINISHED")
    
    def _get_backbone_size(self) -> str:
        """
        Extract backbone size from model config.
        
        Returns:
            str: Backbone size (tiny, small, medium, large)
        """
        
        backbone_str = self.model_config.backbone.value
        
        if 'large' in backbone_str:
            return 'large'
        elif 'medium' in backbone_str or 'base' in backbone_str:
            return 'medium'
        elif 'small' in backbone_str:
            return 'small'
        else:
            return 'tiny'


def load_training_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load training configuration from JSON or use defaults.
    
    Args:
        config_path (Path): Path to config JSON file
        
    Returns:
        dict: Training configuration
    """
    
    default_config = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 100,
        'weight_decay': 1e-5,
        'early_stopping_patience': 5,
        'num_workers': 4,
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            default_config.update(custom_config)
            logger.info("training_config_loaded", config_path=str(config_path))
        except Exception as e:
            logger.warning("training_config_load_failed", error=str(e))
    
    return default_config


def main(args):
    """
    Main training entry point.
    
    Args:
        args: Command-line arguments
    """
    
    logger.info("===== Heimdall Training Pipeline Started =====")
    
    # Initialize MLflow
    mlflow_tracker = initialize_mlflow(settings)
    
    # Load model config
    model_config = ModelConfig(
        backbone=BackboneArchitecture[args.backbone.upper()],
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    
    # Load training config
    training_config = load_training_config(args.config)
    
    # Update from CLI
    if args.learning_rate:
        training_config['learning_rate'] = args.learning_rate
    if args.batch_size:
        training_config['batch_size'] = args.batch_size
    if args.epochs:
        training_config['epochs'] = args.epochs
    
    logger.info(
        "configurations_loaded",
        model_config=str(model_config),
        training_config=training_config,
    )
    
    # Create pipeline
    pipeline = TrainingPipeline(
        mlflow_tracker=mlflow_tracker,
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir,
    )
    
    # For demonstration: use mock data
    # In production, fetch approved sessions from database
    approved_sessions = [f"session_{i}" for i in range(100)]  # Mock
    
    # Setup data loaders
    train_loader, val_loader, test_loader = pipeline.setup_data_loaders(
        approved_sessions=approved_sessions,
        val_split=0.2,
        test_split=0.1,
    )
    
    # Run training
    results = pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_name=args.run_name,
        tags={'environment': settings.environment},
    )
    
    logger.info(
        "===== Training Pipeline Completed =====",
        results=results,
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heimdall RF Source Localization Training Pipeline"
    )
    
    parser.add_argument(
        "--backbone",
        type=str,
        default="CONVNEXT_LARGE",
        help="Backbone architecture (CONVNEXT_LARGE, RESNET_50, VIT_BASE, etc.)",
    )
    
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use ImageNet pretrained weights",
    )
    
    parser.add_argument(
        "--freeze-backbone",
        type=bool,
        default=False,
        help="Freeze backbone during training",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs (overrides config)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.json",
        help="Path to training config JSON",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and artifacts",
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        help="MLflow run name (auto-generated if not provided)",
    )
    
    args = parser.parse_args()
    
    main(args)
