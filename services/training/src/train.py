"""
Training Entry Point Script: Orchestrate complete training pipeline.

This script orchestrates the complete RF source localization training pipeline:
1. Load training/validation sessions from MinIO and PostgreSQL
2. Create PyTorch DataLoaders with proper batching
3. Initialize PyTorch Lightning trainer with checkpoint callbacks
4. Train LocalizationNet with Gaussian NLL loss
5. Export best checkpoint to ONNX format
6. Register model with MLflow Model Registry

Pipeline:
  Sessions (MinIO) → DataLoaders → Lightning Trainer → Best Checkpoint
    ↓                                                        ↓
  PostgreSQL (metadata)                              ONNX Export + MLflow Registry

Performance targets:
- Training throughput: 32 samples/batch (configurable)
- Validation frequency: Every epoch
- Best model selection: Lowest validation loss (early stopping)
- Checkpoint saving: Top 3 models (by validation loss)
- ONNX inference speedup: 1.5-2.5x vs PyTorch

Usage:
    python train.py --epochs 100 --batch_size 32 --lr 1e-3 --val_split 0.2
    python train.py --checkpoint /path/to/checkpoint.ckpt --resume_training
    python train.py --export_only --checkpoint /path/to/best.ckpt

Configuration:
    All parameters configurable via CLI arguments or .env file
    MLflow tracking automatic (experiment: heimdall-localization)
    Checkpoints saved to: /tmp/heimdall_checkpoints/
    ONNX export to: MinIO (heimdall-models bucket)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import logging

# PyTorch & Lightning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# Project imports
import structlog
from config import settings  # Now imports from config/ package
from mlflow_setup import MLflowTracker
from onnx_export import export_and_register_model, ONNXExporter
from models.localization_net import LocalizationNet, LocalizationLightningModule
from data.dataset import HeimdallDataset
from data.features import MEL_SPECTROGRAM_SHAPE

# Configure logging
logger = structlog.get_logger(__name__)
logging.basicConfig(level=logging.INFO)
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.setLevel(logging.WARNING)  # Reduce Lightning verbosity


class TrainingPipeline:
    """
    Orchestrates the complete training workflow.
    
    Responsibilities:
    - Load data from MinIO and PostgreSQL
    - Create data loaders with proper batching
    - Initialize Lightning trainer with callbacks
    - Execute training loop
    - Export and register best model
    """
    
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        num_workers: int = 4,
        accelerator: str = "gpu",
        devices: int = 1,
        checkpoint_dir: Optional[Path] = None,
        experiment_name: str = "heimdall-localization",
        run_name_prefix: str = "rf-localization",
    ):
        """
        Initialize training pipeline.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for data loaders
            learning_rate (float): Learning rate for optimizer
            validation_split (float): Fraction of data for validation
            num_workers (int): Number of PyTorch DataLoader workers
            accelerator (str): Training accelerator ("cpu", "gpu", "auto")
            devices (int): Number of GPUs (if accelerator="gpu")
            checkpoint_dir (Path): Directory for saving checkpoints
            experiment_name (str): MLflow experiment name
            run_name_prefix (str): Prefix for MLflow run name
        """
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.accelerator = accelerator if torch.cuda.is_available() else "cpu"
        self.devices = devices if torch.cuda.is_available() else 1
        
        # Setup checkpoint directory
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/heimdall_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow tracker
        self.mlflow_tracker = self._init_mlflow(
            experiment_name=experiment_name,
            run_name_prefix=run_name_prefix,
        )
        
        # Initialize boto3 S3 client for MinIO
        import boto3
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=settings.mlflow_s3_endpoint_url,
            aws_access_key_id=settings.mlflow_s3_access_key_id,
            aws_secret_access_key=settings.mlflow_s3_secret_access_key,
        )
        
        # Initialize ONNX exporter
        self.onnx_exporter = ONNXExporter(self.s3_client, self.mlflow_tracker)
        
        logger.info(
            "training_pipeline_initialized",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            accelerator=self.accelerator,
            devices=self.devices,
            checkpoint_dir=str(self.checkpoint_dir),
        )
    
    def _init_mlflow(
        self,
        experiment_name: str,
        run_name_prefix: str,
    ) -> MLflowTracker:
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name (str): MLflow experiment name
            run_name_prefix (str): Prefix for run name
        
        Returns:
            MLflowTracker instance
        """
        tracker = MLflowTracker(
            tracking_uri=settings.mlflow_tracking_uri,
            artifact_uri=settings.mlflow_artifact_uri,
            backend_store_uri=settings.mlflow_backend_store_uri,
            registry_uri=settings.mlflow_registry_uri,
            s3_endpoint_url=settings.mlflow_s3_endpoint_url,
            s3_access_key_id=settings.mlflow_s3_access_key_id,
            s3_secret_access_key=settings.mlflow_s3_secret_access_key,
            experiment_name=experiment_name,
        )
        
        # Create run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name_prefix}_{timestamp}"
        
        # Start new run
        tracker.start_run(run_name)
        
        # Log hyperparameters
        tracker.log_params({
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "num_workers": self.num_workers,
            "accelerator": self.accelerator,
            "optimizer": "AdamW",
            "loss_function": "GaussianNLL",
            "model": "ConvNeXt-Large",
        })
        
        return tracker
    
    def load_data(
        self,
        data_dir: str = "/tmp/heimdall_training_data",
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load training and validation data.
        
        Loads IQ recordings from MinIO and ground truth from PostgreSQL,
        creates train/val split, and returns PyTorch DataLoaders.
        
        Args:
            data_dir (str): Directory containing preprocessed data
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("loading_dataset", data_dir=data_dir)
        
        # Create dataset
        dataset = HeimdallDataset(
            data_dir=data_dir,
            split="all",
            augmentation=True,
        )
        
        dataset_size = len(dataset)
        val_size = int(dataset_size * self.validation_split)
        train_size = dataset_size - val_size
        
        logger.info(
            "dataset_loaded",
            total_samples=dataset_size,
            train_samples=train_size,
            val_samples=val_size,
        )
        
        # Split into train/val
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # Reproducible split
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        logger.info(
            "dataloaders_created",
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            batch_size=self.batch_size,
        )
        
        return train_loader, val_loader
    
    def create_lightning_module(self) -> LocalizationLightningModule:
        """
        Create PyTorch Lightning module for training.
        
        Returns:
            LocalizationLightningModule instance configured for training
        """
        logger.info(
            "creating_lightning_module",
            learning_rate=self.learning_rate,
        )
        
        # Initialize model
        model = LocalizationNet()
        
        # Wrap in Lightning module
        lightning_module = LocalizationLightningModule(
            model=model,
            learning_rate=self.learning_rate,
        )
        
        return lightning_module
    
    def create_trainer(
        self,
    ) -> pl.Trainer:
        """
        Create PyTorch Lightning trainer with callbacks.
        
        Callbacks:
        - ModelCheckpoint: Save top 3 models by validation loss
        - EarlyStopping: Stop if val_loss doesn't improve for 10 epochs
        - LearningRateMonitor: Track learning rate in MLflow
        
        Returns:
            Configured Trainer instance
        """
        logger.info(
            "creating_trainer",
            accelerator=self.accelerator,
            devices=self.devices,
            epochs=self.epochs,
        )
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="localization-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,  # Keep top 3 models
            verbose=True,
        )
        
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True,
        )
        
        lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                lr_monitor_callback,
            ],
            log_every_n_steps=10,
            enable_checkpointing=True,
            enable_model_summary=True,
        )
        
        return trainer
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Path:
        """
        Execute training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
        
        Returns:
            Path to best checkpoint
        """
        logger.info("starting_training", epochs=self.epochs)
        
        # Create Lightning module and trainer
        lightning_module = self.create_lightning_module()
        trainer = self.create_trainer()
        
        # Train
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        # Get best checkpoint path
        best_checkpoint_path = trainer.checkpoint_callback.best_model_path
        
        logger.info(
            "training_complete",
            best_checkpoint=best_checkpoint_path,
            best_val_loss=trainer.checkpoint_callback.best_model_score,
        )
        
        # Log final metrics to MLflow
        self.mlflow_tracker.log_metric("best_val_loss", float(trainer.checkpoint_callback.best_model_score))
        self.mlflow_tracker.log_metric("final_epoch", trainer.current_epoch)
        
        return Path(best_checkpoint_path)
    
    def export_and_register(
        self,
        best_checkpoint_path: Path,
        model_name: str = "heimdall-localization-onnx",
    ) -> Dict[str, Any]:
        """
        Export best model to ONNX and register with MLflow.
        
        Pipeline:
        1. Load best checkpoint from training
        2. Export to ONNX format
        3. Validate ONNX (shape, accuracy)
        4. Upload to MinIO
        5. Register with MLflow Model Registry
        
        Args:
            best_checkpoint_path (Path): Path to best checkpoint from training
            model_name (str): Name for ONNX model in MLflow registry
        
        Returns:
            Dict with export results (ONNX path, S3 URI, model version, etc.)
        """
        logger.info(
            "exporting_model",
            checkpoint=str(best_checkpoint_path),
            model_name=model_name,
        )
        
        # Load checkpoint
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        
        # Create model and load state dict
        model = LocalizationNet()
        
        # Handle Lightning checkpoint format
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove "model." prefix added by Lightning
            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        # Export and register
        result = export_and_register_model(
            pytorch_model=model,
            run_id=self.mlflow_tracker.active_run_id,
            s3_client=self.s3_client,
            mlflow_tracker=self.mlflow_tracker,
            model_name=model_name,
        )
        
        # Log export results to MLflow
        self.mlflow_tracker.log_params({
            "onnx_model_name": result.get("model_name", "unknown"),
            "onnx_model_version": result.get("model_version", "unknown"),
            "onnx_file_size_mb": result.get("metadata", {}).get("file_size_mb", 0),
        })
        
        logger.info(
            "model_exported_and_registered",
            model_name=result.get("model_name"),
            model_version=result.get("model_version"),
            s3_uri=result.get("s3_uri"),
        )
        
        return result
    
    def run(
        self,
        data_dir: str = "/tmp/heimdall_training_data",
        export_only: bool = False,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        
        Pipeline:
        1. Load data (if not export_only)
        2. Train model (if not export_only)
        3. Export best checkpoint to ONNX
        4. Register with MLflow
        
        Args:
            data_dir (str): Directory containing training data
            export_only (bool): Skip training, only export from checkpoint
            checkpoint_path (Optional[Path]): Path to checkpoint (for export_only=True)
        
        Returns:
            Dict with pipeline results
        """
        start_time = datetime.now()
        
        try:
            if export_only and checkpoint_path:
                # Only export and register existing checkpoint
                logger.info("running_export_only_mode", checkpoint=str(checkpoint_path))
                result = self.export_and_register(
                    best_checkpoint_path=checkpoint_path,
                )
            else:
                # Full training pipeline
                logger.info("running_full_training_pipeline", data_dir=data_dir)
                
                # 1. Load data
                train_loader, val_loader = self.load_data(data_dir=data_dir)
                
                # 2. Train
                best_checkpoint = self.train(train_loader, val_loader)
                
                # 3. Export and register
                result = self.export_and_register(best_checkpoint_path=best_checkpoint)
            
            # Calculate elapsed time
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Log final status
            logger.info(
                "pipeline_complete",
                elapsed_seconds=elapsed_time,
                success=result.get("success", False),
            )
            
            # Finalize MLflow run
            self.mlflow_tracker.end_run()
            
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "export_result": result,
            }
        
        except Exception as e:
            logger.error(
                "pipeline_error",
                error=str(e),
                exc_info=True,
            )
            
            # End MLflow run with failed status
            self.mlflow_tracker.end_run(status="FAILED")
            
            raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Training entry point for Heimdall RF localization pipeline",
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--validation_split",
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    
    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/heimdall_training_data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    
    # Device parameters
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Training accelerator (default: gpu)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    
    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/heimdall_checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to existing checkpoint (for resume or export)",
    )
    
    # Mode parameters
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Skip training, only export and register checkpoint",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume training from checkpoint",
    )
    
    # MLflow parameters
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="heimdall-localization",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="rf-localization",
        help="Prefix for MLflow run name",
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for training pipeline.
    """
    # Parse arguments
    args = parse_arguments()
    
    logger.info(
        "training_pipeline_started",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        accelerator=args.accelerator,
    )
    
    try:
        # Create pipeline
        pipeline = TrainingPipeline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            experiment_name=args.experiment_name,
            run_name_prefix=args.run_name_prefix,
        )
        
        # Run pipeline
        result = pipeline.run(
            data_dir=args.data_dir,
            export_only=args.export_only,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        )
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETE")
        print("="*80)
        print(f"Success: {result['success']}")
        print(f"Elapsed Time: {result['elapsed_time']:.2f} seconds")
        print(f"Export Result: {json.dumps(result['export_result'], indent=2)}")
        print("="*80 + "\n")
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(
            "training_pipeline_failed",
            error=str(e),
            exc_info=True,
        )
        print(f"\nTraining pipeline failed: {str(e)}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
