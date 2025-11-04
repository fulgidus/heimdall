"""
Celery tasks for training pipeline operations.

Tasks:
- start_training_job: Start model training
- generate_synthetic_data_task: Generate synthetic training data
- evaluate_model_task: Evaluate trained model
- export_model_onnx_task: Export model to ONNX
"""

import uuid
import json
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from io import BytesIO

from celery import Task, shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


class TrainingTask(Task):
    """Base task for training operations."""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 2}
    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True


@shared_task(bind=True, base=TrainingTask)
def start_training_job(self, job_id: str):
    """
    Start training job with real PyTorch triangulation model.
    
    This task:
    1. Loads training configuration and dataset
    2. Initializes TriangulationModel
    3. Creates train/val dataloaders
    4. Trains with Adam + cosine annealing
    5. Implements early stopping and gradient clipping
    6. Saves checkpoints to MinIO
    7. Logs metrics to database
    
    Args:
        job_id: Training job UUID
    
    Returns:
        Dict with training results
    """
    # Import backend storage modules
    sys.path.insert(0, '/app/backend/src')
    from storage.db_manager import get_db_manager
    from storage.minio_client import MinIOClient
    from config import settings as backend_settings
    from events.publisher import get_event_publisher
    from sqlalchemy import text

    logger.info(f"Starting training job {job_id}")
    
    # Initialize event publisher for real-time updates
    event_publisher = get_event_publisher()

    db_manager = get_db_manager()
    
    # Initialize session variables for dataloaders (will be closed in finally block)
    train_session = None
    val_session = None

    try:
        # Update job status to running
        with db_manager.get_session() as session:
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'running', started_at = NOW()
                WHERE id = :job_id
            """)
            session.execute(update_query, {"job_id": job_id})
            session.commit()

        # Load job configuration
        with db_manager.get_session() as session:
            config_query = text("SELECT config FROM heimdall.training_jobs WHERE id = :job_id")
            result = session.execute(config_query, {"job_id": job_id}).fetchone()
            if not result:
                raise ValueError(f"Job {job_id} not found")
            # Config is already a dict from JSONB column
            config = result[0] if isinstance(result[0], dict) else json.loads(result[0])

        # Extract configuration
        dataset_ids = config.get("dataset_ids")
        if not dataset_ids or len(dataset_ids) == 0:
            raise ValueError("dataset_ids is required in training configuration and must contain at least one dataset")
        
        # For backward compatibility with single dataset_id
        if not dataset_ids and config.get("dataset_id"):
            dataset_ids = [config.get("dataset_id")]

        batch_size = config.get("batch_size", 32)
        num_workers = config.get("num_workers", 4)
        epochs = config.get("epochs", 100)
        learning_rate = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        dropout_rate = config.get("dropout_rate", 0.2)
        early_stop_patience = config.get("early_stop_patience", 20)
        early_stop_delta = config.get("early_stop_delta", 0.001)
        max_grad_norm = config.get("max_grad_norm", 1.0)
        max_gdop = config.get("max_gdop", 5.0)

        logger.info(f"Training config: datasets={dataset_ids}, epochs={epochs}, batch={batch_size}, lr={learning_rate}")

        # Import training components (using absolute imports from /app)
        from src.models.triangulator import TriangulationModel, gaussian_nll_loss, haversine_distance_torch
        from src.data.triangulation_dataloader import create_triangulation_dataloader

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() and config.get("accelerator", "cpu") == "gpu" else "cpu")
        logger.info(f"Using device: {device}")

        # Create persistent sessions for dataloaders (will be closed in finally block)
        # Note: We cannot use context managers here because dataloaders need sessions
        # to stay alive during the entire training loop
        logger.info("Creating train dataloader...")
        train_session = db_manager.SessionLocal()
        train_loader = create_triangulation_dataloader(
            dataset_ids=dataset_ids,
            split="train",
            db_session=train_session,
            batch_size=batch_size,
            num_workers=0,  # Avoid multiprocessing issues with DB connections
            shuffle=True,
            max_receivers=7
        )

        logger.info("Creating validation dataloader...")
        val_session = db_manager.SessionLocal()
        val_loader = create_triangulation_dataloader(
            dataset_ids=dataset_ids,
            split="val",
            db_session=val_session,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            max_receivers=7
        )

        # Update dataset info in job
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        with db_manager.get_session() as session:
            session.execute(
                text("UPDATE heimdall.training_jobs SET train_samples = :train, val_samples = :val, dataset_size = :total WHERE id = :job_id"),
                {"train": train_samples, "val": val_samples, "total": train_samples + val_samples, "job_id": job_id}
            )
            session.commit()

        logger.info(f"Dataset loaded: {train_samples} train, {val_samples} val samples")

        # Publish training started event
        event_publisher.publish_training_started(
            job_id=job_id,
            config=config,
            dataset_size=train_samples + val_samples,
            train_samples=train_samples,
            val_samples=val_samples
        )

        # Initialize model
        model = TriangulationModel(
            encoder_input_dim=6,
            encoder_hidden_dim=64,
            encoder_output_dim=32,
            attention_heads=4,
            head_hidden_dim=64,
            dropout=dropout_rate
        ).to(device)

        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

        # Initialize MinIO client for checkpoints
        minio_client = MinIOClient(
            endpoint_url=backend_settings.minio_url,
            access_key=backend_settings.minio_access_key,
            secret_key=backend_settings.minio_secret_key,
            bucket_name="models"
        )
        minio_client.ensure_bucket_exists()

        # Training state
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        resume_epoch = 0
        
        # Check for pause checkpoint to resume from
        with db_manager.get_session() as session:
            pause_query = text("SELECT pause_checkpoint_path FROM heimdall.training_jobs WHERE id = :job_id")
            pause_result = session.execute(pause_query, {"job_id": job_id}).fetchone()
            pause_checkpoint_path = pause_result[0] if pause_result and pause_result[0] else None
        
        if pause_checkpoint_path:
            logger.info(f"Resuming from pause checkpoint: {pause_checkpoint_path}")
            try:
                # Download checkpoint from MinIO
                checkpoint_key = pause_checkpoint_path.replace("s3://models/", "")
                response = minio_client.s3_client.get_object(Bucket="models", Key=checkpoint_key)
                checkpoint_buffer = BytesIO(response['Body'].read())
                checkpoint = torch.load(checkpoint_buffer, map_location=device)
                
                # Restore model, optimizer, and scheduler state
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                resume_epoch = checkpoint['epoch']
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_epoch = checkpoint.get('best_epoch', 0)
                patience_counter = checkpoint.get('patience_counter', 0)
                
                logger.info(f"Resumed from epoch {resume_epoch}, best_val_loss={best_val_loss:.4f}")
                
                # Clear pause checkpoint path after successful resume
                with db_manager.get_session() as session:
                    session.execute(
                        text("UPDATE heimdall.training_jobs SET pause_checkpoint_path = NULL WHERE id = :job_id"),
                        {"job_id": job_id}
                    )
                    session.commit()
            except Exception as e:
                logger.warning(f"Failed to load pause checkpoint: {e}. Starting from scratch.")
                resume_epoch = 0
        
        # Training loop
        for epoch in range(resume_epoch + 1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_distance_sum = 0.0
            train_batches = 0

            # Training phase
            for batch in train_loader:
                receiver_features = batch["receiver_features"].to(device)
                signal_mask = batch["signal_mask"].to(device)
                target_position = batch["target_position"].to(device)

                optimizer.zero_grad()

                # Forward pass
                position, log_variance = model(receiver_features, signal_mask)

                # Calculate loss
                loss = gaussian_nll_loss(position, log_variance, target_position)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                # Calculate distance error for monitoring
                with torch.no_grad():
                    distances = haversine_distance_torch(
                        position[:, 0], position[:, 1],
                        target_position[:, 0], target_position[:, 1]
                    )
                    train_distance_sum += distances.mean().item()

                train_loss_sum += loss.item()
                train_batches += 1

            train_loss = train_loss_sum / train_batches
            train_rmse = train_distance_sum / train_batches

            # Validation phase
            model.eval()
            val_loss_sum = 0.0
            val_distance_sum = 0.0
            val_distance_good_geom_sum = 0.0
            val_good_geom_count = 0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    receiver_features = batch["receiver_features"].to(device)
                    signal_mask = batch["signal_mask"].to(device)
                    target_position = batch["target_position"].to(device)
                    gdop = batch["metadata"]["gdop"]

                    position, log_variance = model(receiver_features, signal_mask)
                    loss = gaussian_nll_loss(position, log_variance, target_position)

                    distances = haversine_distance_torch(
                        position[:, 0], position[:, 1],
                        target_position[:, 0], target_position[:, 1]
                    )

                    val_loss_sum += loss.item()
                    val_distance_sum += distances.mean().item()

                    # Track GDOP<5 subset for success criteria
                    good_geom_mask = gdop < max_gdop
                    if good_geom_mask.any():
                        val_distance_good_geom_sum += distances[good_geom_mask].mean().item()
                        val_good_geom_count += 1

                    val_batches += 1

            val_loss = val_loss_sum / val_batches
            val_rmse = val_distance_sum / val_batches
            val_rmse_good_geom = (val_distance_good_geom_sum / val_good_geom_count) if val_good_geom_count > 0 else val_rmse

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update learning rate
            scheduler.step()

            # Log to database
            with db_manager.get_session() as session:
                # Update job progress
                session.execute(
                    text("""
                        UPDATE heimdall.training_jobs
                        SET current_epoch = :epoch,
                            progress_percent = :progress,
                            train_loss = :train_loss,
                            val_loss = :val_loss,
                            train_accuracy = :train_rmse,
                            val_accuracy = :val_rmse,
                            learning_rate = :lr,
                            updated_at = NOW()
                        WHERE id = :job_id
                    """),
                    {
                        "epoch": epoch,
                        "progress": (epoch / epochs) * 100.0,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_rmse": train_rmse,
                        "val_rmse": val_rmse,
                        "lr": current_lr,
                        "job_id": job_id
                    }
                )

                # Store epoch metrics
                session.execute(
                    text("""
                        INSERT INTO heimdall.training_metrics (
                            training_job_id, epoch, train_loss, val_loss,
                            train_accuracy, val_accuracy, learning_rate, phase, timestamp
                        )
                        VALUES (
                            :job_id, :epoch, :train_loss, :val_loss,
                            :train_rmse, :val_rmse, :lr, 'train', NOW()
                        )
                    """),
                    {
                        "job_id": job_id,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_rmse": train_rmse,
                        "val_rmse": val_rmse,
                        "lr": current_lr
                    }
                )
                session.commit()

            logger.info(
                f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"train_rmse={train_rmse:.1f}m, val_rmse={val_rmse:.1f}m, val_rmse_gdop<{max_gdop}={val_rmse_good_geom:.1f}m, lr={current_lr:.6f}"
            )

            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'job_id': job_id,
                    'current_epoch': epoch,
                    'total_epochs': epochs,
                    'progress_percent': (epoch / epochs) * 100.0,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse
                }
            )

            # Publish training progress event
            event_publisher.publish_training_progress(
                job_id=job_id,
                epoch=epoch,
                total_epochs=epochs,
                metrics={
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'train_rmse': float(train_rmse),
                    'val_rmse': float(val_rmse),
                    'val_rmse_good_geom': float(val_rmse_good_geom),
                    'learning_rate': float(current_lr)
                },
                is_best=False  # Will update below if this is best
            )

            # Check for best model
            is_best = val_loss < (best_val_loss - early_stop_delta)
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save best model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'config': config
                }

                buffer = BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)

                checkpoint_path = f"checkpoints/{job_id}/best_model.pth"
                minio_client.s3_client.put_object(
                    Bucket="models",
                    Key=checkpoint_path,
                    Body=buffer.getvalue(),
                    ContentType="application/octet-stream"
                )

                # Update job with best model info
                with db_manager.get_session() as session:
                    session.execute(
                        text("UPDATE heimdall.training_jobs SET best_epoch = :epoch, best_val_loss = :loss, checkpoint_path = :path WHERE id = :job_id"),
                        {"epoch": epoch, "loss": val_loss, "path": f"s3://models/{checkpoint_path}", "job_id": job_id}
                    )
                    session.commit()

                logger.info(f"Saved best model checkpoint at epoch {epoch}")
                
                # Publish updated progress event with is_best=True
                event_publisher.publish_training_progress(
                    job_id=job_id,
                    epoch=epoch,
                    total_epochs=epochs,
                    metrics={
                        'train_loss': float(train_loss),
                        'val_loss': float(val_loss),
                        'train_rmse': float(train_rmse),
                        'val_rmse': float(val_rmse),
                        'val_rmse_good_geom': float(val_rmse_good_geom),
                        'learning_rate': float(current_lr)
                    },
                    is_best=True
                )
            else:
                patience_counter += 1

            # Save periodic checkpoints (every 10 epochs)
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'config': config
                }

                buffer = BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)

                checkpoint_path = f"checkpoints/{job_id}/epoch_{epoch}.pth"
                minio_client.s3_client.put_object(
                    Bucket="models",
                    Key=checkpoint_path,
                    Body=buffer.getvalue(),
                    ContentType="application/octet-stream"
                )
                logger.info(f"Saved checkpoint at epoch {epoch}")

            # Early stopping check
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={early_stop_patience})")
                break
            
            # Check for pause request
            with db_manager.get_session() as session:
                status_query = text("SELECT status FROM heimdall.training_jobs WHERE id = :job_id")
                status_result = session.execute(status_query, {"job_id": job_id}).fetchone()
                current_status = status_result[0] if status_result else None
            
            if current_status == 'paused':
                logger.info(f"Pause requested at epoch {epoch}. Saving pause checkpoint...")
                
                # Save pause checkpoint with full training state
                pause_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'config': config
                }
                
                buffer = BytesIO()
                torch.save(pause_checkpoint, buffer)
                buffer.seek(0)
                
                pause_checkpoint_path = f"checkpoints/{job_id}/pause_checkpoint.pth"
                minio_client.s3_client.put_object(
                    Bucket="models",
                    Key=pause_checkpoint_path,
                    Body=buffer.getvalue(),
                    ContentType="application/octet-stream"
                )
                
                # Update job with pause checkpoint path
                with db_manager.get_session() as session:
                    session.execute(
                        text("UPDATE heimdall.training_jobs SET pause_checkpoint_path = :path WHERE id = :job_id"),
                        {"path": f"s3://models/{pause_checkpoint_path}", "job_id": job_id}
                    )
                    session.commit()
                
                logger.info(f"Training paused at epoch {epoch}. Checkpoint saved to {pause_checkpoint_path}")
                
                # Publish pause event
                event_publisher.publish_training_completed(
                    job_id=job_id,
                    status='paused',
                    checkpoint_path=f"s3://models/{pause_checkpoint_path}"
                )
                
                return {
                    "status": "paused",
                    "job_id": job_id,
                    "paused_at_epoch": epoch,
                    "pause_checkpoint_path": f"s3://models/{pause_checkpoint_path}"
                }
        
        # Save final checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'config': config
        }

        buffer = BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        final_checkpoint_path = f"checkpoints/{job_id}/final_model.pth"
        minio_client.s3_client.put_object(
            Bucket="models",
            Key=final_checkpoint_path,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream"
        )

        # Save model metadata to models table
        model_id = uuid.uuid4()
        with db_manager.get_session() as session:
            # Store first dataset_id in synthetic_dataset_id column (has FK constraint)
            # Full list is preserved in hyperparameters JSON
            if not dataset_ids or len(dataset_ids) == 0:
                raise ValueError("No dataset_ids available for model insertion")
            primary_dataset_id = dataset_ids[0]
            
            session.execute(
                text("""
                    INSERT INTO heimdall.models (
                        id, model_name, version, model_type, synthetic_dataset_id,
                        pytorch_model_location, accuracy_meters, accuracy_sigma_meters,
                        loss_value, epoch, is_active, is_production,
                        hyperparameters, training_metrics, trained_by_job_id
                    )
                    VALUES (
                        :id, :name, 1, 'triangulation', :dataset_id,
                        :location, :rmse, :rmse_good, :loss, :epoch,
                        FALSE, FALSE, CAST(:hyperparams AS jsonb), CAST(:metrics AS jsonb), :job_id
                    )
                """),
                {
                    "id": str(model_id),
                    "name": f"triangulation_job_{job_id}",
                    "dataset_id": primary_dataset_id,
                    "location": f"s3://models/{final_checkpoint_path}",
                    "rmse": val_rmse,
                    "rmse_good": val_rmse_good_geom,
                    "loss": val_loss,
                    "epoch": epoch,
                    "hyperparams": json.dumps(config),
                    "metrics": json.dumps({"best_epoch": best_epoch, "best_val_loss": float(best_val_loss), "final_val_rmse": val_rmse}),
                    "job_id": job_id
                }
            )
            session.commit()

        # Update job as completed
        with db_manager.get_session() as session:
            session.execute(
                text("""
                    UPDATE heimdall.training_jobs
                    SET status = 'completed', completed_at = NOW(), progress_percent = 100.0
                    WHERE id = :job_id
                """),
                {"job_id": job_id}
            )
            session.commit()

        logger.info(f"Training job {job_id} completed successfully. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
        
        # Publish training completed event
        event_publisher.publish_training_completed(
            job_id=job_id,
            status='completed',
            best_epoch=best_epoch,
            best_val_loss=float(best_val_loss),
            checkpoint_path=f"s3://models/checkpoints/{job_id}/best_model.pth",
            onnx_model_path=None,
            mlflow_run_id=None
        )
        
        return {
            "status": "completed",
            "job_id": job_id,
            "model_id": str(model_id),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "final_val_rmse": val_rmse
        }

    except Exception as e:
        logger.error(f"Error in training job {job_id}: {e}", exc_info=True)

        # Publish training failed event
        try:
            event_publisher.publish_training_completed(
                job_id=job_id,
                status='failed',
                error_message=str(e)
            )
        except Exception as pub_error:
            logger.error(f"Failed to publish failure event: {pub_error}")

        # Update job as failed
        try:
            with db_manager.get_session() as session:
                session.execute(
                    text("""
                        UPDATE heimdall.training_jobs
                        SET status = 'failed', completed_at = NOW(), error_message = :error
                        WHERE id = :job_id
                    """),
                    {"job_id": job_id, "error": str(e)}
                )
                session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")

        raise
    
    finally:
        # Clean up database sessions used by dataloaders
        if train_session is not None:
            try:
                train_session.close()
                logger.debug("Closed train dataloader session")
            except Exception as e:
                logger.warning(f"Error closing train session: {e}")
        
        if val_session is not None:
            try:
                val_session.close()
                logger.debug("Closed val dataloader session")
            except Exception as e:
                logger.warning(f"Error closing val session: {e}")


@shared_task(bind=True, base=TrainingTask)
def generate_synthetic_data_task(self, job_id: str):
    """
    Generate synthetic training data with IQ samples and feature extraction.
    
    This task:
    1. Loads training configuration from database
    2. Generates IQ samples using RF propagation
    3. Extracts features from IQ samples
    4. Saves features to measurement_features table
    5. Saves first 100 IQ samples to MinIO
    6. Updates job status
    
    Args:
        job_id: Training job UUID
    
    Returns:
        Dict with generation results
    """
    import asyncio

    # Import backend storage modules
    sys.path.insert(0, '/app/backend/src')
    from storage.db_manager import get_db_manager
    from storage.minio_client import MinIOClient
    from config import settings as backend_settings
    from sqlalchemy import text

    logger.info(f"Starting synthetic data generation job {job_id}")

    db_manager = get_db_manager()

    try:
        # Load job configuration
        with db_manager.get_session() as session:
            load_query = text("""
                SELECT config FROM heimdall.training_jobs
                WHERE id = :job_id
            """)
            result = session.execute(load_query, {"job_id": job_id}).fetchone()

            if not result:
                raise ValueError(f"Job {job_id} not found")

            # Config is already a dict from JSONB column
            config = result[0] if isinstance(result[0], dict) else json.loads(result[0])

        # Update status to running and initialize progress tracking
        with db_manager.get_session() as session:
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'running',
                    started_at = NOW(),
                    current_progress = 0,
                    total_progress = :total_samples,
                    progress_message = 'Starting synthetic data generation...'
                WHERE id = :job_id
            """)
            session.execute(update_query, {
                "job_id": job_id,
                "total_samples": config['num_samples']
            })
            session.commit()

        # Generate synthetic data - import training service modules
        from src.data.config import TrainingConfig, get_italian_receivers, generate_random_receivers, BoundingBox
        from src.data.synthetic_generator import generate_synthetic_data_with_iq
        from common.terrain import TerrainLookup

        # Determine receiver generation strategy
        use_random = config.get('use_random_receivers', True)

        if use_random:
            # Generate random receivers
            minio_client = MinIOClient(
                endpoint_url=backend_settings.minio_url,
                access_key=backend_settings.minio_access_key,
                secret_key=backend_settings.minio_secret_key
            )

            terrain = TerrainLookup(use_srtm=True, minio_client=minio_client)

            # Define area based on config or default to Italian region
            area_bbox = BoundingBox(
                lat_min=config.get('area_lat_min', 44.0),
                lat_max=config.get('area_lat_max', 46.0),
                lon_min=config.get('area_lon_min', 7.0),
                lon_max=config.get('area_lon_max', 10.0)
            )

            # Generate random number of receivers
            min_rx = config.get('min_receivers_count', 4)
            max_rx = config.get('max_receivers_count', 10)
            import numpy as np
            num_rx = np.random.randint(min_rx, max_rx + 1)

            receivers = generate_random_receivers(
                bbox=area_bbox,
                num_receivers=num_rx,
                terrain_lookup=terrain,
                seed=config.get('receiver_seed')
            )

            logger.info(f"Generated {num_rx} random receivers for training")
        else:
            # Use fixed Italian receivers for backward compatibility
            receivers = get_italian_receivers()
            logger.info("Using fixed Italian receivers")

        training_config = TrainingConfig.from_receivers(receivers, margin_degrees=0.5)

        # Handle continuation vs new dataset
        is_continuation = config.get('is_continuation', False)
        samples_offset = config.get('samples_offset', 0)
        
        if is_continuation:
            # Reuse existing dataset
            dataset_id_str = config.get('existing_dataset_id')
            if not dataset_id_str:
                raise ValueError("is_continuation=True but no existing_dataset_id provided")
            
            dataset_id = uuid.UUID(dataset_id_str)
            logger.info(f"Continuing dataset {dataset_id} from {samples_offset} samples")
        else:
            # Create dataset record (or get existing by name)
            dataset_id = uuid.uuid4()
            with db_manager.get_session() as session:
                # Check if dataset with this name already exists
                check_query = text("""
                    SELECT id FROM heimdall.synthetic_datasets WHERE name = :name
                """)
                existing = session.execute(check_query, {"name": config['name']}).fetchone()

                if existing:
                    dataset_id = existing[0]
                    logger.info(f"Using existing dataset {dataset_id}")
                else:
                    # Create new dataset
                    dataset_query = text("""
                        INSERT INTO heimdall.synthetic_datasets (
                            id, name, description, num_samples,
                            config, created_by_job_id
                        )
                        VALUES (
                            :id, :name, :description, 0,
                            CAST(:config AS jsonb), :job_id
                        )
                    """)

                    session.execute(
                        dataset_query,
                        {
                            "id": str(dataset_id),
                            "name": config['name'],
                            "description": config.get('description'),
                            "config": json.dumps(config),
                            "job_id": job_id
                        }
                    )
                    session.commit()
                    logger.info(f"Created new dataset {dataset_id}")

        # Initialize event publisher for real-time WebSocket updates
        from backend.src.events.publisher import get_event_publisher
        event_publisher = get_event_publisher()

        # Progress callback (async wrapper for Celery)
        # For continuations, show cumulative progress
        async def progress_callback(current, total):
            # Add offset for continuation jobs
            cumulative_current = samples_offset + current
            # Note: total_progress in DB was set to original total (not remaining)
            # so we calculate progress based on cumulative progress
            total_samples = samples_offset + total if is_continuation else total
            progress_pct = (cumulative_current / total_samples) * 100
            
            if is_continuation:
                message = f'Processing {cumulative_current}/{total_samples} samples (continued from {samples_offset})'
            else:
                message = f'Processing {current}/{total} samples'

            logger.info(f"[PROGRESS] {message} ({progress_pct:.1f}%)")

            # Update Celery state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': cumulative_current if is_continuation else current,
                    'total': total_samples,
                    'message': message,
                    'progress_percent': progress_pct
                }
            )

            # Update database with progress
            try:
                with db_manager.get_session() as session:
                    session.execute(
                        text("""
                            UPDATE heimdall.training_jobs
                            SET current_progress = :current,
                                total_progress = :total,
                                progress_percent = :progress,
                                progress_message = :message
                            WHERE id = :job_id
                        """),
                        {
                            "current": cumulative_current if is_continuation else current,
                            "total": total_samples,
                            "progress": progress_pct,
                            "message": message,
                            "job_id": job_id
                        }
                    )
                    session.commit()
                    logger.info(f"[PROGRESS] Database updated: {cumulative_current if is_continuation else current}/{total_samples}")
            except Exception as e:
                logger.error(f"Failed to update progress in database: {e}", exc_info=True)

            # Publish progress event to RabbitMQ for WebSocket clients
            try:
                event_publisher.publish_dataset_generation_progress(
                    job_id=job_id,
                    current=cumulative_current if is_continuation else current,
                    total=total_samples,
                    message=message
                )
            except Exception as e:
                logger.error(f"Failed to publish progress event: {e}", exc_info=True)

        # Get async database pool
        from sqlalchemy.ext.asyncio import create_async_engine
        from config import settings as backend_settings

        async_engine = create_async_engine(
            backend_settings.database_url.replace('postgresql://', 'postgresql+asyncpg://'),
            echo=False
        )

        # Generate samples with IQ generation and feature extraction
        if is_continuation:
            logger.info(f"Continuing generation: {config['num_samples']} remaining samples "
                       f"(total: {samples_offset + config['num_samples']})")
        else:
            logger.info(f"Generating {config['num_samples']} synthetic samples with IQ generation")

        async def run_generation():
            async with async_engine.begin() as conn:
                stats = await generate_synthetic_data_with_iq(
                    dataset_id=dataset_id,
                    num_samples=config['num_samples'],
                    receivers_config=receivers,
                    training_config=training_config,
                    config=config,
                    conn=conn,
                    progress_callback=progress_callback,
                    seed=config.get('seed'),
                    job_id=job_id  # Pass job_id for cancellation detection
                )
            return stats

        # Run async generation
        stats = asyncio.run(run_generation())

        logger.info(f"Generation complete: {stats['total_generated']} samples, "
                    f"{stats['iq_samples_saved']} IQ samples saved to MinIO")

        # Update dataset record with final counts
        # Count actual samples saved to measurement_features table using dataset_id
        with db_manager.get_session() as session:
            # Count actual samples in measurement_features
            count_query = text("""
                SELECT COUNT(*) FROM heimdall.measurement_features
                WHERE dataset_id = :dataset_id
            """)
            actual_count = session.execute(count_query, {"dataset_id": str(dataset_id)}).scalar() or 0

            # Update dataset metadata
            update_query = text("""
                UPDATE heimdall.synthetic_datasets
                SET num_samples = :num_samples
                WHERE id = :id
            """)
            session.execute(
                update_query,
                {
                    "id": str(dataset_id),
                    "num_samples": actual_count
                }
            )
            session.commit()
            logger.info(f"Updated dataset {dataset_id} with {actual_count} samples (generated: {stats['total_generated']})")

        # Update job as completed
        with db_manager.get_session() as session:
            complete_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'completed', completed_at = NOW(), progress_percent = 100.0
                WHERE id = :job_id
            """)
            session.execute(complete_query, {"job_id": job_id})
            session.commit()

        logger.info(f"Synthetic data generation job {job_id} completed")

        return {
            "status": "completed",
            "job_id": job_id,
            "dataset_id": str(dataset_id),
            "num_samples": stats['total_generated'],
            "iq_samples_saved": stats['iq_samples_saved'],
            "success_rate": stats['success_rate']
        }

    except Exception as e:
        logger.error(f"Error in synthetic data generation job {job_id}: {e}", exc_info=True)

        # Update job as failed
        with db_manager.get_session() as session:
            fail_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'failed', completed_at = NOW(), error_message = :error
                WHERE id = :job_id
            """)
            session.execute(fail_query, {"job_id": job_id, "error": str(e)})
            session.commit()

        raise


@shared_task(bind=True, base=TrainingTask)
def evaluate_model_task(self, model_id: str, dataset_id: str = None):
    """
    Evaluate trained model.
    
    Args:
        model_id: Model UUID
        dataset_id: Optional dataset UUID (uses test split)
    
    Returns:
        Dict with evaluation results
    """
    logger.info(f"Evaluating model {model_id}")

    # Placeholder for Phase 5
    logger.warning("Model evaluation not implemented yet")

    return {"status": "not_implemented", "model_id": model_id}


@shared_task(bind=True, base=TrainingTask)
def export_model_onnx_task(self, model_id: str, optimize: bool = True):
    """
    Export model to ONNX format.
    
    Args:
        model_id: Model UUID
        optimize: Apply ONNX optimizations
    
    Returns:
        Dict with export results
    """
    logger.info(f"Exporting model {model_id} to ONNX (optimize={optimize})")

    # Placeholder for Phase 5
    logger.warning("ONNX export not implemented yet")

    return {"status": "not_implemented", "model_id": model_id}
