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
import math
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from io import BytesIO

from celery import Task, shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


def get_optimal_num_workers():
    """
    Calculate optimal number of DataLoader workers based on CPU cores.
    
    Returns:
        int: Number of workers (min 4, max min(cpu_count, 16))
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        # Use 50-75% of available cores, but cap at 16 to avoid overhead
        # Reserve some cores for the main training process
        optimal = max(4, min(cpu_count // 2, 16))
        logger.info(f"Detected {cpu_count} CPU cores, using {optimal} DataLoader workers")
        return optimal
    except:
        logger.warning("Could not detect CPU count, defaulting to 4 workers")
        return 4


def sanitize_for_json(value):
    """
    Sanitize numeric values for JSON serialization.
    Converts NaN and Infinity to None (which becomes null in JSON).
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def sanitize_dict_for_json(data: dict) -> dict:
    """
    Recursively sanitize a dictionary for JSON serialization.
    Converts NaN and Infinity to None.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = sanitize_dict_for_json(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [sanitize_for_json(v) for v in value]
        else:
            result[key] = sanitize_for_json(value)
    return result


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
    from backend.src.events.publisher import get_event_publisher
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

        # Load job configuration and name
        with db_manager.get_session() as session:
            config_query = text("SELECT config, job_name FROM heimdall.training_jobs WHERE id = :job_id")
            result = session.execute(config_query, {"job_id": job_id}).fetchone()
            if not result:
                raise ValueError(f"Job {job_id} not found")
            # Config is already a dict from JSONB column
            config = result[0] if isinstance(result[0], dict) else json.loads(result[0])
            job_name = result[1]
            logger.info(f"Training job '{job_name}' (ID: {job_id})")

        # Extract configuration
        dataset_ids = config.get("dataset_ids")
        if not dataset_ids or len(dataset_ids) == 0:
            raise ValueError("dataset_ids is required in training configuration and must contain at least one dataset")
        
        # For backward compatibility with single dataset_id
        if not dataset_ids and config.get("dataset_id"):
            dataset_ids = [config.get("dataset_id")]

        batch_size = config.get("batch_size", 32)
        # Auto-detect optimal num_workers if not specified or set to 0
        num_workers_config = config.get("num_workers", 0)
        num_workers = get_optimal_num_workers() if num_workers_config == 0 else num_workers_config
        epochs = config.get("epochs", 100)
        learning_rate = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        dropout_rate = config.get("dropout_rate", 0.2)
        early_stop_patience = config.get("early_stop_patience", 20)
        early_stop_delta = config.get("early_stop_delta", 0.001)
        max_grad_norm = config.get("max_grad_norm", 1.0)
        max_gdop = config.get("max_gdop", 5.0)
        max_receivers = config.get("max_receivers", 7)  # Italian WebSDR system uses 7 receivers

        logger.info(f"Training config: datasets={dataset_ids}, epochs={epochs}, batch={batch_size}, lr={learning_rate}, workers={num_workers}")

        # Import training components (using absolute imports from /app)
        from src.models.triangulator import TriangulationModel, gaussian_nll_loss, haversine_distance_torch
        from src.data.triangulation_dataloader import create_triangulation_dataloader
        from src.data.gpu_cached_dataset import GPUCachedDataset
        from torch.utils.data import DataLoader

        # Determine device
        accelerator = config.get("accelerator", "auto")
        if accelerator == "auto":
            # Auto-detect: prefer GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif accelerator == "gpu":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                logger.warning("GPU requested but CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device} (accelerator={accelerator}, cuda_available={torch.cuda.is_available()})")

        # GPU-CACHED DATASET: Load ALL data to VRAM for 100% GPU utilization!
        preload_to_gpu = config.get("preload_to_gpu", True)
        
        if preload_to_gpu and device.type == "cuda":
            logger.info("🚀 GPU-CACHED MODE: Loading ALL data to VRAM for maximum GPU utilization!")
            
            # Create one-time session for dataset preloading
            with db_manager.get_session() as load_session:
                # Create GPU-cached datasets (loads all data to VRAM)
                logger.info("Loading train dataset to GPU...")
                train_dataset = GPUCachedDataset(
                    dataset_ids=dataset_ids,
                    split="train",
                    db_session=load_session,
                    device=device,
                    max_receivers=7,
                    preload_to_gpu=True
                )
                
                logger.info("Loading validation dataset to GPU...")
                val_dataset = GPUCachedDataset(
                    dataset_ids=dataset_ids,
                    split="val",
                    db_session=load_session,
                    device=device,
                    max_receivers=7,
                    preload_to_gpu=True
                )
            
            # Create DataLoaders (num_workers MUST be 0 for GPU-cached datasets)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Data already on GPU, no I/O needed
                pin_memory=False  # Data already on GPU
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            logger.info(f"✅ GPU-CACHED READY: {len(train_dataset)} train + {len(val_dataset)} val samples in VRAM")
            logger.info(f"💪 GPU will run at 100% utilization with ZERO I/O wait!")
            
        else:
            # FALLBACK: Traditional DB-based dataloader with file cache
            if not preload_to_gpu:
                logger.info("📦 NORMAL MODE: Using DB-based dataloaders with file cache")
            else:
                logger.info("⚠️  GPU not available, falling back to DB-based dataloaders")
            
            # IMPORTANT: DataLoader workers incompatible with DB sessions (pickling error)
            # Solution: Use num_workers=0 BUT enable file cache for 10-50x speedup on epoch 2+
            # - Epoch 1: Slow (DB load), saves to /tmp/heimdall_training_cache
            # - Epoch 2+: FAST (disk cache is 100x faster than DB queries)
            actual_num_workers = 0  # Force single-threaded to avoid DB session pickling issues
            if num_workers > 0:
                logger.info(f"Requested {num_workers} workers, but using 0 due to DB session constraints. Cache will make epoch 2+ FAST!")
            
            # Create persistent sessions for dataloaders (will be closed in finally block)
            # Note: We cannot use context managers here because dataloaders need sessions
            # to stay alive during the entire training loop
            logger.info(f"Creating train dataloader with file cache enabled...")
            train_session = db_manager.SessionLocal()
            train_loader = create_triangulation_dataloader(
                dataset_ids=dataset_ids,
                split="train",
                db_session=train_session,
                batch_size=batch_size,
                num_workers=actual_num_workers,
                shuffle=True,
                max_receivers=7,
                use_cache=True  # Enable file cache for massive speedup
            )

            logger.info(f"Creating validation dataloader with file cache enabled...")
            val_session = db_manager.SessionLocal()
            val_loader = create_triangulation_dataloader(
                dataset_ids=dataset_ids,
                split="val",
                db_session=val_session,
                batch_size=batch_size,
                num_workers=actual_num_workers,
                shuffle=False,
                max_receivers=7,
                use_cache=True  # Enable file cache
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
        
        # Check if this is an evolution job (load parent model weights)
        parent_model_id = config.get("parent_model_id")
        if parent_model_id:
            logger.info(f"Evolution mode: loading weights from parent model {parent_model_id}")
            try:
                with db_manager.get_session() as session:
                    parent_query = text("SELECT pytorch_model_location FROM heimdall.models WHERE id = :model_id")
                    parent_result = session.execute(parent_query, {"model_id": parent_model_id}).fetchone()
                    
                    if parent_result and parent_result[0]:
                        parent_checkpoint_path = parent_result[0]
                        checkpoint_key = parent_checkpoint_path.replace("s3://models/", "")
                        
                        # Download parent checkpoint from MinIO
                        response = minio_client.s3_client.get_object(Bucket="models", Key=checkpoint_key)
                        checkpoint_buffer = BytesIO(response['Body'].read())
                        parent_checkpoint = torch.load(checkpoint_buffer, map_location=device)
                        
                        # Load only model weights (not optimizer/scheduler - fresh start)
                        model.load_state_dict(parent_checkpoint['model_state_dict'])
                        logger.info(f"Loaded parent model weights from {parent_checkpoint_path}")
                        
                        # Optionally preserve best_val_loss from parent for comparison
                        # best_val_loss = parent_checkpoint.get('best_val_loss', float('inf'))
                    else:
                        logger.warning(f"Parent model {parent_model_id} has no checkpoint, starting from scratch")
            except Exception as e:
                logger.error(f"Failed to load parent model checkpoint: {e}. Starting from scratch.", exc_info=True)
        
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
            train_grad_norm_sum = 0.0
            
            # Track time for batch progress updates (send every ~1 second)
            import time
            last_batch_update_time = time.time()
            batch_update_interval = 1.0  # Send update every 1 second
            total_train_batches = len(train_loader)

            # Training phase
            for batch_idx, batch in enumerate(train_loader, 1):
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

                # Calculate gradient norm before clipping
                total_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                train_grad_norm_sum += total_grad_norm

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
                
                # Send batch-level progress update every ~1 second for real-time UI updates
                current_time = time.time()
                if current_time - last_batch_update_time >= batch_update_interval:
                    avg_batch_loss = train_loss_sum / train_batches
                    event_publisher.publish_training_batch_progress(
                        job_id=job_id,
                        epoch=epoch,
                        total_epochs=epochs,
                        batch=batch_idx,
                        total_batches=total_train_batches,
                        current_loss=avg_batch_loss,
                        phase='train'
                    )
                    last_batch_update_time = current_time

            train_loss = train_loss_sum / train_batches
            train_rmse = train_distance_sum / train_batches
            train_avg_grad_norm = train_grad_norm_sum / train_batches

            # Validation phase
            model.eval()
            val_loss_sum = 0.0
            val_distance_sum = 0.0
            val_distance_good_geom_sum = 0.0
            val_good_geom_count = 0
            val_batches = 0
            
            # Collect all validation distances and uncertainties for percentile/distribution calculations
            all_val_distances = []
            all_val_uncertainties = []
            all_val_gdop = []

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
                    
                    # Calculate predicted uncertainty (standard deviation from log_variance)
                    # log_variance has shape [batch, 2] for lat/lon
                    # Take mean across lat/lon dimensions for overall uncertainty
                    predicted_std = torch.exp(log_variance / 2.0).mean(dim=1)  # [batch]
                    
                    # Store for percentile calculations
                    all_val_distances.append(distances.cpu())
                    all_val_uncertainties.append(predicted_std.cpu())
                    all_val_gdop.append(gdop)

                    val_loss_sum += loss.item()
                    val_distance_sum += distances.mean().item()

                    # Track GDOP<5 subset for success criteria
                    good_geom_mask = gdop < max_gdop
                    if good_geom_mask.any():
                        val_distance_good_geom_sum += distances[good_geom_mask].mean().item()
                        val_good_geom_count += 1

                    val_batches += 1

            # Protect against division by zero in validation metrics
            if val_batches > 0:
                val_loss = val_loss_sum / val_batches
                val_rmse = val_distance_sum / val_batches
                val_rmse_good_geom = (val_distance_good_geom_sum / val_good_geom_count) if val_good_geom_count > 0 else val_rmse
                
                # Calculate advanced metrics from collected tensors
                all_val_distances_tensor = torch.cat(all_val_distances)  # [total_val_samples]
                all_val_uncertainties_tensor = torch.cat(all_val_uncertainties)  # [total_val_samples]
                all_val_gdop_tensor = torch.cat(all_val_gdop)  # [total_val_samples]
                
                # Distance percentiles (in meters - SI unit)
                val_distance_p50 = torch.quantile(all_val_distances_tensor, 0.50).item()  # median
                val_distance_p68 = torch.quantile(all_val_distances_tensor, 0.68).item()  # project KPI!
                val_distance_p95 = torch.quantile(all_val_distances_tensor, 0.95).item()  # worst-case
                
                # Uncertainty metrics (in meters - SI unit)
                mean_predicted_uncertainty = all_val_uncertainties_tensor.mean().item()
                
                # Uncertainty calibration: compare predicted uncertainty vs actual error
                # Ideal: predicted_uncertainty ≈ actual_error
                actual_error = all_val_distances_tensor.mean().item()
                uncertainty_calibration_error = abs(mean_predicted_uncertainty - actual_error)
                
                # GDOP metrics
                mean_gdop = all_val_gdop_tensor.mean().item()
                gdop_below_5_count = (all_val_gdop_tensor < 5.0).sum().item()
                gdop_below_5_percent = (gdop_below_5_count / len(all_val_gdop_tensor)) * 100.0
                
                # Model weight norm (L2)
                weight_norm = 0.0
                for p in model.parameters():
                    weight_norm += p.data.norm(2).item() ** 2
                weight_norm = weight_norm ** 0.5
                
            else:
                logger.warning("No validation batches - setting validation metrics to high default values")
                val_loss = 999999.0
                val_rmse = 999999.0
                val_rmse_good_geom = 999999.0
                val_distance_p50 = 999999.0
                val_distance_p68 = 999999.0
                val_distance_p95 = 999999.0
                mean_predicted_uncertainty = 999999.0
                uncertainty_calibration_error = 999999.0
                mean_gdop = 999.0
                gdop_below_5_percent = 0.0
                weight_norm = 0.0
                train_avg_grad_norm = 0.0
            
            # Sanitize metrics to ensure JSON compatibility (NaN/Inf -> high default)
            # Use explicit None check since 0.0 is a valid metric value
            sanitized_val_loss = sanitize_for_json(val_loss)
            sanitized_val_rmse = sanitize_for_json(val_rmse)
            sanitized_val_rmse_good_geom = sanitize_for_json(val_rmse_good_geom)
            
            val_loss = 999999.0 if sanitized_val_loss is None else sanitized_val_loss
            val_rmse = 999999.0 if sanitized_val_rmse is None else sanitized_val_rmse
            val_rmse_good_geom = 999999.0 if sanitized_val_rmse_good_geom is None else sanitized_val_rmse_good_geom

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
                            learning_rate = :lr,
                            updated_at = NOW()
                        WHERE id = :job_id
                    """),
                    {
                        "epoch": epoch,
                        "progress": (epoch / epochs) * 100.0,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": current_lr,
                        "job_id": job_id
                    }
                )

                # Store epoch metrics (including advanced metrics)
                session.execute(
                    text("""
                        INSERT INTO heimdall.training_metrics (
                            training_job_id, epoch, train_loss, val_loss,
                            train_accuracy, val_accuracy, learning_rate, phase, timestamp,
                            train_rmse_m, val_rmse_m, val_rmse_good_geom_m,
                            val_distance_p50_m, val_distance_p68_m, val_distance_p95_m,
                            mean_predicted_uncertainty_m, uncertainty_calibration_error,
                            mean_gdop, gdop_below_5_percent,
                            gradient_norm, weight_norm
                        )
                        VALUES (
                            :job_id, :epoch, :train_loss, :val_loss,
                            :train_rmse_m, :val_rmse_m, :lr, 'train', NOW(),
                            :train_rmse_m, :val_rmse_m, :val_rmse_good_geom_m,
                            :val_distance_p50_m, :val_distance_p68_m, :val_distance_p95_m,
                            :mean_predicted_uncertainty_m, :uncertainty_calibration_error,
                            :mean_gdop, :gdop_below_5_percent,
                            :gradient_norm, :weight_norm
                        )
                    """),
                    {
                        "job_id": job_id,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": current_lr,
                        # Advanced metrics (all in meters - SI unit)
                        "train_rmse_m": train_rmse,
                        "val_rmse_m": val_rmse,
                        "val_rmse_good_geom_m": val_rmse_good_geom,
                        "val_distance_p50_m": val_distance_p50,
                        "val_distance_p68_m": val_distance_p68,
                        "val_distance_p95_m": val_distance_p95,
                        "mean_predicted_uncertainty_m": mean_predicted_uncertainty,
                        "uncertainty_calibration_error": uncertainty_calibration_error,
                        "mean_gdop": mean_gdop,
                        "gdop_below_5_percent": gdop_below_5_percent,
                        "gradient_norm": train_avg_grad_norm,
                        "weight_norm": weight_norm
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

            # Publish training progress event (including advanced localization metrics)
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
                    'learning_rate': float(current_lr),
                    # Advanced localization metrics (Phase 7) - all in meters (SI unit)
                    'train_rmse_m': float(train_rmse),
                    'val_rmse_m': float(val_rmse),
                    'val_rmse_good_geom_m': float(val_rmse_good_geom),
                    'val_distance_p50_m': float(val_distance_p50),
                    'val_distance_p68_m': float(val_distance_p68),
                    'val_distance_p95_m': float(val_distance_p95),
                    'mean_predicted_uncertainty_m': float(mean_predicted_uncertainty),
                    'uncertainty_calibration_error': float(uncertainty_calibration_error),
                    'mean_gdop': float(mean_gdop),
                    'gdop_below_5_percent': float(gdop_below_5_percent),
                    'gradient_norm': float(train_avg_grad_norm),
                    'weight_norm': float(weight_norm)
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

            # Early stopping check (skip if patience is 0 - disabled)
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
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

        # Export best model to ONNX
        onnx_path = None
        onnx_s3_uri = None
        try:
            logger.info(f"Exporting best model (epoch {best_epoch}) to ONNX format...")
            
            # Load best model checkpoint from MinIO
            best_checkpoint_path = f"checkpoints/{job_id}/best_model.pth"
            response = minio_client.s3_client.get_object(Bucket="models", Key=best_checkpoint_path)
            best_checkpoint_data = response['Body'].read()
            best_checkpoint = torch.load(BytesIO(best_checkpoint_data), map_location=device)
            
            # Restore model weights
            model.load_state_dict(best_checkpoint['model_state_dict'])
            model.eval()
            
            # Export to ONNX
            import tempfile
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_onnx_path = Path(tmp_dir) / f"{job_name}.onnx"
                
                # Create dummy input for ONNX export
                # TriangulationModel expects (batch, num_receivers, 6) features:
                # [snr, psd, freq_offset, rx_lat, rx_lon, signal_present]
                dummy_receiver_features = torch.randn(1, max_receivers, 6, device=device)
                # Signal mask must be boolean for ONNX masked_fill compatibility
                dummy_signal_mask = torch.zeros(1, max_receivers, dtype=torch.bool, device=device)
                
                # Export model using legacy exporter (more compatible with dynamic shapes)
                torch.onnx.export(
                    model,
                    (dummy_receiver_features, dummy_signal_mask),
                    str(tmp_onnx_path),
                    export_params=True,
                    input_names=['receiver_features', 'signal_mask'],
                    output_names=['position', 'log_variance'],
                    dynamic_axes={
                        'receiver_features': {0: 'batch_size'},
                        'signal_mask': {0: 'batch_size'},
                        'position': {0: 'batch_size'},
                        'log_variance': {0: 'batch_size'}
                    },
                    opset_version=14,
                    do_constant_folding=True,
                    verbose=False,
                    dynamo=False  # Use legacy TorchScript-based exporter
                )
                
                # Upload ONNX to MinIO
                onnx_minio_path = f"onnx/{job_id}/{job_name}.onnx"
                with open(tmp_onnx_path, 'rb') as f:
                    onnx_data = f.read()
                    minio_client.s3_client.put_object(
                        Bucket="models",
                        Key=onnx_minio_path,
                        Body=onnx_data,
                        ContentType="application/octet-stream"
                    )
                
                onnx_s3_uri = f"s3://models/{onnx_minio_path}"
                onnx_file_size_mb = len(onnx_data) / (1024 * 1024)
                logger.info(f"✅ ONNX export successful: {onnx_s3_uri} ({onnx_file_size_mb:.2f} MB)")
                
        except Exception as e:
            logger.error(f"ONNX export failed: {e}", exc_info=True)
            # Continue without ONNX - not critical for training completion

        # Save model metadata to models table
        model_id = uuid.uuid4()
        with db_manager.get_session() as session:
            # Store first dataset_id in synthetic_dataset_id column (has FK constraint)
            # Full list is preserved in hyperparameters JSON
            if not dataset_ids or len(dataset_ids) == 0:
                raise ValueError("No dataset_ids available for model insertion")
            primary_dataset_id = dataset_ids[0]
            
            # Sanitize metrics for JSON/DB insertion (NaN -> None)
            metrics_dict = {
                "best_epoch": best_epoch,
                "best_val_loss": sanitize_for_json(float(best_val_loss)),
                "final_val_rmse": sanitize_for_json(val_rmse)
            }
            
            # Auto-increment version for this model name
            version_query = text("""
                SELECT COALESCE(MAX(version), 0) + 1 
                FROM heimdall.models 
                WHERE model_name = :name
            """)
            version_result = session.execute(version_query, {"name": job_name}).fetchone()
            next_version = version_result[0] if version_result else 1
            
            logger.info(f"Saving model '{job_name}' as version {next_version}")
            
            session.execute(
                text("""
                    INSERT INTO heimdall.models (
                        id, model_name, version, model_type, synthetic_dataset_id,
                        pytorch_model_location, onnx_model_location, accuracy_meters, accuracy_sigma_meters,
                        loss_value, epoch, is_active, is_production,
                        hyperparameters, training_metrics, trained_by_job_id, parent_model_id
                    )
                    VALUES (
                        :id, :name, :version, 'triangulation', :dataset_id,
                        :location, :onnx_location, :rmse, :rmse_good, :loss, :epoch,
                        FALSE, FALSE, CAST(:hyperparams AS jsonb), CAST(:metrics AS jsonb), :job_id, :parent_model_id
                    )
                """),
                {
                    "id": str(model_id),
                    "name": job_name,  # Use human-readable job name instead of UUID
                    "version": next_version,  # Auto-incremented version
                    "dataset_id": primary_dataset_id,
                    "location": f"s3://models/{final_checkpoint_path}",
                    "onnx_location": onnx_s3_uri,  # May be None if ONNX export failed
                    "rmse": sanitize_for_json(val_rmse),
                    "rmse_good": sanitize_for_json(val_rmse_good_geom),
                    "loss": sanitize_for_json(val_loss),
                    "epoch": epoch,
                    "hyperparams": json.dumps(config),
                    "metrics": json.dumps(metrics_dict),
                    "job_id": job_id,
                    "parent_model_id": config.get("parent_model_id")  # Track model lineage
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

        # Publish job status update event
        from backend.src.events.publisher import get_event_publisher
        publisher = get_event_publisher()
        publisher.publish_training_job_update(
            job_id=job_id,
            status='running',
            action='started',
            current_progress=0,
            total_progress=config['num_samples']
        )
        logger.info(f"Published job start event for {job_id}")

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
                            config, created_by_job_id, dataset_type
                        )
                        VALUES (
                            :id, :name, :description, 0,
                            CAST(:config AS jsonb), :job_id, CAST(:dataset_type AS dataset_type_enum)
                        )
                    """)

                    session.execute(
                        dataset_query,
                        {
                            "id": str(dataset_id),
                            "name": config['name'],
                            "description": config.get('description'),
                            "config": json.dumps(config),
                            "job_id": job_id,
                            "dataset_type": config.get('dataset_type', 'feature_based')
                        }
                    )
                    session.commit()
                    logger.info(f"Created new dataset {dataset_id} (type: {config.get('dataset_type', 'feature_based')})")
                    
                    # CRITICAL: Wait for PostgreSQL to make commit visible to async transactions
                    # Without this, save_iq_metadata_to_db() will fail with foreign key violation
                    # because the async transaction can't see the dataset yet
                    import time
                    time.sleep(0.5)
                    logger.info(f"Dataset {dataset_id} committed and visible to all transactions")

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
                    job_id=job_id,  # Pass job_id for cancellation detection
                    dataset_type=config.get('dataset_type', 'feature_based')  # Pass dataset type (iq_raw or feature_based)
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

        # Calculate and update storage size (PostgreSQL + MinIO)
        try:
            with db_manager.get_session() as session:
                # Calculate PostgreSQL storage size using DB function
                pg_size_query = text("SELECT heimdall.calculate_dataset_storage_size(:dataset_id)")
                pg_size = session.execute(pg_size_query, {"dataset_id": str(dataset_id)}).scalar() or 0
                
                # Calculate MinIO storage size for IQ samples (if dataset_type='iq_raw')
                minio_size = 0
                dataset_type = config.get('dataset_type', 'feature_based')
                
                if dataset_type == 'iq_raw' and stats.get('iq_samples_saved', 0) > 0:
                    # Get MinIO client to calculate object sizes
                    minio_client = MinIOClient(
                        endpoint_url=backend_settings.minio_url,
                        access_key=backend_settings.minio_access_key,
                        secret_key=backend_settings.minio_secret_key,
                        bucket_name="heimdall-iq-samples"
                    )
                    
                    try:
                        # List all objects for this dataset in MinIO
                        prefix = f"synthetic/{dataset_id}/"
                        objects = minio_client.s3_client.list_objects_v2(
                            Bucket="heimdall-iq-samples",
                            Prefix=prefix
                        )
                        
                        # Sum up sizes
                        for obj in objects.get('Contents', []):
                            minio_size += obj.get('Size', 0)
                        
                        logger.info(f"MinIO storage for dataset {dataset_id}: {minio_size / (1024**2):.2f} MB ({stats['iq_samples_saved']} IQ samples)")
                    except Exception as e:
                        logger.warning(f"Failed to calculate MinIO size for dataset {dataset_id}: {e}")
                
                # Total storage = PostgreSQL + MinIO
                total_size = pg_size + minio_size
                
                # Update dataset with storage size
                update_size_query = text("""
                    UPDATE heimdall.synthetic_datasets
                    SET storage_size_bytes = :size
                    WHERE id = :dataset_id
                """)
                session.execute(update_size_query, {
                    "dataset_id": str(dataset_id),
                    "size": total_size
                })
                session.commit()
                
                logger.info(f"Storage size calculated for dataset {dataset_id}: "
                           f"PostgreSQL={pg_size / (1024**2):.2f}MB, "
                           f"MinIO={minio_size / (1024**2):.2f}MB, "
                           f"Total={total_size / (1024**2):.2f}MB")
        except Exception as e:
            logger.error(f"Failed to calculate storage size for dataset {dataset_id}: {e}", exc_info=True)
            # Non-critical error, continue with job completion

        # Update job as completed
        with db_manager.get_session() as session:
            complete_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'completed', completed_at = NOW(), progress_percent = 100.0
                WHERE id = :job_id
            """)
            session.execute(complete_query, {"job_id": job_id})
            session.commit()

        # Publish job completion event
        publisher.publish_training_job_update(
            job_id=job_id,
            status='completed',
            action='completed',
            result={
                "dataset_id": str(dataset_id),
                "num_samples": stats['total_generated'],
                "iq_samples_saved": stats['iq_samples_saved'],
                "success_rate": stats['success_rate']
            }
        )
        logger.info(f"Published job completion event for {job_id}")

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

        # Publish job failure event
        publisher.publish_training_job_update(
            job_id=job_id,
            status='failed',
            action='failed',
            error_message=str(e)
        )
        logger.error(f"Published job failure event for {job_id}")

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
