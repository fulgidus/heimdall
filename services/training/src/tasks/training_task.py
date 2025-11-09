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
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from io import BytesIO

from celery import Task
from celery.contrib.abortable import AbortableTask
from celery.utils.log import get_task_logger

# Import the celery_app instance
from ..celery_app import celery_app

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


@celery_app.task(bind=True, base=AbortableTask, name='src.tasks.training_task.start_training_job')
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
        # For backward compatibility: support both dataset_id (singular) and dataset_ids (plural)
        dataset_ids = config.get("dataset_ids")
        if not dataset_ids and config.get("dataset_id"):
            dataset_ids = [config.get("dataset_id")]
        
        if not dataset_ids or len(dataset_ids) == 0:
            raise ValueError("dataset_ids is required in training configuration and must contain at least one dataset")

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
        max_receivers = config.get("max_receivers", 10)  # Support up to 10 receivers (synthetic data + future expansion)

        logger.info(f"Training config: datasets={dataset_ids}, epochs={epochs}, batch={batch_size}, lr={learning_rate}, workers={num_workers}")

        # Import training components (using absolute imports from /app)
        from src.models.triangulator import TriangulationModel, gaussian_nll_loss, haversine_distance_torch
        from src.models.model_factory import create_model_from_registry, get_model_input_requirements
        from src.data.triangulation_dataloader import create_triangulation_dataloader, create_iq_dataloader
        from src.data.gpu_cached_dataset import GPUCachedDataset
        from torch.utils.data import DataLoader
        from storage.minio_client import MinIOClient

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
        
        # ========================================================================
        # DEVICE INDICATOR: Show clear CPU/GPU training indicator
        # ========================================================================
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            logger.info(f"🚀 Training on GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        else:
            logger.info(f"⚠️  Training on CPU (no GPU available or requested)")
        
        logger.info(f"Device: {device} (accelerator={accelerator}, cuda_available={torch.cuda.is_available()})")

        # ===============================================================================
        # STEP 1: Determine Model Architecture and Data Requirements
        # ===============================================================================
        # Get model architecture from config (default to triangulation_model for backward compatibility)
        model_architecture = config.get("model_architecture", "triangulation_model")
        logger.info(f"📐 Model architecture: {model_architecture}")
        
        # Get model input requirements to determine which dataloader to use
        model_requirements = get_model_input_requirements(model_architecture)
        logger.info(f"📊 Model data requirements: {model_requirements}")
        
        # Determine dataloader type based on model requirements
        requires_iq = model_requirements.get("iq_raw", False)
        requires_spectrogram = model_requirements.get("spectrogram", False)
        use_iq_dataloader = requires_iq or requires_spectrogram
        
        if use_iq_dataloader:
            logger.info("🎵 Using IQ/Spectrogram dataloader for this model")
        else:
            logger.info("📊 Using feature-based dataloader for this model")

        # ===============================================================================
        # STEP 2: Create Dataloaders (IQ or Feature-based)
        # ===============================================================================
        # Choose dataloader based on model requirements
        if use_iq_dataloader:
            # =========================================================================
            # IQ/SPECTROGRAM DATALOADER (for CNN/Transformer models on raw IQ data)
            # =========================================================================
            logger.info("🎵 Creating IQ/Spectrogram dataloaders...")
            
            # IQ models need MinIO client to load raw IQ samples
            minio_client = MinIOClient(
                endpoint_url=backend_settings.minio_url,
                access_key=backend_settings.minio_access_key,
                secret_key=backend_settings.minio_secret_key,
                bucket_name="heimdall-synthetic-iq"
            )
            
            # Create DB sessions for IQ dataloaders
            train_session = db_manager.SessionLocal()
            val_session = db_manager.SessionLocal()
            
            # Note: IQ models typically use smaller batch sizes due to memory requirements
            iq_batch_size = min(batch_size, 32)  # Cap at 32 for IQ data
            if iq_batch_size < batch_size:
                logger.info(f"Reducing batch size from {batch_size} to {iq_batch_size} for IQ data (memory optimization)")
            
            train_loader = create_iq_dataloader(
                dataset_ids=dataset_ids,
                split="train",
                db_session=train_session,
                minio_client=minio_client,
                batch_size=iq_batch_size,
                num_workers=0,  # Single-threaded due to DB session constraints
                shuffle=True,
                max_receivers=max_receivers,
                use_cache=True  # File cache for spectrograms
            )
            
            val_loader = create_iq_dataloader(
                dataset_ids=dataset_ids,
                split="val",
                db_session=val_session,
                minio_client=minio_client,
                batch_size=iq_batch_size,
                num_workers=0,
                shuffle=False,
                max_receivers=max_receivers,
                use_cache=True
            )
            
            logger.info(f"✅ IQ dataloaders created: {len(train_loader.dataset)} train + {len(val_loader.dataset)} val samples")
            
        else:
            # =========================================================================
            # FEATURE-BASED DATALOADER (for MLP/feature models)
            # =========================================================================
            # GPU-CACHED DATASET: Load ALL data to VRAM for 100% GPU utilization!
            preload_to_gpu = config.get("preload_to_gpu", True)
            
            if preload_to_gpu and device.type == "cuda":
                logger.info("🚀 GPU-CACHED MODE: Loading ALL data to VRAM for maximum GPU utilization!")
                
                # Create one-time session for dataset preloading
                with db_manager.get_session() as load_session:
                    # Create GPU-cached datasets (loads all data to VRAM)
                    # Apply quality filters from config
                    min_snr_db = config.get("min_snr_db", -999.0)
                    
                    logger.info("Loading train dataset to GPU...")
                    train_dataset = GPUCachedDataset(
                        dataset_ids=dataset_ids,
                        split="train",
                        db_session=load_session,
                        device=device,
                        max_receivers=max_receivers,
                        preload_to_gpu=True,
                        min_snr_db=min_snr_db,
                        max_gdop=max_gdop
                    )
                    
                    logger.info("Loading validation dataset to GPU...")
                    val_dataset = GPUCachedDataset(
                        dataset_ids=dataset_ids,
                        split="val",
                        db_session=load_session,
                        device=device,
                        max_receivers=max_receivers,
                        preload_to_gpu=True,
                        min_snr_db=min_snr_db,
                        max_gdop=max_gdop
                    )
                
                # Import collate function for GPU-cached datasets
                from ..data.gpu_cached_dataset import collate_gpu_cached
                
                # Create DataLoaders (num_workers MUST be 0 for GPU-cached datasets)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,  # Data already on GPU, no I/O needed
                    pin_memory=False,  # Data already on GPU
                    collate_fn=collate_gpu_cached  # Custom collate with centroid support
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    collate_fn=collate_gpu_cached  # Custom collate with centroid support
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
                    max_receivers=max_receivers,
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
                    max_receivers=max_receivers,
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

        # ===============================================================================
        # STEP 3: Initialize Model from Registry
        # ===============================================================================
        # Note: model_architecture was already determined in STEP 1 (lines 199-218)
        # Initialize model using factory
        model = create_model_from_registry(
            model_id=model_architecture,
            max_receivers=max_receivers,
            dropout=dropout_rate
        ).to(device)

        logger.info(f"✅ Model '{model_architecture}' initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # CHECKPOINT 1: Model initialized successfully
        try:
            with open("/tmp/heimdall_trace.txt", "a") as f:
                f.write(f"CHECKPOINT 1: Model initialized, params={sum(p.numel() for p in model.parameters())}\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write checkpoint 1: {e}")

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
        
        # CHECKPOINT 2: MinIO ready
        try:
            with open("/tmp/heimdall_trace.txt", "a") as f:
                f.write(f"CHECKPOINT 2: MinIO client initialized\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write checkpoint 2: {e}")

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
        
        # CHECKPOINT 2A: Parent model loading complete
        try:
            with open("/tmp/heimdall_trace.txt", "a") as f:
                f.write(f"CHECKPOINT 2A: Parent model check complete (parent_model_id={parent_model_id})\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write checkpoint 2A: {e}")
        
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
        
        # CHECKPOINT 2B: Pause checkpoint check complete
        try:
            with open("/tmp/heimdall_trace.txt", "a") as f:
                f.write(f"CHECKPOINT 2B: Pause checkpoint check complete (resume_epoch={resume_epoch})\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write checkpoint 2B: {e}")
        
        # CHECKPOINT 3: Before training loop
        try:
            with open("/tmp/heimdall_trace.txt", "a") as f:
                f.write(f"CHECKPOINT 3: Entering training loop, resume_epoch={resume_epoch}, epochs={epochs}\n")
                f.flush()
        except Exception as e:
            logger.error(f"Failed to write checkpoint 3: {e}")
        
        # Training loop
        for epoch in range(resume_epoch + 1, epochs + 1):
            # CHECKPOINT 4: Inside epoch loop
            try:
                with open("/tmp/heimdall_trace.txt", "a") as f:
                    f.write(f"CHECKPOINT 4: Epoch {epoch} started\n")
                    f.flush()
            except Exception as e:
                logger.error(f"Failed to write checkpoint 4: {e}")
            
            # Check for cancellation at start of each epoch
            if self.is_aborted():
                logger.info(f"Training job {job_id} cancelled at epoch {epoch}. Stopping gracefully...")
                
                # Update job status to cancelled
                with db_manager.get_session() as session:
                    session.execute(
                        text("UPDATE heimdall.training_jobs SET status = 'cancelled', completed_at = NOW() WHERE id = :job_id"),
                        {"job_id": job_id}
                    )
                    session.commit()
                
                # Publish cancellation event
                event_publisher.publish_training_completed(
                    job_id=job_id,
                    status='cancelled',
                    best_epoch=best_epoch,
                    best_val_loss=float(best_val_loss)
                )
                
                return {
                    "status": "cancelled",
                    "job_id": job_id,
                    "cancelled_at_epoch": epoch - 1,
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss)
                }
            
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

            # Write to file to verify execution (logger may be suppressed)
            with open("/tmp/heimdall_execution_trace.txt", "a") as f:
                f.write(f"PRE-BATCH: Epoch={epoch}, model_arch={model_architecture}, batches={total_train_batches}\n")
            
            logger.error(f"[PRE-BATCH DEBUG] About to enter training loop. Epoch={epoch}, model_architecture={model_architecture}, total_batches={total_train_batches}")
            sys.stderr.write(f"[PRE-BATCH DEBUG] About to enter training loop. Epoch={epoch}, model_architecture={model_architecture}\n")
            sys.stderr.flush()
            
            # Training phase
            for batch_idx, batch in enumerate(train_loader, 1):
                # Write to file to verify batch loop entry
                with open("/tmp/heimdall_execution_trace.txt", "a") as f:
                    f.write(f"BATCH {batch_idx}: Epoch={epoch}\n")
                
                logger.error(f"[BATCH DEBUG] Entering batch {batch_idx}/{total_train_batches} for epoch {epoch}")
                print(f"[DEBUG] Entering batch {batch_idx} for epoch {epoch}", flush=True)
                
                # IMMEDIATE DEBUG - Check batch right after DataLoader yields it
                logger.error(f"[BATCH RAW] Keys in batch: {batch.keys()}")
                if "iq_samples" in batch:
                    iq_raw = batch["iq_samples"]
                    logger.error(f"[BATCH RAW] iq_samples shape: {iq_raw.shape}")
                    logger.error(f"[BATCH RAW] iq_samples dtype: {iq_raw.dtype}")
                    logger.error(f"[BATCH RAW] iq_samples device: {iq_raw.device}")
                    logger.error(f"[BATCH RAW] iq_samples min/max BEFORE .to(device): {iq_raw.min():.6e} / {iq_raw.max():.6e}")
                    logger.error(f"[BATCH RAW] iq_samples mean/std: {iq_raw.mean():.6e} / {iq_raw.std():.6e}")
                    sys.stderr.write(f"[BATCH RAW] iq_samples stats: min={iq_raw.min():.6e}, max={iq_raw.max():.6e}\n")
                    sys.stderr.flush()
                # Check for cancellation every 10 batches for responsive cancellation
                if batch_idx % 10 == 0 and self.is_aborted():
                    logger.info(f"Training job {job_id} cancelled during batch {batch_idx}. Stopping immediately...")
                    
                    # Update job status to cancelled
                    with db_manager.get_session() as session:
                        session.execute(
                            text("UPDATE heimdall.training_jobs SET status = 'cancelled', completed_at = NOW() WHERE id = :job_id"),
                            {"job_id": job_id}
                        )
                        session.commit()
                    
                    # Publish cancellation event
                    event_publisher.publish_training_completed(
                        job_id=job_id,
                        status='cancelled',
                        best_epoch=best_epoch,
                        best_val_loss=float(best_val_loss)
                    )
                    
                    return {
                        "status": "cancelled",
                        "job_id": job_id,
                        "cancelled_at_epoch": epoch,
                        "cancelled_at_batch": batch_idx,
                        "best_epoch": best_epoch,
                        "best_val_loss": float(best_val_loss)
                    }
                
                # Handle both IQ and feature-based batch structures
                if use_iq_dataloader:
                    # IQ models: use iq_samples as input (raw IQ time-series)
                    model_input = batch["iq_samples"].to(device)
                    
                    # DEBUG AFTER .to(device)
                    logger.error(f"[POST-DEVICE] model_input shape: {model_input.shape}")
                    logger.error(f"[POST-DEVICE] model_input device: {model_input.device}")
                    logger.error(f"[POST-DEVICE] model_input min/max AFTER .to(device): {model_input.min():.6e} / {model_input.max():.6e}")
                    logger.error(f"[POST-DEVICE] model_input mean/std: {model_input.mean():.6e} / {model_input.std():.6e}")
                    sys.stderr.write(f"[POST-DEVICE] model_input stats: min={model_input.min():.6e}, max={model_input.max():.6e}, device={model_input.device}\n")
                    sys.stderr.flush()
                else:
                    # Feature-based models: use receiver_features as input
                    model_input = batch["receiver_features"].to(device)
                
                signal_mask = batch["signal_mask"].to(device)
                target_position = batch["target_position"].to(device)

                optimizer.zero_grad()

                # CRITICAL DEBUG: Check model_architecture value
                sys.stderr.write(f"[CRITICAL] model_architecture='{model_architecture}', type={type(model_architecture)}, epoch={epoch}, batch={batch_idx}\n")
                sys.stderr.flush()
                logger.error(f"[CRITICAL] model_architecture='{model_architecture}', type={type(model_architecture)}, epoch={epoch}, batch={batch_idx}")

                # Forward pass - handle HeimdallNet/HeimdallNetPro special case
                if model_architecture == "heimdall_net" or model_architecture == "heimdall_net_pro":
                    try:
                        sys.stderr.write(f"[CRITICAL DEBUG] Entered HeimdallNet block for epoch {epoch}, batch {batch_idx}\n")
                        sys.stderr.flush()
                        logger.error(f"[CRITICAL DEBUG] Entered HeimdallNet block for epoch {epoch}, batch {batch_idx}")
                        
                        # HeimdallNet requires (iq_data, features, positions, receiver_ids, mask)
                        # Extract from IQ dataloader batch
                        # BUGFIX: Use model_input which was already transferred to GPU correctly
                        # Calling .to(device) twice on the same tensor corrupts it!
                        iq_data = model_input  # Already on device from line 652
                        
                        # IMMEDIATE CHECK: Verify iq_data right after assignment
                        logger.error(f"[IMMEDIATE CHECK 1] iq_data right after assignment: min={iq_data.min():.6e}, max={iq_data.max():.6e}, shape={iq_data.shape}")
                        logger.error(f"[IMMEDIATE CHECK 2] iq_data is model_input: {iq_data is model_input}")
                        logger.error(f"[IMMEDIATE CHECK 3] model_input: min={model_input.min():.6e}, max={model_input.max():.6e}")
                        
                        receiver_positions_2d = batch["receiver_positions"].to(device)  # (B, N, 2) - lat, lon only
                        
                        # Check again after loading positions
                        logger.error(f"[AFTER POSITIONS] iq_data: min={iq_data.min():.6e}, max={iq_data.max():.6e}")
                        
                        # Verify iq_data is correct after fix
                        logger.error(f"[BUGFIX VERIFY] iq_data is model_input: {iq_data is model_input}")
                        logger.error(f"[BUGFIX VERIFY] iq_data data_ptr: {iq_data.data_ptr()}, model_input data_ptr: {model_input.data_ptr()}")
                        logger.error(f"[BUGFIX VERIFY] iq_data min/max after fix: {iq_data.min():.6e} / {iq_data.max():.6e}")
                        logger.error(f"[DEBUG] Successfully moved iq_data and positions to device")
                        logger.error(f"[DEBUG] IQ data shape: {iq_data.shape}")
                        
                        # Build features tensor: [SNR, PSD, freq_offset, lat, lon, alt]
                        # Use REAL RF features from dataloader (fixes 69km RMSE bug!)
                        batch_size_curr, num_receivers_curr = iq_data.shape[0], iq_data.shape[1]
                        features = torch.zeros(batch_size_curr, num_receivers_curr, 6, device=device)
                        
                        # Extract real RF features from batch
                        receiver_snr = batch["receiver_snr"].to(device)  # (batch, num_receivers)
                        receiver_psd = batch["receiver_psd"].to(device)  # (batch, num_receivers)
                        receiver_freq_offset = batch["receiver_freq_offset"].to(device)  # (batch, num_receivers)
                        
                        # Log first sample's RF features for debugging
                        if batch_idx == 0 and current_epoch % 5 == 0:
                            logger.error(f"[RF FEATURES DEBUG] Sample 0 SNR: {receiver_snr[0, :5].cpu().numpy()}")
                            logger.error(f"[RF FEATURES DEBUG] Sample 0 PSD: {receiver_psd[0, :5].cpu().numpy()}")
                            logger.error(f"[RF FEATURES DEBUG] Sample 0 Freq Offset: {receiver_freq_offset[0, :5].cpu().numpy()}")
                        
                        features[:, :, 0] = receiver_snr  # Real SNR from WebSDR metadata
                        features[:, :, 1] = receiver_psd  # Real PSD from WebSDR metadata
                        features[:, :, 2] = receiver_freq_offset  # Real freq offset from WebSDR metadata
                        features[:, :, 3:5] = receiver_positions_2d  # lat, lon (delta coordinates)
                        features[:, :, 5] = 0.0  # Dummy altitude (0 m) - TODO: add real altitude later
                        
                        logger.error(f"[DEBUG] Built features tensor")
                        
                        # Expand positions to 3D (add altitude) for HeimdallNet
                        receiver_positions_3d = torch.zeros(batch_size_curr, num_receivers_curr, 3, device=device)
                        receiver_positions_3d[:, :, :2] = receiver_positions_2d
                        receiver_positions_3d[:, :, 2] = 0.0  # Dummy altitude
                        
                        # Receiver IDs: Sequential assignment (0, 1, 2, ..., N-1)
                        receiver_ids = torch.arange(num_receivers_curr, device=device).unsqueeze(0).expand(batch_size_curr, -1)
                        receiver_ids = torch.clamp(receiver_ids, 0, max_receivers - 1)
                        
                        logger.error(f"[DEBUG] Built receiver_ids and positions_3d")
                        
                        # Debug logging for first 3 batches to capture potential issues
                        if epoch == 1 and batch_idx <= 3:
                            debug_msg = f"\n[HeimdallNet Debug] Epoch {epoch}, Batch {batch_idx} BEFORE forward pass:\n"
                            debug_msg += f"  iq_data: shape={iq_data.shape}, min={iq_data.min().item():.4f}, max={iq_data.max().item():.4f}, mean={iq_data.mean().item():.4f}, has_nan={torch.isnan(iq_data).any()}, has_inf={torch.isinf(iq_data).any()}\n"
                            debug_msg += f"  features: shape={features.shape}, min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}, has_nan={torch.isnan(features).any()}, has_inf={torch.isinf(features).any()}\n"
                            debug_msg += f"  receiver_positions_3d: shape={receiver_positions_3d.shape}, min={receiver_positions_3d.min().item():.4f}, max={receiver_positions_3d.max().item():.4f}, has_nan={torch.isnan(receiver_positions_3d).any()}\n"
                            debug_msg += f"  receiver_ids: shape={receiver_ids.shape}, min={receiver_ids.min().item()}, max={receiver_ids.max().item()}, unique_count={len(torch.unique(receiver_ids))}\n"
                            debug_msg += f"  signal_mask: shape={signal_mask.shape}, true_count={signal_mask.sum().item()}, false_count={(~signal_mask).sum().item()}\n"
                            debug_msg += f"  target_position: shape={target_position.shape}, min={target_position.min().item():.4f}, max={target_position.max().item():.4f}\n"
                            
                            # Write to both stdout and file
                            logger.error(debug_msg)  # Use logger.error to ensure it appears
                            with open("/tmp/heimdall_debug.log", "a") as f:
                                f.write(debug_msg)
                                f.flush()
                        
                        logger.error(f"[DEBUG] About to call model forward pass")
                        
                        # Forward pass with all 5 arguments
                        # MASK SEMANTICS CLARIFICATION:
                        # - DataLoader produces: True = padding, False = active
                        # - HeimdallNet.forward() expects: True = active, False = padding (per docstring)
                        # - GeometryEncoder expects: True = active (uses & operator on mask)
                        # - SetAttentionAggregator internally converts to PyTorch format before calling MultiheadAttention
                        # 
                        # Therefore: INVERT the mask when passing to model
                        model_mask = ~signal_mask  # Invert: True = active, False = padding
                        position, log_variance = model(iq_data, features, receiver_positions_3d, receiver_ids, model_mask)
                        
                        logger.error(f"[DEBUG] Model forward pass completed successfully")
                        
                        # Debug logging for outputs
                        if epoch == 1 and batch_idx <= 3:
                            debug_msg = f"  Output position: shape={position.shape}, min={position.min().item():.4f}, max={position.max().item():.4f}, mean={position.mean().item():.4f}, has_nan={torch.isnan(position).any()}\n"
                            debug_msg += f"  Output log_variance: shape={log_variance.shape}, min={log_variance.min().item():.4f}, max={log_variance.max().item():.4f}, mean={log_variance.mean().item():.4f}, has_nan={torch.isnan(log_variance).any()}\n"
                            logger.error(debug_msg)
                            with open("/tmp/heimdall_debug.log", "a") as f:
                                f.write(debug_msg)
                                f.flush()
                    
                    except Exception as e:
                        logger.error(f"[EXCEPTION IN HEIMDALLNET BLOCK] {type(e).__name__}: {str(e)}")
                        logger.error(f"[EXCEPTION] Epoch {epoch}, Batch {batch_idx}")
                        import traceback
                        logger.error(f"[TRACEBACK] {traceback.format_exc()}")
                        raise  # Re-raise to maintain original behavior
                else:
                    # Standard models: (model_input, signal_mask)
                    position, log_variance = model(model_input, signal_mask)

                # Debug loss inputs for first 3 batches
                if epoch == 1 and batch_idx <= 3:
                    logger.error(f"[LOSS DEBUG] Batch {batch_idx}:")
                    logger.error(f"  position: min={position.min().item():.6e}, max={position.max().item():.6e}, mean={position.mean().item():.6e}, has_nan={torch.isnan(position).any()}")
                    logger.error(f"  log_variance: min={log_variance.min().item():.6e}, max={log_variance.max().item():.6e}, mean={log_variance.mean().item():.6e}, has_nan={torch.isnan(log_variance).any()}")
                    logger.error(f"  target_position: min={target_position.min().item():.6e}, max={target_position.max().item():.6e}, mean={target_position.mean().item():.6e}, has_nan={torch.isnan(target_position).any()}")
                
                # Calculate loss
                loss = gaussian_nll_loss(position, log_variance, target_position)
                
                # Check for loss explosion (NaN, Inf, or extremely high values)
                loss_value = loss.item()
                if epoch == 1 and batch_idx <= 3:
                    logger.error(f"[LOSS DEBUG] Loss value for batch {batch_idx}: {loss_value:.6e}, is_nan={math.isnan(loss_value)}, is_inf={math.isinf(loss_value)}")
                
                if math.isnan(loss_value) or math.isinf(loss_value) or loss_value > 100000:
                    error_msg = (
                        f"Training failed: Loss explosion detected at epoch {epoch}, batch {batch_idx}. "
                        f"Loss value: {loss_value}. This typically indicates:\n"
                        f"1. Feature normalization issues (check SNR, PSD, freq_offset scaling)\n"
                        f"2. Learning rate too high (current: {optimizer.param_groups[0]['lr']})\n"
                        f"3. Gradient explosion (consider lowering learning rate or checking data quality)\n"
                        f"4. Invalid input data (check for NaN/Inf in dataset)"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

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
                # IMPORTANT: Model predicts STANDARDIZED DELTA METERS (z-score normalized)
                # Must denormalize using z-score inverse, convert to degrees, then add centroid
                with torch.no_grad():
                    centroids = batch["metadata"]["centroids"].to(device)  # [batch, 2] (lat, lon)
                    
                    # Import coordinate conversion constants from dataset
                    from src.data.gpu_cached_dataset import METERS_PER_DEG_LAT, METERS_PER_DEG_LON
                    
                    # Get standardization parameters from batch metadata (per-split)
                    # These are computed during dataset initialization and passed through collate_fn
                    coord_mean_lat_meters = batch["metadata"]["coord_mean_lat_meters"]
                    coord_mean_lon_meters = batch["metadata"]["coord_mean_lon_meters"]
                    coord_std_lat_meters = batch["metadata"]["coord_std_lat_meters"]
                    coord_std_lon_meters = batch["metadata"]["coord_std_lon_meters"]
                    
                    # Step 1: Inverse z-score standardization to get meters
                    # position and target_position are z-score standardized → denormalize: x_meters = z * std + mean
                    position_lat_meters = position[:, 0] * coord_std_lat_meters + coord_mean_lat_meters
                    position_lon_meters = position[:, 1] * coord_std_lon_meters + coord_mean_lon_meters
                    target_lat_meters = target_position[:, 0] * coord_std_lat_meters + coord_mean_lat_meters
                    target_lon_meters = target_position[:, 1] * coord_std_lon_meters + coord_mean_lon_meters
                    
                    # Step 2: Convert meters back to degrees
                    position_lat_deg = position_lat_meters / METERS_PER_DEG_LAT
                    position_lon_deg = position_lon_meters / METERS_PER_DEG_LON
                    target_lat_deg = target_lat_meters / METERS_PER_DEG_LAT
                    target_lon_deg = target_lon_meters / METERS_PER_DEG_LON
                    
                    # Step 3: Reconstruct absolute coordinates: absolute = delta_degrees + centroid
                    position_absolute_lat = position_lat_deg + centroids[:, 0]
                    position_absolute_lon = position_lon_deg + centroids[:, 1]
                    target_absolute_lat = target_lat_deg + centroids[:, 0]
                    target_absolute_lon = target_lon_deg + centroids[:, 1]
                    
                    # Step 4: Calculate haversine distance
                    distances = haversine_distance_torch(
                        position_absolute_lat, position_absolute_lon,
                        target_absolute_lat, target_absolute_lon
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
                    # Handle both IQ and feature-based batch structures
                    if use_iq_dataloader:
                        # IQ models: use iq_samples as input (raw IQ time-series)
                        model_input = batch["iq_samples"].to(device)
                    else:
                        # Feature-based models: use receiver_features as input
                        model_input = batch["receiver_features"].to(device)
                    
                    signal_mask = batch["signal_mask"].to(device)
                    target_position = batch["target_position"].to(device)
                    gdop = batch["metadata"]["gdop"]

                    # Forward pass - handle HeimdallNet/HeimdallNetPro special case
                    if model_architecture == "heimdall_net" or model_architecture == "heimdall_net_pro":
                        # HeimdallNet requires (iq_data, features, positions, receiver_ids, mask)
                        iq_data = batch["iq_samples"].to(device)
                        receiver_positions_2d = batch["receiver_positions"].to(device)  # (B, N, 2) - lat, lon only
                        
                        # Build features tensor: [SNR, PSD, freq_offset, lat, lon, alt]
                        batch_size_curr, num_receivers_curr = iq_data.shape[0], iq_data.shape[1]
                        features = torch.zeros(batch_size_curr, num_receivers_curr, 6, device=device)
                        features[:, :, 0] = 20.0  # Dummy SNR
                        features[:, :, 1] = -80.0  # Dummy PSD
                        features[:, :, 2] = 0.0  # Dummy freq offset
                        features[:, :, 3:5] = receiver_positions_2d  # lat, lon
                        features[:, :, 5] = 0.0  # Dummy altitude
                        
                        # Expand positions to 3D (add altitude) for HeimdallNet
                        receiver_positions_3d = torch.zeros(batch_size_curr, num_receivers_curr, 3, device=device)
                        receiver_positions_3d[:, :, :2] = receiver_positions_2d
                        receiver_positions_3d[:, :, 2] = 0.0  # Dummy altitude
                        
                        # Receiver IDs
                        receiver_ids = torch.arange(num_receivers_curr, device=device).unsqueeze(0).expand(batch_size_curr, -1)
                        receiver_ids = torch.clamp(receiver_ids, 0, max_receivers - 1)
                        
                        position, log_variance = model(iq_data, features, receiver_positions_3d, receiver_ids, signal_mask)
                    else:
                        # Standard models
                        position, log_variance = model(model_input, signal_mask)
                    
                    loss = gaussian_nll_loss(position, log_variance, target_position)
                    
                    # Check for validation loss explosion
                    loss_value = loss.item()
                    if math.isnan(loss_value) or math.isinf(loss_value) or loss_value > 100000:
                        error_msg = (
                            f"Training failed: Validation loss explosion detected at epoch {epoch}. "
                            f"Loss value: {loss_value}. This indicates severe model instability. "
                            f"Consider: lowering learning rate, checking dataset quality, or restarting training."
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Reconstruct absolute coordinates for distance calculation
                    # Model predicts STANDARDIZED DELTA METERS (z-score normalized)
                    # Must denormalize using z-score inverse, convert to degrees, then add centroid
                    centroids = batch["metadata"]["centroids"].to(device)  # [batch, 2] (lat, lon)
                    
                    # Import coordinate conversion constants from dataset
                    from src.data.gpu_cached_dataset import METERS_PER_DEG_LAT, METERS_PER_DEG_LON
                    
                    # Get standardization parameters from batch metadata (per-split)
                    # These are computed during dataset initialization and passed through collate_fn
                    coord_mean_lat_meters = batch["metadata"]["coord_mean_lat_meters"]
                    coord_mean_lon_meters = batch["metadata"]["coord_mean_lon_meters"]
                    coord_std_lat_meters = batch["metadata"]["coord_std_lat_meters"]
                    coord_std_lon_meters = batch["metadata"]["coord_std_lon_meters"]
                    
                    # Step 1: Inverse z-score standardization to get meters
                    # position and target_position are z-score standardized → denormalize: x_meters = z * std + mean
                    position_lat_meters = position[:, 0] * coord_std_lat_meters + coord_mean_lat_meters
                    position_lon_meters = position[:, 1] * coord_std_lon_meters + coord_mean_lon_meters
                    target_lat_meters = target_position[:, 0] * coord_std_lat_meters + coord_mean_lat_meters
                    target_lon_meters = target_position[:, 1] * coord_std_lon_meters + coord_mean_lon_meters
                    
                    # Step 2: Convert meters back to degrees
                    position_lat_deg = position_lat_meters / METERS_PER_DEG_LAT
                    position_lon_deg = position_lon_meters / METERS_PER_DEG_LON
                    target_lat_deg = target_lat_meters / METERS_PER_DEG_LAT
                    target_lon_deg = target_lon_meters / METERS_PER_DEG_LON
                    
                    # Step 3: Reconstruct absolute coordinates: absolute = delta_degrees + centroid
                    position_absolute_lat = position_lat_deg + centroids[:, 0]
                    position_absolute_lon = position_lon_deg + centroids[:, 1]
                    target_absolute_lat = target_lat_deg + centroids[:, 0]
                    target_absolute_lon = target_lon_deg + centroids[:, 1]
                    
                    # Step 4: Calculate haversine distance
                    distances = haversine_distance_torch(
                        position_absolute_lat, position_absolute_lon,
                        target_absolute_lat, target_absolute_lon
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
                
                # Create dummy inputs based on model architecture
                if model_architecture == "heimdall_net" or model_architecture == "heimdall_net_pro":
                    # HeimdallNet expects 5 inputs: (iq_data, features, positions, receiver_ids, mask)
                    seq_len = 200000  # Default 200kHz @ 1 second
                    dummy_iq_data = torch.randn(1, max_receivers, 2, seq_len, device=device)
                    dummy_features = torch.randn(1, max_receivers, 6, device=device)
                    dummy_positions = torch.randn(1, max_receivers, 3, device=device)
                    dummy_receiver_ids = torch.arange(max_receivers, device=device).unsqueeze(0)
                    dummy_mask = torch.zeros(1, max_receivers, dtype=torch.bool, device=device)
                    
                    torch.onnx.export(
                        model,
                        (dummy_iq_data, dummy_features, dummy_positions, dummy_receiver_ids, dummy_mask),
                        str(tmp_onnx_path),
                        export_params=True,
                        input_names=['iq_data', 'features', 'positions', 'receiver_ids', 'mask'],
                        output_names=['position', 'uncertainty'],
                        dynamic_axes={
                            'iq_data': {0: 'batch_size', 3: 'seq_len'},
                            'features': {0: 'batch_size'},
                            'positions': {0: 'batch_size'},
                            'receiver_ids': {0: 'batch_size'},
                            'mask': {0: 'batch_size'},
                            'position': {0: 'batch_size'},
                            'uncertainty': {0: 'batch_size'}
                        },
                        opset_version=14,
                        do_constant_folding=True,
                        verbose=False,
                        dynamo=False
                    )
                else:
                    # TriangulationModel expects 2 inputs: (receiver_features, signal_mask)
                    dummy_receiver_features = torch.randn(1, max_receivers, 6, device=device)
                    dummy_signal_mask = torch.zeros(1, max_receivers, dtype=torch.bool, device=device)
                    
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
                        dynamo=False
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


@celery_app.task(bind=True, base=AbortableTask, name='src.tasks.training_task.generate_synthetic_data_task')
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
    
    # OPTION B: Setup signal handler for graceful shutdown (SIGTERM)
    # This allows us to save partial progress when user cancels job
    import signal
    shutdown_requested = {'value': False}
    
    def handle_shutdown_signal(signum, frame):
        """Handle SIGTERM/SIGINT gracefully by saving partial progress."""
        sig_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
        logger.warning(f"Received {sig_name}, saving partial progress before shutdown...")
        shutdown_requested['value'] = True
    
    # Register signal handlers (SIGTERM = graceful shutdown, SIGINT = Ctrl+C)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    logger.info("Signal handlers registered for graceful shutdown (SIGTERM/SIGINT)")

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
        # For dataset expansion, fetch samples_offset from database
        expand_dataset_id = config.get('expand_dataset_id')
        samples_offset = 0
        
        if expand_dataset_id:
            # Fetch current sample count from database
            with db_manager.get_session() as session:
                count_query = text("""
                    SELECT num_samples FROM heimdall.synthetic_datasets
                    WHERE id = :dataset_id
                """)
                result = session.execute(count_query, {"dataset_id": expand_dataset_id}).fetchone()
                if result:
                    samples_offset = result[0]
                    logger.info(f"Expanding dataset {expand_dataset_id}: current samples = {samples_offset}")
                else:
                    raise ValueError(f"Dataset {expand_dataset_id} not found for expansion")
        
        total_progress_value = samples_offset + config['num_samples']
        
        with db_manager.get_session() as session:
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'running',
                    started_at = NOW(),
                    current_progress = :current_progress,
                    total_progress = :total_samples,
                    progress_message = :progress_message
                WHERE id = :job_id
            """)
            progress_message = f'Continuing from {samples_offset} samples...' if samples_offset > 0 else 'Starting synthetic data generation...'
            session.execute(update_query, {
                "job_id": job_id,
                "current_progress": samples_offset,
                "total_samples": total_progress_value,
                "progress_message": progress_message
            })
            session.commit()

        # Publish job status update event with complete job data
        from backend.src.events.publisher import get_event_publisher
        publisher = get_event_publisher()
        publisher.publish_training_job_update(
            job_id=job_id,
            status='running',
            action='started',
            current_progress=samples_offset,
            total_progress=total_progress_value,
            # Include complete job data for frontend to update directly
            job_name=config.get('job_name', 'Unnamed Job'),
            job_type='synthetic_generation',
            progress_percent=0.0,
            progress_message=f'Continuing from {samples_offset} samples...' if samples_offset > 0 else 'Starting synthetic data generation...',
            dataset_id=expand_dataset_id,  # For expansion jobs, otherwise None
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
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

        # Handle dataset expansion vs new dataset creation
        # Use expand_dataset_id as single source of truth (no fragile flags)
        expand_dataset_id = config.get('expand_dataset_id')
        
        if expand_dataset_id:
            # Expanding existing dataset
            dataset_id = uuid.UUID(expand_dataset_id)
            logger.info(f"Expanding dataset {dataset_id} (current samples: {samples_offset})")
        else:
            # Create new dataset
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
        # Now accepts attempted parameter to show total attempts vs valid samples
        async def progress_callback(current, total, attempted=None):
            # Check for cancellation during generation
            if self.is_aborted():
                logger.info(f"Synthetic data generation job {job_id} cancelled at {current}/{total} samples")
                raise asyncio.CancelledError("Task cancelled by user")
            
            # Add offset for dataset expansion (samples_offset > 0 means expanding)
            cumulative_current = samples_offset + current
            total_samples = samples_offset + total
            
            # Clamp progress to not exceed 100% (batch generation may overshoot target)
            cumulative_current = min(cumulative_current, total_samples)
            current = min(current, total)
            progress_pct = min((cumulative_current / total_samples) * 100, 100.0)
            
            # Calculate success rate if attempted is provided
            success_rate = (current / attempted * 100) if attempted and attempted > 0 else 100.0
            
            # Show different messages for expansion vs new dataset
            if samples_offset > 0:
                if attempted:
                    message = f'Generated {cumulative_current}/{total_samples} valid samples (continued from {samples_offset}, attempted: {attempted}, {success_rate:.1f}% success)'
                else:
                    message = f'Processing {cumulative_current}/{total_samples} samples (continued from {samples_offset})'
            else:
                if attempted:
                    message = f'Generated {current}/{total} valid samples (attempted: {attempted}, {success_rate:.1f}% success)'
                else:
                    message = f'Processing {current}/{total} samples'

            logger.info(f"[PROGRESS] {message} ({progress_pct:.1f}%)")

            # Update Celery state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': cumulative_current,
                    'total': total_samples,
                    'attempted': attempted,
                    'success_rate': success_rate,
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
                            "current": cumulative_current,
                            "total": total_samples,
                            "progress": progress_pct,
                            "message": message,
                            "job_id": job_id
                        }
                    )
                    session.commit()
                    logger.info(f"[PROGRESS] Database updated: {cumulative_current}/{total_samples}")
            except Exception as e:
                logger.error(f"Failed to update progress in database: {e}", exc_info=True)

            # Publish progress event to RabbitMQ for WebSocket clients
            try:
                event_publisher.publish_dataset_generation_progress(
                    job_id=job_id,
                    current=cumulative_current,
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
        if samples_offset > 0:
            logger.info(f"Continuing generation: {config['num_samples']} remaining samples "
                       f"(total: {samples_offset + config['num_samples']})")
        else:
            logger.info(f"Generating {config['num_samples']} synthetic samples with IQ generation")

        async def run_generation():
            # Use .connect() instead of .begin() to allow manual transaction control
            # This enables incremental commits after each batch for partial progress preservation
            async with async_engine.connect() as conn:
                # Start transaction manually
                await conn.begin()
                try:
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
                        dataset_type=config.get('dataset_type', 'feature_based'),  # Pass dataset type (iq_raw or feature_based)
                        use_gpu=config.get('use_gpu', False),  # DEFAULT: CPU-only (False). Set use_gpu=True in config to enable GPU
                        shutdown_requested=shutdown_requested,  # Pass signal handler flag for fast cancellation
                        samples_offset=samples_offset,  # Pass offset for dataset expansion
                        disable_safety_checks=config.get('disable_safety_checks', False)  # Allow bypassing consecutive failure check
                    )
                    # Final commit after all batches
                    await conn.commit()
                except Exception as e:
                    # Rollback on error (though partial commits already happened)
                    await conn.rollback()
                    raise
            return stats

        # Run async generation
        stats = asyncio.run(run_generation())

        logger.info(f"[GENERATION COMPLETE DEBUG] Generation stats: {stats}")
        logger.info(f"[GENERATION COMPLETE DEBUG] total_generated={stats['total_generated']}, reached_target={stats.get('reached_target')}, stopped_reason={stats.get('stopped_reason')}")

        logger.info(f"Generation complete: {stats['total_generated']} samples, "
                    f"{stats['iq_samples_saved']} IQ samples saved to MinIO")

        # Update dataset record with final counts
        # Count actual samples from the correct table based on dataset_type
        with db_manager.get_session() as session:
            # Get dataset_type to determine which table to query
            dataset_type = config.get('dataset_type', 'feature_based')
            
            logger.info(f"[FINAL COUNT DEBUG] Querying final sample count for dataset_id={dataset_id}, dataset_type={dataset_type}")
            logger.info(f"[FINAL COUNT DEBUG] samples_offset={samples_offset}, stats_generated={stats['total_generated']}")
            
            # Count actual samples from measurement_features (source of truth for training)
            # Both 'iq_raw' and 'feature_based' datasets require features for training
            count_query = text("""
                SELECT COUNT(*) FROM heimdall.measurement_features
                WHERE dataset_id = :dataset_id
            """)
            
            actual_count = session.execute(count_query, {"dataset_id": str(dataset_id)}).scalar() or 0
            
            logger.info(f"[FINAL COUNT DEBUG] Query returned actual_count={actual_count} from database")

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
            logger.info(f"Updated dataset {dataset_id} with {actual_count} samples (generated: {stats['total_generated']}, type: {dataset_type})")

        # Calculate and update storage size (PostgreSQL + MinIO)
        # Strategy: Calculate PostgreSQL size first and commit it immediately
        # Then attempt MinIO calculation (which may fail), but PostgreSQL size is already saved
        pg_size = 0
        minio_size = 0
        total_size = 0
        
        try:
            # Step 1: Calculate and save PostgreSQL storage size (always succeeds)
            with db_manager.get_session() as session:
                logger.info(f"Calculating PostgreSQL storage for dataset {dataset_id}...")
                pg_size_query = text("SELECT heimdall.calculate_dataset_storage_size(:dataset_id)")
                pg_size = session.execute(pg_size_query, {"dataset_id": str(dataset_id)}).scalar() or 0
                
                logger.info(f"PostgreSQL storage for dataset {dataset_id}: {pg_size / (1024**2):.2f} MB")
                
                # Update dataset with PostgreSQL size immediately (commit partial result)
                update_size_query = text("""
                    UPDATE heimdall.synthetic_datasets
                    SET storage_size_bytes = :size
                    WHERE id = :dataset_id
                """)
                session.execute(update_size_query, {
                    "dataset_id": str(dataset_id),
                    "size": pg_size
                })
                session.commit()
                logger.info(f"Saved PostgreSQL storage size ({pg_size} bytes) to database")
            
            # Step 2: Calculate MinIO storage size (may fail, but PostgreSQL size already saved)
            dataset_type = config.get('dataset_type', 'feature_based')
            
            if dataset_type == 'iq_raw' and stats.get('iq_samples_saved', 0) > 0:
                try:
                    logger.info(f"Calculating MinIO storage for dataset {dataset_id}...")
                    
                    # Get MinIO client
                    minio_client = MinIOClient(
                        endpoint_url=backend_settings.minio_url,
                        access_key=backend_settings.minio_access_key,
                        secret_key=backend_settings.minio_secret_key,
                        bucket_name="heimdall-iq-samples"
                    )
                    
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
                    
                    # Update with total size (PostgreSQL + MinIO)
                    total_size = pg_size + minio_size
                    with db_manager.get_session() as session:
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
                        logger.info(f"Updated total storage size: PostgreSQL={pg_size / (1024**2):.2f}MB + MinIO={minio_size / (1024**2):.2f}MB = {total_size / (1024**2):.2f}MB")
                    
                except Exception as minio_error:
                    logger.warning(f"Failed to calculate MinIO size for dataset {dataset_id}: {minio_error}", exc_info=True)
                    logger.info(f"Storage size calculation completed with PostgreSQL only: {pg_size / (1024**2):.2f}MB")
                    total_size = pg_size  # Use PostgreSQL size only
            else:
                # Feature-based dataset: only PostgreSQL storage
                total_size = pg_size
                logger.info(f"Storage size calculation complete (feature-based dataset): {total_size / (1024**2):.2f}MB")
            
            # Publish dataset update event to trigger UI refresh
            try:
                event_publisher.publish_dataset_updated(
                    dataset_id=str(dataset_id),
                    num_samples=actual_count,
                    action="expanded" if samples_offset > 0 else "created",
                    storage_size_bytes=total_size,
                    dataset_type=config.get('dataset_type', 'feature_based')
                )
                logger.info(f"Published dataset update event for dataset {dataset_id} (samples={actual_count}, storage={total_size})")
            except Exception as publish_error:
                logger.error(f"Failed to publish dataset update event: {publish_error}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Failed to calculate storage size for dataset {dataset_id}: {e}", exc_info=True)
            # Non-critical error, continue with job completion
            
            # Still publish dataset update event even if storage calculation completely failed
            try:
                event_publisher.publish_dataset_updated(
                    dataset_id=str(dataset_id),
                    num_samples=actual_count,
                    action="expanded" if samples_offset > 0 else "created",
                    dataset_type=config.get('dataset_type', 'feature_based')
                )
                logger.info(f"Published dataset update event for dataset {dataset_id} (samples={actual_count}, storage calc failed)")
            except Exception as publish_error:
                logger.error(f"Failed to publish dataset update event: {publish_error}", exc_info=True)

        # Check if generation stopped early due to consecutive failures
        stopped_reason = stats.get('stopped_reason')
        if stopped_reason == 'consecutive_failures':
            # Mark job as failed with detailed error message
            rejection_stats = {
                'min_receivers': stats.get('rejected_min_receivers', 0),
                'min_snr': stats.get('rejected_min_snr', 0),
                'gdop': stats.get('rejected_gdop', 0)
            }
            error_message = (
                f"Job stopped after 50 consecutive batches with 0 valid samples. "
                f"Generated {stats['total_generated']} samples before stopping. "
                f"Rejection reasons: min_receivers={rejection_stats['min_receivers']}, "
                f"min_snr={rejection_stats['min_snr']}, gdop={rejection_stats['gdop']}. "
                f"Consider relaxing constraints: increase GDOP range, decrease min_snr, or decrease min_receivers."
            )
            
            with db_manager.get_session() as session:
                fail_query = text("""
                    UPDATE heimdall.training_jobs
                    SET status = 'failed', completed_at = NOW(), 
                        error_message = :error_message,
                        dataset_id = :dataset_id
                    WHERE id = :job_id
                """)
                session.execute(fail_query, {
                    "job_id": job_id,
                    "dataset_id": str(dataset_id),
                    "error_message": error_message
                })
                session.commit()

            # Publish job failure event
            event_publisher.publish_training_job_update(
                job_id=job_id,
                status='failed',
                action='failed',
                current_progress=stats['total_generated'],
                total_progress=config.get('num_samples', stats['total_generated']),
                progress_percent=0.0,
                progress_message=error_message,
                job_name=config.get('job_name', 'Unnamed Job'),
                job_type='synthetic_generation',
                dataset_id=str(dataset_id),
                result={
                    "dataset_id": str(dataset_id),
                    "num_samples": stats['total_generated'],
                    "iq_samples_saved": stats['iq_samples_saved'],
                    "success_rate": stats['success_rate'],
                    "stopped_reason": stopped_reason,
                    "rejection_stats": rejection_stats
                }
            )
            logger.warning(f"Job {job_id} marked as failed due to consecutive failures")

            return {
                "status": "failed",
                "job_id": job_id,
                "dataset_id": str(dataset_id),
                "num_samples": stats['total_generated'],
                "error": error_message
            }

        # Update job as completed (only if not stopped early)
        with db_manager.get_session() as session:
            complete_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'completed', completed_at = NOW(), progress_percent = 100.0,
                    dataset_id = :dataset_id
                WHERE id = :job_id
            """)
            session.execute(complete_query, {"job_id": job_id, "dataset_id": str(dataset_id)})
            session.commit()

        # Publish job completion event with complete data
        event_publisher.publish_training_job_update(
            job_id=job_id,
            status='completed',
            action='completed',
            current_progress=stats['total_generated'],
            total_progress=config.get('num_samples', stats['total_generated']),
            progress_percent=100.0,
            progress_message=f"Completed: {stats['total_generated']} samples generated",
            job_name=config.get('job_name', 'Unnamed Job'),
            job_type='synthetic_generation',
            dataset_id=str(dataset_id),
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

        # Count and preserve any samples generated before failure (if dataset_id is defined)
        # This handles partial progress: if job fails after generating some samples,
        # we update num_samples to reflect partial results instead of losing all work
        if 'dataset_id' in locals():
            try:
                with db_manager.get_session() as session:
                    # Count actual samples in measurement_features
                    count_query = text("""
                        SELECT COUNT(*) FROM heimdall.measurement_features
                        WHERE dataset_id = :dataset_id
                    """)
                    actual_count = session.execute(count_query, {"dataset_id": str(dataset_id)}).scalar() or 0

                    # Update dataset with partial results
                    if actual_count > 0:
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
                        logger.info(f"Preserved {actual_count} samples generated before failure for dataset {dataset_id}")

                        # Publish dataset update event for partial results
                        try:
                            # Get event publisher (may not be initialized if failure was early)
                            from backend.src.events.publisher import get_event_publisher
                            failure_publisher = get_event_publisher()
                            dataset_type = config.get('dataset_type', 'feature_based') if 'config' in locals() else 'feature_based'
                            failure_publisher.publish_dataset_updated(
                                dataset_id=str(dataset_id),
                                num_samples=actual_count,
                                action="partial_failure",
                                dataset_type=dataset_type
                            )
                            logger.info(f"Published dataset update event for partial results (samples={actual_count})")
                        except Exception as publish_error:
                            logger.error(f"Failed to publish dataset update event: {publish_error}", exc_info=True)
                    else:
                        logger.info(f"No samples to preserve for dataset {dataset_id} (count=0)")

            except Exception as count_error:
                logger.error(f"Failed to count/preserve partial samples: {count_error}", exc_info=True)
        else:
            logger.info("Job failed before dataset_id was defined - no partial samples to preserve")

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
        try:
            # Get event publisher (may not be initialized if failure was early)
            from backend.src.events.publisher import get_event_publisher
            failure_publisher = get_event_publisher()
            failure_publisher.publish_training_job_update(
                job_id=job_id,
                status='failed',
                action='failed',
                error_message=str(e)
            )
            logger.error(f"Published job failure event for {job_id}")
        except Exception as publish_error:
            logger.error(f"Failed to publish job failure event: {publish_error}", exc_info=True)

        raise


@celery_app.task(bind=True, base=TrainingTask, name='src.tasks.training_task.evaluate_model_task')
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


@celery_app.task(bind=True, base=TrainingTask, name='src.tasks.training_task.export_model_onnx_task')
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
