"""
Celery tasks for training pipeline operations.

Tasks:
- start_training_job: Start model training
- generate_synthetic_data_task: Generate synthetic training data
- evaluate_model_task: Evaluate trained model
- export_model_onnx_task: Export model to ONNX
"""

import uuid
from datetime import datetime, timezone
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
    Start training job.
    
    This task is called from the API to start model training.
    It coordinates the training pipeline.
    
    Args:
        job_id: Training job UUID
    
    Returns:
        Dict with training results
    """
    from ..storage.db_manager import get_db_manager
    from sqlalchemy import text
    
    logger.info(f"Starting training job {job_id}")
    
    db_manager = get_db_manager()
    
    try:
        # Update job status
        with db_manager.get_session() as session:
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'running', started_at = NOW()
                WHERE id = :job_id
            """)
            session.execute(update_query, {"job_id": job_id})
            session.commit()
        
        # Training logic would go here
        # For Phase 5, this is a placeholder
        logger.warning(f"Training job {job_id}: Training logic not implemented yet")
        
        # Update job as completed
        with db_manager.get_session() as session:
            complete_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'completed', completed_at = NOW(), progress_percent = 100.0
                WHERE id = :job_id
            """)
            session.execute(complete_query, {"job_id": job_id})
            session.commit()
        
        return {"status": "completed", "job_id": job_id}
    
    except Exception as e:
        logger.error(f"Error in training job {job_id}: {e}", exc_info=True)
        
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
def generate_synthetic_data_task(self, job_id: str):
    """
    Generate synthetic training data.
    
    This task:
    1. Loads training configuration from database
    2. Generates synthetic samples using RF propagation
    3. Saves samples to TimescaleDB
    4. Updates job status
    
    Args:
        job_id: Training job UUID
    
    Returns:
        Dict with generation results
    """
    from ..storage.db_manager import get_db_manager
    from sqlalchemy import text
    import json
    
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
            
            config = json.loads(result[0])
        
        # Update status to running
        with db_manager.get_session() as session:
            update_query = text("""
                UPDATE heimdall.training_jobs
                SET status = 'running', started_at = NOW()
                WHERE id = :job_id
            """)
            session.execute(update_query, {"job_id": job_id})
            session.commit()
        
        # Generate synthetic data
        # Import here to avoid circular dependencies
        import sys
        import os
        
        # Add training service to path (hacky but works for Phase 5)
        training_path = "/home/runner/work/heimdall/heimdall/services/training/src"
        if training_path not in sys.path:
            sys.path.insert(0, training_path)
        
        from data.config import TrainingConfig, get_italian_receivers
        from data.propagation import RFPropagationModel
        from data.terrain import TerrainLookup
        from data.synthetic_generator import SyntheticDataGenerator, save_samples_to_db, calculate_quality_metrics
        
        # Create training config
        receivers = get_italian_receivers()
        training_config = TrainingConfig.from_receivers(receivers, margin_degrees=0.5)
        
        # Create generator
        propagation = RFPropagationModel()
        terrain = TerrainLookup(use_srtm=False)
        generator = SyntheticDataGenerator(training_config, propagation, terrain)
        
        # Progress callback
        def progress_callback(current, total, message):
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'message': message,
                    'progress_percent': (current / total) * 100
                }
            )
        
        # Generate samples
        logger.info(f"Generating {config['num_samples']} synthetic samples")
        samples = generator.generate_samples(
            num_samples=config['num_samples'],
            inside_ratio=config.get('inside_ratio', 0.7),
            train_ratio=config.get('train_ratio', 0.7),
            val_ratio=config.get('val_ratio', 0.15),
            test_ratio=config.get('test_ratio', 0.15),
            frequency_mhz=config.get('frequency_mhz', 145.0),
            tx_power_dbm=config.get('tx_power_dbm', 37.0),
            min_snr_db=config.get('min_snr_db', 3.0),
            min_receivers=config.get('min_receivers', 3),
            max_gdop=config.get('max_gdop', 10.0),
            progress_callback=progress_callback
        )
        
        logger.info(f"Generated {len(samples)} samples")
        
        # Calculate quality metrics
        quality_metrics = calculate_quality_metrics(samples)
        
        # Calculate split counts
        train_count = sum(1 for s in samples if s.split == 'train')
        val_count = sum(1 for s in samples if s.split == 'val')
        test_count = sum(1 for s in samples if s.split == 'test')
        
        # Create dataset record
        dataset_id = uuid.uuid4()
        with db_manager.get_session() as session:
            dataset_query = text("""
                INSERT INTO heimdall.synthetic_datasets (
                    id, name, description, num_samples, train_count, val_count, test_count,
                    config, quality_metrics, created_by_job_id
                )
                VALUES (
                    :id, :name, :description, :num_samples, :train_count, :val_count, :test_count,
                    :config::jsonb, :quality_metrics::jsonb, :job_id
                )
            """)
            
            session.execute(
                dataset_query,
                {
                    "id": str(dataset_id),
                    "name": config['name'],
                    "description": config.get('description'),
                    "num_samples": len(samples),
                    "train_count": train_count,
                    "val_count": val_count,
                    "test_count": test_count,
                    "config": json.dumps(config),
                    "quality_metrics": json.dumps(quality_metrics),
                    "job_id": job_id
                }
            )
            session.commit()
        
        logger.info(f"Created dataset {dataset_id}")
        
        # Save samples to database
        with db_manager.get_session() as session:
            num_saved = await save_samples_to_db(samples, dataset_id, session)
            logger.info(f"Saved {num_saved} samples to database")
        
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
            "num_samples": len(samples),
            "quality_metrics": quality_metrics
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
