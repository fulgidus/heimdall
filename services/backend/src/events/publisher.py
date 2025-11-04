"""
RabbitMQ Event Publisher

Publishes events from Celery tasks and other backend components to RabbitMQ
for consumption by WebSocket managers and other services.

Architecture Decision:
- Use RabbitMQ (not Redis Pub/Sub) for event broadcasting
- Topic exchange for flexible routing
- Non-durable messages (ephemeral events)
- Separate from Celery task queue (different exchange)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from kombu import Connection, Exchange, Producer

logger = logging.getLogger(__name__)

# RabbitMQ configuration
BROKER_URL = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@rabbitmq:5672/')

# Events exchange (separate from Celery tasks)
EVENTS_EXCHANGE = Exchange(
    'heimdall.events',
    type='topic',
    durable=False,  # Ephemeral events, no persistence needed
    auto_delete=False
)


class EventPublisher:
    """
    Publisher for real-time events to RabbitMQ.

    Usage:
        publisher = EventPublisher()
        publisher.publish_websdr_health(health_data)
        publisher.publish_signal_detection(signal_data)
    """

    def __init__(self, broker_url: Optional[str] = None):
        """
        Initialize event publisher.

        Args:
            broker_url: RabbitMQ connection URL (default: from env)
        """
        self.broker_url = broker_url or BROKER_URL
        self.exchange = EVENTS_EXCHANGE

    def _publish(self, routing_key: str, data: Dict[str, Any]) -> None:
        """
        Internal method to publish event to RabbitMQ.

        Args:
            routing_key: Routing key for topic exchange (e.g., 'websdr.health.update')
            data: Event payload (must be JSON-serializable)
        """
        try:
            with Connection(self.broker_url) as connection:
                with connection.Producer() as producer:
                    producer.publish(
                        data,
                        exchange=self.exchange,
                        routing_key=routing_key,
                        serializer='json',
                        declare=[self.exchange],
                        retry=True,
                        retry_policy={
                            'max_retries': 3,
                            'interval_start': 0,
                            'interval_step': 0.5,
                            'interval_max': 2,
                        }
                    )
                    logger.debug(f"Published event: {routing_key}")

        except Exception as e:
            logger.error(f"Failed to publish event {routing_key}: {e}", exc_info=True)
            # Don't raise - event publishing failures should not crash the publisher

    def publish_websdr_health(self, health_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Publish WebSDR health status update.

        Args:
            health_data: Dict mapping websdr_id to health status
                Example: {
                    "1": {"websdr_id": "1", "name": "Torino", "status": "online", ...},
                    "2": {"websdr_id": "2", "name": "Milano", "status": "offline", ...}
                }
        """
        event = {
            'event': 'websdrs:health',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'health_status': health_data
            }
        }

        self._publish('websdr.health.update', event)
        logger.info(f"Published WebSDR health update for {len(health_data)} stations")

    def publish_service_health(self, service_name: str, status: str, **kwargs) -> None:
        """
        Publish microservice health status update.

        Args:
            service_name: Service name (backend, training, inference)
            status: Health status (healthy, unhealthy, degraded)
            **kwargs: Additional health data (response_time_ms, error, etc.)
        """
        event = {
            'event': 'service:health',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'service': service_name,
                'status': status,
                **kwargs
            }
        }

        self._publish(f'service.health.{service_name}', event)
        logger.debug(f"Published service health: {service_name} = {status}")

    def publish_signal_detection(
        self,
        frequency_hz: float,
        signal_strength_db: float,
        station_id: str,
        **kwargs
    ) -> None:
        """
        Publish signal detection event.

        Args:
            frequency_hz: Detected frequency
            signal_strength_db: Signal strength
            station_id: WebSDR station that detected signal
            **kwargs: Additional signal data
        """
        event = {
            'event': 'signal:detected',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'frequency_hz': frequency_hz,
                'signal_strength_db': signal_strength_db,
                'station_id': station_id,
                **kwargs
            }
        }

        self._publish('signal.detected', event)
        logger.info(f"Published signal detection: {frequency_hz} Hz @ {station_id}")

    def publish_localization_result(
        self,
        latitude: float,
        longitude: float,
        accuracy_meters: float,
        **kwargs
    ) -> None:
        """
        Publish localization result.

        Args:
            latitude: Estimated TX latitude
            longitude: Estimated TX longitude
            accuracy_meters: Estimated accuracy
            **kwargs: Additional localization data
        """
        event = {
            'event': 'localization:complete',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'latitude': latitude,
                'longitude': longitude,
                'accuracy_meters': accuracy_meters,
                **kwargs
            }
        }

        self._publish('localization.complete', event)
        logger.info(f"Published localization result: ({latitude}, {longitude}) ±{accuracy_meters}m")

    def publish_training_started(
        self,
        job_id: str,
        config: Dict[str, Any],
        dataset_size: int,
        train_samples: int,
        val_samples: int
    ) -> None:
        """
        Publish training job started event.

        Args:
            job_id: Training job UUID
            config: Training configuration
            dataset_size: Total dataset size
            train_samples: Number of training samples
            val_samples: Number of validation samples
        """
        event = {
            'event': 'training:started',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'job_id': job_id,
                'status': 'running',
                'config': config,
                'dataset_size': dataset_size,
                'train_samples': train_samples,
                'val_samples': val_samples
            }
        }

        self._publish(f'training.started.{job_id}', event)
        logger.info(f"Published training started: job {job_id} ({train_samples} train, {val_samples} val)")

    def publish_training_progress(
        self,
        job_id: str,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Publish training progress update (per epoch).

        Args:
            job_id: Training job UUID
            epoch: Current epoch
            total_epochs: Total epochs
            metrics: Training metrics (train_loss, val_loss, train_acc, val_acc, lr, etc.)
            is_best: Whether this is the best epoch so far
        """
        event = {
            'event': 'training:progress',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'job_id': job_id,
                'epoch': epoch,
                'total_epochs': total_epochs,
                'progress_percent': (epoch / total_epochs * 100) if total_epochs > 0 else 0,
                'metrics': metrics,
                'is_best': is_best
            }
        }

        self._publish(f'training.progress.{job_id}', event)
        logger.info(f"Published training progress: job {job_id}, epoch {epoch}/{total_epochs}, metrics={metrics}")

    def publish_training_completed(
        self,
        job_id: str,
        status: str,
        best_epoch: Optional[int] = None,
        best_val_loss: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        onnx_model_path: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Publish training job completed event.

        Args:
            job_id: Training job UUID
            status: Final status (completed, failed, cancelled)
            best_epoch: Best epoch number
            best_val_loss: Best validation loss
            checkpoint_path: Path to best checkpoint
            onnx_model_path: Path to ONNX model
            mlflow_run_id: MLflow run ID
            error_message: Error message if failed
        """
        event = {
            'event': 'training:completed',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'job_id': job_id,
                'status': status,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'checkpoint_path': checkpoint_path,
                'onnx_model_path': onnx_model_path,
                'mlflow_run_id': mlflow_run_id,
                'error_message': error_message
            }
        }

        self._publish(f'training.completed.{job_id}', event)
        logger.info(f"Published training completed: job {job_id}, status={status}")

    def publish_dataset_generation_progress(
        self,
        job_id: str,
        current: int,
        total: int,
        message: str
    ) -> None:
        """
        Publish synthetic dataset generation progress.

        Args:
            job_id: Job UUID
            current: Current progress (samples generated)
            total: Total samples to generate
            message: Progress message
        """
        event = {
            'event': 'dataset:generation_progress',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'job_id': job_id,
                'current': current,
                'total': total,
                'progress_percent': (current / total * 100) if total > 0 else 0,
                'message': message
            }
        }

        self._publish(f'dataset.generation.{job_id}', event)
        logger.debug(f"Published dataset generation progress: job {job_id}, {current}/{total}")

    def publish_terrain_tile_progress(
        self,
        tile_name: str,
        status: str,
        current: int,
        total: int,
        error: Optional[str] = None,
        file_size: Optional[int] = None
    ) -> None:
        """
        Publish terrain tile download progress.

        Args:
            tile_name: Tile name (e.g., 'N44E007')
            status: Tile status ('downloading', 'ready', 'failed')
            current: Current tile number
            total: Total tiles to download
            error: Optional error message
            file_size: Optional file size in bytes
        """
        event = {
            'event': 'terrain:tile_progress',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'tile_name': tile_name,
                'status': status,
                'current': current,
                'total': total,
                'progress_percent': (current / total * 100) if total > 0 else 0,
                'error': error,
                'file_size': file_size
            }
        }

        self._publish('terrain.tile.progress', event)
        logger.debug(f"Published terrain tile progress: {tile_name} - {status} ({current}/{total})")


# Singleton instance for convenience
_publisher = None

def get_event_publisher() -> EventPublisher:
    """Get singleton EventPublisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher
