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

    def publish_training_progress(
        self,
        job_id: str,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Publish training progress update.

        Args:
            job_id: Training job UUID
            epoch: Current epoch
            total_epochs: Total epochs
            metrics: Training metrics (loss, accuracy, etc.)
        """
        event = {
            'event': 'training:progress',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'job_id': job_id,
                'epoch': epoch,
                'total_epochs': total_epochs,
                'progress_percent': (epoch / total_epochs * 100) if total_epochs > 0 else 0,
                'metrics': metrics
            }
        }

        self._publish(f'training.progress.{job_id}', event)
        logger.debug(f"Published training progress: job {job_id}, epoch {epoch}/{total_epochs}")


# Singleton instance for convenience
_publisher = None

def get_event_publisher() -> EventPublisher:
    """Get singleton EventPublisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher
