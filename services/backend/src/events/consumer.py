"""
RabbitMQ Event Consumer

Consumes events from RabbitMQ and broadcasts them to WebSocket clients.
Runs as a background task in the FastAPI application.

Architecture:
- Listens to 'heimdall.events' exchange
- Routes events to appropriate WebSocket handlers
- Non-blocking async consumer
"""

import asyncio
import logging
from typing import Callable, Dict

from kombu import Connection, Queue, Exchange, Consumer
from kombu.mixins import ConsumerMixin

logger = logging.getLogger(__name__)


class RabbitMQEventConsumer(ConsumerMixin):
    """
    RabbitMQ consumer for broadcasting events to WebSocket clients.

    Implements ConsumerMixin for robust connection handling and auto-reconnect.
    """

    def __init__(self, broker_url: str, websocket_manager):
        """
        Initialize consumer.

        Args:
            broker_url: RabbitMQ connection URL
            websocket_manager: WebSocket connection manager for broadcasting
        """
        self.connection = Connection(broker_url)
        self.ws_manager = websocket_manager

        # Events exchange (same as publisher)
        self.events_exchange = Exchange(
            'heimdall.events',
            type='topic',
            durable=False,
            auto_delete=False
        )

        # Queue for this consumer instance
        # Auto-delete when consumer disconnects (ephemeral queue)
        self.events_queue = Queue(
            'heimdall.websocket.events',
            exchange=self.events_exchange,
            routing_key='#',  # Subscribe to ALL events
            auto_delete=True,
            exclusive=False,
            durable=False
        )

        self._event_loop = None

    def get_consumers(self, Consumer, channel):
        """
        Define consumers for this connection.

        Called by ConsumerMixin when connection is established.
        """
        return [
            Consumer(
                queues=[self.events_queue],
                callbacks=[self.on_message],
                accept=['json'],
                prefetch_count=10  # Process up to 10 messages at once
            )
        ]

    def on_message(self, body, message):
        """
        Callback when message arrives from RabbitMQ.

        Args:
            body: Deserialized message body (dict)
            message: Kombu message object
        """
        try:
            event_type = body.get('event', 'unknown')
            logger.debug(f"Received event from RabbitMQ: {event_type}")

            # Broadcast to WebSocket clients (async)
            if self._event_loop and self.ws_manager:
                asyncio.run_coroutine_threadsafe(
                    self.ws_manager.broadcast(body),
                    self._event_loop
                )

                # Log successful broadcast
                num_clients = len(self.ws_manager.active_connections)
                if num_clients > 0:
                    logger.debug(f"Broadcasted {event_type} to {num_clients} WebSocket clients")

            # Acknowledge message
            message.ack()

        except Exception as e:
            logger.error(f"Error processing RabbitMQ message: {e}", exc_info=True)
            # Reject message without requeue (dead letter if configured)
            message.reject(requeue=False)

    def on_connection_error(self, exc, interval):
        """Called when connection error occurs."""
        logger.warning(f"RabbitMQ connection error: {exc}. Retrying in {interval}s...")

    def on_connection_revived(self):
        """Called when connection is re-established."""
        logger.info("RabbitMQ connection revived")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Set the asyncio event loop for broadcasting.

        Must be called before starting consumer.

        Args:
            loop: FastAPI event loop
        """
        self._event_loop = loop


async def start_rabbitmq_consumer(broker_url: str, websocket_manager):
    """
    Start RabbitMQ event consumer in background.

    This function runs as a FastAPI background task.

    Args:
        broker_url: RabbitMQ connection URL
        websocket_manager: WebSocket manager instance
    """
    logger.info("Starting RabbitMQ events consumer for WebSocket broadcasting")

    consumer = RabbitMQEventConsumer(broker_url, websocket_manager)

    # Set event loop for async broadcasting
    consumer.set_event_loop(asyncio.get_running_loop())

    # Run consumer in thread pool (blocking I/O)
    def run_consumer():
        try:
            logger.info("RabbitMQ consumer started successfully")
            consumer.run()  # Blocking call with auto-reconnect
        except KeyboardInterrupt:
            logger.info("RabbitMQ consumer stopped by user")
        except Exception as e:
            logger.error(f"RabbitMQ consumer crashed: {e}", exc_info=True)

    # Run in executor to avoid blocking FastAPI
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_consumer)
