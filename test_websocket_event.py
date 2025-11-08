#!/usr/bin/env python3
"""
Test script to manually publish a dataset:generation_progress event to RabbitMQ
and verify it flows through to WebSocket clients.

Usage:
    python test_websocket_event.py
"""

import json
import uuid
from datetime import datetime

from kombu import Connection, Exchange, Producer

# RabbitMQ configuration (use 'rabbitmq' hostname when running inside Docker)
BROKER_URL = "amqp://guest:guest@rabbitmq:5672//"

# Events exchange (must match backend EventPublisher)
events_exchange = Exchange(
    'heimdall.events',
    type='topic',
    durable=False,
    auto_delete=False
)

def publish_test_event():
    """Publish a test dataset generation progress event."""
    
    # Create test event (matches EventPublisher.publish_dataset_generation_progress format)
    test_job_id = str(uuid.uuid4())
    event = {
        'event': 'dataset:generation_progress',
        'timestamp': datetime.utcnow().isoformat(),
        'data': {
            'job_id': test_job_id,
            'current': 50,
            'total': 100,
            'progress_percent': 50.0,
            'message': 'Test progress event from manual script'
        }
    }
    
    print(f"📤 Publishing test event to RabbitMQ:")
    print(f"   Exchange: heimdall.events")
    print(f"   Routing Key: dataset.generation.{test_job_id}")
    print(f"   Event: {json.dumps(event, indent=2)}")
    
    # Publish to RabbitMQ
    with Connection(BROKER_URL) as conn:
        with Producer(conn) as producer:
            producer.publish(
                event,
                exchange=events_exchange,
                routing_key=f'dataset.generation.{test_job_id}',
                serializer='json',
                declare=[events_exchange]
            )
    
    print("✅ Event published successfully!")
    print(f"\n🔍 Check frontend browser console for:")
    print(f"   [WebSocket] Incoming event: dataset:generation_progress")
    print(f"   [SyntheticTab] ===== DATASET GENERATION PROGRESS EVENT =====")
    print(f"\n📊 If you don't see these logs, the WebSocket consumer is not receiving events from RabbitMQ.")

if __name__ == '__main__':
    publish_test_event()
