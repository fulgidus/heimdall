#!/usr/bin/env python3
"""
Test script for training event broadcasting integration.

Tests:
1. RabbitMQ exchange and queue exist
2. Event publisher can send messages
3. WebSocket endpoint is accessible
4. End-to-end event flow (publisher -> RabbitMQ -> consumer -> WebSocket)

Usage:
    python scripts/test_training_events.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Optional

import requests
import websocket
from kombu import Connection, Exchange, Producer

# Configuration
BACKEND_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001/ws"
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
RABBITMQ_MGMT_URL = "http://localhost:15672"

# Test results
test_results = []


def log_test(name: str, passed: bool, message: str = ""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((name, passed, message))
    print(f"{status}: {name}")
    if message:
        print(f"   {message}")


def test_backend_health():
    """Test 1: Backend service is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        passed = response.status_code == 200
        log_test("Backend Health", passed, f"Status: {response.status_code}")
        return passed
    except Exception as e:
        log_test("Backend Health", False, f"Error: {e}")
        return False


def test_rabbitmq_connection():
    """Test 2: RabbitMQ is accessible."""
    try:
        conn = Connection(RABBITMQ_URL)
        conn.connect()
        conn.release()
        log_test("RabbitMQ Connection", True, "Connected successfully")
        return True
    except Exception as e:
        log_test("RabbitMQ Connection", False, f"Error: {e}")
        return False


def test_rabbitmq_exchange():
    """Test 3: heimdall.events exchange exists."""
    try:
        response = requests.get(
            f"{RABBITMQ_MGMT_URL}/api/exchanges/%2F/heimdall.events",
            auth=("guest", "guest"),
            timeout=5
        )
        passed = response.status_code == 200
        if passed:
            exchange = response.json()
            log_test(
                "RabbitMQ Exchange",
                True,
                f"Type: {exchange['type']}, Durable: {exchange['durable']}"
            )
        else:
            log_test("RabbitMQ Exchange", False, f"Status: {response.status_code}")
        return passed
    except Exception as e:
        log_test("RabbitMQ Exchange", False, f"Error: {e}")
        return False


def test_event_publisher():
    """Test 4: Event publisher can send test event."""
    try:
        connection = Connection(RABBITMQ_URL)
        exchange = Exchange('heimdall.events', type='topic', durable=False)
        
        test_event = {
            'event': 'test:event',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'test_id': 'integration_test',
                'message': 'Test event from test_training_events.py'
            }
        }
        
        with connection.Producer() as producer:
            producer.publish(
                test_event,
                exchange=exchange,
                routing_key='test.event',
                serializer='json',
                declare=[exchange]
            )
        
        log_test("Event Publisher", True, "Test event published to RabbitMQ")
        return True
    except Exception as e:
        log_test("Event Publisher", False, f"Error: {e}")
        return False


def test_websocket_connection():
    """Test 5: WebSocket endpoint is accessible."""
    try:
        ws = websocket.create_connection(WS_URL, timeout=5)
        ws.close()
        log_test("WebSocket Connection", True, "Connected successfully")
        return True
    except Exception as e:
        log_test("WebSocket Connection", False, f"Error: {e}")
        return False


def test_end_to_end_event_flow():
    """Test 6: End-to-end event flow (publish -> consume -> WebSocket)."""
    try:
        print("\n🔄 Starting end-to-end event flow test...")
        
        # Connect to WebSocket
        print("   Connecting to WebSocket...")
        ws = websocket.create_connection(WS_URL, timeout=10)
        
        # Publish test event
        print("   Publishing test event to RabbitMQ...")
        connection = Connection(RABBITMQ_URL)
        exchange = Exchange('heimdall.events', type='topic', durable=False)
        
        test_event = {
            'event': 'training:progress',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {
                'job_id': 'test-job-123',
                'epoch': 1,
                'total_epochs': 10,
                'progress_percent': 10.0,
                'metrics': {
                    'train_loss': 0.5,
                    'val_loss': 0.6,
                    'train_rmse': 100.0,
                    'val_rmse': 110.0
                },
                'is_best': False
            }
        }
        
        with connection.Producer() as producer:
            producer.publish(
                test_event,
                exchange=exchange,
                routing_key='training.progress.test-job-123',
                serializer='json',
                declare=[exchange]
            )
        
        print("   Event published. Waiting for WebSocket message...")
        
        # Wait for message (timeout after 5 seconds)
        ws.settimeout(5.0)
        try:
            message = ws.recv()
            received_event = json.loads(message)
            
            # Verify event content
            passed = (
                received_event.get('event') == 'training:progress' and
                received_event.get('data', {}).get('job_id') == 'test-job-123'
            )
            
            if passed:
                log_test(
                    "End-to-End Event Flow",
                    True,
                    f"Event received via WebSocket: {received_event['event']}"
                )
            else:
                log_test(
                    "End-to-End Event Flow",
                    False,
                    f"Event content mismatch: {received_event}"
                )
        except websocket.WebSocketTimeoutException:
            log_test(
                "End-to-End Event Flow",
                False,
                "Timeout waiting for WebSocket message (consumer may not be running)"
            )
            passed = False
        
        ws.close()
        return passed
        
    except Exception as e:
        log_test("End-to-End Event Flow", False, f"Error: {e}")
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p, _ in test_results if p)
    total = len(test_results)
    
    for name, passed_flag, message in test_results:
        status = "✅" if passed_flag else "❌"
        print(f"{status} {name}")
        if message and not passed_flag:
            print(f"   └─ {message}")
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs above for details.")
        return 1


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRAINING EVENT BROADCASTING INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Run tests
    test_backend_health()
    test_rabbitmq_connection()
    test_rabbitmq_exchange()
    test_event_publisher()
    test_websocket_connection()
    test_end_to_end_event_flow()
    
    # Print summary
    exit_code = print_summary()
    
    print("\n📝 Notes:")
    print("   - If end-to-end test fails, check:")
    print("     1. RabbitMQ consumer is running (check backend logs)")
    print("     2. WebSocket manager is initialized")
    print("     3. Event consumer has event loop reference")
    print()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
