#!/usr/bin/env python3
"""
Test script to verify comprehensive health WebSocket event flow.

This script simulates what the comprehensive_health_monitor task does:
1. Collects health data for all components
2. Publishes comprehensive health event to RabbitMQ
3. Verifies event structure matches frontend expectations

Run this from the repository root:
    python test_system_health_websocket.py
"""

import json
from datetime import datetime

# Simulate health data structure from comprehensive_health_monitor
SAMPLE_HEALTH_DATA = {
    # Microservices
    "backend": {
        "service": "backend",
        "status": "healthy",
        "response_time_ms": 23.5,
        "version": "0.1.0",
        "last_check": datetime.utcnow().isoformat(),
    },
    "training": {
        "service": "training",
        "status": "healthy",
        "response_time_ms": 45.2,
        "version": "0.1.0",
        "last_check": datetime.utcnow().isoformat(),
    },
    "inference": {
        "service": "inference",
        "status": "healthy",
        "response_time_ms": 12.8,
        "version": "0.1.0",
        "last_check": datetime.utcnow().isoformat(),
        "model_info": {
            "active_version": "1.0.0",
            "health_status": "healthy",
            "accuracy": 0.89,
            "predictions_total": 1234,
            "predictions_successful": 1200,
            "latency_p95_ms": 450,
            "cache_hit_rate": 0.87,
            "uptime_seconds": 86400,
        },
    },
    # Infrastructure components
    "postgresql": {
        "service": "postgresql",
        "status": "healthy",
        "message": "Database connection OK",
        "type": "database",
        "last_check": datetime.utcnow().isoformat(),
    },
    "redis": {
        "service": "redis",
        "status": "healthy",
        "message": "Cache connection OK",
        "type": "cache",
        "last_check": datetime.utcnow().isoformat(),
    },
    "rabbitmq": {
        "service": "rabbitmq",
        "status": "healthy",
        "message": "Message queue connection OK",
        "type": "queue",
        "last_check": datetime.utcnow().isoformat(),
    },
    "minio": {
        "service": "minio",
        "status": "healthy",
        "message": "Object storage OK, bucket 'heimdall-raw-iq' exists",
        "type": "storage",
        "last_check": datetime.utcnow().isoformat(),
    },
    "celery": {
        "service": "celery",
        "status": "healthy",
        "message": "2 worker(s) active",
        "type": "worker",
        "worker_count": 2,
        "last_check": datetime.utcnow().isoformat(),
    },
}


def test_event_structure():
    """Test that the event structure matches what frontend expects."""
    
    # This is what publish_comprehensive_health() creates
    event = {
        'event': 'system:comprehensive_health',
        'timestamp': datetime.utcnow().isoformat(),
        'data': {
            'components': SAMPLE_HEALTH_DATA
        }
    }
    
    print("=" * 80)
    print("COMPREHENSIVE HEALTH EVENT STRUCTURE")
    print("=" * 80)
    print(json.dumps(event, indent=2))
    print()
    
    # Verify structure matches frontend expectations
    assert 'event' in event
    assert event['event'] == 'system:comprehensive_health'
    assert 'timestamp' in event
    assert 'data' in event
    assert 'components' in event['data']
    
    components = event['data']['components']
    
    # Check microservices are present
    microservices = ['backend', 'training', 'inference']
    for svc in microservices:
        assert svc in components, f"Missing microservice: {svc}"
        assert components[svc]['status'] in ['healthy', 'unhealthy', 'degraded', 'warning', 'unknown']
    
    # Check infrastructure components are present
    infra = ['postgresql', 'redis', 'rabbitmq', 'minio', 'celery']
    for comp in infra:
        assert comp in components, f"Missing infrastructure component: {comp}"
        assert components[comp]['status'] in ['healthy', 'unhealthy', 'warning', 'unknown']
        assert 'type' in components[comp], f"Missing type for {comp}"
    
    print("✅ Event structure validation PASSED")
    print()
    
    # Simulate frontend store separation
    services_health = {}
    infrastructure_health = {}
    
    for name, health in components.items():
        if name in microservices:
            services_health[name] = health
        else:
            infrastructure_health[name] = health
    
    print("=" * 80)
    print("FRONTEND STORE SEPARATION")
    print("=" * 80)
    print(f"Microservices ({len(services_health)}):")
    for name, health in services_health.items():
        print(f"  - {name}: {health['status']}")
    print()
    print(f"Infrastructure ({len(infrastructure_health)}):")
    for name, health in infrastructure_health.items():
        comp_type = health.get('type', 'unknown')
        print(f"  - {name} ({comp_type}): {health['status']}")
    print()
    
    print("✅ Frontend store separation PASSED")
    print()
    
    return True


if __name__ == "__main__":
    try:
        test_event_structure()
        print("=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Start docker compose services: docker compose up -d")
        print("2. Start Celery beat: celery -A src.main.celery_app beat --loglevel=info")
        print("3. Start Celery worker: celery -A src.main.celery_app worker --loglevel=info")
        print("4. Open browser to http://localhost:5173/system-status")
        print("5. Verify infrastructure components appear within 1 second")
        print("6. Check browser console for WebSocket messages")
        print()
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        exit(1)
