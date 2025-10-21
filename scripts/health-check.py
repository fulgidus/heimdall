#!/usr/bin/env python3
"""
Health check script for Heimdall infrastructure services.
Verifies that all critical components are running and responsive.
"""

import sys
import time
import psycopg2
import pika
import redis
from urllib.parse import urlparse
from minio import Minio
from minio.error import S3Error
import requests
from pathlib import Path

# Configuration
SERVICES = {
    'postgresql': {
        'host': 'localhost',
        'port': 5432,
        'user': 'heimdall_user',
        'password': 'changeme',
        'database': 'heimdall'
    },
    'rabbitmq': {
        'host': 'localhost',
        'port': 5672,
        'user': 'guest',
        'password': 'guest',
        'vhost': '/'
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'password': 'changeme'
    },
    'minio': {
        'endpoint': 'localhost:9000',
        'access_key': 'minioadmin',
        'secret_key': 'minioadmin',
        'secure': False
    },
    'prometheus': {
        'url': 'http://localhost:9090'
    },
    'grafana': {
        'url': 'http://localhost:3000'
    }
}

class HealthChecker:
    """Utility class to check service health"""
    
    def __init__(self):
        self.results = {}
    
    def check_postgresql(self):
        """Check PostgreSQL connectivity"""
        try:
            conn = psycopg2.connect(
                host=SERVICES['postgresql']['host'],
                port=SERVICES['postgresql']['port'],
                user=SERVICES['postgresql']['user'],
                password=SERVICES['postgresql']['password'],
                database=SERVICES['postgresql']['database'],
                connect_timeout=5
            )
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            cursor.close()
            conn.close()
            self.results['PostgreSQL'] = '✅ OK - ' + version[0].split(',')[0]
            return True
        except Exception as e:
            self.results['PostgreSQL'] = f'❌ FAILED - {str(e)}'
            return False
    
    def check_rabbitmq(self):
        """Check RabbitMQ connectivity"""
        try:
            credentials = pika.PlainCredentials(
                SERVICES['rabbitmq']['user'],
                SERVICES['rabbitmq']['password']
            )
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=SERVICES['rabbitmq']['host'],
                    port=SERVICES['rabbitmq']['port'],
                    credentials=credentials,
                    connection_attempts=3,
                    retry_delay=2,
                    socket_timeout=5
                )
            )
            channel = connection.channel()
            channel.queue_declare(queue='test', auto_delete=True)
            connection.close()
            self.results['RabbitMQ'] = '✅ OK - Connection successful'
            return True
        except Exception as e:
            self.results['RabbitMQ'] = f'❌ FAILED - {str(e)}'
            return False
    
    def check_redis(self):
        """Check Redis connectivity"""
        try:
            r = redis.Redis(
                host=SERVICES['redis']['host'],
                port=SERVICES['redis']['port'],
                password=SERVICES['redis']['password'],
                socket_connect_timeout=5,
                decode_responses=True
            )
            pong = r.ping()
            if pong:
                self.results['Redis'] = '✅ OK - PONG received'
                return True
        except Exception as e:
            self.results['Redis'] = f'❌ FAILED - {str(e)}'
            return False
    
    def check_minio(self):
        """Check MinIO connectivity and buckets"""
        try:
            client = Minio(
                SERVICES['minio']['endpoint'],
                access_key=SERVICES['minio']['access_key'],
                secret_key=SERVICES['minio']['secret_key'],
                secure=SERVICES['minio']['secure']
            )
            
            # List buckets to verify connectivity
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets.buckets]
            
            required_buckets = [
                'heimdall-raw-iq',
                'heimdall-models',
                'heimdall-mlflow',
                'heimdall-datasets'
            ]
            
            missing = [b for b in required_buckets if b not in bucket_names]
            if missing:
                self.results['MinIO'] = f'⚠️  WARNING - Missing buckets: {missing}'
                return False
            else:
                self.results['MinIO'] = f'✅ OK - All {len(required_buckets)} buckets present'
                return True
        except Exception as e:
            self.results['MinIO'] = f'❌ FAILED - {str(e)}'
            return False
    
    def check_prometheus(self):
        """Check Prometheus API"""
        try:
            response = requests.get(
                f"{SERVICES['prometheus']['url']}/api/v1/status/runtimeinfo",
                timeout=5
            )
            if response.status_code == 200:
                self.results['Prometheus'] = '✅ OK - API responding'
                return True
            else:
                self.results['Prometheus'] = f'❌ FAILED - Status {response.status_code}'
                return False
        except Exception as e:
            self.results['Prometheus'] = f'❌ FAILED - {str(e)}'
            return False
    
    def check_grafana(self):
        """Check Grafana API"""
        try:
            response = requests.get(
                f"{SERVICES['grafana']['url']}/api/health",
                timeout=5
            )
            if response.status_code == 200:
                self.results['Grafana'] = '✅ OK - API responding'
                return True
            else:
                self.results['Grafana'] = f'❌ FAILED - Status {response.status_code}'
                return False
        except Exception as e:
            self.results['Grafana'] = f'❌ FAILED - {str(e)}'
            return False
    
    def run_all_checks(self):
        """Run all health checks"""
        print("🏥 Heimdall Infrastructure Health Check\n")
        print("=" * 60)
        
        self.check_postgresql()
        self.check_rabbitmq()
        self.check_redis()
        self.check_minio()
        self.check_prometheus()
        self.check_grafana()
        
        # Print results
        for service, status in self.results.items():
            print(f"{service:20} {status}")
        
        print("=" * 60)
        
        # Overall status
        failed = [s for s, st in self.results.items() if '❌' in st]
        if not failed:
            print("✅ All services healthy!\n")
            return 0
        else:
            print(f"❌ {len(failed)} service(s) failed: {', '.join(failed)}\n")
            return 1

if __name__ == '__main__':
    checker = HealthChecker()
    sys.exit(checker.run_all_checks())
