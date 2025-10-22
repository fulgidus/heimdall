# Troubleshooting Guide

## Common Issues

## Installation Issues

### Docker won't start

**Error**: `Docker daemon not running`

**Solution**:
```bash
# Start Docker
# Windows (PowerShell)
Start-Service Docker

# macOS
open /Applications/Docker.app

# Linux
sudo systemctl start docker
```

### Containers fail to start

**Error**: `Ports already in use` or `Cannot connect to Docker daemon`

**Solution**:
```bash
# Find process using port
netstat -tulpn | grep :8000

# Kill process
kill -9 <PID>

# Or change port in .env
# Restart containers
docker-compose restart
```

### Memory issues

**Error**: `OOMKilled` or `Cannot allocate memory`

**Solution**:
```bash
# Check Docker memory allocation
docker system df

# Increase Docker memory in settings
# Settings > Resources > Memory: 4GB+

# Clear unused containers/images
docker system prune -a
```

## Database Issues

### Cannot connect to database

**Error**: `connection refused` or `authentication failed`

**Solution**:
```bash
# Check database is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify credentials in .env
grep DATABASE_URL .env

# Test connection
docker-compose exec postgres psql -U heimdall_user -d heimdall -c "SELECT 1"
```

### Database migration failed

**Error**: `Alembic migration error` or `Schema mismatch`

**Solution**:
```bash
# Check migration status
alembic current

# View migrations
ls db/migrations/versions/

# Rollback failed migration
alembic downgrade -1

# Re-run migration
alembic upgrade head

# If stuck, reset database
make db-reset  # WARNING: Deletes all data!
```

### Slow queries

**Error**: API responses slow or timeouts

**Solution**:
```sql
-- Check slow queries
SELECT query, calls, mean_time FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC;

-- Create missing indexes
CREATE INDEX idx_signal_measurements_created_at 
ON signal_measurements(created_at DESC);

-- Analyze table stats
ANALYZE signal_measurements;

-- Vacuum to reclaim space
VACUUM signal_measurements;
```

## API Issues

### API endpoint returns 404

**Error**: `404 Not Found`

**Solution**:
```bash
# Check API is running
curl http://localhost:8000/health

# Check endpoint is correct
curl http://localhost:8000/api/v1/tasks

# View available routes
curl http://localhost:8000/docs

# Check logs for routing errors
docker-compose logs api-gateway | grep -i error
```

### API request timeout

**Error**: `504 Gateway Timeout` or `Request timed out`

**Solution**:
```bash
# Check worker pool size
docker-compose ps | grep worker

# Scale up workers
docker-compose up -d --scale worker=5

# Check queue depth
docker-compose exec rabbitmq rabbitmqctl list_queues name messages

# Increase timeout in config
# Set REQUEST_TIMEOUT=300 in .env
```

### Authentication/Authorization errors

**Error**: `401 Unauthorized` or `403 Forbidden`

**Solution**:
```bash
# Verify API key
echo $API_KEY

# Check token expiration
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/health

# Generate new token
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update .env
API_KEY=new_key_here
```

## WebSDR Issues

### Cannot connect to WebSDR

**Error**: `Connection refused` or `HTTP 503 Service Unavailable`

**Solution**:
```bash
# Check WebSDR status
curl http://sdr1.ik1jns.it:8076/status

# Verify network connectivity
ping sdr1.ik1jns.it

# Check firewall rules
sudo ufw allow out to any port 8076

# Use VPN if needed (some stations block certain regions)
```

### WebSDR signal quality low

**Error**: No valid signals detected

**Solution**:
- **Try different frequencies**: Use nearby amateur frequencies
- **Try different stations**: Not all stations receive all frequencies
- **Check time**: Some frequencies busier at certain times
- **Increase duration**: Longer acquisitions = better quality data

### WebSDR network unstable

**Error**: Intermittent timeouts, partial data

**Solution**:
```python
# Increase retry count in config
WEBSDR_CONFIG = {
    'retry_count': 5,  # Increase from 3
    'timeout': 15,     # Increase from 10
}

# Use exponential backoff
# Check network connectivity
docker-compose exec api-gateway ping 8.8.8.8
```

## Performance Issues

### High CPU usage

**Error**: High CPU, slow responses

**Solution**:
```bash
# Check which process uses CPU
docker stats

# Profile slow operations
python -m cProfile -s cumulative script.py > profile.txt

# Check for long-running queries
docker-compose logs ml-detector | grep -i slow
```

### High memory usage

**Error**: OOM errors or memory leak

**Solution**:
```bash
# Check memory per container
docker stats

# Identify memory leak
# Look for unbounded arrays or caches

# Clear cache
docker-compose exec redis redis-cli FLUSHALL

# Limit memory per container
# Update docker-compose.yml with mem_limit
```

### GPU not being used

**Error**: GPU utilization 0%, CUDA errors

**Solution**:
```bash
# Check GPU is visible
nvidia-smi

# Check CUDA available
python -c "import torch; print(torch.cuda.is_available())"

# Check driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Update drivers
# Follow nvidia.com instructions for your OS

# Force GPU usage in config
INFERENCE_DEVICE=cuda
```

## Message Queue Issues

### RabbitMQ connection errors

**Error**: `Connection refused` or `AMQP protocol error`

**Solution**:
```bash
# Check RabbitMQ running
docker-compose ps rabbitmq

# Check logs
docker-compose logs rabbitmq

# Reset RabbitMQ
docker-compose exec rabbitmq rabbitmqctl reset
docker-compose restart rabbitmq

# Monitor queue
docker-compose exec rabbitmq rabbitmqctl list_queues
```

### Tasks stuck in queue

**Error**: Tasks not being processed

**Solution**:
```bash
# Check worker count
docker-compose ps worker | wc -l

# Check queue size
docker-compose exec rabbitmq rabbitmqctl list_queues name messages

# Purge queue (WARNING: Deletes tasks!)
docker-compose exec rabbitmq rabbitmqctl purge_queue celery

# Restart workers
docker-compose restart worker
```

## Redis Cache Issues

### Redis connection refused

**Error**: `ConnectionRefusedError: [Errno 111] Connection refused`

**Solution**:
```bash
# Check Redis running
docker-compose ps redis

# Check port mapping
docker-compose port redis 6379

# Check logs
docker-compose logs redis

# Restart Redis
docker-compose restart redis
```

### Memory full

**Error**: `OOM command not allowed when used memory > maxmemory`

**Solution**:
```bash
# Check memory usage
docker-compose exec redis redis-cli INFO memory

# Increase maxmemory
docker-compose exec redis redis-cli CONFIG SET maxmemory 1gb

# Enable eviction policy
docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

## Log & Debugging

### View container logs

```bash
# All containers
docker-compose logs -f

# Specific service
docker-compose logs -f api-gateway

# Last N lines
docker-compose logs --tail 100 signal-processor

# Since specific time
docker-compose logs --since 2025-10-22T10:00:00
```

### Enable debug logging

```bash
# Edit .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart

# View detailed logs
docker-compose logs -f | grep DEBUG
```

### Capture network traffic

```bash
# Using tcpdump
docker-compose exec api-gateway tcpdump -i eth0 -w capture.pcap

# View with Wireshark
# Transfer file and open in Wireshark
```

## Getting Help

### Collect diagnostic information

```bash
#!/bin/bash
# Collect diagnostics for bug report

# System info
echo "=== System Info ===" > diagnostics.txt
uname -a >> diagnostics.txt

# Docker info
echo -e "\n=== Docker Info ===" >> diagnostics.txt
docker --version >> diagnostics.txt
docker compose version >> diagnostics.txt

# Container status
echo -e "\n=== Container Status ===" >> diagnostics.txt
docker-compose ps >> diagnostics.txt

# Recent logs
echo -e "\n=== Recent Logs ===" >> diagnostics.txt
docker-compose logs --tail 100 >> diagnostics.txt

# Environment (redacted)
echo -e "\n=== Environment ===" >> diagnostics.txt
grep -v PASSWORD .env | grep -v SECRET | grep -v API_KEY >> diagnostics.txt
```

### Report a bug

1. Collect diagnostics (see above)
2. Open [GitHub Issue](https://github.com/fulgidus/heimdall/issues)
3. Include:
   - Error message
   - Steps to reproduce
   - Diagnostics output
   - Expected behavior

---

**Related**: [FAQ](./faqs.md) | [Support](../README.md)
